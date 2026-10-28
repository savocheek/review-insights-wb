"""Small probe before the full run: which sentence encoder and which clustering give meaningful complaint groups.

10k phrases of 1–2★ WB-train reviews (CPU, in the background; no sentiment filter here — scoring them with the
sentiment model on CPU took longer than the probe itself). For each encoder: speed; agreement with rough keyword
groups (size, smell, threads/seams, colour, wrong item, defect, packaging, delivery) — share of a phrase's 10 nearest
neighbours from the same group, and NMI of the clustering with these groups; cluster keywords for a manual look.
Clusterings: KMeans k = 40/50/60 and HDBSCAN (sklearn) on PCA-50.

Usage: python topics_probe.py → reports/topics_probe.md, reports/topics_probe.json (the choice for topics.py)
"""
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score

from phrases import clauses, norm
from topics import ENCODERS, cluster, ctfidf, embed

torch.set_num_threads(8)
PHRASES = 10_000
GROUPS = {"размер": r"маломер|большемер|размер|мал[аоы]? |велик[аио]? ", "запах": r"запах|воня|пахн|вонь",
          "нитки и швы": r"нитк|шов|швы|строчк|распорол|расходит", "цвет": r"цвет|оттен|линял|полинял|красит",
          "не тот товар": r"не тот|не то |другой товар|друг(ой|ую|ое) цвет|прислали не|пересорт",
          "брак": r"брак|дыр|сломан|слома|трещин|не работа", "упаковка": r"упаков|коробк|пакет|помят",
          "доставка": r"доставк|курьер|пвз|пункт выдачи|опозда|долго ехал"}


def keyword_group(text):
    hits = [g for g, rx in GROUPS.items() if re.search(rx, text.lower())]
    return hits[0] if len(hits) == 1 else None


def main():
    wb = pd.read_parquet("data/wb.parquet", columns=["stars", "text", "split"])
    rev = wb[(wb["split"] == "train") & (wb["stars"] <= 2)].sample(6000, random_state=1)
    ph = pd.DataFrame({"text": [c for t in rev["text"] for c in clauses(t)]})
    ph["norm"] = ph["text"].map(norm)
    ph = ph.drop_duplicates("norm").sample(frac=1, random_state=1).head(PHRASES).reset_index(drop=True)
    grp = ph["text"].map(keyword_group)
    lab = grp.notna().values
    gid = pd.factorize(grp[lab])[0]
    print(f"probe phrases: {len(ph):,}, with one keyword group: {lab.sum():,}", flush=True)

    res, md_clusters = [], []
    for enc in ENCODERS:
        t0 = time.time()
        X = embed(ph["text"].tolist(), enc, "cpu", batch=128)
        speed = len(ph) / (time.time() - t0)
        L = X[lab]
        sim = L @ L.T
        np.fill_diagonal(sim, -1)
        nn = np.argsort(-sim, 1)[:, :10]
        knn = float((gid[nn] == gid[:, None]).mean())
        runs = {}
        for k in (40, 50, 60):
            runs[f"KMeans k={k}"] = cluster(X, k)[0]
        Z = PCA(50, random_state=0).fit_transform(X)
        runs["HDBSCAN (PCA-50, min 30)"] = HDBSCAN(min_cluster_size=30, min_samples=10).fit_predict(Z)
        for m, labels in runs.items():
            noise = float((labels == -1).mean())
            k = int(labels.max()) + 1
            res.append({"encoder": enc, "method": m, "phrases_per_s_cpu": round(speed), "knn10_same_group": round(knn, 3),
                        "nmi_keyword_groups": round(float(normalized_mutual_info_score(gid, labels[lab])), 3),
                        "clusters": k, "noise": round(noise, 3)})
            print(json.dumps(res[-1], ensure_ascii=False), flush=True)
            if m == "KMeans k=50" or m.startswith("HDBSCAN"):
                keep = labels >= 0
                kw = ctfidf(ph["text"][keep].tolist(), labels[keep], np.ones(keep.sum()), k, top=5)
                sizes = np.bincount(labels[keep], minlength=k)
                md_clusters += [f"\n**{enc} · {m}** (по убыванию размера)\n"]
                md_clusters += [f"- {sizes[c]}: {', '.join(kw[c])}" for c in np.argsort(-sizes)[:25]]

    best = max((r for r in res if r["method"].startswith("KMeans")), key=lambda r: (r["nmi_keyword_groups"], r["knn10_same_group"]))
    choice = {"encoder": best["encoder"], "k": int(best["method"].split("=")[1])}
    Path("reports/topics_probe.json").write_text(json.dumps({"runs": res, "choice": choice}, ensure_ascii=False, indent=1),
                                                 encoding="utf-8")
    md = ["# Проба: энкодер и кластеризация жалоб\n",
          f"{len(ph):,} фраз из отзывов 1–2★ (WB-train, без фильтра тональности), CPU. Грубые группы по ключевым словам "
          f"(размер, запах, нитки, цвет, не тот товар, брак, упаковка, доставка) есть у {lab.sum():,} фраз: по ним "
          "считаются доля 10 ближайших соседей из той же группы и NMI кластеров с группами. Это грубая проверка "
          "(ключевые слова сами лексические), поэтому ниже ещё ключевые слова кластеров — для просмотра глазами.\n",
          "| Энкодер | Кластеризация | фраз/с (CPU) | kNN-10 той же группы | NMI с группами | Кластеров | Шум |",
          "|---|---|---|---|---|---|---|"]
    md += [f"| {r['encoder']} | {r['method']} | {r['phrases_per_s_cpu']} | {r['knn10_same_group']} | "
           f"{r['nmi_keyword_groups']} | {r['clusters']} | {r['noise']:.0%} |" for r in res]
    md += [f"\nВыбор для полного прогона (по NMI среди KMeans): **{choice['encoder']}, KMeans k = {choice['k']}**.\n"]
    Path("reports/topics_probe.md").write_text("\n".join(md + md_clusters) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
