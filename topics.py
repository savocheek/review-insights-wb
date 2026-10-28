"""Groups of frequent customer problems: complaint phrases → sentence embeddings → clusters → report.

Input: data/complaints.parquet (complaints.py). Embeddings of a ready Russian sentence encoder (mean pooling,
L2-normalised, the encoder's own task prefix), KMeans on them. For every group: c-TF-IDF keywords, 5 typical
phrases (closest to the centre), share of all complaints (counted with duplicates), stars, cohesion (mean cosine
to the centre); the least cohesive groups are flagged as mixed. Names of groups are drafts until the user
approves them in topics_label.py.

Usage: python topics.py [--encoder NAME] [--k 50] → data/topics.parquet, data/topic_centers.npy,
       reports/topics.json, reports/topics.md
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from transformers import AutoModel, AutoTokenizer

ENCODERS = {"deepvk/USER-base": "query: ", "sergeyzh/rubert-mini-frida": "categorize: "}
LABELS_FILE = Path("data/topic_labels.jsonl")
DRAFTS_FILE = Path("reports/topic_drafts.json")
STOP = set("""и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было вот
от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас нибудь опять уж вам ведь
там потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже
себе под будет ж тогда кто этот того потому этого какой совсем ним здесь этом один почти мой тем чтобы нее сейчас были
куда зачем всех никогда можно при наконец два об другой хоть после над больше тот через эти нас про всего них какая
много разве три эту моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им более всегда конечно всю
между это очень просто вообще который которые которая которое весь вся всё свой своей своих также ещё есть""".split())
_TOKEN = re.compile(r"[а-яё]{3,}")


@torch.no_grad()
def embed(texts, name, device, batch=256, max_length=64):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(device).eval()
    prefix = ENCODERS.get(name, "")
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    for i in range(0, len(texts), batch):
        idx = order[i:i + batch]
        enc = tok([prefix + texts[j] for j in idx], truncation=True, max_length=max_length, padding=True,
                  return_tensors="pt").to(device)
        with torch.autocast(device, dtype=torch.float16, enabled=device != "cpu"):
            h = model(**enc).last_hidden_state.float()
        m = enc["attention_mask"].unsqueeze(-1).float()
        v = (h * m).sum(1) / m.sum(1)
        out[idx] = torch.nn.functional.normalize(v, dim=-1).cpu().numpy()
    return out


def tokens(text):
    return [w for w in _TOKEN.findall(text.lower().replace("ё", "е")) if w not in STOP]


def ctfidf(docs, labels, weights, k, top=8):
    """Class-based TF-IDF (BERTopic): a group's term frequency × log(1 + mean words per group / term frequency)."""
    tf = [Counter() for _ in range(k)]
    for d, c, w in zip(docs, labels, weights):
        for t in set(tokens(d)):
            tf[c][t] += w
    total = Counter()
    for c in tf:
        total.update(c)
    avg = sum(sum(c.values()) for c in tf) / k
    out = []
    for c in tf:
        n = sum(c.values()) or 1
        score = {t: (f / n) * np.log(1 + avg / total[t]) for t, f in c.items() if f >= 3}
        out.append([t for t, _ in sorted(score.items(), key=lambda x: -x[1])[:top]])
    return out


def cluster(X, k, seed=0, fit_max=200_000):
    rng = np.random.default_rng(seed)
    fit = X[rng.choice(len(X), min(len(X), fit_max), replace=False)]
    km = KMeans(k, n_init=3, random_state=seed).fit(fit)
    centers = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    labels = np.concatenate([(X[i:i + 50_000] @ centers.T).argmax(1) for i in range(0, len(X), 50_000)])
    return labels, centers.astype(np.float32)


def describe(df, X, labels, centers):
    k = len(centers)
    sims = (X * centers[labels]).sum(1)
    kw = ctfidf(df["text"].tolist(), labels, df["n"].values, k)
    total = df["n"].sum()
    stars = df[[f"s{i}" for i in range(1, 6)]].values
    groups = []
    for c in range(k):
        idx = np.where(labels == c)[0]
        top = idx[np.argsort(-sims[idx])]
        ex, seen = [], set()
        for i in top:
            key = df["norm"].iat[i]
            if key not in seen:
                seen.add(key), ex.append(df["text"].iat[i])
            if len(ex) == 5:
                break
        st = stars[idx].sum(0)
        groups.append({"id": c, "phrases": int(len(idx)), "mentions": int(df["n"].values[idx].sum()),
                       "share": round(float(df["n"].values[idx].sum() / total), 4),
                       "stars": [round(float(x), 3) for x in st / max(st.sum(), 1)],
                       "cohesion": round(float(sims[idx].mean()), 3), "keywords": kw[c], "examples": ex})
    cut = np.quantile([g["cohesion"] for g in groups], 0.15)
    for g in groups:
        g["mixed"] = bool(g["cohesion"] <= cut)
    return sorted(groups, key=lambda g: -g["mentions"])


def decisions():
    """Last user decision per group from topics_label.py: {id: {"action": name|merge|junk, ...}}."""
    out = {}
    if LABELS_FILE.exists():
        for line in LABELS_FILE.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            out[r["id"]] = r
    return out


def resolve(groups):
    """Group id → (name, approved). Merges follow the chain to the target group; junk → None."""
    dec = decisions()
    drafts = {int(k): v for k, v in json.loads(DRAFTS_FILE.read_text(encoding="utf-8")).items()} if DRAFTS_FILE.exists() else {}
    out = {}
    for g in groups:
        c, hops = g["id"], 0
        while dec.get(c, {}).get("action") == "merge" and hops < 100:
            c, hops = dec[c]["into"], hops + 1
        d = dec.get(c, {})
        if dec.get(g["id"], {}).get("action") == "junk" or d.get("action") == "junk":
            out[g["id"]] = (None, True)
        elif d.get("action") == "name":
            out[g["id"]] = (d["name"], True)
        else:
            out[g["id"]] = (drafts.get(c) or " / ".join(next(x for x in groups if x["id"] == c)["keywords"][:3]), False)
    return out


def write_md(meta=None):
    info = json.loads(Path("reports/topics.json").read_text(encoding="utf-8"))
    groups, meta = info["groups"], meta or info["meta"]
    names = resolve(groups)
    approved = sum(1 for g in groups if names[g["id"]][1])
    md = ["# Частые проблемы покупателей Wildberries\n",
          f"Жалобы — негативные фразы из отзывов WB-train: {meta['phrases']:,} уникальных фраз, {meta['mentions']:,} "
          f"упоминаний с повторами. Модель тональности фраз: {meta['sentiment_model']}. Эмбеддинги: {meta['encoder']}, "
          f"кластеризация: {meta['method']}.\n",
          f"**Названия групп — черновик**, пока их не утвердит человек в `topics_label.py` "
          f"(утверждено {approved} из {len(groups)}). Черновое название — от модели-помощника или три ключевых слова; "
          f"«смешанная» — 15% самых неоднородных групп (низкое среднее сходство с центром).\n",
          "| # | Группа | Доля жалоб | Фраз | 1★ / 2★ / 3★ / 4★ / 5★ | Ключевые слова |", "|---|---|---|---|---|---|"]
    for g in groups:
        name, ok = names[g["id"]]
        if name is None:
            continue
        label = name + ("" if ok else " *(черновик)*") + (" · смешанная" if g["mixed"] else "")
        md.append(f"| {g['id']} | {label} | {g['share']:.1%} | {g['phrases']:,} | "
                  + " / ".join(f"{x:.0%}" for x in g["stars"]) + f" | {', '.join(g['keywords'][:6])} |")
    md.append("\n## Примеры (5 фраз, ближайших к центру группы)\n")
    for g in groups:
        name, ok = names[g["id"]]
        if name is None:
            continue
        md += [f"**{g['id']}. {name}**{'' if ok else ' (черновик)'} — {g['share']:.1%}, сходство {g['cohesion']}\n"]
        md += [f"- {e}" for e in g["examples"]] + [""]
    junk = [g["id"] for g in groups if names[g["id"]][0] is None]
    if junk:
        md.append(f"Выброшены как мусор: {', '.join(map(str, junk))}.")
    Path("reports/topics.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder")
    p.add_argument("--k", type=int)
    a = p.parse_args()
    probe = json.loads(Path("reports/topics_probe.json").read_text(encoding="utf-8"))
    encoder, k = a.encoder or probe["choice"]["encoder"], a.k or probe["choice"]["k"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_parquet("data/complaints.parquet")
    cache = Path(f"data/complaint_emb_{encoder.split('/')[-1]}.npy")
    if cache.exists():
        X = np.load(cache).astype(np.float32)
    else:
        X = embed(df["text"].tolist(), encoder, device)
        np.save(cache, X.astype(np.float16))
    labels, centers = cluster(X, k)
    groups = describe(df, X, labels, centers)
    df.assign(topic=labels)[["norm", "topic"]].to_parquet("data/topics.parquet")
    np.save("data/topic_centers.npy", centers)
    meta = {"phrases": int(len(df)), "mentions": int(df["n"].sum()), "encoder": encoder, "k": k,
            "method": f"KMeans, k = {k}", "sentiment_model": json.loads(Path("data/complaints_meta.json").read_text(
                encoding="utf-8"))["sentiment_model"]}
    Path("reports/topics.json").write_text(json.dumps({"meta": meta, "groups": groups}, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    write_md(meta)
    print(f"{len(groups)} groups; mixed: {[g['id'] for g in groups if g['mixed']]}")


if __name__ == "__main__":
    main()
