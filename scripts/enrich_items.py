"""150 "suspicious" reviews for hand labelling, picked by the student's need score (the user's request, 30.10).

Pool: data/wb.parquet without the 20k teacher corpus, the 500 needs_reply items and every review in the LLM probes;
text 20-1000 characters, no duplicate texts. A random 300k of it goes through the student (data/student/model).
Pick: 75 with score >= 0.85 and 75 with 0.3-0.85, spread evenly over the range, at most 2 per product.
Draft = the student's prediction (need at the validation threshold, reasons >= 0.5).
Writes labeling/needs_reply_items_enriched.json in the format of needs_reply_items.json.
Usage (from the repo root): python scripts/enrich_items.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

sys.path.insert(0, ".")
from llm_probe import REASON_KEYS  # noqa: E402
from student_curve import SEED, Student, jsonl, predict  # noqa: E402

M = Path("data/student/model")
OUT = Path("labeling/needs_reply_items_enriched.json")
SAMPLE, PER_PRODUCT = 300_000, 2
BINS = [(np.linspace(0.85, 1.0001, 4), 25), (np.linspace(0.3, 0.85, 6), 15)]  # (bin edges, reviews per bin): 75 + 75


def excluded():
    ids = {r["id"] for r in jsonl("data/teacher/corpus.jsonl")} | {r["id"] for r in jsonl("data/needs_reply_items.jsonl")}
    for p in Path("data/llm_probe").glob("*.jsonl"):
        ids |= {r["id"] for r in jsonl(p) if "id" in r}
    return {int(i) for i in ids}


def main():
    torch.set_num_threads(8)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wb = pd.read_parquet("data/wb.parquet", columns=["nmId", "stars", "text"])
    skip = excluded()
    seen_texts = set(wb.loc[wb.index.isin(skip), "text"].str.strip())
    wb = wb[~wb.index.isin(skip) & wb["text"].str.len().between(20, 1000)]
    wb = wb.assign(t=wb["text"].str.strip())
    wb = wb[~wb["t"].isin(seen_texts)].drop_duplicates("t", keep=False)  # a text seen twice is likely a template
    pool = wb.sample(min(SAMPLE, len(wb)), random_state=SEED)
    print(f"device {device}; excluded ids {len(skip):,}; pool {len(wb):,}; scored {len(pool):,}", flush=True)

    model = Student(str(M))
    model.load_state_dict(torch.load(M / "heads.pt"), strict=False)
    model.to(device)
    tok = AutoTokenizer.from_pretrained(M)
    need, rea, _ = predict(model, tok, pool["text"].tolist(), device, batch=256)
    pool = pool.assign(score=need)
    thr = json.loads(Path("data/student/results.json").read_text(encoding="utf-8"))["threshold"]
    keys = list(REASON_KEYS)
    pos = {i: k for k, i in enumerate(pool.index)}

    rng, per_product, picked = np.random.default_rng(SEED), {}, []
    for edges, n in BINS:
        for lo, hi in zip(edges[:-1], edges[1:]):
            cand = pool[(pool["score"] >= lo) & (pool["score"] < hi)]
            took = 0
            for i in rng.permutation(cand.index.to_numpy()):
                nm = cand.at[i, "nmId"]
                if per_product.get(nm, 0) >= PER_PRODUCT:
                    continue
                per_product[nm] = per_product.get(nm, 0) + 1
                picked.append(i)
                took += 1
                if took == n:
                    break
            print(f"score {lo:.2f}-{min(hi, 1):.2f}: {len(cand):,} in the pool, took {took}")
    items = [{"id": int(i), "text": pool.at[i, "text"], "stars": int(pool.at[i, "stars"]), "nmId": int(pool.at[i, "nmId"]),
              "draft": {"needs_reply": bool(pool.at[i, "score"] >= thr),
                        "reasons": [keys[j] for j in range(len(keys)) if rea[pos[i]][j] >= 0.5]},
              "batch": "enriched", "score": round(float(pool.at[i, "score"]), 3)}
             for i in rng.permutation(picked)]
    base = json.loads(Path("labeling/needs_reply_items.json").read_text(encoding="utf-8"))
    assert not {x["id"] for x in items} & {x["id"] for x in base["items"]} and len({x["id"] for x in items}) == len(items)
    OUT.write_text(json.dumps({"reasons": base["reasons"], "none": base["none"], "items": items}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"wrote {OUT}: {len(items)} reviews, {sum(x['score'] >= 0.85 for x in items)} with score >= 0.85")


if __name__ == "__main__":
    main()
