"""Complaint phrases from WB reviews: phrases of 1–3★ reviews and of 4–5★ reviews with a "но/минус/жаль…" turn,
kept if the sentiment model calls the phrase negative; identical phrases merged (counted per star).

The sentiment model is the WB fine-tuned student (stage 2) if it beat the source student on WB-val, otherwise the
source student. --split test scores the products of WB-test the same way (for the seller report).

Usage: python complaints.py [--split train|test] [--max-positive 400000]
       → data/complaints.parquet (train) | data/complaints_test.parquet, data/complaints_meta.json
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from finetune import predict
from phrases import clauses, norm, polarity

SOURCE = "../ru-review-sentiment/model"
WB = "models/wb_student"
TURN = re.compile(r"\b(но|однако|минус|минусы|жаль|к сожалению|единственное|единственный|недостат|не очень|"
                  r"не понрав|разочар|брак|маломер|большемер)", re.I)
NEG = -0.3  # polarity below this = negative phrase (as polarity_label in the reviews project)


def sentiment_model():
    log = Path(WB) / "train_log.json"
    if log.exists():
        t = json.loads(log.read_text(encoding="utf-8"))
        if t["best_val_f1"] > t["history"][0]["val_f1"]:
            return WB, f"WB-дообученный ученик (val F1 {t['history'][0]['val_f1']} → {t['best_val_f1']})"
    return SOURCE, "исходный ученик ruBert-base (WB-дообученный не лучше на WB-val или ещё не обучен)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--max-positive", type=int, default=400_000)
    p.add_argument("--limit", type=int, help="first N reviews only (quick check)")
    a = p.parse_args()
    wb = pd.read_parquet("data/wb.parquet", columns=["nmId", "stars", "text", "split"])
    wb = wb[wb["split"] == a.split]
    low = wb[wb["stars"] <= 3]
    high = wb[(wb["stars"] >= 4) & wb["text"].str.contains(TURN)]
    if a.split == "train" and len(high) > a.max_positive:
        high = high.sample(a.max_positive, random_state=0)
    reviews = pd.concat([low, high])
    if a.limit:
        reviews = reviews.head(a.limit)
    print(f"reviews: 1–3★ {len(low):,}, 4–5★ with a turn {len(high):,}", flush=True)

    rows = [(r, nm, s, c) for r, nm, s, t in zip(reviews.index, reviews["nmId"], reviews["stars"], reviews["text"])
            for c in clauses(t)]
    ph = pd.DataFrame(rows, columns=["review", "nmId", "stars", "text"])
    ph["norm"] = ph["text"].map(norm)
    ph = ph[ph["norm"].str.len() >= 5]
    print(f"phrases: {len(ph):,}, unique: {ph['norm'].nunique():,}", flush=True)

    uniq = ph.drop_duplicates("norm")[["norm", "text"]].reset_index(drop=True)
    path, desc = sentiment_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device)
    uniq["pol"] = polarity(predict(model, tok, uniq["text"].tolist(), device, batch=256)).round(3)
    neg = set(uniq.loc[uniq["pol"] < NEG, "norm"])

    ph = ph[ph["norm"].isin(neg)]
    if a.split == "test":  # per-product occurrences for the seller report
        ph.merge(uniq[["norm", "pol"]], on="norm").to_parquet("data/complaints_test.parquet")
        print(f"test complaint phrases: {len(ph):,}")
        return
    counts = pd.crosstab(ph["norm"], ph["stars"]).reindex(columns=range(1, 6), fill_value=0)
    counts.columns = [f"s{i}" for i in range(1, 6)]
    out = uniq[uniq["norm"].isin(neg)].set_index("norm").join(counts)
    out["n"] = counts.sum(1).reindex(out.index).values
    out["products"] = ph.groupby("norm")["nmId"].nunique().reindex(out.index).values
    out = out.reset_index()
    out.to_parquet("data/complaints.parquet")
    meta = {"sentiment_model": desc, "reviews_low": int(len(low)), "reviews_turn": int(len(high)),
            "phrases": int(len(rows)), "unique_phrases": int(len(uniq)), "complaints_unique": int(len(out)),
            "complaints_mentions": int(out["n"].sum()), "negative_threshold": NEG}
    Path("data/complaints_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
