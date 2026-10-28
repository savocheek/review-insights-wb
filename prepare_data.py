"""Wildberries reviews → clean data, product-level splits and a statistics report.

Source: nyuuzyou/wb-feedbacks (CC0), two parquet shards of the HF conversion (refs/convert/parquet):
0000 and 0005, ~2.5M reviews; the shards are sorted by nmId, so two distant ones cover different product ranges.
Cleaning: empty and very short texts (< 3 words or < 15 characters), exact duplicate texts, non-Russian texts
(< 60% Cyrillic among letters). Sentiment as in the reviews project: 1–2★ negative, 3★ neutral, 4–5★ positive.
Splits by product: every nmId goes to one split only (80/10/10 by a hash of nmId), so a model never sees reviews
of a test product in training.
Independent test: ai-forever/ru-reviews-classification (Apache-2.0) test split, English reviews dropped.

Usage: python prepare_data.py → data/wb.parquet, data/rureviews_test.parquet, reports/data_stats.md
"""
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

SHARDS = ("0000", "0005")
LABELS = ["negative", "neutral", "positive"]
_LETTER = re.compile(r"[A-Za-zА-Яа-яЁё]")
_CYR = re.compile(r"[А-Яа-яЁё]")


def cyrillic_share(text):
    letters = len(_LETTER.findall(text))
    return len(_CYR.findall(text)) / letters if letters else 0.0


def split_of(nm_id):
    h = int(hashlib.md5(str(nm_id).encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 80 else "val" if h < 90 else "test"


def load_wb():
    parts = []
    for s in SHARDS:
        p = hf_hub_download("nyuuzyou/wb-feedbacks", f"default/partial-train/{s}.parquet", repo_type="dataset",
                            revision="refs/convert/parquet", local_dir="data/raw")
        parts.append(pd.read_parquet(p, columns=["nmId", "productValuation", "text", "answer"]))
    return pd.concat(parts, ignore_index=True)


def clean(df):
    steps = [("исходно", len(df))]
    df["text"] = df["text"].fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    df["answer"] = df["answer"].fillna("").str.strip()
    df = df[df["text"] != ""]
    steps.append(("без пустых", len(df)))
    df = df[(df["text"].str.len() >= 15) & (df["text"].str.split().str.len() >= 3)]
    steps.append(("без очень коротких (< 3 слов или < 15 символов)", len(df)))
    df = df.drop_duplicates("text")
    steps.append(("без дублей текста", len(df)))
    df = df[df["text"].map(cyrillic_share) >= 0.6]
    steps.append(("только русские (≥ 60% кириллицы среди букв)", len(df)))
    df = df[df["productValuation"].between(1, 5)].copy()
    steps.append(("с оценкой 1–5★", len(df)))
    df["stars"] = df["productValuation"].astype(int)
    df["label"] = np.select([df["stars"] <= 2, df["stars"] == 3], [0, 1], 2)
    df["split"] = df["nmId"].map(split_of)
    return df[["nmId", "stars", "label", "text", "answer", "split"]].reset_index(drop=True), steps


def stats(df, steps, rr, rr_dropped):
    md = ["# Данные: отзывы Wildberries\n",
          f"Источник: [nyuuzyou/wb-feedbacks](https://huggingface.co/datasets/nyuuzyou/wb-feedbacks) (CC0), "
          f"шарды {', '.join(SHARDS)} parquet-конвертации HF.\n", "## Очистка\n", "| Шаг | Отзывов |", "|---|---|"]
    md += [f"| {name} | {n:,} |" for name, n in steps]
    n = len(df)
    md += ["\n## Звёзды\n", "| Звёзды | Отзывов | Доля | С ответом продавца | Из них шаблонных |", "|---|---|---|---|---|"]
    ans = df[df["answer"] != ""]
    dup = ans["answer"].map(ans["answer"].value_counts()) > 1  # the same answer text used for ≥ 2 reviews
    for s in range(1, 6):
        d = df[df["stars"] == s]
        a = ans[ans["stars"] == s]
        md.append(f"| {s}★ | {len(d):,} | {len(d) / n:.1%} | {len(a) / max(len(d), 1):.1%} | "
                  f"{dup[a.index].mean() if len(a) else 0:.1%} |")
    md.append(f"| всего | {n:,} | 100% | {len(ans) / n:.1%} | {dup.mean():.1%} |")
    md += ["\nТональность (1–2★ негатив, 3★ нейтрально, 4–5★ позитив): " +
           ", ".join(f"{l} {(df['label'] == i).mean():.1%}" for i, l in enumerate(LABELS)) + ".",
           "«Шаблонный» ответ — текст ответа продавца, который дословно повторяется хотя бы в двух отзывах.\n"]
    top = ans["answer"].value_counts().head(5)
    md += ["Самые частые ответы продавцов:\n"] + [f"- {c:,} × «{' '.join(t.split())[:120]}»" for t, c in top.items()]
    words = df["text"].str.split().str.len()
    md += ["\n## Длина текстов (слов)\n", "| p10 | p25 | медиана | p75 | p90 | p99 |", "|---|---|---|---|---|---|",
           "| " + " | ".join(str(int(words.quantile(q))) for q in (.1, .25, .5, .75, .9, .99)) + " |"]
    per = df[df["nmId"] != 0].groupby("nmId").size()  # nmId 0 = product not specified in the source
    md += ["\n## Товары\n", f"Товаров (nmId): {per.size:,}. Отзывов на товар: медиана {int(per.median())}, "
           f"p90 {int(per.quantile(.9))}, максимум {per.max():,}; товаров с ≥ 50 отзывами: {(per >= 50).sum():,} "
           f"(в них {per[per >= 50].sum() / n:.0%} отзывов). Ещё {(df['nmId'] == 0).sum():,} отзывов с nmId = 0 "
           f"(товар не указан): они одной группой ушли в train.\n",
           "## Сплиты (по товарам: каждый nmId целиком в одном сплите)\n",
           "| Сплит | Товаров | Отзывов | негатив | нейтрально | позитив |", "|---|---|---|---|---|---|"]
    for sp in ("train", "val", "test"):
        d = df[df["split"] == sp]
        md.append(f"| {sp} | {d['nmId'].nunique():,} | {len(d):,} | " +
                  " | ".join(f"{(d['label'] == i).sum():,}" for i in range(3)) + " |")
    md += ["\n## Независимый тест: RuReviews\n",
           f"[ai-forever/ru-reviews-classification](https://huggingface.co/datasets/ai-forever/ru-reviews-classification) "
           f"(Apache-2.0), сплит test: {len(rr) + rr_dropped:,} отзывов, убрано англоязычных: {rr_dropped}, осталось "
           f"{len(rr):,} (" + ", ".join(f"{l} {(rr['label'] == i).sum():,}" for i, l in enumerate(LABELS)) + ")."]
    return "\n".join(md) + "\n"


def main():
    df, steps = clean(load_wb())
    df.to_parquet("data/wb.parquet")
    p = hf_hub_download("ai-forever/ru-reviews-classification", "default/test/0000.parquet", repo_type="dataset",
                        revision="refs/convert/parquet", local_dir="data/raw/rureviews")
    rr = pd.read_parquet(p)
    assert dict(zip(rr["label"], rr["label_text"])) == dict(enumerate(LABELS))
    keep = rr["text"].map(cyrillic_share) >= 0.6
    rr_dropped = int((~keep).sum())
    rr = rr[keep][["text", "label"]].reset_index(drop=True)
    rr.to_parquet("data/rureviews_test.parquet")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/data_stats.md").write_text(stats(df, steps, rr, rr_dropped), encoding="utf-8")
    print(Path("reports/data_stats.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
