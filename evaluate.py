"""Sentiment quality on Wildberries (full split, product-level) and RuReviews test: macro-F1, accuracy, confusion.

Any sentiment model works (the source project's multi-task models too: only their sentiment output is used).
With --onnx the model's int8 ONNX is scored on CPU instead (WB on a stratified 30k sample, RuReviews in full).
Results are appended to reports/eval.jsonl and the table is rebuilt in reports/eval.md.

Usage: python evaluate.py MODEL_DIR [MODEL_DIR …] [--split test|val] [--onnx PATH --name NAME]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from finetune import LABELS, predict

RU = ["негатив", "нейтр.", "позитив"]


def datasets(split, sample=None):
    wb = pd.read_parquet("data/wb.parquet")
    wb = wb[wb["split"] == split]
    if sample:
        wb = wb.groupby("label", group_keys=False).apply(lambda g: g.sample(min(len(g), sample // 3), random_state=0))
    out = {f"WB-{split}": (wb["text"].tolist(), wb["label"].values)}
    if split == "test":
        rr = pd.read_parquet("data/rureviews_test.parquet")
        out["RuReviews-test"] = (rr["text"].tolist(), rr["label"].values)
    return out


def onnx_predict(path, tok, texts, batch=64):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 8
    sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), 3), dtype=np.float32)
    for i in range(0, len(texts), batch):
        idx = order[i:i + batch]
        e = tok([texts[j] for j in idx], truncation=True, max_length=256, padding=True, return_tensors="np")
        feed = {k: e[k].astype(np.int64) for k in ("input_ids", "attention_mask")}
        feed["token_type_ids"] = e.get("token_type_ids", np.zeros_like(e["input_ids"])).astype(np.int64)
        out[idx] = sess.run(["sentiment"], feed)[0]
    return out


def metrics(gold, pred):
    return {"f1_macro": round(float(f1_score(gold, pred, average="macro")), 4),
            "accuracy": round(float(accuracy_score(gold, pred)), 4),
            "f1_per_class": [round(float(x), 4) for x in f1_score(gold, pred, average=None)],
            "confusion": confusion_matrix(gold, pred, labels=[0, 1, 2]).tolist(), "n": int(len(gold))}


def report():
    rows = [json.loads(x) for x in Path("reports/eval.jsonl").read_text(encoding="utf-8").splitlines()]
    md = ["# Тональность на товарах\n", "| Модель | Набор | n | macro-F1 | accuracy | F1 негатив / нейтр. / позитив |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['data']} | {r['n']:,} | {r['f1_macro']:.3f} | {r['accuracy']:.3f} | "
                  + " / ".join(f"{x:.3f}" for x in r["f1_per_class"]) + " |")
    md.append("\n## Матрицы ошибок (строки — правда, столбцы — ответ модели)\n")
    for r in rows:
        md += [f"**{r['model']} · {r['data']}**\n", "| | " + " | ".join(RU) + " |", "|---|---|---|---|"]
        md += [f"| {RU[i]} | " + " | ".join(f"{x:,}" for x in row) + " |" for i, row in enumerate(r["confusion"])]
        md.append("")
    Path("reports/eval.md").write_text("\n".join(md), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+")
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--onnx")
    p.add_argument("--name")
    a = p.parse_args()
    Path("reports").mkdir(exist_ok=True)
    data = datasets(a.split, sample=30_000 if a.onnx else None)
    device = "cuda" if torch.cuda.is_available() and not a.onnx else "cpu"
    for m in a.models:
        tok = AutoTokenizer.from_pretrained(m)
        if a.onnx:
            run = lambda texts: onnx_predict(a.onnx, tok, texts)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(m).to(device)
            run = lambda texts: predict(model, tok, texts, device)
        for name, (texts, gold) in data.items():
            r = {"model": a.name or Path(m).name, "data": name, **metrics(gold, run(texts).argmax(1))}
            print(json.dumps(r, ensure_ascii=False), flush=True)
            with open("reports/eval.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if not a.onnx:
            del model
            torch.cuda.empty_cache()
    report()


if __name__ == "__main__":
    main()
