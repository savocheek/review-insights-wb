"""Fine-tune the reviews-project student (ruBert-base, distilled from ruRoberta-large) on Wildberries reviews.

Sentiment head only (the 1–5★/place-category heads of the source model are not loaded). Train set is balanced
by class: up to --per-class reviews of each class from WB-train, at most --per-product reviews per product and
class, so a few big products do not dominate. Validation every half epoch on a balanced WB-val sample
(macro-F1); training stops after --patience evaluations without improvement, the best one is kept.

Usage: python finetune.py [--init ../ru-review-sentiment/model] [--out models/wb_student] [--per-class 55000]
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

LABELS = ["negative", "neutral", "positive"]
SEED = 42


def balanced(df, per_class, per_product, seed):
    df = df.sample(frac=1, random_state=seed)
    df = df.groupby(["nmId", "label"], sort=False).head(per_product)
    return pd.concat([g.head(per_class) for _, g in df.groupby("label")]).sample(frac=1, random_state=seed)


@torch.no_grad()
def predict(model, tok, texts, device, batch=128):
    model.eval()
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), 3), dtype=np.float32)
    for i in range(0, len(texts), batch):
        idx = order[i:i + batch]
        enc = tok([texts[j] for j in idx], truncation=True, max_length=256, padding=True, return_tensors="pt").to(device)
        with torch.autocast(device, dtype=torch.float16, enabled=device != "cpu"):
            out[idx] = torch.softmax(model(**enc).logits.float(), -1).cpu().numpy()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", default="../ru-review-sentiment/model")
    p.add_argument("--out", default="models/wb_student")
    p.add_argument("--per-class", type=int, default=55_000)
    p.add_argument("--per-product", type=int, default=30)
    p.add_argument("--val-per-class", type=int, default=5_000)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--patience", type=int, default=2)
    args = p.parse_args()
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wb = pd.read_parquet("data/wb.parquet")
    train = balanced(wb[wb["split"] == "train"], args.per_class, args.per_product, SEED)
    val = balanced(wb[wb["split"] == "val"], args.val_per_class, args.per_product, SEED)
    print("train by class:", train["label"].value_counts().sort_index().tolist(),
          "val by class:", val["label"].value_counts().sort_index().tolist(), flush=True)

    tok = AutoTokenizer.from_pretrained(args.init)
    model = AutoModelForSequenceClassification.from_pretrained(args.init).to(device)
    texts, labels = train["text"].tolist(), torch.tensor(train["label"].values)
    vtexts, vgold = val["text"].tolist(), val["label"].values
    per_epoch = math.ceil(len(texts) / args.batch)
    steps = math.ceil(per_epoch * args.epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=device != "cpu")
    ce = torch.nn.CrossEntropyLoss()

    def val_f1():
        return round(float(f1_score(vgold, predict(model, tok, vtexts, device).argmax(1), average="macro")), 4)

    history = [{"step": 0, "val_f1": val_f1()}]
    print("start", history[0], flush=True)
    best, best_f1, bad, step, t0 = copy.deepcopy(model.state_dict()), history[0]["val_f1"], 0, 0, time.time()
    eval_every = per_epoch // 2
    rng = np.random.default_rng(SEED)
    while step < steps and bad < args.patience:
        order = rng.permutation(len(texts))
        model.train()
        for pos in range(per_epoch):
            if step >= steps or bad >= args.patience:
                break
            idx = order[pos * args.batch:(pos + 1) * args.batch]
            enc = tok([texts[i] for i in idx], truncation=True, max_length=256, padding=True, return_tensors="pt").to(device)
            with torch.autocast(device, dtype=torch.float16, enabled=device != "cpu"):
                loss = ce(model(**enc).logits.float(), labels[idx].to(device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt), scaler.update(), sched.step()
            step += 1
            if step % 500 == 0:
                print(f"step {step}/{steps} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
            if step % eval_every == 0 or step == steps:
                rec = {"step": step, "epoch": round(step / per_epoch, 2), "val_f1": val_f1()}
                history.append(rec)
                print("eval", rec, flush=True)
                if rec["val_f1"] > best_f1:
                    best_f1, bad = rec["val_f1"], 0
                    best = copy.deepcopy(model.state_dict())
                else:
                    bad += 1
                model.train()

    model.load_state_dict(best)
    out = Path(args.out)
    model.save_pretrained(out), tok.save_pretrained(out)
    (out / "train_log.json").write_text(json.dumps(
        {"init": args.init, "train_by_class": train["label"].value_counts().sort_index().tolist(),
         "per_product": args.per_product, "best_val_f1": best_f1, "history": history,
         "minutes": round((time.time() - t0) / 60, 1)}, indent=1), encoding="utf-8")
    print("best val F1", best_f1, flush=True)


if __name__ == "__main__":
    main()
