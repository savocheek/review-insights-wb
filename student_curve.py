"""Student learning curve on the teacher's labels: does quality still grow from 10k to ~18k reviews?

Data: pass a of the teacher (Qwen3-14B, prompt v4) on the 20k WB-train corpus. ~2k reviews are held out by
product (nmId — no product is both in test and train) as the teacher test, ~1k more by product as validation
for early stopping. The rest is shuffled once; the points are nested subsets of it (5k ⊂ 10k ⊂ all).

Student: the ruBert-base student of ru-review-sentiment (same init for every point), three heads on the
[CLS] vector — need (binary), reasons (8, multi-label), sentiment (4). Validation every half epoch on
need F1 + reasons micro-F1, stop after --patience evaluations without improvement, the best one is kept.
The need threshold is picked on validation; reasons use 0.5.

Metrics per point: need F1/precision/recall and reasons micro/macro-F1 against the held-out teacher labels,
sentiment accuracy, need against the user's hand labels (51 — few) and training time.

Usage: python student_curve.py [--sizes 5000,10000,all] [--epochs 4] [--cpu-smoke]
Writes data/student_curve/results.jsonl (local) and reports/student_curve.md.
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from llm_probe import REASON_KEYS, REASON_LIST_V2, SENTIMENTS, gold_labels, need_of, reasons_of

SEED = 42
D = Path("data/student_curve")
REPORT = Path("reports/student_curve.md")


def jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def load():
    corpus = {r["id"]: r for r in jsonl("data/teacher/corpus.jsonl")}
    rows = []
    for r in jsonl("data/teacher/label_a.jsonl"):
        ans = r.get("answer")
        if not ans:
            continue
        reasons = {x["reason"] for x in reasons_of(ans)}
        it = corpus[r["id"]]
        rows.append({"id": r["id"], "nmId": it["nmId"], "text": it["text"], "need": float(need_of(ans)),
                     "reasons": [float(k in reasons) for k in REASON_LIST_V2],
                     "sentiment": SENTIMENTS.index(ans["sentiment"])})
    return rows


def split_by_product(rows, n_test, n_val, rng):
    by = {}
    for r in rows:
        by.setdefault(r["nmId"], []).append(r)
    products = list(by)
    rng.shuffle(products)
    test, val, rest = [], [], []
    for p in products:
        (test if len(test) < n_test else val if len(val) < n_val else rest).extend(by[p])
    rng.shuffle(rest)
    return test, val, rest


class Student(torch.nn.Module):
    def __init__(self, init):
        super().__init__()
        self.enc = AutoModel.from_pretrained(init)
        h = self.enc.config.hidden_size
        self.drop = torch.nn.Dropout(0.1)
        self.need = torch.nn.Linear(h, 1)
        self.reasons = torch.nn.Linear(h, len(REASON_LIST_V2))
        self.sent = torch.nn.Linear(h, len(SENTIMENTS))

    def forward(self, **enc):
        x = self.drop(self.enc(**enc).last_hidden_state[:, 0])
        return self.need(x).squeeze(-1), self.reasons(x), self.sent(x)


@torch.no_grad()
def predict(model, tok, texts, device, batch=128):
    model.eval()
    order = np.argsort([len(t) for t in texts])
    need = np.zeros(len(texts), dtype=np.float32)
    rea = np.zeros((len(texts), len(REASON_LIST_V2)), dtype=np.float32)
    sent = np.zeros(len(texts), dtype=np.int64)
    for i in range(0, len(texts), batch):
        idx = order[i:i + batch]
        enc = tok([texts[j] for j in idx], truncation=True, max_length=256, padding=True, return_tensors="pt").to(device)
        with torch.autocast(device, dtype=torch.float16, enabled=device != "cpu"):
            n, r, s = model(**enc)
        need[idx] = torch.sigmoid(n.float()).cpu().numpy()
        rea[idx] = torch.sigmoid(r.float()).cpu().numpy()
        sent[idx] = s.float().argmax(-1).cpu().numpy()
    return need, rea, sent


def need_scores(gold, pred):
    return {"f1": round(float(f1_score(gold, pred, zero_division=0)), 3),
            "precision": round(float(precision_score(gold, pred, zero_division=0)), 3),
            "recall": round(float(recall_score(gold, pred, zero_division=0)), 3)}


def best_threshold(gold, proba):
    return max(np.arange(0.1, 0.91, 0.05), key=lambda t: f1_score(gold, proba >= t, zero_division=0))


def train_point(train, val, args, device):
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(args.init)
    model = Student(args.init).to(device)
    texts = [r["text"] for r in train]
    y_need = torch.tensor([r["need"] for r in train])
    y_rea = torch.tensor([r["reasons"] for r in train])
    y_sent = torch.tensor([r["sentiment"] for r in train])
    vtexts = [r["text"] for r in val]
    v_need = np.array([r["need"] for r in val])
    v_rea = np.array([r["reasons"] for r in val])
    per_epoch = math.ceil(len(texts) / args.batch)
    steps = math.ceil(per_epoch * args.epochs)
    heads = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": args.lr},
                             {"params": heads, "lr": args.lr * 20}], weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=device != "cpu")
    bce, ce = torch.nn.BCEWithLogitsLoss(), torch.nn.CrossEntropyLoss()

    def val_score():
        n, r, _ = predict(model, tok, vtexts, device)
        need_f1 = f1_score(v_need, n >= best_threshold(v_need, n), zero_division=0)
        return round(float((need_f1 + f1_score(v_rea, r >= 0.5, average="micro", zero_division=0)) / 2), 4)

    best, best_v, bad, step, t0, history = None, -1.0, 0, 0, time.time(), []
    eval_every = max(per_epoch // 2, 1)
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
                n, r, s = model(**enc)
            loss = (bce(n.float(), y_need[idx].to(device)) + bce(r.float(), y_rea[idx].to(device))
                    + ce(s.float(), y_sent[idx].to(device)))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt), scaler.update(), sched.step()
            step += 1
            if step % eval_every == 0 or step == steps:
                v = val_score()
                history.append({"step": step, "epoch": round(step / per_epoch, 2), "val": v})
                print(f"  step {step}/{steps} loss {loss.item():.3f} val {v} {time.time() - t0:.0f}s", flush=True)
                if v > best_v:
                    best_v, bad, best = v, 0, copy.deepcopy(model.state_dict())
                else:
                    bad += 1
                model.train()
    model.load_state_dict(best)
    n, _, _ = predict(model, tok, vtexts, device)
    return model, tok, float(best_threshold(v_need, n)), history, round((time.time() - t0) / 60, 1)


def evaluate(model, tok, thr, test, user, device):
    n, r, s = predict(model, tok, [x["text"] for x in test], device)
    g_rea = np.array([x["reasons"] for x in test])
    out = {"need": need_scores(np.array([x["need"] for x in test]), n >= thr),
           "reasons_micro_f1": round(float(f1_score(g_rea, r >= 0.5, average="micro", zero_division=0)), 3),
           "reasons_macro_f1": round(float(f1_score(g_rea, r >= 0.5, average="macro", zero_division=0)), 3),
           "sentiment_acc": round(float(np.mean(s == np.array([x["sentiment"] for x in test]))), 3)}
    if user:
        un, ur, _ = predict(model, tok, [x["text"] for x in user], device)
        ug = np.array([x["need"] for x in user])
        out["user_need"] = need_scores(ug, un >= thr) | {"accuracy": round(float(np.mean((un >= thr) == ug)), 3),
                                                         "n": len(user), "positives": int(ug.sum())}
        ugr = np.array([x["reasons"] for x in user])
        out["user_reasons_micro_f1"] = round(float(f1_score(ugr, ur >= 0.5, average="micro", zero_division=0)), 3)
    return out


def user_items():
    texts = {str(r["id"]): r["text"] for r in jsonl("data/needs_reply_items.jsonl")}
    return [{"id": g["id"], "text": texts[g["id"]], "need": float(g["need"]),
             "reasons": [float(k in g["reasons"]) for k in REASON_KEYS]}
            for g in gold_labels() if g["id"] in texts]


def report(results, meta):
    md = ["# Кривая обучения студента на метках учителя\n",
          f"Вопрос: растёт ли качество студента заметно при переходе с 10k к ~{meta['pool'] // 1000}k меткам учителя, "
          "то есть есть ли смысл размечать 20k и больше.\n",
          f"- Метки: проход a учителя (Qwen3-14B, промпт v4, без размышлений) — {meta['labelled']:,} валидных ответов из 20k.",
          f"- Тест учителя: {meta['test']:,} отзывов, отложены по товарам (nmId) — ни один товар теста не попал в обучение; "
          f"валидация для ранней остановки — ещё {meta['val']:,} по товарам. Обучающий пул — {meta['pool']:,}, точки — вложенные подвыборки (5k ⊂ 10k ⊂ всё).",
          f"- Студент: ruBert-base из ru-review-sentiment, одинаковый старт для всех точек; головы need + причины (8, multi-label) + тональность (4); "
          f"до {meta['epochs']} эпох, проверка каждые полэпохи, остановка после {meta['patience']} проверок без улучшения. "
          "Порог need подбирается на валидации, причины — 0.5.",
          f"- Метки пользователя: {meta['user']} отзывов, need = да у {meta['user_pos']} — **очень мало, один отзыв меняет recall на "
          f"{100 / max(meta['user_pos'], 1):.0f} п.п.**; ориентир, не метрика.\n",
          "| Обучение | need F1 | precision | recall | причины micro-F1 | macro-F1 | тональность acc | need с пользователем: acc / P / R | время, мин | эпох до лучшей |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        m, u = r["test"], r["test"].get("user_need", {})
        best_ep = max(r["history"], key=lambda h: h["val"])["epoch"] if r["history"] else "—"
        md.append(f"| {r['n']:,} | {m['need']['f1']:.3f} | {m['need']['precision']:.3f} | {m['need']['recall']:.3f} | "
                  f"{m['reasons_micro_f1']:.3f} | {m['reasons_macro_f1']:.3f} | {m['sentiment_acc']:.3f} | "
                  f"{u.get('accuracy', 0):.0%} / {u.get('precision', 0):.0%} / {u.get('recall', 0):.0%} | {r['minutes']} | {best_ep} |")
    if meta.get("conclusion"):
        md += ["\n## Вывод\n", meta["conclusion"]]
    REPORT.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("wrote", REPORT)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", default="../ru-review-sentiment/model")
    p.add_argument("--sizes", default="5000,10000,all")
    p.add_argument("--test", type=int, default=2000)
    p.add_argument("--val", type=int, default=1000)
    p.add_argument("--epochs", type=float, default=4)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--cpu-smoke", action="store_true", help="tiny run on the CPU (8 threads) to check the code")
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args()
    D.mkdir(parents=True, exist_ok=True)
    out = D / ("smoke.jsonl" if args.cpu_smoke else "results.jsonl")
    if args.report_only:
        meta = json.loads((D / "meta.json").read_text(encoding="utf-8"))
        return report(jsonl(out), meta)
    device = "cuda" if torch.cuda.is_available() and not args.cpu_smoke else "cpu"
    torch.set_num_threads(8)

    rows = load()
    test, val, pool = split_by_product(rows, args.test, args.val, np.random.default_rng(SEED))
    user = user_items()
    if args.cpu_smoke:
        test, val, pool, user, args.epochs, args.batch = test[:16], val[:16], pool[:32], user[:8], 1, 8
        sizes = [16, len(pool)]
    else:
        sizes = [len(pool) if s == "all" else int(s) for s in args.sizes.split(",")]
    meta = {"labelled": len(rows), "test": len(test), "val": len(val), "pool": len(pool), "epochs": args.epochs,
            "patience": args.patience, "user": len(user), "user_pos": int(sum(x["need"] for x in user))}
    print("device", device, meta, flush=True)
    meta_path = D / "meta.json"
    if not args.cpu_smoke:
        old = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        meta_path.write_text(json.dumps(old | meta, ensure_ascii=False, indent=1), encoding="utf-8")
    out.write_text("", encoding="utf-8")
    results = []
    for n in sizes:
        print(f"== {n} reviews", flush=True)
        model, tok, thr, history, minutes = train_point(pool[:n], val, args, device)
        rec = {"n": n, "threshold": round(thr, 2), "minutes": minutes, "history": history,
               "test": evaluate(model, tok, thr, test, user, device)}
        print(json.dumps(rec["test"], ensure_ascii=False), flush=True)
        results.append(rec)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    if not args.cpu_smoke:
        report(results, meta)


if __name__ == "__main__":
    main()
