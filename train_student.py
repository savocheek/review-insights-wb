"""The student on all of the teacher's final labels (after the checker): ruBert-base, need + reasons + sentiment.

Data: data/teacher/labels.jsonl (merge: the checker's fix where it checked, else 14B pass a). ~2k reviews held out by
product (nmId) as the teacher test, ~1k by product for early stopping and the need threshold, the rest is training.
Training as in student_curve.py (same heads, schedule, early stopping).

Report: need F1/P/R and reasons F1 against the held-out teacher labels (also by label source); need against the
user's hand labels batch by batch (first 51 / 19 / 31 / later, by git revision of labeling/needs_reply_labels.json)
and a precision/recall table over the need threshold on all of the user's labels.

Usage: python train_student.py [--epochs 4] [--cpu-smoke]
Writes data/student/model (local, not in git), data/student/results.json and reports/student.md.
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from llm_probe import REASON_LIST_V2, SENTIMENTS, need_of, reasons_of
from student_curve import SEED, evaluate, jsonl, need_scores, predict, split_by_product, train_point, user_items

D = Path("data/student")
REPORT = Path("reports/student.md")
BATCHES = [("первые 51", "e547116~1"), ("партия 19", "e547116"), ("партия 31", "f96e38a")]


def load():
    rows = []
    for r in jsonl("data/teacher/labels.jsonl"):
        ans = r["label"]
        reasons = {x["reason"] for x in reasons_of(ans)}
        rows.append({"id": r["id"], "nmId": r["nmId"], "text": r["text"], "source": r["source"],
                     "need": float(need_of(ans)), "reasons": [float(k in reasons) for k in REASON_LIST_V2],
                     "sentiment": SENTIMENTS.index(ans["sentiment"])})
    return rows


def ids_at(rev):
    out = subprocess.run(["git", "show", f"{rev}:labeling/needs_reply_labels.json"], capture_output=True, check=True)
    return {g["id"] for g in json.loads(out.stdout.decode("utf-8"))["labels"]}


def user_batches(user):
    seen, groups = set(), []
    for name, rev in BATCHES:
        ids = ids_at(rev)
        groups.append((name, [u for u in user if u["id"] in ids and u["id"] not in seen]))
        seen |= ids
    rest = [u for u in user if u["id"] not in seen]
    if rest:
        groups.append(("новые после f96e38a", rest))
    return groups + [("все метки", user)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", default="../ru-review-sentiment/model")
    p.add_argument("--test", type=int, default=2000)
    p.add_argument("--val", type=int, default=1000)
    p.add_argument("--epochs", type=float, default=4)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--cpu-smoke", action="store_true", help="tiny run on the CPU (8 threads) to check the code")
    args = p.parse_args()
    D.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and not args.cpu_smoke else "cpu"
    torch.set_num_threads(8)

    rows = load()
    test, val, train = split_by_product(rows, args.test, args.val, np.random.default_rng(SEED))
    user = user_items()
    if args.cpu_smoke:
        test, val, train, args.epochs, args.batch = test[:16], val[:16], train[:32], 1, 8
    print("device", device, {"labels": len(rows), "train": len(train), "val": len(val), "test": len(test), "user": len(user)}, flush=True)
    model, tok, thr, history, minutes = train_point(train, val, args, device)
    res = {"labels": len(rows), "train": len(train), "val": len(val), "test_n": len(test), "threshold": round(thr, 2),
           "minutes": minutes, "history": history, "test": evaluate(model, tok, thr, test, [], device)}

    # the held-out teacher labels by source: checked by Qwen3.8 vs 14B only
    n, _, _ = predict(model, tok, [x["text"] for x in test], device)
    res["test_by_source"] = {}
    for s in sorted({x["source"].rsplit("-", 1)[-1] if "qwen" in x["source"] else x["source"] for x in test}):
        idx = [k for k, x in enumerate(test) if (x["source"].rsplit("-", 1)[-1] if "qwen" in x["source"] else x["source"]) == s]
        g = np.array([test[k]["need"] for k in idx])
        res["test_by_source"][s] = need_scores(g, n[idx] >= thr) | {"n": len(idx), "positives": int(g.sum())}

    # the user's labels batch by batch + P/R over the threshold
    un, ur, _ = predict(model, tok, [x["text"] for x in user], device)
    prob = {u["id"]: (float(un[k]), ur[k]) for k, u in enumerate(user)}
    res["user"] = []
    for name, us in user_batches(user):
        g = np.array([u["need"] for u in us])
        pr = np.array([prob[u["id"]][0] for u in us])
        gr = np.array([u["reasons"] for u in us]) if us else np.zeros((0, len(REASON_LIST_V2)))
        rr = np.array([prob[u["id"]][1] for u in us]) if us else np.zeros((0, len(REASON_LIST_V2)))
        res["user"].append({"batch": name, "n": len(us), "positives": int(g.sum()),
                            "accuracy": round(float(np.mean((pr >= thr) == g)), 3) if us else 0,
                            "fp": int(((pr >= thr) & (g == 0)).sum()), "fn": int(((pr < thr) & (g == 1)).sum()),
                            **need_scores(g, pr >= thr),
                            "reasons_micro_f1": round(float(f1_score(gr, rr >= 0.5, average="micro", zero_division=0)), 3)})
    g = np.array([u["need"] for u in user])
    res["user_curve"] = [{"threshold": round(float(t), 2), "flagged": int((un >= t).sum()),
                          "fp": int(((un >= t) & (g == 0)).sum()), "fn": int(((un < t) & (g == 1)).sum()),
                          **need_scores(g, un >= t)} for t in np.arange(0.05, 0.96, 0.05)]
    print(json.dumps({k: res[k] for k in ("test", "test_by_source", "user")}, ensure_ascii=False), flush=True)
    if args.cpu_smoke:
        return
    model.enc.save_pretrained(D / "model")
    tok.save_pretrained(D / "model")
    torch.save({k: v for k, v in model.state_dict().items() if not k.startswith("enc.")}, D / "model" / "heads.pt")
    (D / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    report(res)


def report(res):
    t = res["test"]
    src = source_rows(res["test_by_source"])
    md = ["# Студент на итоговых метках учителя\n",
          f"- Метки: {res['labels']:,} итоговых меток учителя (Qwen3-14B, проверка Qwen3.8-27B промптом v4 для отзывов, где 14B "
          "поставила «нужен человек» хотя бы в одном проходе, и 5% контроля).",
          f"- Обучение {res['train']:,}, валидация {res['val']:,}, тест {res['test_n']:,}; тест и валидация отложены по товарам (nmId).",
          f"- Студент: ruBert-base из ru-review-sentiment, головы need + причины (8) + тональность (4), как в student_curve. "
          f"Порог need подобран на валидации: **{res['threshold']}**. Время обучения {res['minutes']} мин.\n",
          "## Против отложенных меток учителя\n",
          "| need F1 | precision | recall | причины micro-F1 | macro-F1 | тональность acc |", "|---|---|---|---|---|---|",
          f"| {t['need']['f1']:.3f} | {t['need']['precision']:.3f} | {t['need']['recall']:.3f} | {t['reasons_micro_f1']:.3f} | "
          f"{t['reasons_macro_f1']:.3f} | {t['sentiment_acc']:.3f} |\n",
          "| Источник метки в тесте | Отзывов | need = да | need F1 | P | R |", "|---|---|---|---|---|---|"] + src
    md += ["\n## Против меток пользователя (порог с валидации)\n",
           "Меток мало, особенно need = да: один отзыв меняет recall на десятки п.п. Это ориентир.\n",
           "| Партия | Отзывов | need = да | Совпало | Лишних | Пропусков | need F1 | P | R | причины micro-F1 |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {u['batch']} | {u['n']} | {u['positives']} | {u['accuracy']:.0%} | {u['fp']} | {u['fn']} | {u['f1']:.2f} | "
           f"{u['precision']:.0%} | {u['recall']:.0%} | {u['reasons_micro_f1']:.2f} |" for u in res["user"]]
    md += ["\n## Порог need на всех метках пользователя\n",
           "| Порог | Отмечено «нужен человек» | Лишних | Пропусков | P | R | F1 |", "|---|---|---|---|---|---|---|"]
    md += [f"| {c['threshold']:.2f} | {c['flagged']} | {c['fp']} | {c['fn']} | {c['precision']:.0%} | {c['recall']:.0%} | {c['f1']:.2f} |"
           for c in res["user_curve"]]
    REPORT.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("wrote", REPORT)


def source_rows(by):
    return [f"| {s} | {v['n']:,} | {v['positives']:,} | {v['f1']:.3f} | {v['precision']:.3f} | {v['recall']:.3f} |" for s, v in by.items()]


if __name__ == "__main__":
    main()
