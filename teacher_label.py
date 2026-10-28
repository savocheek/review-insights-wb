"""LLM teacher labels for WB reviews: a small model labels everything, a large one checks the hard cases.

  corpus   ~20k WB-train reviews: 1–3★ heavy, 4–5★ with a "но/минус/жаль…" turn, ~20% random; without the 500
           hand-labelling reviews (data/needs_reply_items.jsonl, the user's gold set) → data/teacher/corpus.jsonl
  label    the labeller (llama.cpp server, prompt v2 with few-shot, grammar = llm_probe.SCHEMA_V2), N parallel requests; --pass a|b:
           pass a at temperature 0.2, pass b at 0.8 (disagreement between them marks hard cases)
  select   hard cases (~30–35%): the two passes disagree, long reviews, mixed sentiment, a borderline
           "needs reply", sarcasm/negation markers; plus a random 5% of the easy ones as a control
  verify   the checker gets the review and the labeller's JSON and answers {verdict, why, fixed}
  merge    final label: the checker's fix where it checked, otherwise the labeller's; source = 14b / 32b-confirmed /
           32b-fixed → data/teacher/labels.jsonl, reports/teacher_labels.md
Every step appends line by line and resumes after a crash.

Usage: python teacher_label.py corpus | label --pass a|b [--workers 4] | select | verify [--workers 1] | merge
"""
import argparse
import json
import random
import re
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from complaints import TURN
from llm_probe import SCHEMA_V2, fewshot_block, need_of, norm, parse, quotes, reasons_of, short, system, validate

D = Path("data/teacher")
N_CORPUS = 20_000
LONG_WORDS = 80
SARCASM = re.compile(r"\b(спасибо за|ну конечно|ага|супер|класс|отлично|прекрасно)\b.*(сломал|порвал|развалил|брак|"
                     r"не работа|через (день|два дня|неделю))|\bне то чтобы\b|\bвроде бы\b|\)\)\)|!{3,}|\?{2,}", re.I)

VERIFY_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["verdict", "why", "fixed"],
    "properties": {"verdict": {"enum": ["согласен", "исправлено"]}, "why": {"type": "string"},
                   "fixed": {"anyOf": [{"type": "null"}, SCHEMA_V2]}}}
SYSTEM = system("v2")
VERIFY_SYSTEM = SYSTEM.split("\n\nФормат:")[0] + "\n\nПримеры правильных разборов:\n\n" + fewshot_block() + """

Тебе дают отзыв и разбор, который сделала другая модель. Проверь его по правилам выше: всё ли верно в problems
(группы, цитаты дословно из отзыва, нет ли выдуманных или пропущенных проблем), в reasons (все ли причины и верны ли они), в needs_human (нужен ли живой человек сверх шаблона,
объяснение why) и в sentiment.
Если всё верно — {"verdict": "согласен", "why": "", "fixed": null}.
Если есть ошибка — {"verdict": "исправлено", "why": "что исправлено, 3–12 слов", "fixed": <полный исправленный разбор
в том же формате>}. Не исправляй мелочи вкуса (другая, но тоже точная цитата) — только ошибки по смыслу."""


def jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def post(port, messages, schema, temperature, max_tokens=1500):
    body = {"messages": messages, "temperature": temperature, "top_p": 0.95, "max_tokens": max_tokens,
            "response_format": {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        res = json.loads(r.read())
    return res["choices"][0]["message"].get("content") or "", res.get("usage", {}), time.time() - t0


def corpus(a):
    D.mkdir(parents=True, exist_ok=True)
    gold = {json.loads(x)["id"] for x in Path("data/needs_reply_items.jsonl").read_text(encoding="utf-8").splitlines()}
    wb = pd.read_parquet("data/wb.parquet", columns=["nmId", "stars", "text", "split"])
    wb = wb[(wb["split"] == "train") & ~wb.index.isin(gold) & (wb["text"].str.split().str.len() <= 300)]
    turn = (wb["stars"] >= 4) & wb["text"].str.contains(TURN)
    parts, taken = [], set()

    def take(df, n, kind):
        df = df[~df.index.isin(taken)].sample(min(n, len(df)), random_state=0)
        taken.update(df.index)
        parts.append(df.assign(kind=kind))

    take(wb[wb["stars"] == 1], 3_000, "1★")
    take(wb[wb["stars"] == 2], 3_000, "2★")
    take(wb[wb["stars"] == 3], 4_000, "3★")
    take(wb[turn], 6_000, "4–5★ с поворотом")
    take(wb, N_CORPUS - sum(map(len, parts)), "случайный")
    df = pd.concat(parts).sample(frac=1, random_state=0)
    with (D / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for i, r in df.iterrows():
            f.write(json.dumps({"id": int(i), "nmId": int(r["nmId"]), "stars": int(r["stars"]), "kind": r["kind"],
                                "text": r["text"]}, ensure_ascii=False) + "\n")
    print(df["kind"].value_counts().to_string())


def run_parallel(todo, fn, out, workers):
    with ThreadPoolExecutor(workers) as ex, out.open("a", encoding="utf-8") as f:
        for n, rec in enumerate(ex.map(fn, todo), 1):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if n % 200 == 0:
                print(f"{time.strftime('%H:%M')} {n}/{len(todo)}", flush=True)


def label(a):
    out = D / f"label_{a.pass_}.jsonl"
    done = {r["id"] for r in jsonl(out)}
    todo = [it for it in jsonl(D / "corpus.jsonl") if it["id"] not in done]
    temp = 0.2 if a.pass_ == "a" else 0.8

    def one(it):
        user = "/no_think\nОтзыв:\n" + it["text"]
        content, usage, sec = post(a.port, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                                   SCHEMA_V2, temp)
        obj = parse(content)
        errs = ["no JSON"] if obj is None else validate(obj)
        return {"id": it["id"], "answer": obj if not errs else None, "errors": errs, "seconds": round(sec, 1),
                "bad_quotes": [q for q in (quotes(obj) if not errs else []) if norm(q) not in norm(it["text"])]}

    print(f"pass {a.pass_}: {len(todo)} to label", flush=True)
    run_parallel(todo, one, out, a.workers)


def signature(ans):
    return ({p["problem"] for p in ans["problems"]}, need_of(ans), {r["reason"] for r in reasons_of(ans)}, ans["sentiment"])


def select(a):
    corpus_ = {it["id"]: it for it in jsonl(D / "corpus.jsonl")}
    A = {r["id"]: r for r in jsonl(D / "label_a.jsonl")}
    B = {r["id"]: r for r in jsonl(D / "label_b.jsonl")}
    rows = []
    for i, it in corpus_.items():
        if i not in A:
            continue
        ra, rb = A[i]["answer"], B.get(i, {}).get("answer")
        why = []
        if ra is None:
            why.append("невалидный ответ")
        elif rb is not None and signature(ra) != signature(rb):
            why.append("прогоны расходятся")
        if len(it["text"].split()) > LONG_WORDS:
            why.append("длинный")
        if ra and ra["sentiment"] == "смешанный":
            why.append("смешанная тональность")
        if ra and need_of(ra) and {r["reason"] for r in reasons_of(ra)} <= {"сильный негатив с конкретикой", "доставка Wildberries"}:
            why.append("пограничный ответ нужен")
        if ra and not need_of(ra) and it["stars"] <= 2 and len(reasons_of(ra)) >= 2:
            why.append("1–2★, несколько причин, человек не нужен")
        if SARCASM.search(it["text"]):
            why.append("сарказм/ирония по признакам")
        if ra and A[i]["bad_quotes"]:
            why.append("цитата не из текста")
        rows.append((i, why))
    hard = [(i, w) for i, w in rows if w]
    easy = [i for i, w in rows if not w]
    if a.control_only:  # the labeller is not worse than the checker: the checker only measures its error
        hard, easy = [], [i for i, _ in rows]
    rng = random.Random(0)
    control = rng.sample(easy, int(0.05 * len(easy)))
    with (D / "to_verify.jsonl").open("w", encoding="utf-8") as f:
        for i, w in hard:
            f.write(json.dumps({"id": i, "group": "трудный", "why": w}, ensure_ascii=False) + "\n")
        for i in control:
            f.write(json.dumps({"id": i, "group": "контроль 5%", "why": []}, ensure_ascii=False) + "\n")
    c = Counter(x for _, w in hard for x in w)
    print(f"labelled {len(rows)}, hard {len(hard)} ({len(hard) / max(len(rows), 1):.0%}), control {len(control)}")
    print(c.most_common())


def verify(a):
    corpus_ = {it["id"]: it for it in jsonl(D / "corpus.jsonl")}
    A = {r["id"]: r for r in jsonl(D / "label_a.jsonl")}
    out = D / "verify.jsonl"
    done = {r["id"] for r in jsonl(out)}
    todo = [t for t in jsonl(D / "to_verify.jsonl") if t["id"] not in done]

    def one(t):
        it, ans = corpus_[t["id"]], A[t["id"]]["answer"]
        user = (f"Отзыв:\n{it['text']}\n\nРазбор другой модели:\n"
                + (json.dumps(ans, ensure_ascii=False) if ans else "(модель не дала валидного ответа — сделай разбор сам)"))
        content, usage, sec = post(a.port, [{"role": "system", "content": VERIFY_SYSTEM},
                                            {"role": "user", "content": ("/no_think\n" if a.qwen else "") + user}], VERIFY_SCHEMA, 0.2, 2000)
        obj = parse(content)
        ok = isinstance(obj, dict) and obj.get("verdict") in ("согласен", "исправлено")
        fixed = obj.get("fixed") if ok else None
        if ok and obj["verdict"] == "исправлено" and (fixed is None or validate(fixed)):
            ok = False
        return {"id": t["id"], "group": t["group"], "checker": a.checker, "why_selected": t["why"], "verdict": obj.get("verdict") if ok else None,
                "why": obj.get("why", "") if ok else "", "fixed": fixed if ok else None, "seconds": round(sec, 1),
                "raw": None if ok else content[:2000]}

    print(f"verify: {len(todo)} to check", flush=True)
    run_parallel(todo, one, out, a.workers)


def merge(a):
    corpus_ = {it["id"]: it for it in jsonl(D / "corpus.jsonl")}
    A = {r["id"]: r for r in jsonl(D / "label_a.jsonl")}
    V = {r["id"]: r for r in jsonl(D / "verify.jsonl")}
    final, src = [], Counter()
    for i, it in corpus_.items():
        v = V.get(i)
        if v and v["verdict"] == "исправлено":
            ans, s = v["fixed"], f"{v['checker']}-fixed"
        elif v and v["verdict"] == "согласен" and A.get(i, {}).get("answer"):
            ans, s = A[i]["answer"], f"{v['checker']}-confirmed"
        elif A.get(i, {}).get("answer"):
            ans, s = A[i]["answer"], "14b"
        else:
            continue
        src[s] += 1
        final.append({"id": i, "nmId": it["nmId"], "stars": it["stars"], "text": it["text"], "label": ans, "source": s})
    with (D / "labels.jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in final)

    checked = [v for v in V.values() if v["verdict"]]
    md = ["# Разметка LLM-учителем\n",
          f"Корпус: {len(corpus_):,} отзывов WB-train (без 500 эталонных отзывов пользователя и без val/test). "
          f"Размечено: {len(final):,}. Разметчик — Qwen3-14B (без размышлений, grammar), проверяющий — "
          f"{next(iter(V.values()))['checker'] if V else '—'}.\n",
          "| Источник метки | Отзывов |", "|---|---|"] + [f"| {k} | {n:,} |" for k, n in src.most_common()]
    md += ["\n## Как часто проверяющий исправлял\n", "| Группа | Проверено | Исправлено |", "|---|---|---|"]
    for g in ("трудный", "контроль 5%"):
        vs = [v for v in checked if v["group"] == g]
        md.append(f"| {g} | {len(vs):,} | {sum(v['verdict'] == 'исправлено' for v in vs) / max(len(vs), 1):.1%} |")
    md.append("\nДоля исправлений на случайных 5% «лёгких» — оценка ошибки 14B там, где 32B не проверял.\n")
    why = Counter(x for v in checked if v["group"] == "трудный" for x in v["why_selected"])
    fixed_by = Counter(x for v in checked if v["group"] == "трудный" and v["verdict"] == "исправлено" for x in v["why_selected"])
    md += ["| Почему отобран как трудный | Отзывов | Исправлено |", "|---|---|---|"]
    md += [f"| {k} | {n:,} | {fixed_by[k] / n:.0%} |" for k, n in why.most_common()]
    md += ["\n## Распределение итоговых меток\n"]
    probs = Counter(p["problem"] for r in final for p in r["label"]["problems"])
    reas = Counter(x["reason"] for r in final for x in reasons_of(r["label"]))
    sent = Counter(r["label"]["sentiment"] for r in final)
    need = sum(need_of(r["label"]) for r in final)
    md += [f"Нужен живой человек: {need:,} из {len(final):,} ({need / max(len(final), 1):.0%}). Тональность: "
           + ", ".join(f"{k} {n / len(final):.0%}" for k, n in sent.most_common()) + ".\n",
           "| Проблема | Упоминаний |", "|---|---|"] + [f"| {k} | {n:,} |" for k, n in probs.most_common()]
    md += ["\n| Причина (о чём отзыв) | Отзывов |", "|---|---|"] + [f"| {k} | {n:,} |" for k, n in reas.most_common()]
    md += ["\n## Типичные исправления (примеры)\n"]
    rng = random.Random(1)
    fixes = [v for v in checked if v["verdict"] == "исправлено"]
    for v in rng.sample(fixes, min(15, len(fixes))):
        it = corpus_[v["id"]]
        md += [f"**{v['id']}** ({it['stars']}★, {', '.join(v['why_selected']) or 'контроль'}) — {v['why']}\n",
               f"> {it['text'][:600]}\n", f"- 14B: {short(A[v['id']]['answer'])}", f"- {v['checker']}: {short(v['fixed'])}", ""]
    Path("reports/teacher_labels.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"{len(final)} labels; {dict(src)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["corpus", "label", "select", "verify", "merge"])
    p.add_argument("--pass", dest="pass_", choices=["a", "b"], default="a")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--checker", default="32b", help="name of the checking model in the source field")
    p.add_argument("--no-qwen", dest="qwen", action="store_false", help="checker is not Qwen: no /no_think")
    p.add_argument("--control-only", action="store_true", help="select: only a random 5%% control set")
    a = p.parse_args()
    {"corpus": corpus, "label": label, "select": select, "verify": verify, "merge": merge}[a.cmd](a)


if __name__ == "__main__":
    main()
