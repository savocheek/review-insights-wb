"""LLM teacher probe: a local LLM reads a whole WB review and returns problems, "needs a human answer" and sentiment
as strict JSON. Talks to a llama.cpp server (OpenAI-compatible /v1/chat/completions) started separately.

  run      200 reviews of data/needs_reply_items.jsonl (WB-train) → data/llm_probe/<name>.jsonl
           --think: Qwen3 thinking mode (the JSON is taken after </think>, no grammar during thinking)
  blind    60 reviews × all runs, model names hidden and shuffled per review → data/llm_probe/blind.md (+ key)
  report   validity, speed, agreement between models, rubric scores → reports/llm_probe.md

Usage: python llm_probe.py run NAME [--port 8080] [--think] [--n 200]
       python llm_probe.py blind | report
"""
import argparse
import itertools
import json
import random
import re
import time
import urllib.request
from pathlib import Path

from needs_reply_prep import REASONS

OUT = Path("data/llm_probe")
# Meaningful complaint groups of stage 3 (reports/topic_drafts.json without junk, context and product-type groups;
# near-duplicates merged). "другое" takes anything else with a short description.
PROBLEMS = ["брак", "пришло сломанным или разбитым", "повреждённая упаковка", "пришло не то", "некомплект",
            "маломерит", "размер не соответствует (велик, не по сетке)", "цвет не такой", "не соответствует фото или описанию",
            "плохое качество, неаккуратно сделано", "торчат нитки, кривые швы", "следы клея, неаккуратная склейка",
            "неприятный запах", "протекает, высыпается", "испортилось после стирки", "быстро сломалось или износилось",
            "расклеилось или расползлось", "нет эффекта, не работает как заявлено", "не подошло", "не стоит своих денег",
            "возврат", "претензии к продавцу или магазину", "отзывы обманули ожидания", "другое"]
REASON_LIST = list(REASONS)
SENTIMENTS = ["негатив", "нейтрально", "позитив", "смешанный"]

SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["problems", "needs_reply", "sentiment"],
    "properties": {
        "problems": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["quote", "problem", "detail"],
            "properties": {"quote": {"type": "string"}, "problem": {"enum": PROBLEMS}, "detail": {"type": "string"}}}},
        "needs_reply": {
            "type": "object", "additionalProperties": False, "required": ["need", "reasons"],
            "properties": {"need": {"type": "boolean"}, "reasons": {"type": "array", "items": {
                "type": "object", "additionalProperties": False, "required": ["quote", "reason"],
                "properties": {"quote": {"type": "string"}, "reason": {"enum": REASON_LIST}}}}}},
        "sentiment": {"enum": SENTIMENTS}}}

SYSTEM = f"""Ты разбираешь отзывы покупателей Wildberries для продавца. Прочитай отзыв целиком и верни строго JSON.

1. problems — проблемы с товаром или заказом, о которых пишет покупатель. Для каждой:
   - quote: точная цитата из отзыва (дословно, без изменений, 2–15 слов), где сказано о проблеме;
   - problem: одна группа из списка: {"; ".join(PROBLEMS)};
   - detail: для «другое» — суть в 2–6 словах, иначе пустая строка.
   Похвала, нейтральное описание («пришло в запаянном пакете», «размер подошёл») и советы другим покупателям без
   претензии — не проблемы. Если проблем нет — пустой список. Не придумывай того, чего нет в тексте.
2. needs_reply — нужно ли продавцу ответить лично (не шаблоном). need = true, если без ответа продавец потеряет деньги,
   покупателя или репутацию, или есть риск для здоровья. reasons — одна или несколько причин с точной цитатой:
   - брак/поломка: товар неисправен, повреждён, сломался;
   - возврат/деньги: хочет вернуть товар или деньги, проблемы с оплатой или возвратом;
   - не пришло / не то / некомплект: не получил заказ, пришёл другой товар, цвет или размер, чего-то не хватает;
   - вопрос к продавцу: настоящий вопрос, на который может ответить продавец (риторический — не считается);
   - здоровье/безопасность: аллергия, сыпь, ожог, травма, химический или резкий запах, опасность;
   - ложная информация о товаре: не соответствует описанию, фото, размерной сетке, составу; подделка;
   - сильный негатив с конкретикой: резко отрицательный отзыв с понятной претензией, отпугнёт покупателей.
   Претензии только к доставке Wildberries (курьер опоздал) — ответ не нужен. Если need = false, reasons пустой.
3. sentiment — общая тональность отзыва: негатив, нейтрально, позитив или смешанный (есть и заметная похвала,
   и заметная претензия).

Формат: {{"problems": [{{"quote": "...", "problem": "...", "detail": ""}}], "needs_reply": {{"need": true, "reasons": [{{"quote": "...", "reason": "..."}}]}}, "sentiment": "..."}}"""


def norm(t):
    return " ".join(re.sub(r"[^\w\s]", " ", t.lower().replace("ё", "е")).split())


def validate(obj):
    """Problems with the answer against SCHEMA (empty list = valid)."""
    err = []
    if not isinstance(obj, dict) or set(obj) != {"problems", "needs_reply", "sentiment"}:
        return ["top-level keys"]
    if obj["sentiment"] not in SENTIMENTS:
        err.append("sentiment")
    for p in obj["problems"] if isinstance(obj["problems"], list) else [None]:
        if not isinstance(p, dict) or p.get("problem") not in PROBLEMS or not isinstance(p.get("quote"), str):
            err.append("problem item")
    nr = obj["needs_reply"]
    if not isinstance(nr, dict) or not isinstance(nr.get("need"), bool) or not isinstance(nr.get("reasons"), list):
        err.append("needs_reply")
    else:
        for r in nr["reasons"]:
            if not isinstance(r, dict) or r.get("reason") not in REASON_LIST or not isinstance(r.get("quote"), str):
                err.append("reason item")
    return err


def quotes(obj):
    return [p["quote"] for p in obj.get("problems", [])] + [r["quote"] for r in obj.get("needs_reply", {}).get("reasons", [])]


def items(n):
    rows = [json.loads(x) for x in Path("data/needs_reply_items.jsonl").read_text(encoding="utf-8").splitlines()]
    return rows[:n]


def chat(port, text, think, grammar, qwen):
    user = ("/no_think\n" if qwen and not think else "") + f"Отзыв:\n{text}"  # Qwen3 soft switch
    body = {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            "temperature": 0.6 if think else 0.2, "top_p": 0.95, "max_tokens": 6000 if think else 1200}
    if grammar:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "review", "schema": SCHEMA}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        res = json.loads(r.read())
    msg = res["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    if "</think>" in content:  # server returned thinking inline
        reasoning, content = content.split("</think>", 1)
    return content.strip(), reasoning, res.get("usage", {}), time.time() - t0


def parse(content):
    m = re.search(r"\{.*\}", content, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def run(a):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{a.name}.jsonl"
    done = {json.loads(x)["id"] for x in path.read_text(encoding="utf-8").splitlines()} if path.exists() else set()
    for it in items(a.n):
        if it["id"] in done:
            continue
        content, reasoning, usage, sec = chat(a.port, it["text"], a.think, grammar=not a.think,
                                                qwen="qwen" in a.name.lower())
        obj = parse(content)
        errs = ["no JSON"] if obj is None else validate(obj)
        bad_quotes = [q for q in (quotes(obj) if obj and not errs else []) if norm(q) not in norm(it["text"])]
        rec = {"id": it["id"], "model": a.name, "answer": obj, "valid": not errs, "errors": errs, "raw": content[:3000],
               "reasoning_chars": len(reasoning), "quotes_not_in_text": bad_quotes, "seconds": round(sec, 1),
               "completion_tokens": usage.get("completion_tokens"), "prompt_tokens": usage.get("prompt_tokens")}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{it['id']} {sec:.0f}s valid={not errs} quotes_bad={len(bad_quotes)}", flush=True)


def load_runs():
    return {p.stem: {json.loads(x)["id"]: json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()}
            for p in sorted(OUT.glob("*.jsonl")) if p.stem not in ("blind_scores",)}


def short(ans):
    if not ans:
        return "— нет валидного JSON —"
    pr = "; ".join(f"{p['problem']}{(': ' + p['detail']) if p.get('detail') else ''} «{p['quote']}»" for p in ans["problems"]) or "нет"
    nr = ans["needs_reply"]
    rs = "; ".join(f"{r['reason']} «{r['quote']}»" for r in nr["reasons"])
    return f"проблемы: {pr} · ответ нужен: {'да' if nr['need'] else 'нет'}{(' — ' + rs) if rs else ''} · тон: {ans['sentiment']}"


def blind(a):
    runs, rng = load_runs(), random.Random(7)
    md, key = ["# Слепая оценка: ответы моделей без имён (порядок перемешан в каждом отзыве)\n"], {}
    for it in items(60):
        names = [n for n in runs if it["id"] in runs[n]]
        rng.shuffle(names)
        key[it["id"]] = dict(zip("ABCDEFGH", names))
        md += [f"## {it['id']}\n", f"> {it['text']}\n"]
        md += [f"- **{l}**: {short(runs[n][it['id']]['answer'] if runs[n][it['id']]['valid'] else None)}" for l, n in key[it["id"]].items()]
        md.append("")
    (OUT / "blind.md").write_text("\n".join(md), encoding="utf-8")
    (OUT / "blind_key.json").write_text(json.dumps(key, ensure_ascii=False), encoding="utf-8")
    print(f"{len(key)} reviews → {OUT / 'blind.md'}")


def report(a):
    runs = load_runs()
    names = list(runs)
    meta = json.loads((OUT / "meta.json").read_text(encoding="utf-8")) if (OUT / "meta.json").exists() else {}
    scores = {}
    if (OUT / "blind_scores.jsonl").exists():
        key = json.loads((OUT / "blind_key.json").read_text(encoding="utf-8"))
        for x in (OUT / "blind_scores.jsonl").read_text(encoding="utf-8").splitlines():
            s = json.loads(x)
            n = key[str(s["id"])][s["letter"]]
            scores.setdefault(n, []).append(s)
    md = ["# LLM-учитель: проба локальных моделей\n", meta.get("intro", ""), "",
          "| Модель | Лицензия | VRAM / RAM | сек на отзыв | ток/с | валидный JSON | цитаты не из текста | "
          "оценка ассистента (0–4) | проблемы / ответ нужен верно |", "|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        r = list(runs[n].values())
        valid = [x for x in r if x["valid"]]
        nq = sum(len(quotes(x["answer"])) for x in valid)
        bad = sum(len(x["quotes_not_in_text"]) for x in valid)
        tps = sum(x["completion_tokens"] or 0 for x in r) / max(sum(x["seconds"] for x in r), 1)
        sc = scores.get(n, [])
        rub = (f"{sum(s['total'] for s in sc) / len(sc):.2f} (n={len(sc)})" if sc else "—")
        acc = (f"{sum(s['problems_ok'] for s in sc) / len(sc):.0%} / {sum(s['need_ok'] for s in sc) / len(sc):.0%}" if sc else "—")
        m = meta.get("models", {}).get(n, {})
        md.append(f"| {n} | {m.get('license', '?')} | {m.get('memory', '?')} | {sum(x['seconds'] for x in r) / len(r):.1f} | "
                  f"{tps:.1f} | {len(valid) / len(r):.0%} ({len(r)}) | {bad / max(nq, 1):.1%} | {rub} | {acc} |")
    md += ["\n## Согласие моделей между собой (на отзывах, где у обеих валидный JSON)\n",
           "| Пара | тональность | ответ нужен | проблемы (Жаккар групп) |", "|---|---|---|---|"]
    for x, y in itertools.combinations(names, 2):
        ids = [i for i in runs[x] if i in runs[y] and runs[x][i]["valid"] and runs[y][i]["valid"]]
        if not ids:
            continue
        A = [runs[x][i]["answer"] for i in ids]
        B = [runs[y][i]["answer"] for i in ids]
        sent = sum(p["sentiment"] == q["sentiment"] for p, q in zip(A, B)) / len(ids)
        need = sum(p["needs_reply"]["need"] == q["needs_reply"]["need"] for p, q in zip(A, B)) / len(ids)
        jac = []
        for p, q in zip(A, B):
            s1, s2 = {t["problem"] for t in p["problems"]}, {t["problem"] for t in q["problems"]}
            jac.append(1.0 if not s1 and not s2 else len(s1 & s2) / len(s1 | s2))
        md.append(f"| {x} × {y} | {sent:.0%} | {need:.0%} | {sum(jac) / len(jac):.2f} |")
    if meta.get("conclusion"):
        md += ["\n## Вывод\n", meta["conclusion"]]
    md += ["\n## Промпт\n", "```", SYSTEM, "```", "\nJSON-схема (grammar llama.cpp): `llm_probe.SCHEMA`.\n",
           "## 20 примеров: отзыв → ответы моделей\n"]
    for it in items(200)[60:80]:
        md += [f"**{it['id']}** — {it['stars']}★\n", f"> {it['text']}\n"]
        md += [f"- *{n}*: {short(runs[n][it['id']]['answer'] if it['id'] in runs[n] and runs[n][it['id']]['valid'] else None)}" for n in names]
        md.append("")
    Path("reports/llm_probe.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("reports/llm_probe.md")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["run", "blind", "report"])
    p.add_argument("name", nargs="?")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--think", action="store_true")
    p.add_argument("--n", type=int, default=200)
    a = p.parse_args()
    {"run": run, "blind": blind, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
