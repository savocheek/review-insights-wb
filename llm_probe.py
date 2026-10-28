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

# v2, after the user's first hand labels (labeling/needs_reply_labels.json): every review gets a template answer anyway,
# so "need" means a live person on top of it (the user says yes for ~16%, v1 models for ~55%); reasons describe
# what the review is about and are given even when no person is needed; the WB delivery is a reason of its own.
REASON_KEYS = {"defect": REASON_LIST[0], "refund": REASON_LIST[1], "delivery": REASON_LIST[2], "question": REASON_LIST[3],
               "safety": REASON_LIST[4], "misinfo": REASON_LIST[5], "strong_negative": REASON_LIST[6],
               "wb_logistics": "доставка Wildberries"}
REASON_LIST_V2 = list(REASON_KEYS.values())
SCHEMA_V2 = {
    "type": "object", "additionalProperties": False, "required": ["problems", "reasons", "needs_human", "sentiment"],
    "properties": {
        "problems": SCHEMA["properties"]["problems"],
        "reasons": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["quote", "reason"],
            "properties": {"quote": {"type": "string"}, "reason": {"enum": REASON_LIST_V2}}}},
        "needs_human": {
            "type": "object", "additionalProperties": False, "required": ["why", "need"],
            "properties": {"why": {"type": "string"}, "need": {"type": "boolean"}}},
        "sentiment": {"enum": SENTIMENTS}}}
# Few-shot examples: user's labels (need, reasons), the rest written by the assistant. These reviews are excluded
# from every comparison with the user's labels.
FEWSHOT = [
    (114132, {"problems": [{"quote": "пришла без банта", "problem": "некомплект", "detail": ""}],
              "reasons": [{"quote": "Блузка пришла без банта", "reason": REASON_KEYS["defect"]}],
              "needs_human": {"why": "мелкий недостаток, покупатель уже отказался от товара — хватит шаблона", "need": False},
              "sentiment": "негатив"}),
    (1830, {"problems": [{"quote": "весь скомкан. Грязный", "problem": "повреждённая упаковка", "detail": ""},
                         {"quote": "в одном месте порезан", "problem": "брак", "detail": ""}],
            "reasons": [{"quote": "как будто на складе валялся без упоковки", "reason": REASON_KEYS["wb_logistics"]}],
            "needs_human": {"why": "претензия к хранению и доставке, решения покупатель не просит", "need": False},
            "sentiment": "смешанный"}),
    (1338970, {"problems": [{"quote": "Сушит волосы очень", "problem": "другое", "detail": "сушит волосы"}],
               "reasons": [{"quote": "Сушит волосы очень", "reason": REASON_KEYS["safety"]},
                           {"quote": "Больше не буду брать", "reason": REASON_KEYS["strong_negative"]}],
               "needs_human": {"why": "неприятный эффект без вреда здоровью, решения покупатель не ждёт", "need": False},
               "sentiment": "негатив"}),
    (1682074, {"problems": [{"quote": "стик очень грубый", "problem": "другое", "detail": "грубый скраб царапает кожу"}],
               "reasons": [],
               "needs_human": {"why": "мелкое замечание в положительном отзыве", "need": False},
               "sentiment": "смешанный"}),
    (1582267, {"problems": [{"quote": "Вместо вишнёвого чехла пришёл чёрный", "problem": "цвет не такой", "detail": ""},
                            {"quote": "Вырез под вспышки и фонарь не подходит", "problem": "не подошло", "detail": ""}],
               "reasons": [{"quote": "Вместо вишнёвого чехла пришёл чёрный", "reason": REASON_KEYS["wb_logistics"]}],
               "needs_human": {"why": "пересорт, покупатель решения не просит — хватит шаблона", "need": False},
               "sentiment": "негатив"}),
    (2237565, {"problems": [{"quote": "Пришёл разбитый", "problem": "пришло сломанным или разбитым", "detail": ""}],
               "reasons": [{"quote": "Пришёл разбитый", "reason": REASON_KEYS["wb_logistics"]},
                           {"quote": "Пришёл разбитый", "reason": REASON_KEYS["defect"]}],
               "needs_human": {"why": "разбили в пути, покупатель уже отказался — деньги вернутся сами", "need": False},
               "sentiment": "негатив"}),
    (158072, {"problems": [{"quote": "вскрытой", "problem": "повреждённая упаковка", "detail": ""},
                           {"quote": "один бокал с треснутой ножкой", "problem": "пришло сломанным или разбитым", "detail": ""},
                           {"quote": "всего 5 бокалов", "problem": "некомплект", "detail": ""}],
              "reasons": [{"quote": "Бокалы пришли в ужасной коричневой коробке , вскрытой", "reason": REASON_KEYS["wb_logistics"]},
                          {"quote": "один бокал с треснутой ножкой", "reason": REASON_KEYS["defect"]}],
              "needs_human": {"why": "набор испорчен, покупатель потерял деньги и ждёт замены или компенсации", "need": True},
              "sentiment": "негатив"}),
    (1855713, {"problems": [{"quote": "микрофон не какой", "problem": "плохое качество, неаккуратно сделано", "detail": ""},
                            {"quote": "Ещё и сломанный внутри", "problem": "брак", "detail": ""},
                            {"quote": "Они не хотят его принимать обратно", "problem": "возврат", "detail": ""}],
               "reasons": [{"quote": "сломанный внутри", "reason": REASON_KEYS["defect"]},
                           {"quote": "Они не хотят его принимать обратно", "reason": REASON_KEYS["refund"]},
                           {"quote": "Для чего мне это", "reason": REASON_KEYS["question"]}],
               "needs_human": {"why": "брак и отказ в возврате, покупатель будет добиваться своего — риск денег и репутации",
                               "need": True},
               "sentiment": "негатив"}),
]
FEWSHOT_IDS = {str(i) for i, _ in FEWSHOT}


# v3: v2 with the "needs a person" rules tightened after 14B-v2 disagreed with the user on 10 of 43: problems at
# delivery (wrong item, shortage, broken on arrival, not delivered) are settled by a WB return without the seller.
NEED_V2 = """   need = true, только если при одном шаблонном ответе продавец потеряет деньги, покупателя или репутацию, или есть
   риск для здоровья: покупатель ждёт решения (замена, возврат денег, компенсация), спорит с продавцом или ему
   отказали, испорчен дорогой товар или набор, задан вопрос по товару, на который нужен ответ, есть вред здоровью.
   need = false: мелкие недостатки, размер или товар просто не подошёл, покупатель уже сам оформил возврат или отказ,
   претензии к доставке Wildberries без просьбы о решении, вкусовые оценки, похвала. Обычно живой человек нужен лишь
   одному-двум отзывам из десяти."""
NEED_V3 = """   need = true, только если при одном шаблонном ответе продавец потеряет деньги, покупателя или репутацию, или есть
   риск для здоровья:
   - покупатель прямо обращается к продавцу: задаёт вопрос по товару («может производитель разъяснит?») или просит
     что-то сделать («не учитывайте в проценте выкупа», «верните деньги»);
   - продавец или пункт выдачи отказали в возврате, покупатель спорит, грозит жалобой;
   - вред или риск для здоровья (аллергия, сыпь, ребёнок, просрочка);
   - товар сломался или развалился в использовании, и отзыв резко отпугнёт других покупателей;
   - испорчен дорогой товар или набор, покупатель потерял деньги.
   need = false:
   - проблемы при получении — пришёл не тот товар, цвет или размер, пересорт, недостача, пришёл сломанным или разбитым,
     не привезли: покупатель оформляет отказ или возврат через Wildberries, продавцу отвечать лично не нужно, даже если
     покупатель возмущён, ставит много восклицательных знаков или пишет «разберитесь», «обратите внимание»;
   - мелкие недостатки, размер или товар просто не подошёл, покупатель уже сам оформил возврат или отказ;
   - претензии к доставке и упаковке, вкусовые оценки, эмоции без конкретики, похвала.
   Обычно живой человек нужен лишь одному-двум отзывам из десяти; сомневаешься — need = false."""


def fewshot_block():
    texts = {x["id"]: x["text"] for x in map(json.loads, Path("data/needs_reply_items.jsonl").read_text(encoding="utf-8").splitlines())}
    return "\n\n".join(f"Отзыв: {texts[i]}\nОтвет: {json.dumps(ans, ensure_ascii=False)}" for i, ans in FEWSHOT)


SYSTEM_V2 = f"""Ты разбираешь отзывы покупателей Wildberries для продавца. Каждый отзыв и так получит вежливый шаблонный
ответ. Твоя задача — найти проблемы и решить, нужен ли сверх шаблона живой человек. Верни строго JSON.

1. problems — проблемы с товаром или заказом, о которых пишет покупатель. Для каждой:
   - quote: точная цитата из отзыва (дословно, без изменений, 2–15 слов), где сказано о проблеме;
   - problem: одна группа из списка: {"; ".join(PROBLEMS)};
   - detail: для «другое» — суть в 2–6 словах, иначе пустая строка.
   Похвала, нейтральное описание («пришло в запаянном пакете», «размер подошёл») и советы другим покупателям без
   претензии — не проблемы. Если проблем нет — пустой список. Не придумывай того, чего нет в тексте.
2. reasons — о чём этот отзыв для продавца; ставь, даже если живой человек не нужен. Каждая причина с точной цитатой:
   - брак/поломка: товар неисправен, повреждён, сломался;
   - возврат/деньги: хочет вернуть товар или деньги, проблемы с оплатой или возвратом;
   - не пришло / не то / некомплект: ошибка продавца в заказе — другой товар, цвет или размер, чего-то не хватает,
     пустая или полупустая упаковка;
   - вопрос к продавцу: настоящий вопрос, на который может ответить продавец (риторический — не считается);
   - здоровье/безопасность: аллергия, сыпь, ожог, травма, химический или резкий запах, опасность;
   - ложная информация о товаре: не соответствует описанию, фото, размерной сетке, составу; подделка;
   - сильный негатив с конкретикой: резко отрицательный отзыв с понятной претензией, отпугнёт покупателей;
   - доставка Wildberries: виноваты доставка или склад — не привезли, курьер, ПВЗ, помяли, вскрыли или разбили в пути,
     грязная или порванная упаковка, пересорт. Если пришёл не тот товар и непонятно, кто виноват, ставь обе причины.
3. needs_human — нужен ли живой человек сверх шаблона. Сначала why — одно короткое предложение, потом need.
   need = true, только если при одном шаблонном ответе продавец потеряет деньги, покупателя или репутацию, или есть
   риск для здоровья: покупатель ждёт решения (замена, возврат денег, компенсация), спорит с продавцом или ему
   отказали, испорчен дорогой товар или набор, задан вопрос по товару, на который нужен ответ, есть вред здоровью.
   need = false: мелкие недостатки, размер или товар просто не подошёл, покупатель уже сам оформил возврат или отказ,
   претензии к доставке Wildberries без просьбы о решении, вкусовые оценки, похвала. Обычно живой человек нужен лишь
   одному-двум отзывам из десяти.
4. sentiment — общая тональность отзыва: негатив, нейтрально, позитив или смешанный (есть и заметная похвала,
   и заметная претензия).

Формат: {{"problems": [{{"quote": "...", "problem": "...", "detail": ""}}], "reasons": [{{"quote": "...", "reason": "..."}}], "needs_human": {{"why": "...", "need": false}}, "sentiment": "..."}}

Примеры:

{{FEWSHOT}}"""


def norm(t):
    return " ".join(re.sub(r"[^\w\s]", " ", t.lower().replace("ё", "е")).split())


def validate(obj):
    """Problems with the answer against SCHEMA or SCHEMA_V2 (empty list = valid)."""
    err = []
    if isinstance(obj, dict) and set(obj) == {"problems", "reasons", "needs_human", "sentiment"}:
        return validate_v2(obj)
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


def validate_v2(obj):
    err = [] if obj["sentiment"] in SENTIMENTS else ["sentiment"]
    for p in obj["problems"] if isinstance(obj["problems"], list) else [None]:
        if not isinstance(p, dict) or p.get("problem") not in PROBLEMS or not isinstance(p.get("quote"), str):
            err.append("problem item")
    for r in obj["reasons"] if isinstance(obj["reasons"], list) else [None]:
        if not isinstance(r, dict) or r.get("reason") not in REASON_LIST_V2 or not isinstance(r.get("quote"), str):
            err.append("reason item")
    if not isinstance(obj["needs_human"], dict) or not isinstance(obj["needs_human"].get("need"), bool):
        err.append("needs_human")
    return err


def reasons_of(obj):
    return obj["reasons"] if "reasons" in obj else obj["needs_reply"]["reasons"]


def need_of(obj):
    return (obj["needs_human"] if "needs_human" in obj else obj["needs_reply"])["need"]


def quotes(obj):
    return [p["quote"] for p in obj.get("problems", [])] + [r["quote"] for r in reasons_of(obj)]


def system(prompt):
    if prompt == "v1":
        return SYSTEM
    s = SYSTEM_V2 if prompt == "v2" else SYSTEM_V2.replace(NEED_V2, NEED_V3)
    shots = fewshot_block()
    if prompt == "v4":
        shots += "\n\n" + "\n\n".join(f"Отзыв: {t}\nОтвет: {json.dumps(ans, ensure_ascii=False)}" for t, ans in FEWSHOT_INVENTED)
    return s.replace("{FEWSHOT}", shots)


# v4 = v3 + two invented reviews (not from the 500): an angry wrong-item delivery that still needs no person
FEWSHOT_INVENTED = [
    ("Прислали не тот размер!!! Заказывала 42, пришёл 48. Сколько можно, разберитесь уже со своими упаковщиками!!!",
     {"problems": [{"quote": "Заказывала 42, пришёл 48", "problem": "пришло не то", "detail": ""}],
      "reasons": [{"quote": "Прислали не тот размер", "reason": REASON_KEYS["delivery"]},
                  {"quote": "разберитесь уже со своими упаковщиками", "reason": REASON_KEYS["wb_logistics"]}],
      "needs_human": {"why": "пересорт при получении, возврат через Wildberries; возмущение не делает человека нужным",
                      "need": False},
      "sentiment": "негатив"}),
    ("Коробка пришла пустая, где товар? Деньги списали. Отказалась на пункте выдачи.",
     {"problems": [{"quote": "Коробка пришла пустая", "problem": "некомплект", "detail": ""}],
      "reasons": [{"quote": "Коробка пришла пустая", "reason": REASON_KEYS["delivery"]},
                  {"quote": "Отказалась на пункте выдачи", "reason": REASON_KEYS["wb_logistics"]}],
      "needs_human": {"why": "недостача при получении, покупатель уже отказался — деньги вернутся", "need": False},
      "sentiment": "негатив"}),
]


def gold_labels():
    path = Path("labeling/needs_reply_labels.json")
    return json.loads(path.read_text(encoding="utf-8"))["labels"] if path.exists() else []


def items(n, subset="first"):
    """first: the first n of the 500; gold: the user's labelled reviews except the few-shot ones;
    random: the 100 random reviews of the 500 (an estimate of the real stream)."""
    rows = [json.loads(x) for x in Path("data/needs_reply_items.jsonl").read_text(encoding="utf-8").splitlines()]
    if subset == "gold":
        ids = {g["id"] for g in gold_labels()} - FEWSHOT_IDS
        return [r for r in rows if str(r["id"]) in ids]
    if subset == "random":
        return [r for r in rows if r["kind"] == "случайный"]
    return rows[:n]


def chat(port, text, think, grammar, qwen, prompt="v1"):
    user = ("/no_think\n" if qwen and not think else "") + f"Отзыв:\n{text}"  # Qwen3 soft switch
    body = {"messages": [{"role": "system", "content": system(prompt)}, {"role": "user", "content": user}],
            "temperature": 0.6 if think else 0.2, "top_p": 0.95, "max_tokens": 6000 if think else 1200}
    if grammar:
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": "review", "schema": SCHEMA if prompt == "v1" else SCHEMA_V2}}
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
    for it in [x for s in a.set.split(",") for x in items(a.n, s)]:
        if it["id"] in done:
            continue
        done.add(it["id"])
        content, reasoning, usage, sec = chat(a.port, it["text"], a.think, grammar=not a.think,
                                                qwen="qwen" in a.name.lower(), prompt=a.prompt)
        obj = parse(content)
        errs = ["no JSON"] if obj is None else validate(obj)
        bad_quotes = [q for q in (quotes(obj) if obj and not errs else []) if norm(q) not in norm(it["text"])]
        rec = {"id": it["id"], "model": a.name, "prompt": a.prompt, "answer": obj, "valid": not errs, "errors": errs, "raw": content[:3000],
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
    rs = "; ".join(f"{r['reason']} «{r['quote']}»" for r in reasons_of(ans))
    why = f" ({ans['needs_human']['why']})" if "needs_human" in ans else ""
    return (f"проблемы: {pr} · {'человек' if 'needs_human' in ans else 'ответ'} нужен: {'да' if need_of(ans) else 'нет'}"
            f"{why}{(' — ' + rs) if rs else ''} · тон: {ans['sentiment']}")


def gold_section(runs, texts):
    """The models against the user's hand labels (few-shot reviews excluded) + the need rate on the 100 random reviews."""
    gold = {g["id"]: g for g in gold_labels() if g["id"] not in FEWSHOT_IDS}
    if not gold:
        return []
    key_of = {t: k for k, t in REASON_KEYS.items()}
    md = ["\n## Сверка с ручной разметкой пользователя\n",
          f"Эталон — метки пользователя: {len(gold)} отзывов (размечено {len(gold_labels())}, из них {len(FEWSHOT_IDS)} "
          "ушли в few-shot примеры промпта v2 и здесь не считаются). need = нужен живой человек сверх шаблона; у "
          f"пользователя need = да у {sum(g['need'] for g in gold.values())} из {len(gold)}. Причины у пользователя "
          "ставятся и при need = нет; у промпта v1 причины есть только при «ответ нужен», поэтому полнота причин v1 занижена "
          "by design, а «доставка Wildberries» в v1 отсутствует.\n",
          "| Прогон | отзывов | need: точность | precision | recall | need=да у модели | причины: precision | recall | "
          "need=да на 100 случайных |", "|---|---|---|---|---|---|---|---|---|"]
    diffs = {}
    for n, r in runs.items():
        ids = [i for i in r if str(i) in gold and r[i]["valid"]]
        rnd = [x for x in r.values() if texts[x["id"]]["kind"] == "случайный" and x["valid"]]
        rate = f"{sum(need_of(x['answer']) for x in rnd) / len(rnd):.0%} (n={len(rnd)})" if rnd else "—"
        if not ids:
            continue
        pairs = [(gold[str(i)]["need"], need_of(r[i]["answer"])) for i in ids]
        tp = sum(g and m for g, m in pairs); fp = sum(m and not g for g, m in pairs); fn = sum(g and not m for g, m in pairs)
        rt = rp = rg = 0
        for i in ids:
            gk = set(gold[str(i)]["reasons"]); mk = {key_of[x["reason"]] for x in reasons_of(r[i]["answer"])}
            rt += len(gk & mk); rp += len(mk); rg += len(gk)
        md.append(f"| {n} | {len(ids)} | {sum(g == m for g, m in pairs) / len(pairs):.0%} | {tp / max(tp + fp, 1):.0%} | "
                  f"{tp / max(tp + fn, 1):.0%} | {tp + fp} | {rt / max(rp, 1):.0%} | {rt / max(rg, 1):.0%} | {rate} |")
        diffs[n] = [i for i in ids if gold[str(i)]["need"] != need_of(r[i]["answer"])]
    for n, ids in diffs.items():
        if not ids:
            continue
        md.append(f"\n### Расхождения по need: {n} ({len(ids)})\n")
        for i in ids:
            g = gold[str(i)]
            md += [f"**{i}** — {texts[i]['stars']}★ · пользователь: человек {'нужен' if g['need'] else 'не нужен'}"
                   f"{(', причины: ' + ', '.join(g['reasons'])) if g['reasons'] else ''}\n",
                   f"> {texts[i]['text']}\n", f"- модель: {short(runs[n][i]['answer'])}", ""]
    return md


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
        need = sum(need_of(p) == need_of(q) for p, q in zip(A, B)) / len(ids)
        jac = []
        for p, q in zip(A, B):
            s1, s2 = {t["problem"] for t in p["problems"]}, {t["problem"] for t in q["problems"]}
            jac.append(1.0 if not s1 and not s2 else len(s1 & s2) / len(s1 | s2))
        md.append(f"| {x} × {y} | {sent:.0%} | {need:.0%} | {sum(jac) / len(jac):.2f} |")
    texts = {x["id"]: x for x in items(500)}
    md += gold_section(runs, texts)
    if meta.get("conclusion"):
        md += ["\n## Вывод\n", meta["conclusion"]]
    md += ["\n## Промпт v4 (рекомендуемый)\n",
           "v2 = схема reasons + needs_human и 8 few-shot примеров из меток пользователя; v3 = v2 с уточнёнными правилами "
           "need (проблемы при получении решает возврат WB); v4 = v3 + 2 придуманных примера (не из 500 отзывов).\n",
           "```", system("v4"), "```", "\nJSON-схема (grammar llama.cpp): `llm_probe.SCHEMA_V2`.\n",
           "\n## Промпт v1 (первый прогон всех моделей)\n", "```", SYSTEM, "```", "\nJSON-схема: `llm_probe.SCHEMA`.\n",
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
    p.add_argument("--prompt", choices=["v1", "v2", "v3", "v4"], default="v1")
    p.add_argument("--set", default="first", help="first | gold | random, or several joined by commas")
    a = p.parse_args()
    {"run": run, "blind": blind, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
