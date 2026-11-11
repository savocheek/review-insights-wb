"""Demo report data for a seller of maternity bandages and goods (the user's request, 31.10).

Four products of the niche (found by review text, the base has no categories): every review goes through the student
(need score, reason probabilities, sentiment); Qwen3-14B (llama.cpp server on :8080, started separately) names
4-6 "liked" and 3-5 "questions" themes per product with the reviews behind each theme; the share of a theme is
counted from those reviews. Unreviewed themes and supporting quotes are withheld from the public export;
the raw output stays in data/demo_maternity/unreviewed_export.json. Names, phones, e-mails and links are masked.
Usage (from the repo root):  python scripts/demo_maternity.py student   (GPU: the student)
                             python scripts/demo_maternity.py themes    (GPU: llama-server with Qwen3-14B)
                             python scripts/demo_maternity.py tag       (same server, 4 slots: every review × themes)
                             python scripts/demo_maternity.py build     → reports/demo_maternity.json
"""
import json
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from llm_probe import REASON_KEYS, SENTIMENTS  # noqa: E402
from demo_export import save_export  # noqa: E402

PRODUCTS = {1017642: "бандаж для беременных", 11709626: "бандаж до и после родов",
            1169996: "колготки для беременных", 10009941: "сорочка для роддома и кормления"}
D = Path("data/demo_maternity")
OUT = Path("reports/demo_maternity.json")
NEED_THR, REASON_THR, MAX_THEME_REVIEWS = 0.85, 0.5, 300

NAMES = ("Анна Анастасия Алена Алёна Александра Алина Валентина Валерия Вера Виктория Галина Дарья Даша Евгения Екатерина "
         "Катя Елена Лена Елизавета Ирина Ира Карина Кристина Ксения Лариса Любовь Людмила Марина Мария Маша Надежда Наталья "
         "Наташа Нина Оксана Ольга Оля Полина Светлана Света Софья София Татьяна Таня Юлия Юля Яна Александр Алексей Андрей "
         "Антон Артём Артем Владимир Дмитрий Дима Евгений Иван Игорь Максим Михаил Никита Николай Олег Павел Роман Сергей").split()
PII = [(re.compile(r"\S+@\S+"), "[почта]"), (re.compile(r"https?://\S+|www\.\S+"), "[ссылка]"),
       (re.compile(r"\+?\d[\d\s()-]{8,}\d"), "[телефон]"),
       (re.compile(r"\b(" + "|".join(NAMES) + r")[а-яё]{0,3}\b"), "[имя]"),
       # brands (the demo goes to another seller of the niche): every brand found in these products' reviews and answers
       (re.compile(r"\bmom\W?s?\W*balance\b|\bmom\W?s\b|\bbalance\b|\bmombalance\S*|\bbandageformam\S*|\bfest\b|\bdea\s*mia\b|\bmia\b|\bflammber\b|"
                   r"(?<=бренд )mom\b|(?<=бренду )mom\b|(?<=бренда )mom\b", re.I), "[бренд]")]


def clean(text, limit=None):
    for rx, sub in PII:
        text = rx.sub(sub, text)
    text = " ".join(text.split())
    return text if not limit or len(text) <= limit else text[:limit - 1].rsplit(" ", 1)[0] + "…"


def reviews():
    w = pd.read_parquet("data/wb.parquet", columns=["nmId", "stars", "text", "answer"])
    return w[w["nmId"].isin(PRODUCTS)]


def student():
    import torch
    from transformers import AutoTokenizer
    from student_curve import Student, predict
    M = Path("data/student/model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Student(str(M))
    model.load_state_dict(torch.load(M / "heads.pt"), strict=False)
    model.to(device)
    w = reviews()
    need, rea, sent = predict(model, AutoTokenizer.from_pretrained(M), w["text"].tolist(), device)
    D.mkdir(parents=True, exist_ok=True)
    with (D / "student.jsonl").open("w", encoding="utf-8") as f:
        for k, i in enumerate(w.index):
            f.write(json.dumps({"id": int(i), "need": round(float(need[k]), 4), "sentiment": SENTIMENTS[sent[k]],
                                "reasons": {key: round(float(rea[k][j]), 4) for j, key in enumerate(REASON_KEYS)}}) + "\n")
    print(f"student: {len(w)} reviews on {device}")


THEME_PROMPT = """Ты аналитик отзывов маркетплейса. Ниже отзывы покупателей на один товар ({name}), каждый с номером.
Выдели темы, которые повторяются в нескольких отзывах:
- liked: 4–6 тем «что нравится покупателям»;
- questions: 3–5 тем «что вызывает вопросы или недовольство».
Тема — короткая фраза по-русски (3–6 слов), конкретная для этого товара (не «хорошее качество», а например «не давит на живот»).
Для каждой темы перечисли номера ВСЕХ отзывов, где она встречается, и 2 короткие дословные цитаты из разных отзывов (до 120 символов, копируй текст точно).
Ответ — только JSON по схеме."""
THEME = {"type": "object", "additionalProperties": False, "required": ["theme", "reviews", "quotes"],
         "properties": {"theme": {"type": "string"}, "reviews": {"type": "array", "items": {"type": "integer"}},
                        "quotes": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2}}}
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["liked", "questions"],
          "properties": {"liked": {"type": "array", "items": THEME, "minItems": 4, "maxItems": 6},
                         "questions": {"type": "array", "items": THEME, "minItems": 3, "maxItems": 5}}}


def themes():
    w = reviews()
    D.mkdir(parents=True, exist_ok=True)
    out = D / "themes.jsonl"
    done = {json.loads(x)["nmId"] for x in out.read_text(encoding="utf-8").splitlines()} if out.exists() else set()
    for nm, name in PRODUCTS.items():
        if nm in done:
            continue
        s = w[w["nmId"] == nm].sample(frac=1, random_state=0).head(MAX_THEME_REVIEWS)
        listing = "\n".join(f"[{k}] ({r.stars}★) {clean(r.text, 400)}" for k, r in enumerate(s.itertuples(), 1))
        ans, usage = ask(THEME_PROMPT.format(name=name), listing, SCHEMA, 12000)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"nmId": nm, "ids": [int(i) for i in s.index], "answer": ans, "usage": usage},
                               ensure_ascii=False) + "\n")
        print(f"themes {nm}: {len(s)} reviews, {usage}", flush=True)


def ask(system, user, schema, max_tokens):
    body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": "/no_think\n" + user}],
            "temperature": 0.2, "top_p": 0.95, "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}}}
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        res = json.loads(r.read())
    return json.loads(res["choices"][0]["message"]["content"].split("</think>")[-1]), res.get("usage")


TAG_PROMPT = """Ты аналитик отзывов маркетплейса. Товар: {name}. Темы отзывов:
{themes}
Прочитай один отзыв и перечисли номера тем, которые в нём ЯВНО упомянуты (покупатель прямо пишет об этом).
Общая похвала («всё отлично», «рекомендую») не означает конкретную тему. Если ни одна тема не упомянута — пустой список."""


def theme_list(t):
    return [x["theme"] for x in t["answer"]["liked"] + t["answer"]["questions"]]


def tag(workers=4):
    """The model's own review lists for a theme are unreliable (it lists nearly every positive review for each
    "liked" theme), so every review is tagged with the product's themes in its own request; shares come from that."""
    from concurrent.futures import ThreadPoolExecutor
    w = reviews()
    th = [json.loads(x) for x in (D / "themes.jsonl").read_text(encoding="utf-8").splitlines()]
    out = D / "tags.jsonl"
    done = {json.loads(x)["id"] for x in out.read_text(encoding="utf-8").splitlines()} if out.exists() else set()
    for t in th:
        names = theme_list(t)
        system = TAG_PROMPT.format(name=PRODUCTS[t["nmId"]], themes="\n".join(f"{k}. {x}" for k, x in enumerate(names, 1)))
        schema = {"type": "object", "additionalProperties": False, "required": ["themes"],
                  "properties": {"themes": {"type": "array", "items": {"type": "integer", "enum": list(range(1, len(names) + 1))}}}}
        todo = [i for i in t["ids"] if i not in done]

        def one(i):
            ans, _ = ask(system, f"Отзыв ({w.at[i, 'stars']}★): {clean(w.at[i, 'text'], 1000)}", schema, 100)
            return {"nmId": t["nmId"], "id": i, "themes": sorted(set(ans["themes"]))}
        with ThreadPoolExecutor(workers) as ex, out.open("a", encoding="utf-8") as f:
            for n, rec in enumerate(ex.map(one, todo), 1):
                f.write(json.dumps(rec) + "\n")
                f.flush()
        print(f"tags {t['nmId']}: {len(todo)} reviews, {len(names)} themes", flush=True)


def norm(t):
    return re.sub(r"[^а-яёa-z0-9]+", " ", t.lower()).strip()


def theme_rows(items, offset, texts, tags):
    """Share = reviews tagged with the theme / reviews read. Quotes: the model's quotes often come from the wrong part
    of a review, so they are taken from the tagged reviews themselves, single-theme and short ones first."""
    rows = []
    for k, t in enumerate(items, offset + 1):
        own = [j for j in range(len(texts)) if k in tags[j]]
        quotes = [clean(texts[j], 120) for j in sorted(own, key=lambda j: (len(tags[j]) > 1, len(texts[j]) < 25, len(texts[j])))[:2]]
        rows.append({"theme": clean(t["theme"]), "share": round(len(own) / len(texts), 3), "reviews": len(own), "quotes": quotes})
    return sorted(rows, key=lambda r: -r["share"])


def answer_kinds(answers):
    """Seller answers → "template" / "individual" / None (no answer). Template: after normalising (lower case,
    no names, digits, punctuation) it equals or nearly equals (difflib ratio >= 0.9 or a common prefix >= 80%
    of the shorter text) at least 2 other answers on the same product."""
    from difflib import SequenceMatcher
    from os.path import commonprefix
    names = re.compile(r"\b(" + "|".join(NAMES) + r")[а-яё]{0,3}\b", re.I)
    norm_a = [" ".join(re.sub(r"[^а-яёa-z]+", " ", names.sub(" ", a.lower())).split()) if a and a.strip() else None
              for a in answers]

    def near(x, y):
        if not x or not y:
            return x == y
        return x == y or len(commonprefix([x, y])) >= 0.8 * min(len(x), len(y)) or SequenceMatcher(None, x, y).ratio() >= 0.9
    kinds = []
    for i, x in enumerate(norm_a):
        if x is None:
            kinds.append(None)
            continue
        same = sum(1 for j, y in enumerate(norm_a) if j != i and y and near(x, y))
        kinds.append("template" if same >= 2 else "individual")
    return kinds


def build():
    w = reviews()
    st = {r["id"]: r for r in map(json.loads, (D / "student.jsonl").read_text(encoding="utf-8").splitlines())}
    th = {r["nmId"]: r for r in map(json.loads, (D / "themes.jsonl").read_text(encoding="utf-8").splitlines())}
    tg = {r["id"]: set(r["themes"]) for r in map(json.loads, (D / "tags.jsonl").read_text(encoding="utf-8").splitlines())}
    titles = {k: t for k, t in REASON_KEYS.items()}
    products = []
    for nm, name in PRODUCTS.items():
        s = w[w["nmId"] == nm]
        p = [st[int(i)] for i in s.index]
        n = len(s)
        sent = pd.Series([x["sentiment"] for x in p]).value_counts()
        reasons = []
        for key in REASON_KEYS:
            hits = sorted([(x[1]["reasons"][key], x[0]) for x in zip(s["text"], p) if x[1]["reasons"][key] >= REASON_THR],
                          reverse=True)
            reasons.append({"reason": key, "title": titles[key], "share": round(len(hits) / n, 3), "reviews": len(hits),
                            "quotes": [clean(t, 120) for _, t in hits[:2]]})
        t = th[nm]
        texts = [w.at[i, "text"] for i in t["ids"]]
        tags = [tg[i] for i in t["ids"]]
        n_liked = len(t["answer"]["liked"])
        kinds = answer_kinds(s["answer"].fillna("").tolist())
        answered = [k for k in kinds if k]
        example = {k: next((clean(a, 200) for a, kk in zip(s["answer"], kinds) if kk == k), None) for k in ("template", "individual")}
        need = sorted([{"text": clean(txt, 300), "stars": int(stars), "score": round(x["need"], 3),
                        "reasons": [key for key in REASON_KEYS if x["reasons"][key] >= REASON_THR], "seller_answer": kind or "none"}
                       for txt, stars, x, kind in zip(s["text"], s["stars"], p, kinds) if x["need"] >= NEED_THR],
                      key=lambda r: -r["score"])
        products.append({
            "nmId": nm, "product": name, "n": n,
            "stars": {str(k): int(v) for k, v in s["stars"].value_counts().sort_index().items()},
            "avg_stars": round(float(s["stars"].mean()), 2),
            "sentiment": {k: round(float(sent.get(k, 0)) / n, 3) for k in SENTIMENTS},
            "reasons": sorted(reasons, key=lambda r: -r["share"]),
            "themes": {"reviews_read": len(texts), "liked": theme_rows(t["answer"]["liked"], 0, texts, tags),
                       "questions": theme_rows(t["answer"]["questions"], n_liked, texts, tags)},
            "seller_answered": round(len(answered) / n, 3),
            "seller_answers": {"answered": len(answered),
                               "template_share": round(answered.count("template") / max(len(answered), 1), 3),
                               "individual_share": round(answered.count("individual") / max(len(answered), 1), 3),
                               "template_example": example["template"], "individual_example": example["individual"]},
            "need_human": {"threshold": NEED_THR, "count": len(need), "share": round(len(need) / n, 3),
                           "answered": {k: sum(r["seller_answer"] == k for r in need) for k in ("individual", "template", "none")},
                           "reviews": need}})
    meta = {"source": "WB reviews (data/wb.parquet), no dates in the data; niche found by review text",
            "student": "ruBert-base on 20k teacher labels (reports/student.md): need score, reasons ≥ 0.5, sentiment",
            "themes": "Qwen3-14B Q4_K_M, themes named in one request per product over up to 300 reviews, then every review tagged with the themes in its own request; share = tagged reviews / reviews read",
            "seller_answers": "template = after normalising (lower case, no names, digits, punctuation) equal or near "
                              "(difflib ratio >= 0.9 or common prefix >= 80%) to >= 2 other answers on the product",
            "privacy": "names, phones, e-mails and links masked in every quote"}
    raw = {"meta": meta, "products": products}
    # Preserve the unreviewed output locally for rechecking every theme assignment.
    (D / "unreviewed_export.json").write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")
    save_export(raw, OUT, OUT.with_suffix('.html'))
    print(f"wrote {OUT}: " + ", ".join(f"{x['nmId']} n={x['n']} need={x['need_human']['count']}" for x in products))


if __name__ == "__main__":
    {"student": student, "themes": themes, "tag": tag, "build": build}[sys.argv[1]]()
