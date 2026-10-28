"""Stage 4 prep, CPU only: "does this review need a human answer?"

1. Weak signals from the seller answers: every answer is none / template / unique-short / unique-substantive.
   Template = its skeleton (lower case, digits → 0, mid-sentence capitalised words such as names → Х) is used for at
   least two reviews. Substantive = a unique answer of ≥ 20 words that deals with the case (exchange, return,
   defect, size, composition, a request to contact…). A hint of what sellers think deserves a real answer, not truth.
   How this relates to stars and to the reason keywords → reports/answer_signals.md.
2. ~500 WB-train reviews for hand labelling: ~80% "interesting" (keywords of every reason, questions, low stars,
   substantive seller answers), ~20% random. Rule pre-labels (keywords) are drafts for the labeller to fix.
   → data/needs_reply_items.jsonl (the page needs_reply_label.py hides stars and the seller answer).

Usage: python needs_reply_prep.py
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Reason → keyword stems (the labeller's guide with criteria and examples: docs/needs_reply.md)
REASONS = {
    "брак/поломка": r"брак|бракован|слома|сломал|не работа|перестал|треснул|трещин|дыр[аку]|порвал|разошл|оторвал|"
                    r"отвалил|протека|не включа|не заряжа|скол",
    "возврат/деньги": r"возврат|вернуть|верните|вернула|вернул|деньг|списал|оплат|компенсац|претенз|не прин[яи]",
    "не пришло / не то / некомплект": r"не пришл|не пришел|не доставил|пришл[аио]? не т|прислали не|не то[тй]? |"
                                     r"другой цвет|другого цвета|другой размер|некомплект|не хватает|нет в комплект|"
                                     r"пуст[аяо]я? коробк|пересорт",
    "вопрос к продавцу": r"\?|подскажите|подскажете|как (?:пользоваться|стирать|настроить|собрать)|где (?:найти|взять)",
    "здоровье/безопасность": r"аллерг|сыпь|раздражен|ожог|обжег|обожгл|травм|порезал|химическ|токсичн|отрав|"
                             r"удушлив|резкий запах|воня|вонь|опасн|задохн|током",
    "ложная информация о товаре": r"не соответству|не как на фото|не как в описан|описани[ею]|на фото друг|"
                                  r"состав не|не тот состав|маломер|большемер|размер не соответ|размерн[аяой]+ сетк|"
                                  r"обман|введен[ыо] в заблужден|фейк|подделк|не оригинал",
    "сильный негатив с конкретикой": r"ужас|кошмар|отврат|никогда больше|не советую|не рекомендую|не покупайте|"
                                     r"выброшен|деньги на ветер|позор",
}
NONE = "ответ не нужен"
SUBSTANTIVE = re.compile(r"обмен|возврат|верн|брак|замен|размер|состав|обратит|свяжит|напишите|служб|поддержк|"
                         r"проверим|передали|разбер|компенс|инструкц|гаранти|сервис|\?", re.I)
_RX = {r: re.compile(p, re.I) for r, p in REASONS.items()}
N_ITEMS, RANDOM_SHARE = 500, 0.2


def skeleton(t):
    t = re.sub(r"(?<![.!?]\s)(?<!^)\b[А-ЯЁ][а-яё]+", "Х", t)
    t = re.sub(r"\d+", "0", t.lower())
    return " ".join(re.sub(r"[^\w\s]", " ", t).split())


def answer_kinds(answers):
    sk = answers.map(skeleton)
    reused = sk.map(sk[answers != ""].value_counts()).fillna(0) > 1
    words = answers.str.split().str.len()
    kind = np.select([answers == "", reused, words < 20, answers.str.contains(SUBSTANTIVE)],
                     ["нет ответа", "шаблон", "уникальный короткий", "уникальный по существу"], "уникальный общий")
    return pd.Series(kind, index=answers.index)


def rule_reasons(text):
    return [r for r, rx in _RX.items() if rx.search(text)]


def signals_md(df):
    kinds = ["нет ответа", "шаблон", "уникальный короткий", "уникальный общий", "уникальный по существу"]
    md = ["# Слабые сигналы: как продавцы отвечают на отзывы\n",
          f"WB, все сплиты, {len(df):,} отзывов. Вид ответа продавца — подсказка, где продавец счёл нужным ответить "
          "по-человечески, а не истина: многие продавцы не отвечают вовсе, а уникальный ответ бывает сгенерирован.\n",
          "Шаблон — «скелет» ответа (нижний регистр, цифры → 0, имена посреди фразы → Х) повторяется хотя бы дважды. "
          "По существу — уникальный ответ от 20 слов про обмен, возврат, брак, размер, состав, просьбу связаться и т. п.\n",
          "## Вид ответа по звёздам\n", "| Звёзды | Отзывов | " + " | ".join(kinds) + " |", "|---" * (len(kinds) + 2) + "|"]
    for s in range(1, 6):
        d = df[df["stars"] == s]["answer_kind"].value_counts(normalize=True)
        md.append(f"| {s}★ | {(df['stars'] == s).sum():,} | " + " | ".join(f"{d.get(k, 0):.1%}" for k in kinds) + " |")
    md += ["\n## Вид ответа по причинам (ключевые слова в отзыве)\n",
           "Среди отзывов, где продавец вообще ответил: доля ответов по существу. Для сравнения — все отвеченные.\n",
           "| Причина (по ключевым словам) | Отзывов | Средняя оценка | Отвечено | Из отвеченных — по существу |",
           "|---|---|---|---|---|"]
    answered = df["answer_kind"] != "нет ответа"
    rows = [("все отзывы", np.ones(len(df), bool))] + [(r, df["reasons"].map(lambda x, r=r: r in x).values) for r in REASONS]
    rows.append(("без ключевых слов", df["reasons"].map(len).values == 0))
    for name, m in rows:
        sub = answered & m
        md.append(f"| {name} | {m.sum():,} | {df['stars'][m].mean():.2f}★ | {sub.sum() / max(m.sum(), 1):.0%} | "
                  f"{(df['answer_kind'][sub] == 'уникальный по существу').mean():.1%} |")
    return "\n".join(md) + "\n"


def main():
    df = pd.read_parquet("data/wb.parquet")
    df["answer_kind"] = answer_kinds(df["answer"])
    df["reasons"] = df["text"].map(rule_reasons)
    Path("reports/answer_signals.md").write_text(signals_md(df), encoding="utf-8")

    rng = np.random.default_rng(0)
    pool = df[(df["split"] == "train") & (df["text"].str.split().str.len() <= 150)]
    picked = {}

    def take(mask, n, kind):
        idx = pool.index[mask & ~pool.index.isin(list(picked))]
        for i in rng.choice(idx, min(n, len(idx)), replace=False):
            picked[int(i)] = kind

    n_interesting = int(N_ITEMS * (1 - RANDOM_SHARE))
    for r in REASONS:
        take(pool["reasons"].map(lambda x, r=r: r in x).values, 35, f"причина: {r}")
    take(pool["answer_kind"].eq("уникальный по существу").values, 55, "ответ продавца по существу")
    take(pool["stars"].le(2).values & pool["reasons"].map(len).eq(0).values, n_interesting - len(picked), "1–2★ без ключевых слов")
    take(np.ones(len(pool), bool), N_ITEMS - len(picked), "случайный")
    items = []
    for i in rng.permutation(list(picked)):
        x = pool.loc[i]
        items.append({"id": int(i), "text": x["text"], "stars": int(x["stars"]), "nmId": int(x["nmId"]),
                      "answer": x["answer"], "answer_kind": x["answer_kind"], "kind": picked[int(i)],
                      "pre_need": bool(x["reasons"]), "pre_reasons": list(x["reasons"])})
    with open("data/needs_reply_items.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(it, ensure_ascii=False) + "\n" for it in items)
    kinds = pd.Series([it["kind"] for it in items]).value_counts()
    print(f"{len(items)} items, pre-labelled 'needs reply': {sum(it['pre_need'] for it in items)}")
    print(kinds.to_string())


if __name__ == "__main__":
    main()
