"""Seller report prototype: "top problems of a product" for a few real WB-test products with many reviews.

Complaint phrases of WB-test (complaints.py --split test) go to the nearest complaint group of topics.py (groups
marked junk by the user are dropped, merged groups count as their target). Per product: reviews and stars, share
of reviews with a complaint, problems by share of complaint mentions with 2–3 phrases each. The data has no review
dates, so there is no dynamics over time. Group names are drafts until approved in topics_label.py.

Usage: python product_report.py [--products 5] → reports/product_report.html
"""
import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from topics import embed, resolve

TOP = 8  # problems shown per product


def bar(share, color="#c2410c"):
    return (f'<div class="bar"><div style="width:{min(share, 1) * 100:.1f}%;background:{color}"></div></div>')


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--products", type=int, default=5)
    a = p.parse_args()
    info = json.loads(Path("reports/topics.json").read_text(encoding="utf-8"))
    groups, meta = info["groups"], info["meta"]
    names = resolve(groups)
    centers = np.load("data/topic_centers.npy")
    wb = pd.read_parquet("data/wb.parquet", columns=["nmId", "stars", "split"])
    test = wb[(wb["split"] == "test") & (wb["nmId"] != 0)]
    counts = test.groupby("nmId").size().sort_values(ascending=False)
    products = counts.index[:a.products].tolist()

    ph = pd.read_parquet("data/complaints_test.parquet")
    ph = ph[ph["nmId"].isin(products)].reset_index(drop=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = embed(ph["text"].tolist(), meta["encoder"], device)
    sim = X @ centers.T
    ph["topic"], ph["sim"] = sim.argmax(1), sim.max(1)
    ph["name"] = ph["topic"].map(lambda t: names[int(t)][0])
    approved = ph["topic"].map(lambda t: names[int(t)][1])
    ph = ph[ph["name"].notna()]

    cards = []
    for nm in products:
        rev = test[test["nmId"] == nm]
        c = ph[ph["nmId"] == nm]
        stars = rev["stars"].value_counts(normalize=True).reindex(range(1, 6), fill_value=0)
        rows = []
        for name, g in sorted(c.groupby("name"), key=lambda x: -len(x[1]))[:TOP]:
            ex = g.sort_values("sim", ascending=False).drop_duplicates("text")["text"].head(3)
            draft = "" if approved[g.index].all() else ' <span class="draft">черновик</span>'
            rows.append(f"<tr><td>{html.escape(name)}{draft}</td><td>{len(g) / len(c):.0%}</td><td>{bar(len(g) / len(c))}</td>"
                        f"<td>{g['review'].nunique()}</td><td><ul>" + "".join(f"<li>{html.escape(e)}</li>" for e in ex)
                        + "</ul></td></tr>")
        cards.append(
            f'<section><h2>Товар <a href="https://www.wildberries.ru/catalog/{nm}/detail.aspx">{nm}</a></h2>'
            f'<p class="meta">{len(rev):,} отзывов · средняя оценка {rev["stars"].mean():.2f}★ · '
            f'отзывов с жалобой: {c["review"].nunique() / len(rev):.0%} · жалоб (фраз): {len(c):,}</p>'
            '<div class="stars">' + "".join(f"<span>{s}★ {stars[s]:.0%}</span>" for s in range(5, 0, -1)) + "</div>"
            '<table><thead><tr><th>Проблема</th><th>Доля жалоб</th><th></th><th>Отзывов</th><th>Примеры</th></tr></thead>'
            "<tbody>" + "".join(rows) + "</tbody></table></section>")

    page = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Топ проблем товара</title><style>
:root {{ --bg:#fafaf9; --fg:#1c1917; --muted:#78716c; --card:#fff; --line:#e7e5e4; --track:#f5f5f4; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#1c1917; --fg:#f5f5f4; --muted:#a8a29e; --card:#292524;
  --line:#44403c; --track:#3a3532; }} }}
body {{ margin:0; padding:24px 16px; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; }}
main {{ max-width:980px; margin:0 auto; }} h1 {{ font-size:24px; margin:0 0 4px; }}
.note {{ color:var(--muted); margin:0 0 20px; }} section {{ background:var(--card); border:1px solid var(--line);
  border-radius:10px; padding:16px; margin-bottom:18px; overflow-x:auto; }}
h2 {{ font-size:18px; margin:0 0 4px; }} a {{ color:inherit; }} .meta {{ color:var(--muted); margin:0 0 8px; }}
.stars span {{ display:inline-block; margin-right:12px; color:var(--muted); font-size:13px; }}
table {{ width:100%; border-collapse:collapse; margin-top:10px; }} th, td {{ text-align:left; vertical-align:top;
  padding:6px 8px; border-top:1px solid var(--line); }} th {{ font-size:13px; color:var(--muted); font-weight:500; }}
td ul {{ margin:0; padding-left:18px; }} .bar {{ width:120px; height:8px; background:var(--track); border-radius:4px;
  margin-top:7px; }} .bar div {{ height:100%; border-radius:4px; }}
.draft {{ font-size:11px; color:var(--muted); border:1px solid var(--line); border-radius:4px; padding:0 4px; }}
</style></head><body><main>
<h1>Топ проблем товара — прототип отчёта для продавца</h1>
<p class="note">{len(products)} товаров из WB-test с наибольшим числом отзывов (модель их не видела при обучении).
Жалоба — негативная фраза отзыва; проблема — её группа из reports/topics.md. <b>Названия групп черновые</b>, пока
их не утвердит человек. Дат у отзывов в данных нет, поэтому динамики по времени нет.</p>
{''.join(cards)}</main></body></html>"""
    Path("reports/product_report.html").write_text(page, encoding="utf-8")
    print("reports/product_report.html:", products)


if __name__ == "__main__":
    main()
