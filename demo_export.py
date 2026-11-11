"""Export a demo without presenting unreviewed LLM themes as verified facts."""
import copy
import html
import json
from pathlib import Path


def reviewed_export(source):
    data = copy.deepcopy(source)
    data['meta']['quality_status'] = 'model_predictions_only; themes and supporting quotes withheld pending review'
    data['meta']['themes'] = 'Withheld pending a review of theme assignments and a recount'
    for p in data['products']:
        n = p['n']
        if n <= 0 or sum(p['stars'].values()) != n:
            raise ValueError('Rating counts must sum to product review count')
        if p['need_human']['count'] != len(p['need_human']['reviews']):
            raise ValueError('Need count and candidate list disagree')
        themes = p['themes']
        themes.setdefault('withheld_theme_count', len(themes['liked']) + len(themes['questions']))
        themes.update(liked=[], questions=[], quality_status='withheld_pending_review')
        for r in p['reasons']:
            r['quotes'] = []
            r['quality_status'] = 'model_prediction; supporting_quotes_withheld'
        p['need_human']['quality_status'] = 'model_candidates_not_human_decisions'
    return data


def render_demo(data):
    esc = lambda x: html.escape(str(x), quote=True)
    cards = []
    for p in data['products']:
        stars = ' · '.join(f'{esc(k)}★: {v}' for k, v in sorted(p['stars'].items()))
        sentiment = ''.join(f'<li>{esc(k)}: {v:.0%}</li>' for k, v in p['sentiment'].items())
        reasons = ''.join(f'<li>{esc(r["title"])}: {r["reviews"]} из {p["n"]}</li>'
                          for r in p['reasons'] if r['reviews'])
        candidates = ''.join(f'<li><p>{esc(r["text"])}</p><small>Оценка модели: {r["score"]:.3f}</small></li>'
                             for r in p['need_human']['reviews'])
        cards.append(f'''<article><div class="eyebrow">ТОВАР {p['nmId']}</div><h2>{esc(p['product'])}</h2>
<div class="numbers"><b>{p['n']} отзывов</b><b>{p['avg_stars']:.2f} / 5</b>
<b>{p['need_human']['count']} кандидатов на ответ</b></div><p class="muted">{stars}</p>
<div class="columns"><section><h3>Предсказания модели: тональность</h3><ul>{sentiment}</ul></section>
<section><h3>Предсказания модели: причины</h3><ul>{reasons or '<li>Не отмечены</li>'}</ul></section></div>
<p>Ответ продавца есть у {p['seller_answered']:.0%} отзывов. Это наличие ответа, а не оценка его качества.</p>
<details><summary>Кандидаты на ответ · порог {p['need_human']['threshold']}</summary>
<p class="muted">Нужность ответа ещё не подтверждена человеком. Оценка модели не является калиброванной вероятностью.</p>
<ul>{candidates or '<li>Нет кандидатов при выбранном пороге</li>'}</ul></details></article>''')
    total = sum(p['n'] for p in data['products'])
    return '''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Демо аналитики отзывов</title><style>
*{box-sizing:border-box}body{margin:0;background:#f1f5f9;color:#17233b;font:16px/1.6 system-ui,sans-serif}
main{max-width:1120px;margin:auto;padding:48px 24px}h1{font-size:36px;line-height:1.2}h2{margin:8px 0 20px}
article{background:white;border:1px solid #dbe4ed;border-radius:16px;padding:28px;margin:24px 0}
.eyebrow{font-size:12px;letter-spacing:2px;color:#64748b}.numbers,.columns{display:flex;gap:28px;flex-wrap:wrap}
.numbers b{background:#edf5fb;padding:12px 16px;border-radius:8px}.columns section{flex:1;min-width:240px}
.muted,small{color:#64748b}.notice{padding:20px;background:#fff7df;border-radius:12px}summary{cursor:pointer;font-weight:600}
li{margin:8px 0}li p{margin-bottom:0}@media(max-width:600px){main{padding:24px 12px}article{padding:20px}h1{font-size:28px}}
</style><main><div class="eyebrow">REVIEW INSIGHTS / ДЕМО</div><h1>Отзывы о товарах для беременных</h1>
<p>''' + f"Товаров: {len(data['products'])} · отзывов: {total}" + '''. Оценки покупателей и предсказания исследовательской модели.</p>
<div class="notice"><b>Темы и подтверждающие цитаты временно скрыты.</b> Проверка нашла ошибки отрицаний и
соответствия размера. Доли тем не публикуются до перепроверки разметки. Причины и тональность ниже —
предсказания модели, не ручной эталон. Дат отзывов нет, динамика не рассчитывается.</div>
''' + ''.join(cards) + '</main></html>'


def save_export(source, json_path, html_path):
    data = reviewed_export(source)
    Path(json_path).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8', newline='\n')
    Path(html_path).write_text(render_demo(data), encoding='utf-8', newline='\n')
    return data
