"""Local page (Gradio) for reviewing the complaint groups of topics.py: one group per screen.

For each group: draft name, keywords, share, stars, typical phrases and random ones. Actions: name the group,
merge it into another group (its phrases count there), or throw it out as junk. Every decision is appended to
data/topic_labels.jsonl at once (last decision per group wins) and reports/topics.md is rebuilt, so a restart
resumes at the first undecided group. Nothing is approved without the user: until then names stay drafts.

Usage: python topics_label.py [--port 7863]
"""
import argparse
import json
import time
from pathlib import Path

import gradio as gr
import pandas as pd

from topics import LABELS_FILE, decisions, resolve, write_md

ACTIONS = ["назвать", "объединить с другой группой", "мусор"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7863)
    a = ap.parse_args()
    groups = json.loads(Path("reports/topics.json").read_text(encoding="utf-8"))["groups"]
    ids = [g["id"] for g in groups]
    by_id = {g["id"]: g for g in groups}
    ph = pd.read_parquet("data/complaints.parquet", columns=["text", "n"]).assign(
        topic=pd.read_parquet("data/topics.parquet")["topic"].values)

    def first_open(start=0):
        done = decisions()
        for k in list(range(start, len(ids))) + list(range(start)):
            if ids[k] not in done:
                return k
        return min(start, len(ids) - 1)

    def targets(i):
        names = resolve(groups)
        return [f"{g} · {names[g][0]}" for g in ids if g != ids[i] and names[g][0] is not None]

    def show(i):
        g, done, names = by_id[ids[i]], decisions(), resolve(groups)
        d = done.get(g["id"])
        head = (f"**Группа {i + 1} / {len(ids)}** · решено {len(done)} из {len(ids)}"
                + (f" · ✅ уже решена: {d['action']}" if d else "") + (" · 🎉 все группы решены" if len(done) == len(ids) else ""))
        info = (f"### {names[g['id']][0] or '—'}{'' if names[g['id']][1] else ' (черновик)'}\n"
                f"Доля жалоб **{g['share']:.1%}** · фраз {g['phrases']:,} · сходство с центром {g['cohesion']}"
                f"{' · смешанная' if g['mixed'] else ''}\n\n"
                f"Звёзды 1–5: {' / '.join(f'{x:.0%}' for x in g['stars'])}\n\n"
                f"Ключевые слова: {', '.join(g['keywords'])}\n\n**Типичные фразы**\n\n"
                + "\n".join(f"- {e}" for e in g["examples"]))
        sample = ph[ph["topic"] == g["id"]].sample(min(15, g["phrases"]), random_state=len(done))
        rnd = "**Случайные фразы группы**\n\n" + "\n".join(f"- {t}" for t in sample["text"])
        action = {"name": ACTIONS[0], "merge": ACTIONS[1], "junk": ACTIONS[2]}.get(d["action"]) if d else ACTIONS[0]
        name = d.get("name") if d and d["action"] == "name" else names[g["id"]][0] or ""
        into = None
        if d and d["action"] == "merge":
            into = next((t for t in targets(i) if t.startswith(f"{d['into']} · ")), None)
        return i, head, info, rnd, action, name, gr.update(choices=targets(i), value=into), ""

    def save(i, action, name, into):
        rec = {"id": ids[i], "ts": round(time.time())}
        if action == ACTIONS[0]:
            if not name.strip():
                raise gr.Error("Введите название группы")
            rec.update(action="name", name=name.strip())
        elif action == ACTIONS[1]:
            if not into:
                raise gr.Error("Выберите группу, с которой объединить")
            rec.update(action="merge", into=int(into.split(" · ")[0]))
        else:
            rec.update(action="junk")
        with LABELS_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        write_md()
        return first_open(i + 1), ""

    with gr.Blocks(title="Группы жалоб") as demo:
        gr.Markdown("## Группы жалоб покупателей: назвать, объединить или выбросить")
        idx = gr.State(first_open())
        head = gr.Markdown()
        with gr.Row():
            info = gr.Markdown()
            rnd = gr.Markdown()
        action = gr.Radio(ACTIONS, label="Что сделать с группой")
        name = gr.Textbox(label="Название (для «назвать»)")
        into = gr.Dropdown([], label="С какой группой объединить (для «объединить»)")
        msg = gr.Markdown()
        with gr.Row():
            back = gr.Button("← Назад")
            go = gr.Button("Сохранить и дальше", variant="primary")
            skip = gr.Button("К первой нерешённой")
        outputs = [idx, head, info, rnd, action, name, into, msg]
        demo.load(show, idx, outputs)
        go.click(save, [idx, action, name, into], [idx, msg]).success(show, idx, outputs)
        back.click(lambda i: max(i - 1, 0), idx, idx).then(show, idx, outputs)
        skip.click(lambda: first_open(), None, idx).then(show, idx, outputs)
    demo.launch(server_port=a.port)


if __name__ == "__main__":
    main()
