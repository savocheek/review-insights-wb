"""Local hand-labelling page (Gradio): does a WB review need a human answer from the seller, and why.

One review per screen, the text only: stars and the seller's answer are hidden so they do not prompt. Reasons
(docs/needs_reply.md) are pre-ticked by keyword rules — a draft to check. Any reason means "needs an answer",
«ответ не нужен» excludes the reasons. Every answer is appended to data/needs_reply_labels.jsonl at once (last
answer per id wins), so a restart resumes at the first unlabelled review.
Hotkeys (any keyboard layout): 1…7 reasons, 0 «ответ не нужен», Enter save and next, Backspace back.

Usage: python needs_reply_label.py [--port 7864]   (items: python needs_reply_prep.py)
"""
import argparse
import json
import time
from pathlib import Path

import gradio as gr

from needs_reply_prep import NONE, REASONS

CHOICES = [*REASONS, NONE]
ITEMS = Path("data/needs_reply_items.jsonl")
OUT = Path("data/needs_reply_labels.jsonl")

HOTKEYS = """<script>
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "TEXTAREA" || (e.target.tagName === "INPUT" && e.target.type === "text")) return;
  if (e.ctrlKey || e.altKey || e.metaKey) return;
  const boxes = [...document.querySelectorAll("#reasons input[type=checkbox]")];
  const digit = e.code.startsWith("Digit") ? +e.code.slice(5) : -1;
  let hit = true;
  if (digit >= 1 && digit <= 7) boxes[digit - 1]?.click();
  else if (digit === 0) boxes[7]?.click();
  else if (e.code === "Enter") document.querySelector("#save")?.click();
  else if (e.code === "Backspace") document.querySelector("#back")?.click();
  else hit = false;
  if (hit) e.preventDefault();
});
</script>"""


def load():
    done = {}
    if OUT.exists():
        for line in OUT.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["id"]] = r
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7864)
    a = ap.parse_args()
    items = [json.loads(x) for x in ITEMS.read_text(encoding="utf-8").splitlines()]

    def first_open(start=0):
        done = load()
        for k in list(range(start, len(items))) + list(range(start)):
            if items[k]["id"] not in done:
                return k
        return min(start, len(items) - 1)

    def show(i):
        done = load()
        x, r = items[i], done.get(items[i]["id"])
        head = (f"**{i + 1} / {len(items)}** · размечено {len(done)} из {len(items)}"
                + (" · ✅ этот уже размечен" if r else " · галочки — черновик по ключевым словам, проверьте")
                + (" · 🎉 всё размечено, можно закрывать" if len(done) == len(items) else ""))
        value = (r["reasons"] or [NONE]) if r else (x["pre_reasons"] or [NONE])
        return i, head, x["text"], gr.update(value=value), ""

    def save(i, reasons):
        if not reasons:
            raise gr.Error("Отметьте причину или «ответ не нужен»")
        reasons = [x for x in reasons if x != NONE]
        rec = {"id": items[i]["id"], "need": bool(reasons), "reasons": reasons, "ts": round(time.time())}
        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return first_open(i + 1), ""

    with gr.Blocks(title="Нужен ответ человека") as demo:
        gr.Markdown("## Нужен ли продавцу ответить на этот отзыв лично? Инструкция: docs/needs_reply.md")
        idx = gr.State(first_open())
        head = gr.Markdown()
        body = gr.Markdown(elem_id="review")
        reasons = gr.CheckboxGroup(CHOICES, label="Причины (1…7), «ответ не нужен» (0)", elem_id="reasons")
        last = gr.State([])
        msg = gr.Markdown()
        with gr.Row():
            back = gr.Button("← Назад (Backspace)", elem_id="back")
            go = gr.Button("Сохранить и дальше (Enter)", variant="primary", elem_id="save")
            skip = gr.Button("К первому неразмеченному")
        outputs = [idx, head, body, reasons, msg]
        # a reason and «ответ не нужен» exclude each other: the one just ticked wins
        reasons.input(lambda new, old: [NONE] if NONE in new and NONE not in old else [x for x in new if x != NONE],
                      [reasons, last], reasons)
        reasons.change(lambda x: x, reasons, last)
        demo.load(show, idx, outputs)
        go.click(save, [idx, reasons], [idx, msg]).success(show, idx, outputs)
        back.click(lambda i: max(i - 1, 0), idx, idx).then(show, idx, outputs)
        skip.click(lambda: first_open(), None, idx).then(show, idx, outputs)
    demo.launch(server_port=a.port, head=HOTKEYS, css="#review { font-size: 19px; line-height: 1.5; }")


if __name__ == "__main__":
    main()
