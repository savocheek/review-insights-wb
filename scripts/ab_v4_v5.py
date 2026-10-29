"""Checker prompt v4 vs v5 (Qwen3.8-27B) on the user's needs-reply labels, batch by batch.

Rule (the user's): v5 wins if it has fewer extra need (false positives) and no more misses (false negatives)
on all of the user's labels outside the few-shot examples; otherwise v4. Prints the table and writes the
chosen version to data/llm_probe/checker_prompt.txt.
Usage (from the repo root): python scripts/ab_v4_v5.py
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, ".")
from llm_probe import FEWSHOT_IDS, REASON_KEYS, gold_labels, load_runs, need_of, reasons_of  # noqa: E402

BATCHES = [("первые 51", "e547116~1"), ("партия 19", "e547116"), ("партия 31", "f96e38a")]


def ids_at(rev):
    out = subprocess.run(["git", "show", f"{rev}:labeling/needs_reply_labels.json"], capture_output=True, check=True)
    return {g["id"] for g in json.loads(out.stdout.decode("utf-8"))["labels"]}


def stats(G, R, ids):
    inv = {v: k for k, v in REASON_KEYS.items()}
    ids = [i for i in ids if int(i) in R and R[int(i)]["answer"]]
    s = {"n": len(ids), "agree": 0, "fp": 0, "fn": 0, "rtp": 0, "rfp": 0, "rfn": 0, "sec": 0.0}
    for i in ids:
        ans, g = R[int(i)]["answer"], G[i]
        s["agree"] += need_of(ans) == g["need"]
        s["fp"] += need_of(ans) and not g["need"]
        s["fn"] += g["need"] and not need_of(ans)
        m, u = {inv[r["reason"]] for r in reasons_of(ans)}, set(g["reasons"])
        s["rtp"], s["rfp"], s["rfn"] = s["rtp"] + len(m & u), s["rfp"] + len(m - u), s["rfn"] + len(u - m)
        s["sec"] += R[int(i)]["seconds"]
    return s


def fmt(s):
    if not s["n"]:
        return "—"
    return (f"{s['agree']}/{s['n']} = {s['agree'] / s['n']:.0%}; лишних {s['fp']}, пропусков {s['fn']}; "
            f"причины P {s['rtp'] / max(s['rtp'] + s['rfp'], 1):.0%} R {s['rtp'] / max(s['rtp'] + s['rfn'], 1):.0%}")


def main():
    G = {g["id"]: g for g in gold_labels()}
    runs = load_runs()
    r4, r5 = runs["qwen3.8-27b-v4"], runs["qwen3.8-27b-v5"]
    seen, groups = set(), []
    for name, rev in BATCHES:
        ids = ids_at(rev)
        groups.append((name, [i for i in G if i in ids and i not in seen and i not in FEWSHOT_IDS]))
        seen |= ids
    rest = [i for i in G if i not in seen and i not in FEWSHOT_IDS]
    if rest:
        groups.append(("новые после f96e38a", rest))
    every = [i for i in G if i not in FEWSHOT_IDS]
    groups.append(("все метки", every))
    print("| Партия | v4 | v5 |\n|---|---|---|")
    for name, ids in groups:
        need_yes = sum(G[i]["need"] for i in ids)
        print(f"| {name} ({len(ids)}, need = да у {need_yes}) | {fmt(stats(G, r4, ids))} | {fmt(stats(G, r5, ids))} |")
    a, b = stats(G, r4, every), stats(G, r5, every)
    full = a["n"] == b["n"] == len(every)
    pick = "v5" if full and b["fp"] < a["fp"] and b["fn"] <= a["fn"] else "v4"
    print(f"\nответы есть: v4 {a['n']}, v5 {b['n']} из {len(every)}; с/отзыв v4 {a['sec'] / max(a['n'], 1):.1f}, "
          f"v5 {b['sec'] / max(b['n'], 1):.1f}")
    print(f"правило: v5, если лишних меньше ({b['fp']} против {a['fp']}) и пропусков не больше ({b['fn']} против {a['fn']})"
          f"{'' if full else ' — ответы неполные, остаётся v4'} → выбор: {pick}")
    Path("data/llm_probe/checker_prompt.txt").write_text(pick, encoding="utf-8")


if __name__ == "__main__":
    main()
