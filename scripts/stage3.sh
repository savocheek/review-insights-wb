#!/usr/bin/env bash
# Stage 3 GPU queue: frequent customer problems. Starts after stage 2 (one GPU job at a time, 60 s cool-downs).
# 1. complaint phrases of WB-train (sentiment model: WB fine-tuned if better on WB-val, else the source student)
# 2. sentence embeddings + KMeans → groups, reports/topics.md
# 3. complaint phrases of WB-test + the seller report prototype reports/product_report.html
# Every piece is committed (no push). Usage (from this repo's root): bash scripts/stage3.sh > stage3.log 2>&1
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 TQDM_DISABLE=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1 OMP_NUM_THREADS=8
PY=${PY:-../ru-review-sentiment/.venv/Scripts/python.exe}
step() { echo "== $(date +%H:%M) $*"; }
cool() { step "cool-down 60 s"; sleep 60; }
commit() { git add "$@" && git commit -qm "$MSG

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git log --oneline -1; }

step "waiting for stage 2"
until grep -qE "^== .* (done|!!)" stage2.log && ! ps -ef | grep -qE "[s]tage2.sh|[r]ound3_resume.sh|[t]opics_probe|[s]moke3"; do
  sleep 60; done
[ -f reports/topics_probe.json ] || { step "!! no reports/topics_probe.json (run topics_probe.py first)"; exit 1; }
rm -f data/complaint_emb_*.npy  # embeddings of the smoke test, if any
cool

# stage 2 PyTorch rows were computed before evaluate.py scored the fixed balanced 30k WB sample: re-run them so
# PyTorch and ONNX are compared on the same data (a re-run replaces the earlier row in reports/eval.md)
step "stage 2: PyTorch on WB-test full + balanced 30k"
SRC=../ru-review-sentiment
"$PY" evaluate.py "$SRC/model" --name student_base | cut -c1-160
"$PY" evaluate.py "$SRC/model_multi" --name teacher_large | cut -c1-160
"$PY" evaluate.py "$SRC/model_multi_big" --name teacher_large_big | cut -c1-160
"$PY" evaluate.py models/wb_student --name wb_student | cut -c1-160
cool

step "complaints (train)"
"$PY" complaints.py > complaints.log 2>&1 || { step "!! complaints failed"; tail -20 complaints.log; exit 1; }
tail -1 complaints.log
cool

step "topics"
"$PY" topics.py > topics.log 2>&1 || { step "!! topics failed"; tail -20 topics.log; exit 1; }
tail -1 topics.log
MSG="Stage 3: complaint groups of WB-train (reports/topics.md, draft names)" commit reports/topics.md reports/topics.json
cool

step "complaints (test) + seller report"
"$PY" complaints.py --split test > complaints_test.log 2>&1 || { step "!! complaints test failed"; tail -20 complaints_test.log; exit 1; }
"$PY" product_report.py > product_report.log 2>&1 || { step "!! product report failed"; tail -20 product_report.log; exit 1; }
tail -1 product_report.log
MSG="Stage 3: seller report prototype for 5 WB-test products" commit reports/product_report.html
step done
