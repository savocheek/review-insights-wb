#!/usr/bin/env bash
# Resume point: need-only verify (v4, 2 slots, auto-restart; continues where it stopped) → merge → student on all ~20k labels.
# Usage (from this repo's root): nohup bash scripts/verify_then_student.sh > /dev/null 2>&1 &
cd /d/claude/review-insights-wb
export PYTHONUNBUFFERED=1 PYTHONUTF8=1
sleep 60
SKIP_LABEL=1 MODE=full SELECT_FLAGS=--need-only CHECKER=Qwen3.8-27B-Q4_K_M.gguf CHECKER_NAME=qwen3.8-27b-v4 CHECK_PROMPT=v4 CHECK_SLOTS=${CHECK_SLOTS:-2} CHECK_CTX=${CHECK_CTX:-8192} \
  bash scripts/teacher_queue.sh >> teacher_queue.log 2>&1 || exit 1
grep -q "^== .* done$" <(tail -3 teacher_queue.log) || exit 1
echo "== $(date +%H:%M) cool-down 60 s, then the student" >> teacher_queue.log; sleep 60
../ru-review-sentiment/.venv/Scripts/python.exe train_student.py > train_student.log 2>&1 \
  && echo "== $(date +%H:%M) student done" >> teacher_queue.log || echo "== $(date +%H:%M) !! student failed" >> teacher_queue.log
