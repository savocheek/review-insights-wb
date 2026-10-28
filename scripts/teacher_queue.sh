#!/usr/bin/env bash
# Teacher labelling on the GPU, one llama.cpp server at a time (60 s cool-downs), every step resumable.
#   MODE=full     14B labels 20k reviews twice (T 0.2 / 0.8), the checker verifies hard cases + 5% control
#   MODE=control  14B labels once, the checker verifies only a random 5% (measures 14B's error)
# Usage (from this repo's root): MODE=full CHECKER=Qwen3.8-27B-Q4_K_M.gguf CHECKER_NAME=qwen3.8-27b bash scripts/teacher_queue.sh > teacher_queue.log 2>&1
#   for a non-Qwen checker add QWEN_FLAG=--no-qwen
export PYTHONUNBUFFERED=1 PYTHONUTF8=1
PY=${PY:-../ru-review-sentiment/.venv/Scripts/python.exe}
LLAMA=/d/llm/llama-vulkan/llama-server.exe
M=/d/llm/models
MODE=${MODE:-full}; CHECKER=${CHECKER:-Qwen3.8-27B-Q4_K_M.gguf}; CHECKER_NAME=${CHECKER_NAME:-qwen3.8-27b}
step() { echo "== $(date +%H:%M) $*"; }
cool() { step "cool-down 60 s"; sleep 60; }

serve() {  # gguf, slots, ctx, log
  "$LLAMA" -m "$M/$1" --device Vulkan0 --fit on --fit-target 1024 -c "$3" -np "$2" --jinja -t 8 --port 8080 > "$4" 2>&1 &
  SERVER=$!
  for _ in $(seq 1 200); do curl -s http://127.0.0.1:8080/health | grep -q ok && return 0; sleep 3; done
  step "!! server did not start"; tail -20 "$4"; kill $SERVER; exit 1
}

[ -f data/teacher/corpus.jsonl ] || "$PY" teacher_label.py corpus
step "14B labels (mode $MODE)"
serve Qwen3-14B-Q4_K_M.gguf 4 32768 llm_server_teacher14.log  # 8k per slot: prompt v2 with few-shot is ~4k tokens
"$PY" teacher_label.py label --pass a --workers 4 || { step "!! label a failed"; kill $SERVER; exit 1; }
kill $SERVER; sleep 5
cool
# an optional GPU window between the passes (e.g. a model probe); the file is removed after it ran
if [ -f scripts/between_passes.sh ]; then
  step "between passes: scripts/between_passes.sh"
  bash scripts/between_passes.sh > between_passes.log 2>&1; rm -f scripts/between_passes.sh
  cool
fi
if [ "$MODE" = full ]; then
  serve Qwen3-14B-Q4_K_M.gguf 4 32768 llm_server_teacher14.log
  "$PY" teacher_label.py label --pass b --workers 4 || { step "!! label b failed"; kill $SERVER; exit 1; }
  kill $SERVER; sleep 5
  cool
fi

step "select"
if [ "$MODE" = full ]; then "$PY" teacher_label.py select; else "$PY" teacher_label.py select --control-only; fi

step "checker $CHECKER_NAME"
serve "$CHECKER" 1 8192 "llm_server_teacher_$CHECKER_NAME.log"
"$PY" teacher_label.py verify --workers 1 --checker "$CHECKER_NAME" $QWEN_FLAG || { step "!! verify failed"; kill $SERVER; exit 1; }
kill $SERVER; sleep 5

step "merge"
"$PY" teacher_label.py merge
git add reports/teacher_labels.md && git commit -qm "Teacher labels report (14B labeller, $CHECKER_NAME checker, mode $MODE)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git log --oneline -1
step done
