#!/usr/bin/env bash
# Prompt v2/v3 (PROMPT=v2|v3; needs a live person, reasons apart, few-shot from the user's labels) on the GPU after scripts/llm_queue.sh:
# Qwen3-14B and Qwen3-32B on the user's labelled reviews (few-shot ones excluded) + the 100 random reviews of the 500.
# Usage (from this repo's root): bash scripts/llm_v2.sh > llm_v2.log 2>&1
export PYTHONUNBUFFERED=1 PYTHONUTF8=1
PY=${PY:-../ru-review-sentiment/.venv/Scripts/python.exe}
LLAMA=/d/llm/llama-vulkan/llama-server.exe
M=/d/llm/models
SET=${SET:-gold,random}; PROMPT=${PROMPT:-v2}; MODELS=${MODELS:-"qwen3-14b:Qwen3-14B-Q4_K_M.gguf qwen3-32b:Qwen3-32B-Q4_K_M.gguf"}
step() { echo "== $(date +%H:%M) $*"; }
cool() { step "cool-down 60 s"; sleep 60; }

serve() {  # gguf, log
  "$LLAMA" -m "$M/$1" --device Vulkan0 --fit on --fit-target 1024 -c 8192 -np 1 --jinja -t 8 --port 8080 > "$2" 2>&1 &
  SERVER=$!
  for _ in $(seq 1 200); do curl -s http://127.0.0.1:8080/health | grep -q ok && return 0; sleep 3; done
  step "!! server did not start"; tail -20 "$2"; kill $SERVER; return 1
}

step "waiting for the probe queue"
until grep -qE "^== .* (done|!!)" llm_queue.log; do sleep 60; done
cool
for m in $MODELS; do
  set -- ${m%%:*} ${m#*:}
  step "$1: prompt $PROMPT on $SET"
  serve "$2" "llm_server_$1_$PROMPT.log" || exit 1
  "$PY" llm_probe.py run "$1-$PROMPT" --prompt "$PROMPT" --set "$SET" | tail -2
  kill $SERVER; sleep 5
  cool
done
"$PY" llm_probe.py report
step done
