#!/usr/bin/env bash
# LLM teacher probe on the GPU, one model at a time (llama.cpp Vulkan server, port 8080, 60 s cool-downs).
# Each model: 200 reviews without thinking (JSON grammar); Qwen3 also 60 reviews with thinking (the rubric set).
# Large models are fitted to VRAM by llama.cpp (--fit), the rest stays in RAM. Weights wait in D:/llm/models.
# Usage (from this repo's root): bash scripts/llm_queue.sh > llm_queue.log 2>&1
export PYTHONUNBUFFERED=1 PYTHONUTF8=1
PY=${PY:-../ru-review-sentiment/.venv/Scripts/python.exe}
LLAMA=/d/llm/llama-vulkan/llama-server.exe
M=/d/llm/models
step() { echo "== $(date +%H:%M) $*"; }
cool() { step "cool-down 60 s"; sleep 60; }

serve() {  # gguf, log
  "$LLAMA" -m "$M/$1" --device Vulkan0 --fit on --fit-target 1024 -c 8192 -np 1 --jinja -t 8 --port 8080 > "$2" 2>&1 &
  SERVER=$!
  for _ in $(seq 1 200); do curl -s http://127.0.0.1:8080/health | grep -q ok && return 0; sleep 3; done
  step "!! server did not start"; tail -20 "$2"; kill $SERVER; return 1
}

probe() {  # name, gguf, think(0/1)
  until grep -q "$2" /d/llm/dl.log && [ -f "$M/$2" ]; do sleep 60; done  # download finished
  step "$1: server"
  serve "$2" "llm_server_$1.log" || return 1
  grep -iE "offload|layers|fit" "llm_server_$1.log" | tail -4
  step "$1: 200 reviews, no thinking"
  "$PY" llm_probe.py run "$1" --n 200 | tail -2
  if [ "$3" = 1 ]; then
    step "$1: 60 reviews, thinking"
    "$PY" llm_probe.py run "$1-think" --think --n 60 | tail -2
  fi
  kill $SERVER; sleep 5
  cool
}

probe qwen3-14b Qwen3-14B-Q4_K_M.gguf 1
probe qwen3-32b Qwen3-32B-Q4_K_M.gguf 1
probe gemma3-27b gemma-3-27b-it-Q4_K_M.gguf 0
"$PY" llm_probe.py blind
step done
