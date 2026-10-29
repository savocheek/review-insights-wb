#!/usr/bin/env bash
# After both 14B passes: Qwen3.8-27B checker prompt v4 vs v5 on the user's labels, a speed test with 1/2/3 slots on
# real verify work (its answers are kept), then the verify step with the chosen prompt and slots. GPU: one server.
# Usage (from this repo's root): bash scripts/before_verify.sh > before_verify.log 2>&1
export PYTHONUNBUFFERED=1 PYTHONUTF8=1
set -o pipefail
PY=${PY:-../ru-review-sentiment/.venv/Scripts/python.exe}
LLAMA=/d/llm/llama-vulkan/llama-server.exe
GGUF=/d/llm/models/Qwen3.8-27B-Q4_K_M.gguf
step() { echo "== $(date +%H:%M) $*"; }
cool() { step "cool-down 60 s"; sleep 60; }

serve() {  # slots, log
  "$LLAMA" -m "$GGUF" --device Vulkan0 --fit on --fit-target 1024 -c $((8192 * $1)) -np "$1" --jinja -t 8 --port 8080 > "$2" 2>&1 &
  SERVER=$!
  for _ in $(seq 1 200); do curl -s http://127.0.0.1:8080/health | grep -q ok && return 0; sleep 3; done
  step "!! server did not start ($1 slots)"; tail -5 "$2"; kill $SERVER; return 1
}
stop() { kill $SERVER; sleep 5; taskkill //F //IM llama-server.exe > /dev/null 2>&1; }

step "A/B: Qwen3.8-27B v4 (missing answers) and v5 on the user's labels + the 100 random reviews"
serve 1 llm_server_qwen3.8-27b_ab.log || exit 1
"$PY" llm_probe.py run qwen3.8-27b-v4 --prompt v4 --set gold | tail -1
"$PY" llm_probe.py run qwen3.8-27b-v5 --prompt v5 --set gold,random | tail -1
stop; cool
"$PY" scripts/ab_v4_v5.py
PROMPT=$(cat data/llm_probe/checker_prompt.txt)
NAME=qwen3.8-27b-$PROMPT
step "checker prompt: $PROMPT (source: $NAME-confirmed / $NAME-fixed)"

# speed test: the same work per slot count, throughput = reviews per minute
BEST=1; BEST_RATE=0
for slots in 1 2 3; do
  n=$((15 * slots + 5))
  step "speed: $slots slots, $n reviews"
  serve "$slots" "llm_server_qwen3.8-27b_speed$slots.log" || break
  t0=$(date +%s)
  "$PY" teacher_label.py verify --workers "$slots" --limit "$n" --checker "$NAME" --prompt "$PROMPT" | tail -1 || { stop; break; }
  rate=$(( n * 600 / ($(date +%s) - t0) ))  # reviews per 10 min
  step "speed: $slots slots → $rate reviews / 10 min"
  stop; cool
  # more slots only if clearly faster (≥ 15%): fewer slots = less VRAM, less offload, less risk of a hang
  if [ "$rate" -gt $(( BEST_RATE * 115 / 100 )) ]; then BEST=$slots; BEST_RATE=$rate; fi
done
step "chosen: $BEST slots, ~$BEST_RATE reviews / 10 min; to check: $(wc -l < data/teacher/to_verify.jsonl), done: $(wc -l < data/teacher/verify.jsonl)"
echo "$PROMPT $BEST $BEST_RATE" > data/llm_probe/checker_config.txt
step done
