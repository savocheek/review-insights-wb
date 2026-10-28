#!/usr/bin/env bash
# Stage 2 GPU queue: starts after the reviews project's round-3 queue (one GPU job at a time, 60 s cool-downs).
# 1. fine-tune the reviews-project student on WB-train (early stopping on WB-val)
# 2. WB-test and RuReviews-test, opened once at the end: student, teacher(s), fine-tuned student
# 3. ONNX int8 of the fine-tuned student on CPU: size, speed, quality
# Usage (from this repo's root): bash scripts/stage2.sh > stage2.log 2>&1
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 TQDM_DISABLE=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1 OMP_NUM_THREADS=8
PY=${PY:-../ru-review-sentiment/.venv/Scripts/python.exe}
SRC=../ru-review-sentiment
step() { echo "== $(date +%H:%M) $*"; }
cool() { step "cool-down 60 s"; sleep 60; }
CPU="env DEVICE=cpu CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES="

step "waiting for round 3"
until grep -qE "^== .* done" "$SRC/round3.log" || [ "$(ps -ef | grep -c '[r]ound3.sh')" = "0" ]; do sleep 60; done
cool

if [ ! -f models/wb_student/train_log.json ]; then
  step "finetune"; "$PY" finetune.py > finetune.log 2>&1 || { step "!! finetune failed"; tail -20 finetune.log; exit 1; }
fi
grep -E "^(train by|start|eval|best)" finetune.log
cool

step "test (opened once)"
"$PY" evaluate.py "$SRC/model" --name student_base
"$PY" evaluate.py "$SRC/model_multi" --name teacher_large
[ -f "$SRC/model_multi_big/multitask.json" ] && "$PY" evaluate.py "$SRC/model_multi_big" --name teacher_large_big
"$PY" evaluate.py models/wb_student --name wb_student

step "ONNX int8 (CPU)"
(cd "$SRC" && $CPU .venv/Scripts/python.exe export_onnx.py ../review-insights-wb/models/wb_student ../review-insights-wb/models/wb_student/onnx) > onnx.log 2>&1
cat models/wb_student/onnx/onnx_report.json
$CPU "$PY" evaluate.py "$SRC/model" --onnx "$SRC/model/onnx/model.int8.onnx" --name student_base_int8
$CPU "$PY" evaluate.py models/wb_student --onnx models/wb_student/onnx/model.int8.onnx --name wb_student_int8
step done
