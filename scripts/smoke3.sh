#!/usr/bin/env bash
# CPU-only checks before stage 3: waits for the CPU ONNX re-export of the reviews project, then the encoder probe
# (10k phrases) and a tiny end-to-end run of the stage-3 scripts (outputs are overwritten by the real run).
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 TQDM_DISABLE=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1 OMP_NUM_THREADS=8
export DEVICE=cpu CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES=
PY=${PY:-../ru-review-sentiment/.venv/Scripts/python.exe}
step() { echo "== $(date +%H:%M) $*"; }
until [ "$(grep -c "onnx exit" ../ru-review-sentiment/onnx_redo.log 2>/dev/null)" = "2" ]; do sleep 30; done
step "probe"; "$PY" topics_probe.py > probe.log 2>&1 && step "probe ok" || step "!! probe failed"
step "smoke"
"$PY" complaints.py --limit 300 && "$PY" topics.py --k 5 && "$PY" complaints.py --split test --limit 3000 \
  && "$PY" product_report.py --products 2 && step "smoke ok" || step "!! smoke failed"
rm -f data/complaint_emb_*.npy
