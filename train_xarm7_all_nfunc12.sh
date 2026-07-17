#!/usr/bin/env bash
# Native N=12 Bernstein SDF training for all 8 xArm7 links (base + link1..7).
# Uses the from-scratch recipe in train_link5_nfunc12.py: recursive-least-squares
# value-fit init + geometric fine-tune (ray/eikonal/direction) for good gradients.
# NOT degree-elevation from the N=8 RDF seed. Candidates land in an isolated dir;
# production models in panda_test/Models are only replaced after validation.
set -euo pipefail
cd "$(dirname "$0")"

PY=/home/flanthier/Github/src/vision_processing/venv_sam3/bin/python3
OUT=panda_test/Models/nfunc12_candidate
LOG=nfunc12_train.log
LINKS=(xarm7_link_base xarm7_link1 xarm7_link2 xarm7_link3 xarm7_link4 xarm7_link5 xarm7_link6 xarm7_link7)

mkdir -p "$OUT"
: > "$LOG"
echo "START $(date)  ->  $OUT" | tee -a "$LOG"
for L in "${LINKS[@]}"; do
  echo "=================================================================" | tee -a "$LOG"
  echo ">>> [$L] $(date +%H:%M:%S)" | tee -a "$LOG"
  t0=$(date +%s)
  "$PY" train_link5_nfunc12.py --link "$L" --n-func 12 \
      --rls-iters 80 --epochs 1500 --ray-points 20000 \
      --output-dir "$OUT" >>"$LOG" 2>&1
  echo ">>> [$L] done in $(( $(date +%s) - t0 ))s" | tee -a "$LOG"
done
echo "ALL_DONE $(date)" | tee -a "$LOG"
