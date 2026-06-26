#!/usr/bin/env bash
set -euo pipefail

export OPD_GRPO_USE_FALLBACK_DATA="${OPD_GRPO_USE_FALLBACK_DATA:-0}"
PYTHON_BIN="${PYTHON_BIN:-/data/xuebinrui/miniconda3/envs/loopai/bin/python}"
SUMMARY_PYTHON_BIN="${SUMMARY_PYTHON_BIN:-python3}"

for cfg in \
  configs/e2e_b1.json \
  configs/e2e_b2.json \
  configs/e2e_b3.json \
  configs/e2e_b4.json \
  configs/e2e_ours.json \
  configs/e2e_b4b.json \
  configs/e2e_ours_minus.json
do
  "$PYTHON_BIN" -m opd_grpo_gradstruct.train_e2e --config "$cfg" --seed 42 --num-steps 1
done

"$SUMMARY_PYTHON_BIN" -m opd_grpo_gradstruct.summarize_e2e --runs-dir runs --out-dir results
