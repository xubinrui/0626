#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/data/xuebinrui/miniconda3/envs/loopai/bin/python}"
SUMMARY_PYTHON_BIN="${SUMMARY_PYTHON_BIN:-python3}"
GPUS="${GPUS:-1,3,5}"
SEED="${SEED:-42}"
NUM_STEPS="${NUM_STEPS:-1}"
LIMIT_PROMPTS="${LIMIT_PROMPTS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
SAVE_MODEL="${SAVE_MODEL:-0}"
RUN_EVAL="${RUN_EVAL:-0}"
USE_FALLBACK_DATA="${USE_FALLBACK_DATA:-1}"
CONDITIONS="${CONDITIONS:-b1 b2 b3 b4 ours b4b ours_minus}"

export CUDA_VISIBLE_DEVICES="$GPUS"
export OPD_GRPO_USE_FALLBACK_DATA="$USE_FALLBACK_DATA"

mkdir -p logs
LOG_PATH="logs/seed${SEED}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "[$(date)] starting OPD-GRPO e2e run"
echo "cwd=$(pwd)"
echo "GPUS=$GPUS"
echo "SEED=$SEED NUM_STEPS=$NUM_STEPS LIMIT_PROMPTS=$LIMIT_PROMPTS MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
echo "SAVE_MODEL=$SAVE_MODEL RUN_EVAL=$RUN_EVAL USE_FALLBACK_DATA=$USE_FALLBACK_DATA"
echo "CONDITIONS=$CONDITIONS"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true

cfg_for_condition() {
  case "$1" in
    b1) echo "configs/e2e_b1.json" ;;
    b2) echo "configs/e2e_b2.json" ;;
    b3) echo "configs/e2e_b3.json" ;;
    b4) echo "configs/e2e_b4.json" ;;
    b4b) echo "configs/e2e_b4b.json" ;;
    ours) echo "configs/e2e_ours.json" ;;
    ours_minus) echo "configs/e2e_ours_minus.json" ;;
    *) echo "unknown condition: $1" >&2; return 1 ;;
  esac
}

train_args=(--seed "$SEED" --num-steps "$NUM_STEPS" --limit-prompts "$LIMIT_PROMPTS" --max-new-tokens "$MAX_NEW_TOKENS")
if [[ "$SAVE_MODEL" == "1" ]]; then
  train_args+=(--save-model)
else
  train_args+=(--no-save-model)
fi

for cond in $CONDITIONS; do
  cfg="$(cfg_for_condition "$cond")"
  echo "[$(date)] train condition=$cond cfg=$cfg"
  "$PYTHON_BIN" -m opd_grpo_gradstruct.train_e2e --config "$cfg" "${train_args[@]}"
done

if [[ "$RUN_EVAL" == "1" ]]; then
  if [[ "$SAVE_MODEL" != "1" ]]; then
    echo "WARNING: RUN_EVAL=1 but SAVE_MODEL=0; eval will fall back to the base student model."
  fi
  for cond in $CONDITIONS; do
    cfg="$(cfg_for_condition "$cond")"
    echo "[$(date)] eval condition=$cond cfg=$cfg"
    "$PYTHON_BIN" -m opd_grpo_gradstruct.eval_e2e --config "$cfg" --seed "$SEED"
  done
fi

echo "[$(date)] summarizing"
"$SUMMARY_PYTHON_BIN" -m opd_grpo_gradstruct.summarize_e2e --runs-dir runs --out-dir results
echo "[$(date)] done; log=$LOG_PATH"
