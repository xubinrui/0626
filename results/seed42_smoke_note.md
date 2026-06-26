# Seed 42 Smoke Result Note

Scope: this is a pipeline validation run only.

- Seeds: `42` only.
- Training scale: 7 conditions x 1 step x 1 fallback GSM8K-style prompt.
- Group size: 8.
- Student: `/data/liuhao/local_models/Qwen3_1B7_Instruct`.
- Teacher: `/data/qinhanyu/models/Qwen3-8B`.
- GPU mapping used for smoke: `CUDA_VISIBLE_DEVICES=3,5`, so config `cuda:0/cuda:1`
  mapped to physical GPUs 3 and 5.

Observed smoke status:

- All 7 training conditions wrote `runs/{condition}/42/train.jsonl`.
- Routed conditions realized hard fraction `0.19921875`, within the requested
  `[0.15, 0.25]` interval.
- B4 and OURS used compressed easy-token OPD channels and logged reconstruction error.
- Collapse flag files were written and are empty for this 1-step smoke.
- `eval_e2e.py` wrote a GSM8K fallback eval record for B1 and skipped MATH/AIME in
  offline smoke mode.
- Summary artifacts were generated in `results/`.

Hypothesis status:

The hypotheses in `HYPOTHESES.md` are not evaluated by this smoke. They require
real held-out eval, longer runs, and at least 3 seeds per condition.
