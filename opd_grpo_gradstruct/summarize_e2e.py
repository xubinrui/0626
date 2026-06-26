from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out-dir", default="results")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    runs = Path(args.runs_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_rows = []
    eval_rows = []
    for train_path in runs.glob("*/*/train.jsonl"):
        train_rows.extend(read_jsonl(train_path))
    for eval_path in runs.glob("*/*/eval.jsonl"):
        eval_rows.extend(read_jsonl(eval_path))
    train = pd.DataFrame(train_rows)
    evals = pd.DataFrame(eval_rows)
    if not evals.empty:
        evals.to_csv(out / "final_summary.csv", index=False)
        skipped = evals["skipped"].astype(bool) if "skipped" in evals.columns else False
        gsm = evals[(evals.get("dataset") == "gsm8k") & (~skipped)]
        if not gsm.empty and {"pass_at_1", "pass_at_8"}.issubset(gsm.columns):
            agg = gsm.groupby("condition").agg(pass_at_1=("pass_at_1", "mean"), pass_at_8=("pass_at_8", "mean")).reset_index()
            plt.figure(figsize=(6, 4))
            plt.scatter(agg["pass_at_8"], agg["pass_at_1"])
            for _, row in agg.iterrows():
                plt.annotate(row["condition"], (row["pass_at_8"], row["pass_at_1"]))
            plt.xlabel("Pass@8")
            plt.ylabel("Pass@1")
            plt.tight_layout()
            plt.savefig(out / "pareto_pass1_vs_pass8.png", dpi=160)
            plt.close()
    else:
        pd.DataFrame().to_csv(out / "final_summary.csv", index=False)

    if not train.empty:
        speed = train.groupby(["condition", "seed"]).agg(
            wall_seconds_mean=("wall_seconds", "mean"),
            wall_seconds_std=("wall_seconds", "std"),
            peak_memory_mb_max=("peak_memory_mb", "max"),
            routing_svd_seconds_mean=("routing_svd_seconds", "mean"),
            recon_error_mean=("recon_error", "mean"),
        )
        speed.to_csv(out / "speed_memory_table.csv")
        plt.figure(figsize=(8, 4))
        for cond, grp in train.groupby("condition"):
            grp = grp.sort_values("step")
            plt.plot(grp["step"], grp["reward_mean"], label=cond)
        plt.xlabel("step")
        plt.ylabel("reward")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "stability_curves.png", dpi=160)
        plt.close()

        comp = train[train["recon_error"] > 0]
        if not comp.empty:
            comp.groupby("condition")["recon_error"].mean().plot(kind="bar")
            plt.ylabel("mean reconstruction error")
            plt.tight_layout()
            plt.savefig(out / "compression_sweep.png", dpi=160)
            plt.close()
        else:
            plt.figure(figsize=(5, 3))
            plt.text(0.5, 0.5, "No compression rows logged", ha="center", va="center")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(out / "compression_sweep.png", dpi=160)
            plt.close()
    else:
        pd.DataFrame().to_csv(out / "speed_memory_table.csv", index=False)


if __name__ == "__main__":
    main()
