from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .io import write_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/smoke_synthetic")
    parser.add_argument("--prompts", type=int, default=6)
    parser.add_argument("--n-tokens", type=int, default=96)
    parser.add_argument("--vocab", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def make_matrix(rng: np.random.Generator, arm: str, n: int, v: int):
    if arm == "grpo":
        rank = 16
        sparse_frac = 0.35
    elif arm == "opd":
        rank = 6
        sparse_frac = 0.15
    else:
        rank = 8
        sparse_frac = 0.20

    u = rng.normal(size=(n, rank)).astype(np.float32)
    z = rng.normal(size=(rank, v)).astype(np.float32)
    low = (u @ z) / np.sqrt(rank * v)
    sparse = np.zeros((n, v), dtype=np.float32)
    rows = rng.choice(n, size=max(1, int(n * sparse_frac)), replace=False)
    cols = rng.integers(0, v, size=(len(rows), 8))
    for i, row in enumerate(rows):
        sparse[row, cols[i]] = rng.normal(0, 0.25, size=cols.shape[1])
    noise = rng.normal(0, 0.002, size=(n, v)).astype(np.float32)
    G = low + sparse + noise
    token_loss = rng.gamma(shape=2.0, scale=1.0, size=n).astype(np.float32)
    entropy = rng.normal(4.0, 0.4, size=n).astype(np.float32)
    token_loss[rows] += 2.0
    entropy[rows] += 0.8
    hard = (token_loss >= np.quantile(token_loss, 0.75)) & (entropy >= np.quantile(entropy, 0.75))
    arrays = {
        "token_ids": rng.integers(0, v, size=n, dtype=np.int64),
        "token_loss": token_loss,
        "student_entropy": entropy,
        "teacher_entropy": entropy + rng.normal(0, 0.1, size=n).astype(np.float32),
        "advantage": rng.normal(0, 1, size=n).astype(np.float32),
        "support_candidate": hard.astype(np.bool_),
    }
    return G.astype(np.float16), arrays


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    record_dir = out / "records"
    record_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for prompt_idx in range(args.prompts):
        for arm in ["grpo", "opd", "joint"]:
            G, arrays = make_matrix(rng, arm, args.n_tokens, args.vocab)
            metadata = {
                "prompt_idx": prompt_idx,
                "arm": arm,
                "pass_at_1_proxy": float(rng.random() < 0.2),
                "pass_at_group_proxy": float(rng.random() < 0.5),
                "synthetic": True,
            }
            write_record(record_dir, arm, prompt_idx, G, arrays, metadata)
    with (out / "README.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Wrote synthetic records to {record_dir}")


if __name__ == "__main__":
    main()

