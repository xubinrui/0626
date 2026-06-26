from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm

from .io import read_record
from .rpca import inexact_alm_rpca, topk_column_submatrix
from .routing import grpo_sparse_support_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory containing .npz records.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--svd-rank", type=int, default=128)
    parser.add_argument("--rpca-topk-cols-per-row", type=int, default=16)
    parser.add_argument("--rpca-max-iter", type=int, default=100)
    parser.add_argument("--sparse-row-threshold-frac", type=float, default=0.05)
    parser.add_argument("--sparse-elem-threshold-frac", type=float, default=1e-3)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--deamplify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--threshold-free-sparsity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stratified-perm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n-strata", type=int, default=10)
    parser.add_argument("--exclude-degenerate-advantage", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--advantage-eps", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def effective_rank_from_energy(s: np.ndarray, threshold: float) -> int:
    if s.size == 0:
        return 0
    energy = s**2
    total = energy.sum()
    if total <= 0:
        return 0
    return int(np.searchsorted(np.cumsum(energy) / total, threshold) + 1)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    union = np.logical_or(a, b).sum()
    return 0.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def permutation_jaccard_pvalue(
    support: np.ndarray,
    hard: np.ndarray,
    permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, np.ndarray]:
    support = support.astype(bool)
    hard = hard.astype(bool)
    observed = jaccard(support, hard)
    if permutations <= 0:
        return observed, np.nan, np.nan, np.array([], dtype=np.float32)
    null = np.empty(permutations, dtype=np.float32)
    for i in range(permutations):
        perm = rng.permutation(hard)
        null[i] = jaccard(support, perm)
    p = (float((null >= observed).sum()) + 1.0) / (permutations + 1.0)
    return observed, float(null.mean()), p, null


def stratified_permutation_test(
    support: np.ndarray,
    hard: np.ndarray,
    row_norms: np.ndarray,
    permutations: int,
    n_strata: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, np.ndarray]:
    support = support.astype(bool)
    hard = hard.astype(bool)
    observed = jaccard(support, hard)
    if permutations <= 0 or hard.size == 0:
        return observed, np.nan, np.nan, np.array([], dtype=np.float32)
    finite_norms = np.nan_to_num(row_norms.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    qs = np.quantile(finite_norms, np.linspace(0, 1, n_strata + 1)[1:-1])
    qs = np.unique(qs)
    strata = np.digitize(finite_norms, qs)
    null = np.empty(permutations, dtype=np.float32)
    for i in range(permutations):
        permuted = np.zeros_like(hard, dtype=bool)
        for stratum in np.unique(strata):
            idx = np.where(strata == stratum)[0]
            permuted[idx] = rng.permutation(hard[idx])
        null[i] = jaccard(support, permuted)
    p = (float((null >= observed).sum()) + 1.0) / (permutations + 1.0)
    return observed, float(null.mean()), p, null


def svd_metrics(matrix: np.ndarray, svd_rank: int, rng: np.random.Generator) -> dict[str, object]:
    fro = float(np.linalg.norm(matrix, ord="fro"))
    matrix_norm = matrix / fro if fro > 0 else matrix
    rank_k = min(svd_rank, max(1, min(matrix_norm.shape) - 1))
    if min(matrix_norm.shape) <= 1 or fro == 0:
        singular_values = np.array([], dtype=np.float32)
    else:
        _, singular_values, _ = randomized_svd(
            matrix_norm,
            n_components=rank_k,
            n_iter=4,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
    rank90 = effective_rank_from_energy(singular_values, 0.90)
    rank99 = effective_rank_from_energy(singular_values, 0.99)
    stable_rank = (
        float((fro**2) / ((singular_values[0] * fro) ** 2 + 1e-30))
        if singular_values.size
        else 0.0
    )
    return {
        "fro": fro,
        "normalized": matrix_norm,
        "singular_values": singular_values,
        "rank90": rank90,
        "rank99": rank99,
        "rank90_frac_min_dim": rank90 / max(1, min(matrix.shape)),
        "rank99_frac_min_dim": rank99 / max(1, min(matrix.shape)),
        "stable_rank": stable_rank,
        "top_rank90_energy": float((singular_values[:rank90] ** 2).sum()) if singular_values.size else 0.0,
    }


def reconstruct_token_to_rollout(rec: dict, n_rows: int) -> np.ndarray | None:
    if "token_to_rollout" in rec:
        return rec["token_to_rollout"].astype(np.int64)
    metadata = rec["metadata"]
    group_size = int(metadata.get("group_size") or 0)
    max_new_tokens = int(metadata.get("max_new_tokens") or 0)
    if group_size > 0 and max_new_tokens > 0 and group_size * max_new_tokens == n_rows:
        return np.repeat(np.arange(group_size, dtype=np.int64), max_new_tokens)
    if group_size > 0 and n_rows % group_size == 0:
        return np.repeat(np.arange(group_size, dtype=np.int64), n_rows // group_size)
    return None


def deamplify_advantage(G: np.ndarray, rec: dict) -> tuple[np.ndarray, bool]:
    metadata = rec["metadata"]
    arm = str(metadata.get("arm", ""))
    if arm == "opd":
        return G, True
    token_to_rollout = reconstruct_token_to_rollout(rec, G.shape[0])
    rollout_advantages = metadata.get("advantages")
    if token_to_rollout is not None and rollout_advantages is not None:
        rollout_advantages = np.asarray(rollout_advantages, dtype=np.float32)
        valid = token_to_rollout < rollout_advantages.shape[0]
        scale = np.ones(G.shape[0], dtype=np.float32)
        scale[valid] = np.abs(rollout_advantages[token_to_rollout[valid]])
    elif "advantage" in rec:
        scale = np.abs(rec["advantage"].astype(np.float32))
    else:
        return G, False
    scale[scale == 0] = 1.0
    return G / scale[:, None], True


def row_concentration(G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    row_l2 = np.linalg.norm(G, axis=1)
    row_l1 = np.abs(G).sum(axis=1)
    l1_l2_ratio = row_l1 / (row_l2 + 1e-12)
    row_fourth = (G**4).sum(axis=1)
    row_pr = np.zeros_like(row_l2, dtype=np.float64)
    nonzero = row_fourth > 0
    row_pr[nonzero] = (row_l2[nonzero] ** 4) / row_fourth[nonzero]
    return l1_l2_ratio.astype(np.float32), row_pr.astype(np.float32)


def read_metadata_only(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    return json.loads(str(data["metadata"]))


def advantage_stats(metadata: dict) -> tuple[float, bool]:
    advantages = metadata.get("advantages")
    if advantages is None:
        return np.nan, False
    arr = np.asarray(advantages, dtype=np.float32)
    if arr.size == 0:
        return np.nan, False
    std = float(arr.std())
    return std, bool(std <= 1e-8)


def should_exclude_degenerate(metadata: dict, eps: float) -> tuple[bool, float]:
    arm = str(metadata.get("arm", ""))
    advantages = metadata.get("advantages")
    if arm not in {"grpo", "joint"} or advantages is None:
        return False, np.nan
    arr = np.asarray(advantages, dtype=np.float32)
    if arr.size == 0:
        return False, np.nan
    std = float(arr.std())
    return std <= eps, std


def analyze_record(path: Path, args: argparse.Namespace, rng: np.random.Generator):
    rec = read_record(path)
    G = rec["G"].astype(np.float32)
    metadata = rec["metadata"]
    adv_std, adv_degenerate = advantage_stats(metadata)

    raw_metrics = svd_metrics(G, args.svd_rank, rng)
    G_norm = raw_metrics["normalized"]
    fro = float(raw_metrics["fro"])

    if args.deamplify:
        G_deamp, deamp_available = deamplify_advantage(G, rec)
    else:
        G_deamp, deamp_available = G, False
    deamp_metrics = svd_metrics(G_deamp, args.svd_rank, rng) if args.deamplify else raw_metrics
    rank90_raw = int(raw_metrics["rank90"])
    rank90_deamp = int(deamp_metrics["rank90"])
    if rank90_raw == 0:
        rank_inflation_ratio = 1.0 if rank90_deamp == 0 else np.inf
    else:
        rank_inflation_ratio = float(rank90_deamp / rank90_raw)

    if args.threshold_free_sparsity:
        l1_l2_ratio, row_pr = row_concentration(G)
        row_pr_median = float(np.median(row_pr)) if row_pr.size else 0.0
        row_pr_q25 = float(np.quantile(row_pr, 0.25)) if row_pr.size else 0.0
        row_pr_q75 = float(np.quantile(row_pr, 0.75)) if row_pr.size else 0.0
        row_l1_l2_median = float(np.median(l1_l2_ratio)) if l1_l2_ratio.size else 0.0
    else:
        row_pr = np.array([], dtype=np.float32)
        row_pr_median = np.nan
        row_pr_q25 = np.nan
        row_pr_q75 = np.nan
        row_l1_l2_median = np.nan

    row_norm, support_info = grpo_sparse_support_scores(
        G,
        rpca_topk_cols_per_row=args.rpca_topk_cols_per_row,
        rpca_max_iter=args.rpca_max_iter,
        sparse_row_threshold_frac=args.sparse_row_threshold_frac,
    )
    sub, cols = topk_column_submatrix(G_norm, args.rpca_topk_cols_per_row)
    sub_norm_sq = float(np.linalg.norm(sub, ord="fro") ** 2)
    full_norm_sq = float(np.linalg.norm(G_norm, ord="fro") ** 2)
    energy_coverage = float(support_info["energy_coverage"])
    _L, S, rpca_info = inexact_alm_rpca(sub, max_iter=args.rpca_max_iter, tol=1e-5)
    s_abs = np.abs(S)
    row_eps = max(1e-7, float(row_norm.max(initial=0.0)) * args.sparse_row_threshold_frac)
    elem_eps = max(1e-7, float(s_abs.max(initial=0.0)) * args.sparse_elem_threshold_frac)
    sparse_rows = row_norm > row_eps
    sparse_elements = s_abs > elem_eps
    s_row_frac = float(sparse_rows.mean()) if sparse_rows.size else 0.0
    s_elem_frac = float(sparse_elements.mean()) if sparse_elements.size else 0.0
    hard = rec["support_candidate"].astype(bool)
    G_row_norms = np.linalg.norm(G, axis=1)
    jaccard_obs, null_jaccard, p_global, global_null = permutation_jaccard_pvalue(
        sparse_rows, hard, args.permutations, rng
    )
    if args.stratified_perm:
        strat_obs, strat_null_jaccard, p_stratified, strat_null = stratified_permutation_test(
            sparse_rows, hard, G_row_norms, args.permutations, args.n_strata, rng
        )
    else:
        strat_obs, strat_null_jaccard, p_stratified = jaccard_obs, np.nan, np.nan
        strat_null = np.array([], dtype=np.float32)

    false_low_rank_advantage = bool(rank_inflation_ratio > 1.5)
    false_sparse_row_pr = bool(row_pr_median > 0.5 * G.shape[1]) if np.isfinite(row_pr_median) else False
    truncation_contaminated = bool(energy_coverage > 0.99)
    false_alignment = bool(p_stratified >= 0.05) if np.isfinite(p_stratified) else False

    row = {
        "file": path.name,
        "prompt_idx": metadata.get("prompt_idx"),
        "arm": metadata.get("arm"),
        "n_tokens": int(G.shape[0]),
        "vocab_size": int(G.shape[1]),
        "fro_norm": fro,
        "rank90": rank90_raw,
        "rank99": int(raw_metrics["rank99"]),
        "rank90_frac_min_dim": raw_metrics["rank90_frac_min_dim"],
        "rank99_frac_min_dim": raw_metrics["rank99_frac_min_dim"],
        "stable_rank": raw_metrics["stable_rank"],
        "top_rank90_energy": raw_metrics["top_rank90_energy"],
        "rank90_raw": rank90_raw,
        "rank99_raw": int(raw_metrics["rank99"]),
        "stable_rank_raw": raw_metrics["stable_rank"],
        "rank90_deamplified": rank90_deamp,
        "rank99_deamplified": int(deamp_metrics["rank99"]),
        "stable_rank_deamplified": deamp_metrics["stable_rank"],
        "rank90_deamplified_frac_min_dim": deamp_metrics["rank90_frac_min_dim"],
        "rank99_deamplified_frac_min_dim": deamp_metrics["rank99_frac_min_dim"],
        "rank_inflation_ratio": rank_inflation_ratio,
        "deamplify_available": deamp_available,
        "advantage_std": adv_std,
        "advantage_degenerate": adv_degenerate,
        "row_l1_l2_median": row_l1_l2_median,
        "row_pr_median": row_pr_median,
        "row_pr_q25": row_pr_q25,
        "row_pr_q75": row_pr_q75,
        "energy_coverage": energy_coverage,
        "rpca_sub_cols": int(len(cols)),
        "rpca_rank_L": int(rpca_info["rank"]),
        "rpca_rank_frac_min_dim": float(rpca_info["rank"] / max(1, min(sub.shape))),
        "rpca_iters": int(rpca_info["iters"]),
        "rpca_recon_error": float(rpca_info["rel_error"]),
        "S_row_threshold": row_eps,
        "S_elem_threshold": elem_eps,
        "S_nonzero_row_frac": s_row_frac,
        "S_nonzero_elem_frac": s_elem_frac,
        "hard_token_frac": float(hard.mean()) if hard.size else 0.0,
        "support_hard_jaccard": jaccard_obs,
        "support_hard_null_jaccard": null_jaccard,
        "support_hard_pvalue": p_global,
        "p_global": p_global,
        "support_hard_stratified_jaccard": strat_obs,
        "support_hard_stratified_null_jaccard": strat_null_jaccard,
        "p_stratified": p_stratified,
        "false_low_rank_advantage": false_low_rank_advantage,
        "false_sparse_row_pr": false_sparse_row_pr,
        "truncation_contaminated": truncation_contaminated,
        "rpca_auxiliary_contaminated": truncation_contaminated,
        "false_alignment": false_alignment,
        "pass_at_1_proxy": metadata.get("pass_at_1_proxy"),
        "pass_at_group_proxy": metadata.get("pass_at_group_proxy"),
    }
    sv_rows = [
        {
            "file": path.name,
            "prompt_idx": metadata.get("prompt_idx"),
            "arm": metadata.get("arm"),
            "rank_index": i + 1,
            "singular_value": float(s),
            "cum_energy": float(
                np.cumsum(raw_metrics["singular_values"] ** 2)[i]
                / max(1e-30, np.sum(raw_metrics["singular_values"] ** 2))
            ),
        }
        for i, s in enumerate(raw_metrics["singular_values"])
    ]
    deamp_sv_rows = [
        {
            "file": path.name,
            "prompt_idx": metadata.get("prompt_idx"),
            "arm": metadata.get("arm"),
            "rank_index": i + 1,
            "singular_value": float(s),
            "cum_energy": float(
                np.cumsum(deamp_metrics["singular_values"] ** 2)[i]
                / max(1e-30, np.sum(deamp_metrics["singular_values"] ** 2))
            ),
        }
        for i, s in enumerate(deamp_metrics["singular_values"])
    ]
    row_pr_rows = [
        {
            "file": path.name,
            "prompt_idx": metadata.get("prompt_idx"),
            "arm": metadata.get("arm"),
            "row_pr": float(v),
        }
        for v in row_pr
    ]
    strat_null_rows = [
        {
            "file": path.name,
            "prompt_idx": metadata.get("prompt_idx"),
            "arm": metadata.get("arm"),
            "null_jaccard": float(v),
            "observed_jaccard": float(strat_obs),
        }
        for v in strat_null
    ]
    return row, sv_rows, deamp_sv_rows, row_pr_rows, strat_null_rows


def write_plots(
    out: Path,
    summary: pd.DataFrame,
    singular: pd.DataFrame,
    deamp_singular: pd.DataFrame,
    row_pr: pd.DataFrame,
    stratified_null: pd.DataFrame,
) -> None:
    plot_dir = out / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if not singular.empty:
        plt.figure(figsize=(7, 4))
        for arm, grp in singular.groupby("arm"):
            med = grp.groupby("rank_index")["cum_energy"].median()
            plt.plot(med.index, med.values, label=arm)
        plt.xlabel("Top-k singular values")
        plt.ylabel("Median cumulative energy")
        plt.ylim(0, 1.02)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "median_cumulative_energy.png", dpi=160)
        plt.close()

    if not summary.empty:
        for col, ylabel, name in [
            ("rank90_frac_min_dim", "rank90 / min(N,V)", "rank90_frac.png"),
            ("S_nonzero_row_frac", "S nonzero row fraction", "sparse_row_frac.png"),
            ("p_global", "H2 global permutation p-value", "h2_pvalue.png"),
        ]:
            plt.figure(figsize=(6, 4))
            arms = list(summary["arm"].dropna().unique())
            data = [summary.loc[summary["arm"] == arm, col].dropna().values for arm in arms]
            plt.boxplot(data, tick_labels=arms, showmeans=True)
            plt.ylabel(ylabel)
            plt.tight_layout()
            plt.savefig(plot_dir / name, dpi=160)
            plt.close()

        arms = list(summary["arm"].dropna().unique())
        x = np.arange(len(arms))
        raw = [summary.loc[summary["arm"] == arm, "rank90_raw"].median() for arm in arms]
        deamp = [summary.loc[summary["arm"] == arm, "rank90_deamplified"].median() for arm in arms]
        width = 0.38
        plt.figure(figsize=(7, 4))
        plt.bar(x - width / 2, raw, width, label="raw")
        plt.bar(x + width / 2, deamp, width, label="deamplified")
        plt.xticks(x, arms)
        plt.ylabel("Median rank90")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "rank90_raw_vs_deamplified.png", dpi=160)
        plt.close()

    if not row_pr.empty:
        plt.figure(figsize=(7, 4))
        for arm, grp in row_pr.groupby("arm"):
            vals = grp["row_pr"].dropna().values
            if vals.size:
                plt.hist(vals, bins=50, alpha=0.45, density=True, label=arm)
        plt.xlabel("Row participation ratio")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "row_pr_distribution.png", dpi=160)
        plt.close()

    if not stratified_null.empty:
        arms = list(stratified_null["arm"].dropna().unique())
        n = max(1, len(arms))
        fig, axes = plt.subplots(n, 1, figsize=(7, 3 * n), squeeze=False)
        for ax, arm in zip(axes[:, 0], arms):
            grp = stratified_null.loc[stratified_null["arm"] == arm]
            ax.hist(grp["null_jaccard"].values, bins=40, alpha=0.75)
            observed = grp["observed_jaccard"].median()
            ax.axvline(observed, color="red", linewidth=2, label="median observed")
            ax.set_title(str(arm))
            ax.set_xlabel("Stratified null Jaccard")
            ax.set_ylabel("Count")
            ax.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "jaccard_null_stratified.png", dpi=160)
        plt.close()

    if not deamp_singular.empty:
        plt.figure(figsize=(7, 4))
        for arm, grp in deamp_singular.groupby("arm"):
            med = grp.groupby("rank_index")["cum_energy"].median()
            plt.plot(med.index, med.values, label=arm)
        plt.xlabel("Top-k singular values")
        plt.ylabel("Median cumulative energy, deamplified")
        plt.ylim(0, 1.02)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "median_cumulative_energy_deamplified.png", dpi=160)
        plt.close()


def main() -> None:
    args = parse_args()
    in_dir = Path(args.input)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    files = sorted(p for p in in_dir.glob("*.npz") if p.name.startswith("prompt_"))
    if not files:
        raise FileNotFoundError(f"No prompt_*.npz records found in {in_dir}")

    rows = []
    sv_rows = []
    deamp_sv_rows = []
    row_pr_rows = []
    strat_null_rows = []
    excluded_rows = []
    for path in tqdm(files, desc="analyze"):
        if args.exclude_degenerate_advantage:
            metadata = read_metadata_only(path)
            exclude, adv_std = should_exclude_degenerate(metadata, args.advantage_eps)
            if exclude:
                excluded_rows.append(
                    {
                        "file": path.name,
                        "prompt_idx": metadata.get("prompt_idx"),
                        "arm": metadata.get("arm"),
                        "advantage_std": adv_std,
                        "reason": "degenerate_advantage",
                    }
                )
                continue
        row, one_sv, one_deamp_sv, one_row_pr, one_strat_null = analyze_record(path, args, rng)
        rows.append(row)
        sv_rows.extend(one_sv)
        deamp_sv_rows.extend(one_deamp_sv)
        row_pr_rows.extend(one_row_pr)
        strat_null_rows.extend(one_strat_null)
    summary = pd.DataFrame(rows)
    singular = pd.DataFrame(sv_rows)
    deamp_singular = pd.DataFrame(deamp_sv_rows)
    row_pr = pd.DataFrame(row_pr_rows)
    stratified_null = pd.DataFrame(strat_null_rows)
    excluded = pd.DataFrame(excluded_rows)
    if summary.empty:
        raise RuntimeError("No records left to analyze after filtering.")
    summary.to_csv(out / "summary.csv", index=False)
    singular.to_csv(out / "singular_values.csv", index=False)
    deamp_singular.to_csv(out / "singular_values_deamplified.csv", index=False)
    row_pr.to_csv(out / "row_pr.csv", index=False)
    stratified_null.to_csv(out / "stratified_null_jaccard.csv", index=False)
    excluded.to_csv(out / "excluded_records.csv", index=False)
    arm_summary = summary.groupby("arm").agg(
        rank90_frac_median=("rank90_frac_min_dim", "median"),
        rank99_frac_median=("rank99_frac_min_dim", "median"),
        stable_rank_median=("stable_rank", "median"),
        rank90_raw=("rank90_raw", "median"),
        rank90_deamplified=("rank90_deamplified", "median"),
        rank99_raw=("rank99_raw", "median"),
        rank99_deamplified=("rank99_deamplified", "median"),
        stable_rank_raw=("stable_rank_raw", "median"),
        stable_rank_deamplified=("stable_rank_deamplified", "median"),
        rank_inflation_ratio=("rank_inflation_ratio", "median"),
        row_pr_median=("row_pr_median", "median"),
        row_pr_q25=("row_pr_q25", "median"),
        row_pr_q75=("row_pr_q75", "median"),
        energy_coverage=("energy_coverage", "median"),
        S_row_frac_median=("S_nonzero_row_frac", "median"),
        rpca_recon_error_median=("rpca_recon_error", "median"),
        h2_pvalue_median=("support_hard_pvalue", "median"),
        p_global=("p_global", "median"),
        p_stratified=("p_stratified", "median"),
        false_low_rank_advantage_rate=("false_low_rank_advantage", "mean"),
        false_sparse_row_pr_rate=("false_sparse_row_pr", "mean"),
        rpca_auxiliary_contaminated_rate=("rpca_auxiliary_contaminated", "mean"),
        false_alignment_rate=("false_alignment", "mean"),
        advantage_degenerate_rate=("advantage_degenerate", "mean"),
        pass_at_1_proxy_mean=("pass_at_1_proxy", "mean"),
        pass_at_group_proxy_mean=("pass_at_group_proxy", "mean"),
        n_records=("file", "count"),
    )
    arm_summary["pseudo_low_rank_trigger"] = arm_summary["rank_inflation_ratio"] > 1.5
    vocab_median = summary.groupby("arm")["vocab_size"].median()
    arm_summary["pseudo_sparse_trigger"] = arm_summary["row_pr_median"] > 0.5 * vocab_median
    arm_summary["rpca_truncation_contaminated_trigger"] = arm_summary["energy_coverage"] > 0.99
    arm_summary["sparsity_primary_metric"] = "row_pr_median"
    arm_summary["rpca_role"] = np.where(
        arm_summary["rpca_truncation_contaminated_trigger"],
        "auxiliary_contaminated_by_topk",
        "auxiliary",
    )
    arm_summary["pseudo_alignment_trigger"] = arm_summary["p_stratified"] >= 0.05
    arm_summary.to_csv(out / "arm_summary.csv")
    write_plots(out, summary, singular, deamp_singular, row_pr, stratified_null)
    print(arm_summary)


if __name__ == "__main__":
    main()
