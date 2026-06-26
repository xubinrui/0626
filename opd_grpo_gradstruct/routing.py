from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.utils.extmath import randomized_svd

from .rpca import inexact_alm_rpca, topk_column_submatrix


@dataclass(frozen=True)
class SparseSupportResult:
    scores: np.ndarray
    mask: np.ndarray
    threshold: float
    hard_fraction: float
    rpca_rank: int
    rpca_iters: int
    rpca_recon_error: float
    energy_coverage: float
    selected_columns: np.ndarray


def _target_count(n_rows: int, target_fraction: float, min_fraction: float, max_fraction: float) -> int:
    if n_rows <= 0:
        return 0
    count = int(round(n_rows * target_fraction))
    min_count = int(np.floor(n_rows * min_fraction))
    max_count = int(np.ceil(n_rows * max_fraction))
    min_count = max(1, min_count)
    max_count = max(min_count, max_count)
    return int(np.clip(count, min_count, min(max_count, n_rows)))


def top_fraction_mask(
    scores: np.ndarray,
    target_fraction: float = 0.20,
    min_fraction: float = 0.15,
    max_fraction: float = 0.25,
) -> tuple[np.ndarray, float, float]:
    scores = np.asarray(scores, dtype=np.float64)
    n_rows = int(scores.shape[0])
    count = _target_count(n_rows, target_fraction, min_fraction, max_fraction)
    mask = np.zeros(n_rows, dtype=bool)
    if count == 0:
        return mask, float("inf"), 0.0
    order = np.argsort(scores, kind="stable")
    chosen = order[-count:]
    mask[chosen] = True
    threshold = float(np.min(scores[chosen])) if chosen.size else float("inf")
    return mask, threshold, float(mask.mean())


def grpo_sparse_support_scores(
    G_grpo: np.ndarray,
    *,
    rpca_topk_cols_per_row: int = 16,
    rpca_max_iter: int = 100,
    sparse_row_threshold_frac: float = 0.05,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return the analysis-phase GRPO sparse-row statistic.

    This mirrors analyze.py: normalize the gradient block by Frobenius norm, keep the
    top-k columns per row as the RPCA auxiliary matrix, run RPCA, and score each token
    row by the L2 norm of the sparse component.
    """

    G = np.asarray(G_grpo, dtype=np.float32)
    if G.ndim != 2:
        raise ValueError(f"G_grpo must be 2D, got shape={G.shape}")
    fro = float(np.linalg.norm(G, ord="fro"))
    G_norm = G / fro if fro > 0 else G
    sub, cols = topk_column_submatrix(G_norm, rpca_topk_cols_per_row)
    sub_norm_sq = float(np.linalg.norm(sub, ord="fro") ** 2)
    full_norm_sq = float(np.linalg.norm(G_norm, ord="fro") ** 2)
    energy_coverage = sub_norm_sq / full_norm_sq if full_norm_sq > 0 else 0.0
    _low_rank, sparse, rpca_info = inexact_alm_rpca(sub, max_iter=rpca_max_iter, tol=1e-5)
    scores = np.linalg.norm(sparse, axis=1).astype(np.float32)
    threshold = max(1e-7, float(scores.max(initial=0.0)) * sparse_row_threshold_frac)
    analysis_mask = scores > threshold
    info: dict[str, object] = {
        "fro_norm": fro,
        "analysis_threshold": threshold,
        "analysis_mask": analysis_mask,
        "analysis_hard_fraction": float(analysis_mask.mean()) if analysis_mask.size else 0.0,
        "rpca_rank": int(rpca_info["rank"]),
        "rpca_iters": int(rpca_info["iters"]),
        "rpca_recon_error": float(rpca_info["rel_error"]),
        "energy_coverage": energy_coverage,
        "selected_columns": cols,
    }
    return scores, info


def grpo_sparse_support_mask(
    G_grpo: np.ndarray,
    *,
    target_fraction: float = 0.20,
    min_fraction: float = 0.15,
    max_fraction: float = 0.25,
    rpca_topk_cols_per_row: int = 16,
    rpca_max_iter: int = 100,
    sparse_row_threshold_frac: float = 0.05,
) -> SparseSupportResult:
    scores, info = grpo_sparse_support_scores(
        G_grpo,
        rpca_topk_cols_per_row=rpca_topk_cols_per_row,
        rpca_max_iter=rpca_max_iter,
        sparse_row_threshold_frac=sparse_row_threshold_frac,
    )
    mask, threshold, hard_fraction = top_fraction_mask(scores, target_fraction, min_fraction, max_fraction)
    return SparseSupportResult(
        scores=scores,
        mask=mask,
        threshold=threshold,
        hard_fraction=hard_fraction,
        rpca_rank=int(info["rpca_rank"]),
        rpca_iters=int(info["rpca_iters"]),
        rpca_recon_error=float(info["rpca_recon_error"]),
        energy_coverage=float(info["energy_coverage"]),
        selected_columns=np.asarray(info["selected_columns"], dtype=np.int64),
    )


def opd_magnitude_mask(
    G_opd: np.ndarray,
    *,
    target_fraction: float = 0.20,
    min_fraction: float = 0.15,
    max_fraction: float = 0.25,
) -> SparseSupportResult:
    scores = np.linalg.norm(np.asarray(G_opd, dtype=np.float32), axis=1).astype(np.float32)
    mask, threshold, hard_fraction = top_fraction_mask(scores, target_fraction, min_fraction, max_fraction)
    return SparseSupportResult(
        scores=scores,
        mask=mask,
        threshold=threshold,
        hard_fraction=hard_fraction,
        rpca_rank=0,
        rpca_iters=0,
        rpca_recon_error=0.0,
        energy_coverage=1.0,
        selected_columns=np.array([], dtype=np.int64),
    )


def random_matched_mask(
    n_rows: int,
    hard_fraction: float,
    rng: np.random.Generator,
    *,
    min_fraction: float = 0.15,
    max_fraction: float = 0.25,
) -> SparseSupportResult:
    count = _target_count(n_rows, hard_fraction, min_fraction, max_fraction)
    mask = np.zeros(n_rows, dtype=bool)
    if count > 0:
        mask[rng.choice(n_rows, size=count, replace=False)] = True
    scores = rng.random(n_rows).astype(np.float32)
    return SparseSupportResult(
        scores=scores,
        mask=mask,
        threshold=float("nan"),
        hard_fraction=float(mask.mean()) if n_rows else 0.0,
        rpca_rank=0,
        rpca_iters=0,
        rpca_recon_error=0.0,
        energy_coverage=1.0,
        selected_columns=np.array([], dtype=np.int64),
    )


def lowrank_reconstruct(
    matrix: np.ndarray,
    rank: int,
    rng: np.random.Generator | int | None = None,
) -> tuple[np.ndarray, float, int]:
    G = np.asarray(matrix, dtype=np.float32)
    if G.ndim != 2:
        raise ValueError(f"matrix must be 2D, got shape={G.shape}")
    if G.size == 0 or min(G.shape) == 0:
        return np.zeros_like(G, dtype=np.float32), 0.0, 0
    kept_rank = int(min(max(rank, 1), min(G.shape)))
    if isinstance(rng, np.random.Generator):
        random_state = int(rng.integers(0, 2**31 - 1))
    else:
        random_state = rng
    U, S, Vt = randomized_svd(
        G,
        n_components=kept_rank,
        n_iter=4,
        random_state=random_state,
    )
    recon = ((U * S) @ Vt).astype(np.float32)
    denom = float(np.linalg.norm(G, ord="fro"))
    err = 0.0 if denom == 0.0 else float(np.linalg.norm(G - recon, ord="fro") / denom)
    return recon, err, kept_rank


ConditionId = Literal["b1", "b2", "b3", "b4", "ours", "b4b", "ours_minus"]


def rule_hash(condition: str, *, compressed_easy: bool, rank: int, lambda_joint: float) -> str:
    import hashlib

    payload = f"{condition}|compressed_easy={compressed_easy}|rank={rank}|lambda={lambda_joint:.8f}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
