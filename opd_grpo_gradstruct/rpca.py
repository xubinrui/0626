from __future__ import annotations

import numpy as np


def shrink(x: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def svd_threshold(x: np.ndarray, tau: float) -> tuple[np.ndarray, int]:
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    keep = s > tau
    if not np.any(keep):
        return np.zeros_like(x), 0
    s2 = s[keep] - tau
    return (u[:, keep] * s2) @ vt[keep, :], int(keep.sum())


def inexact_alm_rpca(
    matrix: np.ndarray,
    lam: float | None = None,
    max_iter: int = 200,
    tol: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    x = np.asarray(matrix, dtype=np.float32)
    m, n = x.shape
    if lam is None:
        lam = 1.0 / np.sqrt(max(m, n))
    norm = np.linalg.norm(x, ord="fro")
    if norm == 0:
        return np.zeros_like(x), np.zeros_like(x), {"rank": 0, "iters": 0, "rel_error": 0.0}

    y = x / max(np.linalg.norm(x, 2), np.linalg.norm(x.ravel(), np.inf) / lam)
    mu = 1.25 / np.linalg.norm(x, 2)
    mu_bar = mu * 1e7
    rho = 1.5
    low_rank = np.zeros_like(x)
    sparse = np.zeros_like(x)
    rank = 0
    rel_error = float("inf")

    for it in range(1, max_iter + 1):
        low_rank, rank = svd_threshold(x - sparse + y / mu, 1.0 / mu)
        sparse = shrink(x - low_rank + y / mu, lam / mu)
        residual = x - low_rank - sparse
        rel_error = float(np.linalg.norm(residual, ord="fro") / norm)
        y = y + mu * residual
        mu = min(mu * rho, mu_bar)
        if rel_error < tol:
            break

    return low_rank, sparse, {"rank": rank, "iters": it, "rel_error": rel_error}


def topk_column_submatrix(matrix: np.ndarray, k_per_row: int) -> tuple[np.ndarray, np.ndarray]:
    if k_per_row <= 0 or k_per_row >= matrix.shape[1]:
        cols = np.arange(matrix.shape[1])
        return matrix.astype(np.float32, copy=False), cols
    abs_x = np.abs(matrix)
    idx = np.argpartition(abs_x, -k_per_row, axis=1)[:, -k_per_row:]
    cols = np.unique(idx.reshape(-1))
    return matrix[:, cols].astype(np.float32, copy=False), cols

