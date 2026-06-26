from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from opd_grpo_gradstruct.io import read_record
from opd_grpo_gradstruct.routing import grpo_sparse_support_scores
from opd_grpo_gradstruct.rpca import inexact_alm_rpca, topk_column_submatrix


def legacy_sparse_scores(G: np.ndarray, topk: int, max_iter: int, threshold_frac: float):
    fro = float(np.linalg.norm(G, ord="fro"))
    G_norm = G / fro if fro > 0 else G
    sub, _cols = topk_column_submatrix(G_norm, topk)
    _L, S, _info = inexact_alm_rpca(sub, max_iter=max_iter, tol=1e-5)
    row_norm = np.linalg.norm(S, axis=1).astype(np.float32)
    row_eps = max(1e-7, float(row_norm.max(initial=0.0)) * threshold_frac)
    return row_norm, row_norm > row_eps


def test_grpo_sparse_support_refactor_matches_smoke_record_bit_for_bit():
    record_dir = Path("/data/xuebinrui/code/opd_grpo_gradstruct/outputs/smoke_synthetic2/records")
    records = sorted(record_dir.glob("*_grpo.npz"))
    if not records:
        pytest.skip(f"no smoke records found at {record_dir}")
    rec = read_record(records[0])
    G = rec["G"].astype(np.float32)
    expected_scores, expected_mask = legacy_sparse_scores(G, topk=16, max_iter=100, threshold_frac=0.05)
    scores, info = grpo_sparse_support_scores(
        G,
        rpca_topk_cols_per_row=16,
        rpca_max_iter=100,
        sparse_row_threshold_frac=0.05,
    )
    np.testing.assert_array_equal(scores, expected_scores)
    np.testing.assert_array_equal(info["analysis_mask"], expected_mask)
