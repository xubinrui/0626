from __future__ import annotations

from pathlib import Path

import pytest

from opd_grpo_gradstruct.data import load_gsm8k_rows_from_arrow_cache


def test_gsm8k_arrow_cache_loads_without_hub():
    cache_root = Path("~/.cache/huggingface/datasets/gsm8k").expanduser()
    if not cache_root.exists():
        pytest.skip(f"no local GSM8K cache at {cache_root}")
    rows = load_gsm8k_rows_from_arrow_cache("train")
    assert len(rows) > 1000
    assert {"question", "answer"}.issubset(rows[0])
