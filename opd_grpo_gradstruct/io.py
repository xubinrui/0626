from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def write_record(
    out_dir: str | Path,
    arm: str,
    prompt_idx: int,
    gradient: np.ndarray,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> Path:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    record_path = path / f"prompt_{prompt_idx:05d}_{arm}.npz"
    meta_json = json.dumps(metadata, ensure_ascii=False)
    np.savez(record_path, G=gradient, metadata=np.array(meta_json), **arrays)
    return record_path


def read_record(path: str | Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    out: dict[str, Any] = {key: data[key] for key in data.files if key != "metadata"}
    out["metadata"] = json.loads(str(data["metadata"]))
    return out
