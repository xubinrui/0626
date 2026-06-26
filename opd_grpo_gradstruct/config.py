from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CollectConfig:
    student_model: str = "Qwen/Qwen3-1.7B"
    teacher_model: str = "Qwen/Qwen3-8B"
    dataset: str = "gsm8k"
    dataset_split: str = "train"
    limit_prompts: int = 48
    seed: int = 7
    group_size: int = 4
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.95
    lambda_joint: float = 0.5
    arms: list[str] = field(default_factory=lambda: ["grpo", "opd", "joint"])
    student_device: str = "cuda:0"
    teacher_device: str = "cuda:1"
    teacher_4bit: bool = True
    student_dtype: str = "bf16"
    teacher_dtype: str = "bf16"
    max_prompt_tokens: int = 512
    batch_eval_tokens: int = 1
    out_dir: str = "outputs/one_day/records"


def load_collect_config(path: str | Path) -> CollectConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return CollectConfig(**data)


def apply_cli_overrides(config: CollectConfig, args: argparse.Namespace) -> CollectConfig:
    for key in [
        "student_model",
        "teacher_model",
        "limit_prompts",
        "group_size",
        "max_new_tokens",
        "student_device",
        "teacher_device",
        "out_dir",
    ]:
        value = getattr(args, key, None)
        if value is not None:
            setattr(config, key, value)
    if getattr(args, "teacher_4bit", None) is not None:
        config.teacher_4bit = args.teacher_4bit
    return config


def dtype_from_name(name: str):
    import torch

    lowered = name.lower()
    if lowered in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if lowered in {"fp16", "float16", "half"}:
        return torch.float16
    if lowered in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")

