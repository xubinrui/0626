from __future__ import annotations

import random
import re
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from datasets import Dataset, load_dataset


@dataclass(frozen=True)
class PromptExample:
    idx: int
    question: str
    answer: str
    dataset: str


_FALLBACK_GSM8K = [
    {
        "question": "Janet has 3 apples and buys 5 more. How many apples does she have?",
        "answer": "Janet has 3 + 5 = 8 apples. #### 8",
    },
    {
        "question": "A box has 12 pencils. Tom gives away 4 pencils. How many pencils remain?",
        "answer": "There are 12 - 4 = 8 pencils left. #### 8",
    },
    {
        "question": "If each notebook costs 6 dollars, how much do 7 notebooks cost?",
        "answer": "The total cost is 6 * 7 = 42 dollars. #### 42",
    },
    {
        "question": "Maria read 9 pages on Monday and twice as many on Tuesday. How many pages did she read in total?",
        "answer": "Tuesday she read 18 pages, so total is 9 + 18 = 27. #### 27",
    },
    {
        "question": "There are 24 cookies split equally among 6 children. How many cookies does each child get?",
        "answer": "Each child gets 24 / 6 = 4 cookies. #### 4",
    },
    {
        "question": "A train travels 45 miles each hour for 3 hours. How far does it travel?",
        "answer": "It travels 45 * 3 = 135 miles. #### 135",
    },
]


def load_prompt_examples(dataset: str, split: str, limit: int, seed: int) -> list[PromptExample]:
    if dataset.lower() != "gsm8k":
        raise ValueError("This one-day implementation currently supports dataset='gsm8k'.")
    if os.environ.get("OPD_GRPO_USE_FALLBACK_DATA") == "1":
        print("WARNING: OPD_GRPO_USE_FALLBACK_DATA=1, using built-in tiny GSM8K-style fallback.")
        rows = list(_FALLBACK_GSM8K)
    else:
        rows = load_gsm8k_rows(split)
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    selected = indices[: min(limit, len(indices))]
    examples: list[PromptExample] = []
    for out_idx, ds_idx in enumerate(selected):
        row = rows[ds_idx]
        examples.append(
            PromptExample(
                idx=out_idx,
                question=row["question"],
                answer=row["answer"],
                dataset="gsm8k",
            )
        )
    return examples


def load_gsm8k_rows(split: str) -> list[dict[str, str]]:
    rows = load_gsm8k_rows_from_arrow_cache(split)
    if rows:
        return rows
    if os.environ.get("OPD_GRPO_ALLOW_DATASET_NETWORK") == "1":
        try:
            ds = load_dataset("gsm8k", "main", split=split)
            return [{"question": row["question"], "answer": row["answer"]} for row in ds]
        except Exception as exc:
            print(f"WARNING: failed to load GSM8K from datasets hub/cache: {exc}")
    print("WARNING: using built-in tiny GSM8K-style fallback for smoke testing only.")
    return list(_FALLBACK_GSM8K)


def load_gsm8k_rows_from_arrow_cache(split: str) -> list[dict[str, str]]:
    cache_root = Path(os.environ.get("OPD_GRPO_GSM8K_CACHE_DIR", "~/.cache/huggingface/datasets/gsm8k")).expanduser()
    split_name = str(split).split("[")[0]
    candidates = sorted(cache_root.glob(f"main/0.0.0/*/gsm8k-{split_name}.arrow"))
    if not candidates:
        return []
    path = candidates[-1]
    ds = Dataset.from_file(str(path))
    print(f"Loaded GSM8K {split_name} from local arrow cache: {path}")
    return [{"question": row["question"], "answer": row["answer"]} for row in ds]


def build_prompt(question: str) -> str:
    return (
        "Solve the following math problem. Show concise reasoning and put the final "
        "answer after '####'.\n\n"
        f"Problem: {question}\n\nSolution:"
    )


_NUM_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")


def normalize_number(text: str) -> str | None:
    matches = _NUM_RE.findall(text.replace("$", ""))
    if not matches:
        return None
    value = matches[-1].replace(",", "")
    if value.endswith(".0"):
        value = value[:-2]
    return value


def gsm8k_gold(answer: str) -> str | None:
    if "####" in answer:
        return normalize_number(answer.split("####")[-1])
    return normalize_number(answer)


def gsm8k_reward(generation: str, answer: str) -> float:
    pred = normalize_number(generation)
    gold = gsm8k_gold(answer)
    if pred is None or gold is None:
        return 0.0
    return 1.0 if pred == gold else 0.0


def group_advantages(rewards: Iterable[float]) -> list[float]:
    vals = list(float(x) for x in rewards)
    if not vals:
        return []
    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / max(1, len(vals))
    std = var**0.5
    if std < 1e-6:
        return [0.0 for _ in vals]
    return [(x - mean) / std for x in vals]
