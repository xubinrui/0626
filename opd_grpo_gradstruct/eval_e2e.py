from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .data import PromptExample, build_prompt, gsm8k_reward, load_prompt_examples, normalize_number
from .modeling import generate_group, load_causal_lm, load_tokenizer
from .train_e2e import load_e2e_config, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--condition")
    parser.add_argument("--model-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def _field(row: dict[str, Any], names: list[str]) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value)
    content = row.get("content")
    if isinstance(content, dict):
        return _field(content, names)
    return ""


def _examples_from_rows(rows: list[dict[str, Any]], dataset: str, limit: int, seed: int) -> list[PromptExample]:
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    out: list[PromptExample] = []
    for out_idx, ds_idx in enumerate(indices[: min(limit, len(indices))]):
        row = rows[ds_idx]
        question = _field(row, ["question", "problem", "instruction", "prompt"])
        answer = _field(row, ["answer", "solution", "response", "target", "final_answer"])
        if question and answer:
            out.append(PromptExample(idx=out_idx, question=question, answer=answer, dataset=dataset))
    return out


def load_jsonl_or_csv(path: str | Path, dataset: str, limit: int, seed: int) -> list[PromptExample]:
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    if p.suffix.lower() == ".csv":
        with p.open("r", encoding="utf-8") as f:
            rows = [dict(row) for row in csv.DictReader(f)]
    else:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return _examples_from_rows(rows, dataset, limit, seed)


def default_local_eval_path(dataset: str) -> Path | None:
    root = Path(__file__).resolve().parent.parent
    name = dataset.lower().replace("-", "")
    candidates = {
        "math": root / "data" / "eval" / "math_smoke.jsonl",
        "aime2024": root / "data" / "eval" / "aime2024_smoke.jsonl",
    }
    return candidates.get(name)


def load_eval_examples(
    dataset: str,
    split: str,
    limit: int,
    seed: int,
    offline_only: bool = False,
    local_path: str | None = None,
) -> list[PromptExample]:
    name = dataset.lower()
    if name == "gsm8k":
        return load_prompt_examples("gsm8k", split, limit, seed)
    if local_path:
        examples = load_jsonl_or_csv(local_path, dataset, limit, seed)
        if examples:
            return examples
    default_path = default_local_eval_path(dataset)
    if default_path is not None:
        examples = load_jsonl_or_csv(default_path, dataset, limit, seed)
        if examples:
            return examples
    if offline_only:
        return []
    try:
        from datasets import load_dataset
    except Exception:
        return []
    rows: list[dict[str, str]] = []
    try:
        if name == "math":
            ds = load_dataset("hendrycks/competition_math", split=split)
            rows = [{"question": r["problem"], "answer": r["solution"]} for r in ds]
        elif name in {"aime2024", "aime-2024"}:
            ds = load_dataset("Maxwell-Jia/AIME_2024", split=split)
            rows = [{"question": r.get("Problem", r.get("problem", "")), "answer": str(r.get("Answer", r.get("answer", "")))} for r in ds]
    except Exception:
        rows = []
    return _examples_from_rows(rows, dataset, limit, seed)


def reward_for_dataset(dataset: str, generation: str, answer: str) -> float:
    if dataset.lower() == "gsm8k":
        return gsm8k_reward(generation, answer)
    pred = normalize_number(generation)
    gold = normalize_number(answer)
    return 1.0 if pred is not None and gold is not None and pred == gold else 0.0


def distinct_n(texts: list[str], n: int) -> float:
    total = 0
    seen: set[tuple[str, ...]] = set()
    for text in texts:
        toks = text.split()
        grams = [tuple(toks[i : i + n]) for i in range(max(0, len(toks) - n + 1))]
        total += len(grams)
        seen.update(grams)
    return 0.0 if total == 0 else len(seen) / total


def bleu_like(candidate: str, references: list[str], n: int = 4) -> float:
    cand = candidate.split()
    if not cand or not references:
        return 0.0
    scores = []
    for k in range(1, n + 1):
        cand_grams = Counter(tuple(cand[i : i + k]) for i in range(max(0, len(cand) - k + 1)))
        if not cand_grams:
            scores.append(0.0)
            continue
        ref_counts: Counter[tuple[str, ...]] = Counter()
        for ref in references:
            toks = ref.split()
            one = Counter(tuple(toks[i : i + k]) for i in range(max(0, len(toks) - k + 1)))
            for gram, count in one.items():
                ref_counts[gram] = max(ref_counts[gram], count)
        overlap = sum(min(count, ref_counts[gram]) for gram, count in cand_grams.items())
        scores.append((overlap + 1e-9) / (sum(cand_grams.values()) + 1e-9))
    return float(math.exp(sum(math.log(max(x, 1e-9)) for x in scores) / n))


def self_bleu(texts: list[str]) -> float:
    if len(texts) <= 1:
        return 0.0
    vals = []
    for i, text in enumerate(texts):
        vals.append(bleu_like(text, texts[:i] + texts[i + 1 :]))
    return float(np.mean(vals)) if vals else 0.0


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    cfg = load_e2e_config(args.config)
    if args.condition:
        cfg["condition"] = args.condition
    if args.seed is not None:
        cfg["seed"] = args.seed
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    condition = str(cfg.get("condition", "unknown")).lower()
    output_template = str(cfg.get("output_dir") or f"runs/{condition}/{seed}")
    output_dir = Path(args.output_dir or output_template.format(condition=condition, seed=seed))
    eval_path = output_dir / "eval.jsonl"
    if eval_path.exists():
        eval_path.unlink()

    model_path = args.model_path or cfg.get("eval_model_path")
    if not model_path:
        final_model = output_dir / "final_model"
        model_path = str(final_model if final_model.exists() else cfg["student_model"])
    tokenizer = load_tokenizer(str(model_path))
    model = load_causal_lm(str(model_path), cfg["student_device"], cfg["student_dtype"], four_bit=False)
    k_values = [int(k) for k in cfg.get("pass_k", [1, 2, 4, 8])]
    sample_count = int(max(k_values))

    for dataset_cfg in cfg.get("eval_datasets", [{"name": "gsm8k", "split": "test", "limit": 8}]):
        dataset = str(dataset_cfg["name"])
        examples = load_eval_examples(
            dataset,
            str(dataset_cfg.get("split", "test")),
            int(dataset_cfg.get("limit", 8)),
            seed,
            offline_only=bool(cfg.get("eval_offline_only", True)),
            local_path=dataset_cfg.get("local_path"),
        )
        if not examples:
            append_jsonl(eval_path, {"condition": condition, "seed": seed, "dataset": dataset, "skipped": True, "reason": "no eval examples available"})
            continue
        all_texts: list[str] = []
        pass_counts = {k: [] for k in k_values}
        for example in tqdm(examples, desc=f"eval {dataset}"):
            _seq, _prompt_len, _gen, texts = generate_group(
                model,
                tokenizer,
                prompt=build_prompt(example.question),
                group_size=sample_count,
                max_prompt_tokens=int(cfg["max_prompt_tokens"]),
                max_new_tokens=int(cfg.get("eval_max_new_tokens", cfg["max_new_tokens"])),
                temperature=float(cfg.get("eval_temperature", cfg["temperature"])),
                top_p=float(cfg.get("eval_top_p", cfg["top_p"])),
                device=cfg["student_device"],
            )
            rewards = [reward_for_dataset(dataset, text, example.answer) for text in texts]
            all_texts.extend(texts)
            for k in k_values:
                pass_counts[k].append(float(any(r > 0.0 for r in rewards[:k])))
        row = {
            "condition": condition,
            "seed": seed,
            "dataset": dataset,
            "n_examples": len(examples),
            "distinct_1": distinct_n(all_texts, 1),
            "distinct_2": distinct_n(all_texts, 2),
            "distinct_3": distinct_n(all_texts, 3),
            "self_bleu": self_bleu(all_texts),
            "skipped": False,
        }
        for k in k_values:
            row[f"pass_at_{k}"] = float(np.mean(pass_counts[k])) if pass_counts[k] else 0.0
        append_jsonl(eval_path, row)


if __name__ == "__main__":
    main()
