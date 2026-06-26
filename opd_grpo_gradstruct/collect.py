from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .config import apply_cli_overrides, load_collect_config
from .data import build_prompt, group_advantages, gsm8k_reward, load_prompt_examples
from .gradients import build_logit_gradients, token_distributions
from .io import write_record
from .modeling import generate_group, load_causal_lm, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--student-model")
    parser.add_argument("--teacher-model")
    parser.add_argument("--limit-prompts", type=int)
    parser.add_argument("--group-size", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--student-device")
    parser.add_argument("--teacher-device")
    parser.add_argument("--teacher-4bit", action=argparse.BooleanOptionalAction)
    parser.add_argument("--out-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = apply_cli_overrides(load_collect_config(args.config), args)
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    with (Path(cfg.out_dir) / "collect_config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    tokenizer = load_tokenizer(cfg.student_model)
    teacher_tokenizer = load_tokenizer(cfg.teacher_model)
    if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise RuntimeError("Student and teacher tokenizers differ. This experiment requires a shared tokenizer.")

    student = load_causal_lm(
        cfg.student_model,
        device=cfg.student_device,
        dtype_name=cfg.student_dtype,
        four_bit=False,
    )
    teacher = load_causal_lm(
        cfg.teacher_model,
        device=cfg.teacher_device,
        dtype_name=cfg.teacher_dtype,
        four_bit=cfg.teacher_4bit,
    )

    examples = load_prompt_examples(cfg.dataset, cfg.dataset_split, cfg.limit_prompts, cfg.seed)
    for example in tqdm(examples, desc="collect"):
        prompt = build_prompt(example.question)
        sequences_cpu, prompt_len, _generated_cpu, texts = generate_group(
            student,
            tokenizer,
            prompt=prompt,
            group_size=cfg.group_size,
            max_prompt_tokens=cfg.max_prompt_tokens,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            device=cfg.student_device,
        )
        rewards = [gsm8k_reward(text, example.answer) for text in texts]
        advantages = group_advantages(rewards)

        student_probs, token_ids, valid_mask = token_distributions(
            student, sequences_cpu, prompt_len, cfg.student_device, tokenizer.pad_token_id
        )
        teacher_probs, teacher_token_ids, teacher_valid_mask = token_distributions(
            teacher, sequences_cpu, prompt_len, cfg.teacher_device, tokenizer.pad_token_id
        )
        if not torch.equal(token_ids, teacher_token_ids):
            raise RuntimeError("Teacher/student token id mismatch on shared rollout.")
        valid_mask = valid_mask & teacher_valid_mask

        for arm in cfg.arms:
            G, arrays = build_logit_gradients(
                arm=arm,
                student_probs=student_probs,
                teacher_probs=teacher_probs,
                token_ids=token_ids,
                rollout_advantages=advantages,
                valid_mask=valid_mask,
                lambda_joint=cfg.lambda_joint,
            )
            if arm == "grpo":
                G = G.astype(np.float16)
            metadata = {
                "prompt_idx": example.idx,
                "arm": arm,
                "dataset": example.dataset,
                "question": example.question,
                "gold_answer": example.answer,
                "generation_texts": texts,
                "rewards": rewards,
                "advantages": advantages,
                "pass_at_1_proxy": float(rewards[0] > 0.0) if rewards else 0.0,
                "pass_at_group_proxy": float(any(r > 0.0 for r in rewards)),
                "student_model": cfg.student_model,
                "teacher_model": cfg.teacher_model,
                "prompt_len": prompt_len,
                "group_size": cfg.group_size,
                "max_new_tokens": cfg.max_new_tokens,
                "temperature": cfg.temperature,
                "top_p": cfg.top_p,
                "lambda_joint": cfg.lambda_joint,
            }
            record_path = write_record(cfg.out_dir, arm, example.idx, G, arrays, metadata)
            tqdm.write(f"wrote {record_path} shape={G.shape}")

        del student_probs, teacher_probs, token_ids, valid_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
