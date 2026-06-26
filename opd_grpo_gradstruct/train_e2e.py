from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW, SGD
from tqdm import trange
from transformers import AutoModelForCausalLM

from .config import dtype_from_name
from .data import build_prompt, group_advantages, gsm8k_reward, load_prompt_examples
from .gradients import token_distributions
from .modeling import generate_group, load_causal_lm, load_tokenizer
from .routing import (
    ConditionId,
    grpo_sparse_support_mask,
    lowrank_reconstruct,
    opd_magnitude_mask,
    random_matched_mask,
    rule_hash,
)


CONDITIONS: set[str] = {"b1", "b2", "b3", "b4", "ours", "b4b", "ours_minus"}


@dataclass
class StepResult:
    loss_proxy: float
    reward_mean: float
    pass_at_group_proxy: float
    reward_direction_before: float
    reward_direction_after: float
    reward_direction_delta: float
    reward_direction_ok: bool
    reward_direction_active: bool
    hard_fraction: float
    recon_error: float
    routing_svd_seconds: float
    entropy: float
    kl_to_teacher: float
    grad_norm: float
    peak_memory_mb: float
    wall_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--condition")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--limit-prompts", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--save-model", action=argparse.BooleanOptionalAction)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_e2e_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    parent = cfg.get("extends")
    if parent:
        parent_path = (cfg_path.parent / parent).resolve()
        return deep_update(load_e2e_config(parent_path), cfg)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_trainable_student(cfg: dict[str, Any]):
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype_from_name(cfg["student_dtype"]),
        "low_cpu_mem_usage": True,
        "device_map": {"": cfg["student_device"]},
    }
    model = AutoModelForCausalLM.from_pretrained(cfg["student_model"], **kwargs)
    model.config.use_cache = False
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
    for param in model.parameters():
        param.requires_grad_(True)
    return model


def make_optimizer(model, cfg: dict[str, Any]):
    params = [p for p in model.parameters() if p.requires_grad]
    opt_name = str(cfg.get("optimizer", "adamw")).lower()
    lr = float(cfg.get("learning_rate", 1e-6))
    wd = float(cfg.get("weight_decay", 0.0))
    if opt_name == "sgd":
        return SGD(params, lr=lr, weight_decay=wd)
    if opt_name == "adamw":
        return AdamW(params, lr=lr, weight_decay=wd)
    raise ValueError(f"Unsupported optimizer: {opt_name}")


def valid_generation_mask(model, target_tokens: torch.Tensor, pad_token_id: int | None) -> torch.Tensor:
    if pad_token_id is None:
        valid = torch.ones_like(target_tokens, dtype=torch.bool)
    else:
        valid = target_tokens.ne(pad_token_id)
    eos = getattr(model.config, "eos_token_id", None)
    if isinstance(eos, int):
        after_eos = torch.cumsum(target_tokens.eq(eos).int(), dim=1) > 1
        valid = valid & (~after_eos)
    return valid


def flat_advantages(valid_mask: torch.Tensor, rollout_advantages: list[float], device: torch.device) -> torch.Tensor:
    row_adv: list[float] = []
    for row_idx, adv in enumerate(rollout_advantages):
        row_adv.extend([float(adv)] * int(valid_mask[row_idx].sum().item()))
    return torch.tensor(row_adv, dtype=torch.float32, device=device)


def build_component_gradients(
    probs: torch.Tensor,
    teacher_probs: torch.Tensor,
    token_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    rollout_advantages: list[float],
    lambda_joint: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    flat_p, flat_tau, flat_tokens, adv = flatten_gradient_inputs(
        probs, teacher_probs, token_ids, valid_mask, rollout_advantages
    )
    g_grpo, g_opd, g_joint = build_needed_component_gradients(
        flat_p,
        flat_tau,
        flat_tokens,
        adv,
        lambda_joint,
        need_grpo=True,
        need_opd=True,
        need_joint=True,
    )
    return g_grpo, g_opd, g_joint, flat_tokens


def flatten_gradient_inputs(
    probs: torch.Tensor,
    teacher_probs: torch.Tensor,
    token_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    rollout_advantages: list[float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    flat_p = probs[valid_mask].float()
    flat_tau = teacher_probs[valid_mask].float()
    flat_tokens = token_ids[valid_mask]
    adv = flat_advantages(valid_mask, rollout_advantages, flat_p.device)
    return flat_p, flat_tau, flat_tokens, adv


def build_needed_component_gradients(
    flat_p: torch.Tensor,
    flat_tau: torch.Tensor,
    flat_tokens: torch.Tensor,
    adv: torch.Tensor,
    lambda_joint: float,
    *,
    need_grpo: bool,
    need_opd: bool,
    need_joint: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = torch.arange(flat_tokens.numel(), device=flat_p.device)

    empty = torch.empty(0, device=flat_p.device, dtype=flat_p.dtype)
    if need_opd:
        g_opd = flat_p - flat_tau
    else:
        g_opd = empty
    if need_grpo:
        g_grpo = flat_p * adv[:, None]
        if flat_tokens.numel():
            g_grpo[rows, flat_tokens] -= adv
    else:
        g_grpo = empty
    if need_joint:
        g_joint = flat_p - (1.0 - lambda_joint) * flat_tau
        if flat_tokens.numel():
            g_joint[rows, flat_tokens] -= lambda_joint * adv
    else:
        g_joint = empty
    return g_grpo, g_opd, g_joint


def needed_gradient_flags(condition: str) -> tuple[bool, bool, bool]:
    if condition == "b1":
        return True, False, False
    if condition == "b2":
        return False, True, False
    if condition == "b3":
        return False, False, True
    return True, True, False


def select_condition_delta(
    condition: ConditionId,
    g_grpo: torch.Tensor,
    g_opd: torch.Tensor,
    g_joint: torch.Tensor,
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[torch.Tensor, dict[str, Any]]:
    start = time.perf_counter()
    lambda_joint = float(cfg["lambda_joint"])
    rank = int(cfg.get("lowrank_rank", 32))
    target_fraction = float(cfg.get("hard_fraction_target", 0.20))
    min_fraction = float(cfg.get("hard_fraction_min", 0.15))
    max_fraction = float(cfg.get("hard_fraction_max", 0.25))
    rpca_topk = int(cfg.get("rpca_topk_cols_per_row", 16))
    rpca_max_iter = int(cfg.get("rpca_max_iter", 100))

    if condition == "b1":
        return g_grpo, {
            "hard_fraction": 0.0,
            "recon_error": 0.0,
            "routing_svd_seconds": 0.0,
            "routing_source": "none",
            "compressed_easy": False,
        }
    if condition == "b2":
        return g_opd, {
            "hard_fraction": 0.0,
            "recon_error": 0.0,
            "routing_svd_seconds": 0.0,
            "routing_source": "none",
            "compressed_easy": False,
        }
    if condition == "b3":
        return g_joint, {
            "hard_fraction": 0.0,
            "recon_error": 0.0,
            "routing_svd_seconds": 0.0,
            "routing_source": "none",
            "compressed_easy": False,
        }

    grpo_np = g_grpo.detach().cpu().numpy().astype(np.float32, copy=False)
    opd_np = g_opd.detach().cpu().numpy().astype(np.float32, copy=False)
    if condition in {"ours", "ours_minus"}:
        route = grpo_sparse_support_mask(
            grpo_np,
            target_fraction=target_fraction,
            min_fraction=min_fraction,
            max_fraction=max_fraction,
            rpca_topk_cols_per_row=rpca_topk,
            rpca_max_iter=rpca_max_iter,
        )
        routing_source = "grpo_sparse_support"
    elif condition == "b4":
        route = opd_magnitude_mask(
            opd_np,
            target_fraction=target_fraction,
            min_fraction=min_fraction,
            max_fraction=max_fraction,
        )
        routing_source = "opd_magnitude"
    elif condition == "b4b":
        ours_route = grpo_sparse_support_mask(
            grpo_np,
            target_fraction=target_fraction,
            min_fraction=min_fraction,
            max_fraction=max_fraction,
            rpca_topk_cols_per_row=rpca_topk,
            rpca_max_iter=rpca_max_iter,
        )
        route = random_matched_mask(grpo_np.shape[0], ours_route.hard_fraction, rng)
        routing_source = "random_matched_to_grpo_fraction"
    else:
        raise ValueError(f"Unknown condition: {condition}")

    hard = torch.from_numpy(route.mask).to(device=g_opd.device, dtype=torch.bool)
    delta = torch.empty_like(g_opd)
    hard_delta = g_grpo + (1.0 - lambda_joint) * g_opd
    delta[hard] = hard_delta[hard]

    compressed_easy = bool(cfg.get("compress_easy_tokens", condition in {"ours", "b4"}))
    recon_error = 0.0
    easy = ~hard
    if int(easy.sum().item()) > 0 and compressed_easy:
        easy_np = opd_np[route.mask == 0]
        recon, recon_error, kept_rank = lowrank_reconstruct(easy_np, rank, rng)
        delta[easy] = torch.from_numpy(recon).to(device=g_opd.device, dtype=g_opd.dtype)
    else:
        kept_rank = 0
        delta[easy] = g_opd[easy]

    elapsed = time.perf_counter() - start
    return delta, {
        "hard_fraction": route.hard_fraction,
        "recon_error": recon_error,
        "routing_svd_seconds": elapsed,
        "routing_source": routing_source,
        "compressed_easy": compressed_easy,
        "lowrank_rank_kept": kept_rank,
        "route_threshold": route.threshold,
        "rpca_rank": route.rpca_rank,
        "rpca_iters": route.rpca_iters,
        "rpca_recon_error": route.rpca_recon_error,
        "route_energy_coverage": route.energy_coverage,
    }


def inject_logit_gradient(logits: torch.Tensor, grad_logits: torch.Tensor) -> None:
    logits.backward(gradient=grad_logits)


def grad_norm(model) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().float().norm().item() ** 2)
    return total**0.5


def reward_surrogate_from_logits(
    flat_logits: torch.Tensor,
    flat_tokens: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    if flat_tokens.numel() == 0:
        return torch.tensor(0.0, device=flat_logits.device, dtype=torch.float32)
    rows = torch.arange(flat_tokens.numel(), device=flat_logits.device)
    logp = F.log_softmax(flat_logits.float(), dim=-1)[rows, flat_tokens]
    return (advantages.float() * logp).mean()


def entropy_kl_means_chunked(flat_p: torch.Tensor, flat_tau: torch.Tensor, chunk_rows: int) -> tuple[float, float]:
    if flat_p.numel() == 0:
        return 0.0, 0.0
    chunk_rows = max(1, int(chunk_rows))
    entropy_sum = 0.0
    kl_sum = 0.0
    n_rows = flat_p.shape[0]
    with torch.no_grad():
        for start in range(0, n_rows, chunk_rows):
            end = min(start + chunk_rows, n_rows)
            p = flat_p[start:end].float()
            tau = flat_tau[start:end].float()
            p_safe = p.clamp_min(1e-30)
            tau_safe = tau.clamp_min(1e-30)
            logp = p_safe.log()
            entropy_sum += float((-(p_safe * logp).sum(dim=-1)).sum().item())
            kl_sum += float((p_safe * (logp - tau_safe.log())).sum(dim=-1).sum().item())
    return entropy_sum / n_rows, kl_sum / n_rows


def valid_logit_delta_dot_chunked(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
    delta: torch.Tensor,
    scale: float,
    chunk_rows: int,
) -> float:
    if delta.numel() == 0:
        return 0.0
    positions = valid_mask.nonzero(as_tuple=False)
    chunk_rows = max(1, int(chunk_rows))
    total = 0.0
    with torch.no_grad():
        for start in range(0, positions.shape[0], chunk_rows):
            end = min(start + chunk_rows, positions.shape[0])
            pos = positions[start:end]
            chunk_logits = logits.detach()[pos[:, 0], pos[:, 1], :].float()
            chunk_delta = delta[start:end].float() * scale
            total += float((chunk_logits * chunk_delta).sum().item())
    return total


def dense_delta_chunk(
    condition: str,
    flat_p: torch.Tensor,
    flat_tau: torch.Tensor,
    flat_tokens: torch.Tensor,
    adv: torch.Tensor,
    lambda_joint: float,
) -> torch.Tensor:
    rows = torch.arange(flat_tokens.numel(), device=flat_p.device)
    if condition == "b1":
        delta = flat_p * adv[:, None]
        if flat_tokens.numel():
            delta[rows, flat_tokens] -= adv
        return delta
    if condition == "b2":
        return flat_p - flat_tau
    if condition == "b3":
        delta = flat_p - (1.0 - lambda_joint) * flat_tau
        if flat_tokens.numel():
            delta[rows, flat_tokens] -= lambda_joint * adv
        return delta
    raise ValueError(f"dense_delta_chunk only supports b1/b2/b3, got {condition}")


def dense_condition_grad_logits_chunked(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
    flat_p: torch.Tensor,
    flat_tau: torch.Tensor,
    flat_tokens: torch.Tensor,
    flat_adv: torch.Tensor,
    condition: str,
    lambda_joint: float,
    eps: float,
    tolerance: float,
    chunk_rows: int,
) -> tuple[torch.Tensor, float, float, float, float, bool, bool]:
    positions = valid_mask.nonzero(as_tuple=False)
    n_rows = int(positions.shape[0])
    grad_logits = torch.zeros_like(logits, dtype=torch.float32)
    if n_rows == 0:
        return grad_logits, 0.0, 0.0, 0.0, 0.0, True, False

    scale = 1.0 / float(n_rows)
    chunk_rows = max(1, int(chunk_rows))
    loss_proxy = 0.0
    before_sum = 0.0
    after_sum = 0.0
    active = bool(flat_adv.abs().sum().item() > 0.0)
    for start in range(0, n_rows, chunk_rows):
        end = min(start + chunk_rows, n_rows)
        pos = positions[start:end]
        delta = dense_delta_chunk(
            condition,
            flat_p[start:end],
            flat_tau[start:end],
            flat_tokens[start:end],
            flat_adv[start:end],
            lambda_joint,
        )
        grad_logits[pos[:, 0], pos[:, 1], :] = delta * scale
        with torch.no_grad():
            chunk_logits = logits.detach()[pos[:, 0], pos[:, 1], :].float()
            loss_proxy += float((chunk_logits * delta.float() * scale).sum().item())
            before = reward_surrogate_from_logits(chunk_logits, flat_tokens[start:end], flat_adv[start:end])
            after = reward_surrogate_from_logits(chunk_logits - eps * delta.detach().float(), flat_tokens[start:end], flat_adv[start:end])
            before_sum += float(before.item()) * (end - start)
            after_sum += float(after.item()) * (end - start)
        del delta

    before_mean = before_sum / n_rows
    after_mean = after_sum / n_rows
    direction_delta = after_mean - before_mean
    return (
        grad_logits,
        loss_proxy,
        before_mean,
        after_mean,
        direction_delta,
        bool(direction_delta >= -tolerance),
        active,
    )


def build_grpo_opd_numpy_chunked(
    flat_p: torch.Tensor,
    flat_tau: torch.Tensor,
    flat_tokens: torch.Tensor,
    flat_adv: torch.Tensor,
    chunk_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_rows, vocab = flat_p.shape
    g_grpo = np.empty((n_rows, vocab), dtype=np.float32)
    g_opd = np.empty((n_rows, vocab), dtype=np.float32)
    chunk_rows = max(1, int(chunk_rows))
    for start in range(0, n_rows, chunk_rows):
        end = min(start + chunk_rows, n_rows)
        p = flat_p[start:end].detach().cpu().numpy().astype(np.float32, copy=True)
        tau = flat_tau[start:end].detach().cpu().numpy().astype(np.float32, copy=False)
        adv = flat_adv[start:end].detach().cpu().numpy().astype(np.float32, copy=False)
        tokens = flat_tokens[start:end].detach().cpu().numpy()
        opd = p - tau
        grpo = p * adv[:, None]
        if tokens.size:
            grpo[np.arange(tokens.size), tokens] -= adv
        g_opd[start:end] = opd
        g_grpo[start:end] = grpo
    return g_grpo, g_opd


def select_condition_delta_numpy(
    condition: str,
    g_grpo: np.ndarray,
    g_opd: np.ndarray,
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    start = time.perf_counter()
    lambda_joint = float(cfg["lambda_joint"])
    rank = int(cfg.get("lowrank_rank", 32))
    target_fraction = float(cfg.get("hard_fraction_target", 0.20))
    min_fraction = float(cfg.get("hard_fraction_min", 0.15))
    max_fraction = float(cfg.get("hard_fraction_max", 0.25))
    rpca_topk = int(cfg.get("rpca_topk_cols_per_row", 16))
    rpca_max_iter = int(cfg.get("rpca_max_iter", 100))

    if condition in {"ours", "ours_minus"}:
        route = grpo_sparse_support_mask(
            g_grpo,
            target_fraction=target_fraction,
            min_fraction=min_fraction,
            max_fraction=max_fraction,
            rpca_topk_cols_per_row=rpca_topk,
            rpca_max_iter=rpca_max_iter,
        )
        routing_source = "grpo_sparse_support"
    elif condition == "b4":
        route = opd_magnitude_mask(
            g_opd,
            target_fraction=target_fraction,
            min_fraction=min_fraction,
            max_fraction=max_fraction,
        )
        routing_source = "opd_magnitude"
    elif condition == "b4b":
        ours_route = grpo_sparse_support_mask(
            g_grpo,
            target_fraction=target_fraction,
            min_fraction=min_fraction,
            max_fraction=max_fraction,
            rpca_topk_cols_per_row=rpca_topk,
            rpca_max_iter=rpca_max_iter,
        )
        route = random_matched_mask(g_grpo.shape[0], ours_route.hard_fraction, rng)
        routing_source = "random_matched_to_grpo_fraction"
    else:
        raise ValueError(f"select_condition_delta_numpy only supports routed conditions, got {condition}")

    hard = route.mask
    easy = ~hard
    delta = np.empty_like(g_opd, dtype=np.float32)
    delta[hard] = g_grpo[hard] + (1.0 - lambda_joint) * g_opd[hard]

    compressed_easy = bool(cfg.get("compress_easy_tokens", condition in {"ours", "b4"}))
    recon_error = 0.0
    if int(easy.sum()) > 0 and compressed_easy:
        recon, recon_error, kept_rank = lowrank_reconstruct(g_opd[easy], rank, rng)
        delta[easy] = recon
    else:
        kept_rank = 0
        delta[easy] = g_opd[easy]

    elapsed = time.perf_counter() - start
    return delta, {
        "hard_fraction": route.hard_fraction,
        "recon_error": recon_error,
        "routing_svd_seconds": elapsed,
        "routing_source": routing_source,
        "compressed_easy": compressed_easy,
        "lowrank_rank_kept": kept_rank,
        "route_threshold": route.threshold,
        "rpca_rank": route.rpca_rank,
        "rpca_iters": route.rpca_iters,
        "rpca_recon_error": route.rpca_recon_error,
        "route_energy_coverage": route.energy_coverage,
    }


def numpy_delta_grad_logits_chunked(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
    flat_tokens: torch.Tensor,
    flat_adv: torch.Tensor,
    delta_np: np.ndarray,
    eps: float,
    tolerance: float,
    chunk_rows: int,
) -> tuple[torch.Tensor, float, float, float, float, bool, bool]:
    positions = valid_mask.nonzero(as_tuple=False)
    n_rows = int(positions.shape[0])
    grad_logits = torch.zeros_like(logits, dtype=torch.float32)
    if n_rows == 0:
        return grad_logits, 0.0, 0.0, 0.0, 0.0, True, False
    scale = 1.0 / float(n_rows)
    chunk_rows = max(1, int(chunk_rows))
    loss_proxy = 0.0
    before_sum = 0.0
    after_sum = 0.0
    active = bool(flat_adv.abs().sum().item() > 0.0)
    device = logits.device
    for start in range(0, n_rows, chunk_rows):
        end = min(start + chunk_rows, n_rows)
        pos = positions[start:end]
        delta = torch.from_numpy(delta_np[start:end]).to(device=device, dtype=torch.float32)
        grad_logits[pos[:, 0], pos[:, 1], :] = delta * scale
        with torch.no_grad():
            chunk_logits = logits.detach()[pos[:, 0], pos[:, 1], :].float()
            loss_proxy += float((chunk_logits * delta * scale).sum().item())
            before = reward_surrogate_from_logits(chunk_logits, flat_tokens[start:end], flat_adv[start:end])
            after = reward_surrogate_from_logits(chunk_logits - eps * delta, flat_tokens[start:end], flat_adv[start:end])
            before_sum += float(before.item()) * (end - start)
            after_sum += float(after.item()) * (end - start)
        del delta
    before_mean = before_sum / n_rows
    after_mean = after_sum / n_rows
    direction_delta = after_mean - before_mean
    return (
        grad_logits,
        loss_proxy,
        before_mean,
        after_mean,
        direction_delta,
        bool(direction_delta >= -tolerance),
        active,
    )


def reward_direction_sanity(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    valid_mask: torch.Tensor,
    delta: torch.Tensor,
    rollout_advantages: list[float],
    eps: float,
    tolerance: float,
    chunk_rows: int = 16,
) -> tuple[float, float, float, bool, bool]:
    positions = valid_mask.nonzero(as_tuple=False)
    flat_tokens = token_ids[valid_mask]
    advantages = flat_advantages(valid_mask, rollout_advantages, logits.device)
    active = bool(advantages.abs().sum().item() > 0.0)
    if positions.numel() == 0:
        return 0.0, 0.0, 0.0, True, active
    before_sum = 0.0
    after_sum = 0.0
    chunk_rows = max(1, int(chunk_rows))
    with torch.no_grad():
        for start in range(0, positions.shape[0], chunk_rows):
            end = min(start + chunk_rows, positions.shape[0])
            pos = positions[start:end]
            chunk_logits = logits.detach()[pos[:, 0], pos[:, 1], :].float()
            chunk_tokens = flat_tokens[start:end]
            chunk_adv = advantages[start:end]
            before = reward_surrogate_from_logits(chunk_logits, chunk_tokens, chunk_adv)
            after = reward_surrogate_from_logits(chunk_logits - eps * delta[start:end].detach().float(), chunk_tokens, chunk_adv)
            before_sum += float(before.item()) * (end - start)
            after_sum += float(after.item()) * (end - start)
    before_mean = before_sum / positions.shape[0]
    after_mean = after_sum / positions.shape[0]
    diff = after_mean - before_mean
    return before_mean, after_mean, diff, bool(diff >= -tolerance), active


def train_one_step(
    student,
    teacher,
    tokenizer,
    example,
    optimizer,
    cfg: dict[str, Any],
    condition: ConditionId,
    rng: np.random.Generator,
) -> StepResult:
    step_start = time.perf_counter()
    device = torch.device(cfg["student_device"])
    student.eval()
    prompt = build_prompt(example.question)
    sequences_cpu, prompt_len, _generated_cpu, texts = generate_group(
        student,
        tokenizer,
        prompt=prompt,
        group_size=int(cfg["group_size"]),
        max_prompt_tokens=int(cfg["max_prompt_tokens"]),
        max_new_tokens=int(cfg["max_new_tokens"]),
        temperature=float(cfg["temperature"]),
        top_p=float(cfg["top_p"]),
        device=cfg["student_device"],
    )
    rewards = [gsm8k_reward(text, example.answer) for text in texts]
    advantages = group_advantages(rewards)

    teacher_probs_cpu, teacher_token_ids, teacher_valid_cpu = token_distributions(
        teacher, sequences_cpu, prompt_len, cfg["teacher_device"], tokenizer.pad_token_id
    )

    student.train()
    optimizer.zero_grad(set_to_none=True)
    sequences = sequences_cpu.to(device)
    outputs = student(sequences)
    logits = outputs.logits[:, prompt_len - 1 : -1, :].float()
    with torch.no_grad():
        probs = F.softmax(logits, dim=-1)
    token_ids = sequences[:, prompt_len:]
    if not torch.equal(token_ids.detach().cpu(), teacher_token_ids):
        raise RuntimeError("Teacher/student token id mismatch on shared rollout.")
    valid_mask = valid_generation_mask(student, token_ids, tokenizer.pad_token_id)
    valid_mask = valid_mask & teacher_valid_cpu.to(device)
    teacher_probs = teacher_probs_cpu.to(device=device, dtype=torch.float32)

    flat_p, flat_tau, flat_tokens, flat_adv = flatten_gradient_inputs(
        probs, teacher_probs, token_ids, valid_mask, advantages
    )
    metric_chunk_rows = int(cfg.get("metric_chunk_rows", 16))
    entropy_mean, kl_mean = entropy_kl_means_chunked(flat_p, flat_tau, metric_chunk_rows)
    if condition in {"b1", "b2", "b3"}:
        (
            grad_logits,
            loss_proxy,
            direction_before,
            direction_after,
            direction_delta,
            direction_ok,
            direction_active,
        ) = dense_condition_grad_logits_chunked(
            logits=logits,
            valid_mask=valid_mask,
            flat_p=flat_p,
            flat_tau=flat_tau,
            flat_tokens=flat_tokens,
            flat_adv=flat_adv,
            condition=condition,
            lambda_joint=float(cfg["lambda_joint"]),
            eps=float(cfg.get("reward_direction_probe_eps", 1e-3)),
            tolerance=float(cfg.get("reward_direction_tolerance", 1e-8)),
            chunk_rows=metric_chunk_rows,
        )
        route_meta = {
            "hard_fraction": 0.0,
            "recon_error": 0.0,
            "routing_svd_seconds": 0.0,
        }
        del flat_p, flat_tau, flat_adv
    else:
        g_grpo_np, g_opd_np = build_grpo_opd_numpy_chunked(
            flat_p=flat_p,
            flat_tau=flat_tau,
            flat_tokens=flat_tokens,
            flat_adv=flat_adv,
            chunk_rows=metric_chunk_rows,
        )
        delta_np, route_meta = select_condition_delta_numpy(condition, g_grpo_np, g_opd_np, cfg, rng)
        del g_grpo_np, g_opd_np, flat_p, flat_tau
        (
            grad_logits,
            loss_proxy,
            direction_before,
            direction_after,
            direction_delta,
            direction_ok,
            direction_active,
        ) = numpy_delta_grad_logits_chunked(
            logits=logits,
            valid_mask=valid_mask,
            flat_tokens=flat_tokens,
            flat_adv=flat_adv,
            delta_np=delta_np,
            eps=float(cfg.get("reward_direction_probe_eps", 1e-3)),
            tolerance=float(cfg.get("reward_direction_tolerance", 1e-8)),
            chunk_rows=metric_chunk_rows,
        )
        del delta_np, flat_adv
    del probs, teacher_probs
    inject_logit_gradient(logits, grad_logits)
    gnorm = grad_norm(student)
    optimizer.step()

    if torch.cuda.is_available() and str(device).startswith("cuda"):
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024**2)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        peak_memory = 0.0

    return StepResult(
        loss_proxy=loss_proxy,
        reward_mean=float(np.mean(rewards)) if rewards else 0.0,
        pass_at_group_proxy=float(any(r > 0.0 for r in rewards)),
        reward_direction_before=direction_before,
        reward_direction_after=direction_after,
        reward_direction_delta=direction_delta,
        reward_direction_ok=direction_ok,
        reward_direction_active=direction_active,
        hard_fraction=float(route_meta["hard_fraction"]),
        recon_error=float(route_meta["recon_error"]),
        routing_svd_seconds=float(route_meta["routing_svd_seconds"]),
        entropy=entropy_mean,
        kl_to_teacher=kl_mean,
        grad_norm=gnorm,
        peak_memory_mb=peak_memory,
        wall_seconds=time.perf_counter() - step_start,
    )


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
    if args.num_steps is not None:
        cfg["num_steps"] = args.num_steps
    if args.limit_prompts is not None:
        cfg["limit_prompts"] = args.limit_prompts
    if args.max_new_tokens is not None:
        cfg["max_new_tokens"] = args.max_new_tokens
    if args.save_model is not None:
        cfg["save_model"] = args.save_model
    condition = str(cfg["condition"]).lower()
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {sorted(CONDITIONS)}, got {condition}")
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    output_template = str(cfg.get("output_dir") or f"runs/{condition}/{seed}")
    output_dir = Path(args.output_dir or output_template.format(condition=condition, seed=seed))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    tokenizer = load_tokenizer(cfg["student_model"])
    teacher_tokenizer = load_tokenizer(cfg["teacher_model"])
    if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise RuntimeError("Student and teacher tokenizers differ. This experiment requires a shared tokenizer.")

    student = load_trainable_student(cfg)
    teacher = load_causal_lm(
        cfg["teacher_model"],
        device=cfg["teacher_device"],
        dtype_name=cfg["teacher_dtype"],
        four_bit=bool(cfg.get("teacher_4bit", False)),
    )
    optimizer = make_optimizer(student, cfg)
    examples = load_prompt_examples(cfg["dataset"], cfg["dataset_split"], int(cfg["limit_prompts"]), seed)
    rng = np.random.default_rng(seed)
    num_steps = int(cfg.get("num_steps", len(examples)))
    train_path = output_dir / "train.jsonl"
    if train_path.exists():
        train_path.unlink()

    init_entropy: float | None = None
    peak_pass_group = 0.0
    collapse_flags: list[dict[str, Any]] = []
    compressed_easy = bool(cfg.get("compress_easy_tokens", condition in {"ours", "b4"}))
    rhash = rule_hash(condition, compressed_easy=compressed_easy, rank=int(cfg.get("lowrank_rank", 32)), lambda_joint=float(cfg["lambda_joint"]))
    for step in trange(num_steps, desc=f"train {condition} seed={seed}"):
        example = examples[step % len(examples)]
        result = train_one_step(student, teacher, tokenizer, example, optimizer, cfg, condition, rng)  # type: ignore[arg-type]
        if init_entropy is None:
            init_entropy = result.entropy
        peak_pass_group = max(peak_pass_group, result.pass_at_group_proxy)
        collapse = False
        if init_entropy and result.entropy < float(cfg.get("collapse_entropy_frac", 0.40)) * init_entropy:
            collapse = True
        if peak_pass_group > 0 and result.pass_at_group_proxy < peak_pass_group - float(cfg.get("collapse_pass8_drop", 0.25)):
            collapse = True
        if collapse:
            collapse_flags.append({"step": step, "entropy": result.entropy, "pass_at_group_proxy": result.pass_at_group_proxy})
        row = {
            "step": step,
            "condition": condition,
            "seed": seed,
            "prompt_idx": example.idx,
            "rule_hash": rhash,
            **result.__dict__,
        }
        append_jsonl(train_path, row)

    with (output_dir / "collapse_flags.json").open("w", encoding="utf-8") as f:
        json.dump(collapse_flags, f, indent=2, ensure_ascii=False)
    if bool(cfg.get("save_model", False)):
        model_dir = output_dir / "final_model"
        student.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)


if __name__ == "__main__":
    main()
