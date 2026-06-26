from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    return -(probs.clamp_min(1e-30) * probs.clamp_min(1e-30).log()).sum(dim=-1)


@torch.inference_mode()
def token_distributions(
    model,
    sequences_cpu: torch.Tensor,
    prompt_len: int,
    device: str,
    pad_token_id: int | None,
):
    sequences = sequences_cpu.to(device)
    outputs = model(sequences)
    logits = outputs.logits[:, prompt_len - 1 : -1, :].float()
    probs = F.softmax(logits, dim=-1)
    target_tokens = sequences[:, prompt_len:]
    if pad_token_id is None:
        valid = torch.ones_like(target_tokens, dtype=torch.bool)
    else:
        valid = target_tokens.ne(pad_token_id)
    eos = getattr(model.config, "eos_token_id", None)
    if isinstance(eos, int):
        after_eos = torch.cumsum(target_tokens.eq(eos).int(), dim=1) > 1
        valid = valid & (~after_eos)
    return probs.cpu(), target_tokens.cpu(), valid.cpu()


def flatten_valid(
    probs: torch.Tensor,
    target_tokens: torch.Tensor,
    valid_mask: torch.Tensor,
):
    flat_probs = probs[valid_mask]
    flat_tokens = target_tokens[valid_mask]
    return flat_probs, flat_tokens


def build_logit_gradients(
    arm: str,
    student_probs: torch.Tensor,
    teacher_probs: torch.Tensor,
    token_ids: torch.Tensor,
    rollout_advantages: list[float],
    valid_mask: torch.Tensor,
    lambda_joint: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if student_probs.shape != teacher_probs.shape:
        raise ValueError(f"Student/teacher probability shape mismatch: {student_probs.shape} vs {teacher_probs.shape}")

    flat_student, flat_tokens = flatten_valid(student_probs, token_ids, valid_mask)
    flat_teacher, _ = flatten_valid(teacher_probs, token_ids, valid_mask)
    row_adv = []
    token_to_rollout = []
    for row_idx, adv in enumerate(rollout_advantages):
        n_valid = int(valid_mask[row_idx].sum().item())
        row_adv.extend([adv] * n_valid)
        token_to_rollout.extend([row_idx] * n_valid)
    adv_t = torch.tensor(row_adv, dtype=torch.float32)

    grad = flat_student.clone()
    rows = torch.arange(flat_tokens.numel())
    if arm == "grpo":
        grad *= adv_t[:, None]
        grad[rows, flat_tokens] -= adv_t
    elif arm == "opd":
        grad -= flat_teacher
    elif arm == "joint":
        grad -= (1.0 - lambda_joint) * flat_teacher
        grad[rows, flat_tokens] -= lambda_joint * adv_t
    else:
        raise ValueError(f"Unknown arm: {arm}")

    token_prob = flat_student[rows, flat_tokens].clamp_min(1e-30)
    token_loss = -token_prob.log()
    student_entropy = _entropy_from_probs(flat_student)
    teacher_entropy = _entropy_from_probs(flat_teacher)
    q75_loss = torch.quantile(token_loss, 0.75) if token_loss.numel() else torch.tensor(0.0)
    q75_entropy = torch.quantile(student_entropy, 0.75) if student_entropy.numel() else torch.tensor(0.0)
    support_candidate = (token_loss >= q75_loss) & (student_entropy >= q75_entropy)

    meta = {
        "token_ids": flat_tokens.numpy().astype(np.int64),
        "token_loss": token_loss.numpy().astype(np.float32),
        "student_entropy": student_entropy.numpy().astype(np.float32),
        "teacher_entropy": teacher_entropy.numpy().astype(np.float32),
        "advantage": adv_t.numpy().astype(np.float32),
        "token_to_rollout": np.asarray(token_to_rollout, dtype=np.int32),
        "support_candidate": support_candidate.numpy().astype(np.bool_),
    }
    return grad.numpy().astype(np.float32), meta
