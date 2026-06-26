from __future__ import annotations

import torch

from opd_grpo_gradstruct.train_e2e import build_component_gradients, reward_direction_sanity


def test_grpo_direction_increases_reward_surrogate():
    logits = torch.tensor([[[2.0, 0.0, -1.0], [0.0, 1.0, -1.0]]], dtype=torch.float32)
    probs = torch.softmax(logits, dim=-1)
    teacher = probs.clone()
    token_ids = torch.tensor([[0, 1]])
    valid = torch.tensor([[True, True]])
    advantages = [1.0]
    g_grpo, _g_opd, _g_joint, _tokens = build_component_gradients(
        probs=probs,
        teacher_probs=teacher,
        token_ids=token_ids,
        valid_mask=valid,
        rollout_advantages=advantages,
        lambda_joint=0.5,
    )
    before, after, delta, ok, active = reward_direction_sanity(
        logits=logits,
        token_ids=token_ids,
        valid_mask=valid,
        delta=g_grpo,
        rollout_advantages=advantages,
        eps=1e-2,
        tolerance=1e-10,
    )
    assert after > before
    assert delta > 0
    assert ok
    assert active
