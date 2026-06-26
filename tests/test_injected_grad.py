from __future__ import annotations

import torch

from opd_grpo_gradstruct.train_e2e import inject_logit_gradient


def test_injected_logit_gradient_matches_finite_difference():
    torch.manual_seed(42)
    layer = torch.nn.Linear(3, 2, bias=False, dtype=torch.double)
    x = torch.randn(4, 3, dtype=torch.double)
    delta = torch.randn(4, 2, dtype=torch.double)

    logits = layer(x)
    inject_logit_gradient(logits, delta)
    autograd = layer.weight.grad.detach().clone()

    eps = 1e-6
    numerical = torch.zeros_like(layer.weight)
    with torch.no_grad():
        for i in range(layer.weight.shape[0]):
            for j in range(layer.weight.shape[1]):
                orig = layer.weight[i, j].item()
                layer.weight[i, j] = orig + eps
                plus = (layer(x) * delta).sum().item()
                layer.weight[i, j] = orig - eps
                minus = (layer(x) * delta).sum().item()
                layer.weight[i, j] = orig
                numerical[i, j] = (plus - minus) / (2 * eps)

    torch.testing.assert_close(autograd, numerical, rtol=1e-6, atol=1e-6)
