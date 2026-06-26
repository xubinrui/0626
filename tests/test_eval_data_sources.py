from __future__ import annotations

from opd_grpo_gradstruct.eval_e2e import load_eval_examples


def test_math_and_aime_smoke_sources_do_not_skip_offline():
    math_examples = load_eval_examples("math", "test", limit=8, seed=42, offline_only=True)
    aime_examples = load_eval_examples("aime2024", "train", limit=8, seed=42, offline_only=True)
    assert len(math_examples) == 2
    assert len(aime_examples) == 2
    assert all(ex.question and ex.answer for ex in math_examples + aime_examples)
