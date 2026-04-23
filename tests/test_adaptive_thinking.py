"""Unit tests for adaptive-thinking format on Claude 4.6 / 4.7 models.

These tests don't hit the API — they just check that `ReasoningConfig`
serialises to the right Anthropic request shape for each model generation.
"""

import pytest

from localrouter.dtypes import ReasoningConfig


ADAPTIVE_MODELS = [
    "claude-opus-4-6",
    "claude-opus-4-6-20251001",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
    "claude-opus-4-7-20260101",  # hypothetical dated variant, still adaptive
    "claude-sonnet-4-7",
    "claude-haiku-4-7",
]

LEGACY_THINKING_MODELS = [
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    "claude-haiku-4-5-20251001",
    "claude-opus-4-5-20251101",
    "claude-sonnet-4-5-20250929",
    "claude-3-7-sonnet-20250219",
]


@pytest.mark.parametrize("model", ADAPTIVE_MODELS)
def test_adaptive_effort_high(model):
    cfg = ReasoningConfig(effort="high")
    out = cfg.to_anthropic_format(model)
    assert out == {"type": "adaptive", "effort": "high"}, out


@pytest.mark.parametrize("model", ADAPTIVE_MODELS)
def test_adaptive_effort_medium(model):
    cfg = ReasoningConfig(effort="medium")
    out = cfg.to_anthropic_format(model)
    assert out == {"type": "adaptive", "effort": "medium"}, out


@pytest.mark.parametrize("model", ADAPTIVE_MODELS)
def test_adaptive_budget_tokens_maps_to_effort(model):
    # budget_tokens on an adaptive model is coerced to the nearest effort level.
    assert ReasoningConfig(budget_tokens=1000).to_anthropic_format(model) == {
        "type": "adaptive",
        "effort": "low",
    }
    assert ReasoningConfig(budget_tokens=5000).to_anthropic_format(model) == {
        "type": "adaptive",
        "effort": "medium",
    }
    assert ReasoningConfig(budget_tokens=16000).to_anthropic_format(model) == {
        "type": "adaptive",
        "effort": "high",
    }


@pytest.mark.parametrize("model", ADAPTIVE_MODELS)
def test_adaptive_effort_none_disables_thinking(model):
    cfg = ReasoningConfig(effort="none")
    assert cfg.to_anthropic_format(model) is None


@pytest.mark.parametrize("model", LEGACY_THINKING_MODELS)
def test_legacy_models_use_enabled_with_budget(model):
    cfg = ReasoningConfig(effort="high")
    out = cfg.to_anthropic_format(model)
    assert out is not None, f"expected thinking on {model}"
    assert out.get("type") == "enabled", out
    assert out.get("budget_tokens", 0) > 0, out


def test_non_thinking_models_return_none():
    cfg = ReasoningConfig(effort="high")
    assert cfg.to_anthropic_format("claude-3-haiku-20240307") is None
    assert cfg.to_anthropic_format("gpt-5") is None  # not an anthropic model
    assert cfg.to_anthropic_format("") is None


def test_opus_4_7_distinct_from_legacy_opus_4():
    # Regression guard: "claude-opus-4-20250514" must not match the adaptive
    # substring check. Before the 4-7 addition, only 4-6 was adaptive, so a
    # naive "claude-opus-4" substring match would have caught too much.
    cfg = ReasoningConfig(effort="high")
    legacy = cfg.to_anthropic_format("claude-opus-4-20250514")
    adaptive = cfg.to_anthropic_format("claude-opus-4-7")
    assert legacy is not None and legacy.get("type") == "enabled"
    assert adaptive is not None and adaptive.get("type") == "adaptive"
