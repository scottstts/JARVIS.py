"""Documented Claude request capabilities (reviewed 2026-10-20)."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace


_STANDARD_EFFORTS = ("low", "medium", "high")
_MAX_EFFORTS = (*_STANDARD_EFFORTS, "max")
_ALL_EFFORTS = (*_STANDARD_EFFORTS, "xhigh", "max")


@dataclass(frozen=True, slots=True)
class AnthropicModelCapabilities:
    adaptive: bool = False
    manual_thinking: bool = True
    thinking_default: bool = False
    can_disable_thinking: bool = True
    between_tools: bool = False
    high_effort_requires_thinking: bool = False
    sampling: bool = True
    forced_tools: bool = True
    efforts: tuple[str, ...] = ()


_CLAUDE_46 = AnthropicModelCapabilities(adaptive=True, efforts=_MAX_EFFORTS)
_ADAPTIVE_ONLY = AnthropicModelCapabilities(
    adaptive=True, manual_thinking=False, sampling=False, efforts=_ALL_EFFORTS,
)
_DEFAULT_THINKING = AnthropicModelCapabilities(
    adaptive=True, manual_thinking=False, thinking_default=True,
    sampling=False, efforts=_ALL_EFFORTS,
)
_ALWAYS_THINKING = AnthropicModelCapabilities(
    adaptive=True, manual_thinking=False, thinking_default=True,
    can_disable_thinking=False, sampling=False,
    efforts=_ALL_EFFORTS,
)
_ALWAYS_THINKING_NO_FORCE = replace(_ALWAYS_THINKING, forced_tools=False)
_MODELS = {
    "claude-opus-4-5": AnthropicModelCapabilities(efforts=_STANDARD_EFFORTS),
    "claude-opus-4-6": _CLAUDE_46,
    "claude-sonnet-4-6": _CLAUDE_46,
    "claude-opus-4-7": _ADAPTIVE_ONLY,
    "claude-opus-4-8": _ADAPTIVE_ONLY,
    "claude-opus-5": AnthropicModelCapabilities(
        adaptive=True, manual_thinking=False, thinking_default=True,
        high_effort_requires_thinking=True, sampling=False, efforts=_ALL_EFFORTS,
    ),
    "claude-opus-5-5": _ALWAYS_THINKING_NO_FORCE,
    "claude-sonnet-5": _DEFAULT_THINKING,
    "claude-sonnet-5-5": AnthropicModelCapabilities(
        adaptive=True, manual_thinking=False, thinking_default=True,
        can_disable_thinking=False, between_tools=True,
        high_effort_requires_thinking=True, sampling=False, forced_tools=False,
        efforts=_ALL_EFFORTS,
    ),
    "claude-haiku-5-5": AnthropicModelCapabilities(
        adaptive=True, manual_thinking=False, thinking_default=True,
        high_effort_requires_thinking=True, sampling=False, efforts=_ALL_EFFORTS,
    ),
    "claude-fable-5": _ALWAYS_THINKING,
    "claude-fable-5-1": _ALWAYS_THINKING_NO_FORCE,
    "claude-mythos-5": _ALWAYS_THINKING,
    "claude-mythos-5-1": _ALWAYS_THINKING_NO_FORCE,
    "claude-mythos-preview": AnthropicModelCapabilities(
        adaptive=True, thinking_default=True, can_disable_thinking=False,
        sampling=False, efforts=_MAX_EFFORTS,
    ),
}


def anthropic_model_capabilities(model: str) -> AnthropicModelCapabilities:
    # Recognize snapshots and hosted IDs without treating an unknown future
    # version (for example opus-5-6) as the base opus-5 model.
    normalized = model.lower()
    for model_id, capabilities in _MODELS.items():
        if re.search(rf"(?<![a-z0-9]){model_id}(?:-20\d{{6}})?(?:-v\d+:\d+)?$", normalized):
            return capabilities
    return AnthropicModelCapabilities()
