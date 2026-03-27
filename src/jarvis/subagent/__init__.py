"""Public API for the subagent subsystem."""

from .manager import SubagentManager
from .primitives import (
    SUBAGENT_PRIMITIVE_NAMES,
    build_subagent_primitive_definitions,
    render_subagent_primitive_docs,
)

__all__ = [
    "SUBAGENT_PRIMITIVE_NAMES",
    "SubagentManager",
    "build_subagent_primitive_definitions",
    "render_subagent_primitive_docs",
]
