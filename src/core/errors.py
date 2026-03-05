"""Error types for the core agent loop."""

from __future__ import annotations


class CoreError(Exception):
    """Base class for core loop errors."""


class CoreConfigurationError(CoreError):
    """Raised when core loop configuration is invalid."""


class ContextBudgetError(CoreError):
    """Raised when request still exceeds budget after compaction attempts."""
