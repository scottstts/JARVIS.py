"""Internal defaults for the bash tool runtime."""

from __future__ import annotations

from pathlib import Path


DEFAULT_BASH_EXECUTABLE = "/bin/bash"
DEFAULT_BASH_DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_BASH_MAX_TIMEOUT_SECONDS = 1800.0
DEFAULT_BASH_FOREGROUND_SOFT_TIMEOUT_SECONDS = 15.0
DEFAULT_BASH_MAX_OUTPUT_CHARS = 40_000
DEFAULT_BASH_JOB_LOG_MAX_BYTES = 4 * 1024 * 1024
DEFAULT_BASH_JOB_TOTAL_STORAGE_BUDGET_BYTES = 128 * 1024 * 1024
DEFAULT_BASH_JOB_RETENTION_SECONDS = 86_400.0
DEFAULT_BASH_DANGEROUSLY_SKIP_PERMISSION = False
DEFAULT_CENTRAL_PYTHON_VENV = Path("/opt/venv")
DEFAULT_CENTRAL_PYTHON_STARTER_PACKAGES = (
    "python-dateutil",
    "pyyaml",
    "pymupdf",
    "pandas",
    "matplotlib",
    "pillow",
    "pydantic",
    "jinja2",
    "rapidfuzz",
    "markdown-it-py",
    "feedparser",
    "openpyxl",
    "python-docx",
    "icalendar",
)
