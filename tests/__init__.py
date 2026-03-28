"""Test package for Jarvis."""

from __future__ import annotations

import os
from pathlib import Path


_PACKAGED_SETTINGS_PATH = Path(__file__).resolve().parents[1] / "src" / "jarvis" / "settings.yml"
os.environ.setdefault("JARVIS_SETTINGS_FILE", str(_PACKAGED_SETTINGS_PATH))
