#!/usr/bin/env bash
set -euo pipefail

docker compose exec \
  -e JARVIS_SETTINGS_FILE=/workspace/settings/settings.yml \
  jarvis_runtime \
  bash -lc 'cd /repo && exec uv run python tests/headless/headless.py "$@"' \
  headless.py "$@"
