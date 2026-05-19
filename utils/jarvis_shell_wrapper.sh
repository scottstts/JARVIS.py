#!/usr/bin/env bash

set -euo pipefail

tool_bin="${HOME:-/root}/.local/bin/jarvis"
dev_bin="/opt/venv/bin/jarvis"

if [[ -x "${tool_bin}" ]]; then
    exec "${tool_bin}" "$@"
fi

exec "${dev_bin}" "$@"