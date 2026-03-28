#!/usr/bin/env bash

set -euo pipefail

tool_bin="${HOME:-/root}/.local/bin/jarvis"
tool_python="${HOME:-/root}/.local/share/uv/tools/jarvis/bin/python"
dev_bin="/opt/venv/bin/jarvis"

if [[ -x "${tool_bin}" && -x "${tool_python}" ]]; then
    package_dir="$("${tool_python}" -c 'from pathlib import Path; import jarvis; print(Path(jarvis.__file__).resolve().parent)')"

    if [[ -z "${JARVIS_SETTINGS_FILE:-}" ]]; then
        export JARVIS_SETTINGS_FILE="${package_dir}/settings.yml"
    fi

    if [[ -z "${JARVIS_IDENTITIES_DIR:-}" ]]; then
        export JARVIS_IDENTITIES_DIR="${package_dir}/identities"
    fi

    exec "${tool_bin}" "$@"
fi

exec "${dev_bin}" "$@"
