#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
host_repo_root="$(cd "${script_dir}/.." && pwd)"

tool_bin="${HOME:-/root}/.local/bin/jarvis"
tool_python="${HOME:-/root}/.local/share/uv/tools/jarvis/bin/python"

usage() {
    cat <<'EOF'
Usage: install_build.sh [--reinstall]

Builds and installs the Jarvis wheel into the jarvis_runtime container.

Options:
  --reinstall  Rebuild the wheel and force reinstall it.
  --help       Show this help.
EOF
}

show_help=0
for arg in "$@"; do
    if [[ "${arg}" == "--help" ]]; then
        show_help=1
        break
    fi
done

if [[ ! -d /repo || ! -f /repo/pyproject.toml ]]; then
    if ((show_help)); then
        usage
        exit 0
    fi
    cd "${host_repo_root}"
    exec docker compose exec jarvis_runtime bash /repo/utils/install_build.sh "$@"
fi

export PATH="${HOME:-/root}/.local/bin:${PATH}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

force_reinstall=0

while (($# > 0)); do
    case "$1" in
        --reinstall)
            force_reinstall=1
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

tool_installed=0
if [[ -x "${tool_bin}" && -x "${tool_python}" ]]; then
    tool_installed=1
fi

install_wheel() {
    local project_version
    local wheel_candidates
    local wheel_path

    cd /repo
    uv build

    project_version="$(
        python -c 'import tomllib; from pathlib import Path; print(tomllib.loads(Path("/repo/pyproject.toml").read_text(encoding="utf-8"))["project"]["version"])'
    )"

    shopt -s nullglob
    wheel_candidates=(/repo/dist/jarvis-"${project_version}"-*.whl)
    shopt -u nullglob

    if ((${#wheel_candidates[@]} == 0)); then
        echo "No built Jarvis wheel found under /repo/dist for version ${project_version}." >&2
        exit 1
    fi

    wheel_path="$(ls -1t "${wheel_candidates[@]}" | head -n 1)"

    if ((force_reinstall)); then
        uv tool install --force "${wheel_path}"
    else
        uv tool install "${wheel_path}"
    fi
}

if ((force_reinstall)) || ((tool_installed == 0)); then
    install_wheel
    tool_installed=1
fi

if [[ ! -x "${tool_bin}" || ! -x "${tool_python}" ]]; then
    echo "Installed Jarvis tool is unavailable after installation." >&2
    exit 1
fi

echo "Jarvis artifact is installed in jarvis_runtime."
echo "Run \`jarvis\` inside the container to start the installed build."
