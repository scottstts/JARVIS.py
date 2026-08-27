#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

cd "${repo_root}"

reinstall=0
if [[ "${1:-}" == "--reinstall" ]]; then
    reinstall=1
    shift
fi

docker compose start

if ((reinstall)); then
    bash utils/install_build.sh --reinstall
fi

exec docker compose exec jarvis_runtime bash -lc 'exec jarvis "$@"' jarvis "$@"
