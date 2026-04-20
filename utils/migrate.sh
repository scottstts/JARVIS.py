#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: migrate.sh [--all]

Create a zip archive from the current directory.

Default targets:
  archive/ memory/ runtime_tools/ settings/

Options:
  --all   Archive every entry in the current directory except node_modules/.
  --help  Show this help.
EOF
}

archive_all=0
script_name="$(basename "${BASH_SOURCE[0]}")"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cwd_real="$(pwd -P)"

while (($# > 0)); do
    case "$1" in
        --all)
            archive_all=1
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

if ! command -v zip >/dev/null 2>&1; then
    echo "zip is required but not installed." >&2
    exit 1
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
archive_name="jarvis-workspace-${timestamp}.zip"

if [[ -e "${archive_name}" ]]; then
    counter=1
    while [[ -e "jarvis-workspace-${timestamp}-${counter}.zip" ]]; do
        ((counter += 1))
    done
    archive_name="jarvis-workspace-${timestamp}-${counter}.zip"
fi

declare -a targets=()

if ((archive_all)); then
    shopt -s dotglob nullglob
    for entry in ./*; do
        [[ -e "${entry}" ]] || continue
        entry_name="${entry#./}"
        if [[ -d "${entry}" && "${entry_name}" == .* ]]; then
            continue
        fi
        if [[ "${cwd_real}" == "${script_dir}" && "${entry_name}" == "${script_name}" ]]; then
            continue
        fi
        if [[ "${entry_name}" == "node_modules" ]]; then
            continue
        fi
        targets+=("${entry_name}")
    done
    shopt -u dotglob nullglob
else
    for entry in archive memory runtime_tools settings; do
        [[ -e "${entry}" ]] || continue
        targets+=("${entry}")
    done
fi

if ((${#targets[@]} == 0)); then
    if ((archive_all)); then
        echo "Nothing to archive in $(pwd)." >&2
    else
        echo "None of the default migration targets exist in $(pwd)." >&2
        echo "Expected one or more of: archive/ memory/ runtime_tools/ settings/" >&2
    fi
    exit 1
fi

declare -a zip_excludes=()
if ((archive_all)); then
    zip_excludes=(
        "node_modules/*"
        "*/node_modules/*"
    )
fi

if ((${#zip_excludes[@]} > 0)); then
    zip -rq "${archive_name}" "${targets[@]}" -x "${zip_excludes[@]}"
else
    zip -rq "${archive_name}" "${targets[@]}"
fi
echo "Created $(pwd)/${archive_name}"
