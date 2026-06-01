#!/usr/bin/env bash
# Recreates ENV (Python 3.12), ENV_old (Python 3.12), and unsloth_env (Python 3.13).
# All environments are created with: virtualenv -p pythonX.Y <name>
# Run from any directory: bash /path/to/setup_envs.sh [--env DIR] [--env-old DIR] [--unsloth DIR] [--all]
# By default creates all three in the current directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_DIR="ENV"
ENV_OLD_DIR="ENV_old"
UNSLOTH_DIR="unsloth_env"

BUILD_ENV=0
BUILD_ENV_OLD=0
BUILD_UNSLOTH=0

usage() {
    echo "Usage: $0 [--env DIR] [--env-old DIR] [--unsloth DIR] [--all]"
    echo "  --env DIR        Create ENV at DIR        (default: ./ENV)"
    echo "  --env-old DIR    Create ENV_old at DIR    (default: ./ENV_old)"
    echo "  --unsloth DIR    Create unsloth_env at DIR (default: ./unsloth_env)"
    echo "  --all            Build all three (default when no flags given)"
    exit 1
}

if [[ $# -eq 0 ]]; then
    BUILD_ENV=1; BUILD_ENV_OLD=1; BUILD_UNSLOTH=1
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env)      ENV_DIR="$2";     BUILD_ENV=1;     shift 2 ;;
        --env-old)  ENV_OLD_DIR="$2"; BUILD_ENV_OLD=1; shift 2 ;;
        --unsloth)  UNSLOTH_DIR="$2"; BUILD_UNSLOTH=1; shift 2 ;;
        --all)      BUILD_ENV=1; BUILD_ENV_OLD=1; BUILD_UNSLOTH=1; shift ;;
        -h|--help)  usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# ── helpers ──────────────────────────────────────────────────────────────────

require_virtualenv() {
    if ! command -v virtualenv &>/dev/null; then
        echo "ERROR: 'virtualenv' not found. Install it first." >&2; exit 1
    fi
}

find_python() {
    local version="$1"   # e.g. "3.12" or "3.13"
    local major="${version%%.*}"
    local minor="${version##*.}"
    for py in "python${version}" "python${major}"; do
        if command -v "$py" &>/dev/null && "$py" --version 2>&1 | grep -q "${major}\.${minor}"; then
            echo "$py"; return
        fi
    done
    echo "ERROR: Python ${version} not found on PATH." >&2; exit 1
}

# ── ENV (Python 3.12, CUDA 12.8 torch) ───────────────────────────────────────

build_env() {
    local dir="$ENV_DIR"
    require_virtualenv
    local py; py="$(find_python 3.12)"
    echo "==> Creating ENV at '$dir' with $py"
    virtualenv -p "$py" "$dir"
    "$dir/bin/pip" install --upgrade pip
    "$dir/bin/pip" install \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        -r "$SCRIPT_DIR/requirements_ENV.txt"
    echo "==> ENV ready:  source $dir/bin/activate"
}

# ── ENV_old (Python 3.12, CUDA 12.1 torch) ───────────────────────────────────

build_env_old() {
    local dir="$ENV_OLD_DIR"
    require_virtualenv
    local py; py="$(find_python 3.12)"
    echo "==> Creating ENV_old at '$dir' with $py"
    virtualenv -p "$py" "$dir"
    "$dir/bin/pip" install --upgrade pip
    "$dir/bin/pip" install \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        -r "$SCRIPT_DIR/requirements_ENV_old.txt"
    echo "==> ENV_old ready:  source $dir/bin/activate"
}

# ── unsloth_env (Python 3.13, CUDA 12.9 torch) ───────────────────────────────

build_unsloth() {
    local dir="$UNSLOTH_DIR"
    require_virtualenv
    local py; py="$(find_python 3.13)"
    echo "==> Creating unsloth_env at '$dir' with $py"
    virtualenv -p "$py" "$dir"
    "$dir/bin/pip" install --upgrade pip
    "$dir/bin/pip" install \
        --extra-index-url https://download.pytorch.org/whl/cu129 \
        -r "$SCRIPT_DIR/requirements_unsloth_env.txt"
    echo "==> unsloth_env ready:  source $dir/bin/activate"
}

# ── main ──────────────────────────────────────────────────────────────────────

[[ $BUILD_ENV     -eq 1 ]] && build_env
[[ $BUILD_ENV_OLD -eq 1 ]] && build_env_old
[[ $BUILD_UNSLOTH -eq 1 ]] && build_unsloth

echo ""
echo "Done."
