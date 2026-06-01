#!/usr/bin/env bash
# Recreates ENV (Python 3.12), ENV_old (Python 3.12), and unsloth_env (Python 3.13 via uv).
# Run from any directory: bash /path/to/setup_envs.sh [--env ENV] [--env-old ENV_old] [--unsloth unsloth_env] [--all]
# By default creates all three environments in the current directory.

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
    echo "  --env DIR        Create ENV at DIR (default: ./ENV)"
    echo "  --env-old DIR    Create ENV_old at DIR (default: ./ENV_old)"
    echo "  --unsloth DIR    Create unsloth_env at DIR (default: ./unsloth_env)"
    echo "  --all            Build all three environments (default if no flags given)"
    exit 1
}

if [[ $# -eq 0 ]]; then
    BUILD_ENV=1; BUILD_ENV_OLD=1; BUILD_UNSLOTH=1
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env)      ENV_DIR="${2:-.ENV}"; BUILD_ENV=1; shift 2 ;;
        --env-old)  ENV_OLD_DIR="${2:-ENV_old}"; BUILD_ENV_OLD=1; shift 2 ;;
        --unsloth)  UNSLOTH_DIR="${2:-unsloth_env}"; BUILD_UNSLOTH=1; shift 2 ;;
        --all)      BUILD_ENV=1; BUILD_ENV_OLD=1; BUILD_UNSLOTH=1; shift ;;
        -h|--help)  usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# ── helpers ──────────────────────────────────────────────────────────────────

require_python312() {
    for py in python3.12 python3; do
        if "$py" --version 2>&1 | grep -q "3\.12"; then
            echo "$py"; return
        fi
    done
    echo "ERROR: Python 3.12 not found. Install it first." >&2; exit 1
}

require_uv() {
    if command -v uv &>/dev/null; then echo "uv"; return; fi
    # check common local install path
    if [[ -x "$HOME/.local/bin/uv" ]]; then echo "$HOME/.local/bin/uv"; return; fi
    echo "ERROR: 'uv' not found. Install from https://github.com/astral-sh/uv" >&2; exit 1
}

pip_install_extra_index() {
    # Some torch packages need the PyTorch index
    local pip="$1"; local req="$2"
    "$pip" install \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        -r "$req"
}

# ── ENV (Python 3.12, pip) ────────────────────────────────────────────────────

build_env() {
    local dir="$ENV_DIR"
    local py; py="$(require_python312)"
    echo "==> Creating ENV at $dir (Python 3.12)"
    "$py" -m venv "$dir"
    "$dir/bin/pip" install --upgrade pip
    "$dir/bin/pip" install \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        -r "$SCRIPT_DIR/requirements_ENV.txt"
    echo "==> ENV ready: source $dir/bin/activate"
}

# ── ENV_old (Python 3.12, pip, CUDA 12.1 torch) ──────────────────────────────

build_env_old() {
    local dir="$ENV_OLD_DIR"
    local py; py="$(require_python312)"
    echo "==> Creating ENV_old at $dir (Python 3.12)"
    "$py" -m venv "$dir"
    "$dir/bin/pip" install --upgrade pip
    "$dir/bin/pip" install \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        -r "$SCRIPT_DIR/requirements_ENV_old.txt"
    echo "==> ENV_old ready: source $dir/bin/activate"
}

# ── unsloth_env (Python 3.13, uv, CUDA 12.9 torch) ───────────────────────────

build_unsloth() {
    local dir="$UNSLOTH_DIR"
    local uv; uv="$(require_uv)"
    echo "==> Creating unsloth_env at $dir (Python 3.13, uv)"
    "$uv" venv --python 3.13 "$dir"
    "$uv" pip install \
        --python "$dir/bin/python" \
        --extra-index-url https://download.pytorch.org/whl/cu129 \
        -r "$SCRIPT_DIR/requirements_unsloth_env.txt"
    echo "==> unsloth_env ready: source $dir/bin/activate"
}

# ── main ──────────────────────────────────────────────────────────────────────

[[ $BUILD_ENV     -eq 1 ]] && build_env
[[ $BUILD_ENV_OLD -eq 1 ]] && build_env_old
[[ $BUILD_UNSLOTH -eq 1 ]] && build_unsloth

echo ""
echo "Done."
