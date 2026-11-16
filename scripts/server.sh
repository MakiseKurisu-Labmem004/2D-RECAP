#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if ! command -v uv >/dev/null 2>&1; then
    echo 'uv is required; install uv and add it to PATH first.' >&2
    exit 1
fi
if [[ ! -x .venv/bin/python ]]; then
    uv venv --python 3.12 .venv
fi
uv pip install --python .venv/bin/python --torch-backend cpu -r requirements.txt
uv pip install --python .venv/bin/python --no-deps -e .
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
.venv/bin/python -m pytest -q
config="${1:-configs/experiments/server_cpu.yaml}"
if (( $# > 0 )); then
    shift
fi
.venv/bin/python -m minimal_recap.run --config "$config" "$@"
