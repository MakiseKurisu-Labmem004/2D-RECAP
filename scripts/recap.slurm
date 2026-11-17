#!/usr/bin/env bash
#SBATCH --job-name=minimal-recap
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=recap-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# No OpenGL required unless exporting replay images. On EGL-capable hosts:
# export MUJOCO_GL=egl
# Prepare the uv environment on a login/setup node before submitting.
config="${1:-configs/experiments/arm3r.yaml}"
if (( $# > 0 )); then
    shift
fi
.venv/bin/python -m minimal_recap.run --config "$config" "$@"
