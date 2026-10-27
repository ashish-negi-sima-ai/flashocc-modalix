#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${FLASHOCC_QAT_VENV:-${ROOT}/qat/.venv}"
[[ -x "${VENV}/bin/python" ]] || { echo "Run qat/setup_env.sh first" >&2; exit 1; }

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    exec "${VENV}/bin/python" "${ROOT}/qat/train.py" "$@"
fi

module load opensource/slurm/24.11.3g
exec srun \
    -c "${SLURM_CPUS:-8}" \
    --mem="${SLURM_MEM:-32G}" \
    --gres="gpu:${SLURM_GPUS:-1}" \
    --constraint="${SLURM_CONSTRAINT:-rtx4000}" \
    -p "${SLURM_PARTITION:-light}" \
    "${VENV}/bin/python" "${ROOT}/qat/train.py" "$@"

