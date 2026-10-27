#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${FLASHOCC_QAT_VENV:-${ROOT}/qat/.venv}"
[[ -x "${VENV}/bin/python" ]] || { echo "Run qat/setup_env.sh first" >&2; exit 1; }

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    exec "${VENV}/bin/python" "${ROOT}/qat/train.py" "$@"
fi

if [[ -n "${SLURM_MODULE:-}" ]]; then module load "${SLURM_MODULE}"; fi
command -v srun >/dev/null 2>&1 || { echo "srun is required; alternatively run qat/train.py directly" >&2; exit 1; }
launch=(srun -c "${SLURM_CPUS:-8}" --mem="${SLURM_MEM:-32G}" --gres="gpu:${SLURM_GPUS:-1}")
if [[ -n "${SLURM_CONSTRAINT:-}" ]]; then launch+=(--constraint="${SLURM_CONSTRAINT}"); fi
if [[ -n "${SLURM_PARTITION:-}" ]]; then launch+=(-p "${SLURM_PARTITION}"); fi
exec "${launch[@]}" "${VENV}/bin/python" "${ROOT}/qat/train.py" "$@"

