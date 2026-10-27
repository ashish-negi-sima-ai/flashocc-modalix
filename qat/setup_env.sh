#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${FLASHOCC_QAT_VENV:-${ROOT}/qat/.venv}"
QAT_SOURCE="${SIMA_QAT_SOURCE:-}"
PYTORCH_INDEX="${PYTORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/${USER}/uv-cache}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

command -v uv >/dev/null 2>&1 || { echo "uv is required" >&2; exit 1; }
[[ -f "${QAT_SOURCE}/pyproject.toml" ]] || { echo "Set SIMA_QAT_SOURCE to the SiMa QAT source directory" >&2; exit 1; }
mkdir -p "${UV_CACHE_DIR}"

if [[ ! -x "${VENV}/bin/python" ]]; then
    uv venv --python "${FLASHOCC_QAT_PYTHON:-python3}" "${VENV}"
fi
uv pip install --python "${VENV}/bin/python" --index "${PYTORCH_INDEX}" \
    torch==2.8.0 torchvision==0.23.0
uv pip install --python "${VENV}/bin/python" "${QAT_SOURCE}"
uv pip install --python "${VENV}/bin/python" -r "${ROOT}/qat/requirements.txt"

"${VENV}/bin/python" - <<'PY'
import sima_qat
import torch
import torchvision
print(f"torch={torch.__version__}")
print(f"torchvision={torchvision.__version__}")
print(f"sima_qat={sima_qat.__file__}")
PY

