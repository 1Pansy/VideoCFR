#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${VIRTUAL_ENV:-}" && -z "${CONDA_PREFIX:-}" ]]; then
  echo "Please activate a virtualenv or conda environment before running setup.sh." >&2
  exit 1
fi

python -m pip install --upgrade pip

# Local project packages.
python -m pip install -e "${ROOT_DIR}/CFR/qwen-vl-utils[decord]"
python -m pip install -e "${ROOT_DIR}/CFR/r1-v[dev]"

# Runtime modules used by the provided scripts.
python -m pip install \
  wandb==0.18.3 \
  tensorboardx \
  torchvision \
  vllm==0.7.2 \
  nltk \
  rouge_score \
  deepspeed

if [[ "${INSTALL_FLASH_ATTN:-1}" == "1" ]]; then
  python -m pip install flash-attn --no-build-isolation
else
  echo "Skipping flash-attn because INSTALL_FLASH_ATTN=0."
fi

# Some environments require a specific Transformers revision. Enable manually
# only when the installed version is incompatible with your checkpoint.
# python -m pip install "git+https://github.com/huggingface/transformers.git@336dc69d63d56f232a183a3e7f52790429b871ef"
