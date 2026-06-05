#!/usr/bin/env bash
set -euo pipefail

CFR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${1:-${MODEL_PATH:-}}"
RUN_NAME="${2:-${RUN_NAME:-VideoCFR}}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "Usage: bash CFR/eval_bench.sh /path/to/model [run_name]" >&2
  exit 1
fi

export DECORD_EOF_RETRY_MAX="${DECORD_EOF_RETRY_MAX:-40960}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python "${CFR_DIR}/eval_bench.py" \
  --model_path "${MODEL_PATH}" \
  --file_name "${RUN_NAME}"
