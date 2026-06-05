#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}/CFR/r1-v"

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
SFT_DATASET="${SFT_DATASET:-./Video-R1-data/Video-R1-COT-165k.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./log/Qwen2.5-VL-7B-Video-7B-cot-sft}"
RUN_NAME="${RUN_NAME:-Qwen2.5-VL-7B-Video-cot-sft}"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12349" \
    src/open_r1/sft_video.py \
    --output_dir "${OUTPUT_DIR}" \
    --model_name_or_path "${BASE_MODEL}" \
    --dataset_name "${SFT_DATASET}" \
    --deepspeed local_scripts/zero2.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-6 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name "${RUN_NAME}" \
    --save_steps 1000 \
    --max_grad_norm 5 \
    --save_only_model true
