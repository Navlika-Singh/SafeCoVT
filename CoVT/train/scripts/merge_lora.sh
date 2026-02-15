#!/bin/bash

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-7B-Instruct}"
MODEL_PATH="${MODEL_PATH:-/home/xxx/default_lora_ckpt}"
SAVE_MODEL_PATH="${SAVE_MODEL_PATH:-/home/xxx/default_merged_model}"
VISUAL_MODEL_ID="${VISUAL_MODEL_ID:-['sam', 'depth', 'dino']}"
VISIBLE_CUDA_DEVICES="${VISIBLE_CUDA_DEVICES:-0}"   

export PYTHONPATH=src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="$VISIBLE_CUDA_DEVICES"

python src/merge_lora_weights.py \
    --model-path "$MODEL_PATH" \
    --model-base "$MODEL_NAME"  \
    --save-model-path "$SAVE_MODEL_PATH" \
    --safe-serialization \
    --anchor-model-id "$VISUAL_MODEL_ID"
