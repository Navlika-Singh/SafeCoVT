#!/bin/bash

python3 -m vllm.entrypoints.openai.api_server \
--model Qwen/Qwen2.5-VL-3B-Instruct \
--host 0.0.0.0 \
--port 8000 \
--tensor-parallel-size 1 \
--trust-remote-code \
--enable-multimodal \
--max-model-len 8192