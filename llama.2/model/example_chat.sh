#!/bin/bash

CHECKPOINT_DIR="/data/llm/llama/Llama-3.2-1B-Instruct"
LLAMA_ROOT_DIR="/data/students/gerry/repos/llama-models"
torchrun ${LLAMA_ROOT_DIR}/models/scripts/example_chat_completion.py $CHECKPOINT_DIR