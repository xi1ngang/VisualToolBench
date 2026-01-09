#!/bin/bash

# Simple script to run model response generation for multi-turn
# Usage: ./run_model_response_multi_turn.sh [MODEL_NAME]

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
MODEL=${1:-"gpt-4o"}
DATASET="$PROJECT_ROOT/data/multi_turn/dataset.json"
OUTPUT_DIR="$PROJECT_ROOT/eval_results_multi_turn"
NUM_WORKERS=4

echo "Running multi-turn inference with model: $MODEL"

python "$SCRIPT_DIR/run_model_response_multi_turn.py" \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --tool_use \
    --system_prompt_level high \
    --max_tool_calls 20 \
    --num_workers $NUM_WORKERS \
    --use_GT_answer 1
