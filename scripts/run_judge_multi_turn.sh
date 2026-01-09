#!/bin/bash

# Simple script to run judge evaluation for multi-turn
# Usage: ./run_judge_multi_turn.sh [MODEL_NAME]

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
MODEL=${1:-"gpt-4o"}
JUDGE_MODEL="gpt-4o"
INPUT_DIR="$PROJECT_ROOT/eval_results_multi_turn"
OUTPUT_DIR="$PROJECT_ROOT/eval_results/multi_turn"
NUM_WORKERS=4

# Input/output files
INPUT_FILE="$INPUT_DIR/${MODEL}_w_tool_system_high_max_tool_calls_20_trial_1.json"
RESULTS_FILE="$OUTPUT_DIR/${MODEL}_eval_results.json"
SUMMARY_FILE="$OUTPUT_DIR/${MODEL}_eval_summary.json"

mkdir -p "$OUTPUT_DIR"

echo "Running judge evaluation for model: $MODEL"
echo "Input: $INPUT_FILE"

python "$SCRIPT_DIR/run_judge_multi_turn.py" \
    --model_response "$INPUT_FILE" \
    --eval_results "$RESULTS_FILE" \
    --eval_summary "$SUMMARY_FILE" \
    --judge_model "$JUDGE_MODEL" \
    --num_workers $NUM_WORKERS
