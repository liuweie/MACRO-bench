#!/bin/bash

# Usage:
#   bash run_stability.sh <NUM_RUNS>
# Example:
#   bash run_stability.sh 10

NUM_RUNS=$1

if [ -z "$NUM_RUNS" ]; then
  echo "Usage: $0 <NUM_RUNS>"
  exit 1
fi

EXP_DIR=./trace_output/exp_repeat_${NUM_RUNS}
mkdir -p ${EXP_DIR}

for ((i=1; i<=NUM_RUNS; i++)); do
  echo "=== Running evaluation round ${i}/${NUM_RUNS} ==="

  python evaluators/evaluator.py \
    -t ./datasets/travel/tasks.yaml \
    -a ./config/travel_agents.json \
    -i ./trace_output/response_1767608368542.jsonl \
    -v

  mv ./trace_output/response_1767608368542_llm_enhanced_results_detailed.json \
     ${EXP_DIR}/response_1767608368542_run${i}_results_detailed.json

  mv ./trace_output/response_1767608368542_llm_enhanced_results_summary.json \
     ${EXP_DIR}/response_1767608368542_run${i}_results_summary.json
done
