#!/bin/bash
# Usage: revision/run_demo_sets.sh <cuda> <set> [<set> ...]
# Example: revision/run_demo_sets.sh 0 A1 A2 A3
# Runs the reason task (5-shot, llm-jp-3.1-13b, generation seed 1) once per
# demonstration set in revision/demo_sets/. Outputs are written by run_llm.py to
# results/reason_llm-jp-31-13b-instruct4_5_seed1_demo<set>.jsonl.

MODEL="llm-jp-3.1-13b-instruct4"
SEED=1
CUDA=$1
shift

cd "$(dirname "$0")/.."
mkdir -p revision/logs
for name in "$@"; do
    echo "$(date '+%F %T') start $name cuda=$CUDA" >> revision/logs/status.txt
    CUDA_VISIBLE_DEVICES=$CUDA python run_llm.py reason \
        --model $MODEL \
        --shots 5 \
        --seed $SEED \
        --demo-set revision/demo_sets/reason_${name}.json \
        > revision/logs/reason_${name}.log 2>&1
    echo "$(date '+%F %T') end $name exit=$?" >> revision/logs/status.txt
done
