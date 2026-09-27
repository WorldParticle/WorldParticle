#!/bin/bash

# Batch inference script for sample trajectories.

# Data path
DATA_DIR="/ssd/data/WorldParticle_Data/fancy_fluid2"

# Checkpoint output path
FOLD_DIR="/ssd/data/WorldParticle/experiments/fluid/fluid-submission/fluid-submission-2"

CONFIG_FILE="${FOLD_DIR}/config.yaml"
CHECKPOINT="${FOLD_DIR}/last.ckpt"
OUTPUT_DIR="${FOLD_DIR}"

TOTAL_STEPS=599
START=0
END=600
DT_LIST=(1)

samples=(
    "113"
)

mkdir -p "$OUTPUT_DIR"

for sample in "${samples[@]}"; do
    echo "=== Sample: $sample ==="

    for dt in "${DT_LIST[@]}"; do
        echo "🚀 Running inference for $sample with dt = $dt ..."

        CUDA_VISIBLE_DEVICES=0 python inference.py \
            --data_dir "$DATA_DIR" \
            --sample "$sample" \
            --config "$CONFIG_FILE" \
            --checkpoint "$CHECKPOINT" \
            --num_steps "$TOTAL_STEPS" \
            --dt "$dt" \
            --output_dir "$OUTPUT_DIR" \
            --start "$START" \
            --end "$END"
    done
done

echo "🎯 All samples finished. Results saved to: $OUTPUT_DIR"
