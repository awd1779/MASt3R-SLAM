#!/bin/bash
# Run complete AAAI evaluation pipeline

# Configuration
DATASET_TYPE="replica"
DATASET_PATH="/path/to/replica_dataset"
SCENE_NAME="apartment_0"
OUTPUT_DIR="./aaai_evaluation_results"

# Input paths (from your SLAM output)
SLAM_OUTPUT_DIR="./results/${SCENE_NAME}"
PREDICTIONS_DIR="${SLAM_OUTPUT_DIR}/predictions"
SEMANTIC_PLY="${SLAM_OUTPUT_DIR}/${SCENE_NAME}_semantic_dense.ply"

# Step 1: Generate test queries
echo "Generating test queries..."
python evaluation/generate_test_queries.py \
    --dataset ${DATASET_TYPE} \
    --output ${OUTPUT_DIR}/test_queries.json \
    --num_queries 100

# Step 2: Prepare predictions (if needed)
# This would extract per-frame predictions from your SLAM output
# python evaluation/prepare_predictions.py ...

# Step 3: Run AAAI evaluation
echo "Running AAAI evaluation..."
python evaluation/aaai_evaluation_runner.py \
    --dataset_type ${DATASET_TYPE} \
    --dataset_path ${DATASET_PATH} \
    --scene_name ${SCENE_NAME} \
    --predictions_dir ${PREDICTIONS_DIR} \
    --semantic_ply ${SEMANTIC_PLY} \
    --test_queries ${OUTPUT_DIR}/test_queries.json \
    --output_dir ${OUTPUT_DIR} \
    --experiment_name "aaai_2025"

# Step 4: Generate additional visualizations (optional)
# python evaluation/visualize_results.py ...

echo "Evaluation complete! Results in ${OUTPUT_DIR}"

# The output will include:
# - LaTeX tables ready for your paper
# - Comparison with baselines
# - Ablation study results
# - Human-readable summary
# - All metrics in JSON format