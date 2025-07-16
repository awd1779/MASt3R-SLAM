#!/bin/bash

# Run semantic SLAM on any Replica dataset with automatic vocabulary loading
# Usage: ./run_replica_semantic_auto.sh [room_0|room_1|apartment_0|etc]

# Get scene name from argument or default to room_0
SCENE="${1:-room_0}"
DATASET_PATH="/home/ubuntu/restart_from_scratch/datasets/${SCENE}"
CONFIG="config/replica_semantic_auto.yaml"
OUTPUT_NAME="replica_${SCENE}_semantic_auto"

# Critical environment fixes from working script
export LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

echo "========================================="
echo "Running Semantic SLAM on Replica ${SCENE}"
echo "Vocabulary will be loaded automatically from info_semantic.json"
echo "========================================="

# Check dataset exists
if [ ! -d "${DATASET_PATH}" ]; then
    echo "Error: Dataset not found at ${DATASET_PATH}"
    echo "Usage: $0 [scene_name]"
    echo "Example: $0 room_1"
    exit 1
fi

# Check for info_semantic.json
if [ -f "${DATASET_PATH}/info_semantic.json" ]; then
    echo "Found info_semantic.json - will load vocabulary automatically"
elif [ -f "${DATASET_PATH}/habitat/info_semantic.json" ]; then
    echo "Found habitat/info_semantic.json - will load vocabulary automatically"
else
    echo "Warning: info_semantic.json not found, will use default vocabulary"
fi

echo ""
echo "Dataset: ${DATASET_PATH}"
echo "Config: ${CONFIG}"
echo "Output: ${OUTPUT_NAME}"
echo ""

# Create log directory
mkdir -p logs

# Run semantic SLAM
cd /home/ubuntu/restart_from_scratch

echo "Starting semantic SLAM with automatic vocabulary..."
python main_semantic.py \
    --dataset ${DATASET_PATH} \
    --config ${CONFIG} \
    --save-as ${OUTPUT_NAME} \
    --verbose 2>&1 | tee logs/replica_${SCENE}_auto_$(date +%Y%m%d_%H%M%S).log

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "Semantic SLAM completed successfully!"
    echo "Results saved with prefix: ${OUTPUT_NAME}"
    echo "========================================="
    
    # List output files
    RESULTS_DIR=$(ls -td results/${OUTPUT_NAME}_* 2>/dev/null | head -1)
    if [ -n "$RESULTS_DIR" ]; then
        echo ""
        echo "Output files in ${RESULTS_DIR}:"
        ls -la ${RESULTS_DIR}/*.ply 2>/dev/null || echo "No PLY files generated yet"
        ls -la ${RESULTS_DIR}/*.txt 2>/dev/null || echo "No trajectory files generated yet"
        ls -la ${RESULTS_DIR}/*.json 2>/dev/null || echo "No JSON files generated yet"
    fi
else
    echo ""
    echo "========================================="
    echo "Error: Semantic SLAM failed with exit code $EXIT_CODE"
    echo "Check the log file in logs/ directory"
    echo "========================================="
fi