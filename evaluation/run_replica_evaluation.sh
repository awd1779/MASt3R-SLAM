#!/bin/bash
# Complete pipeline for evaluating semantic SLAM on Replica dataset

set -e  # Exit on error

# Configuration
REPLICA_SCENE="/path/to/replica_dataset/apartment_0"  # Update this path
OUTPUT_DIR="./replica_evaluation_output"
TRAJECTORY_DIR="$OUTPUT_DIR/trajectories"
SLAM_OUTPUT_DIR="$OUTPUT_DIR/slam_results"
EVAL_RESULTS_DIR="$OUTPUT_DIR/evaluation_results"

echo "=== Replica 3D Semantic Segmentation Evaluation Pipeline ==="
echo "Scene: $REPLICA_SCENE"
echo "Output directory: $OUTPUT_DIR"

# Step 1: Generate trajectories using Habitat-Sim
echo -e "\n[Step 1] Generating camera trajectories..."
python evaluation/generate_replica_trajectory.py \
    --scene_path "$REPLICA_SCENE" \
    --output_dir "$TRAJECTORY_DIR" \
    --num_frames 100 \
    --trajectory_type both

# Step 2: Update semantic SLAM config with Replica vocabulary
echo -e "\n[Step 2] Updating semantic SLAM configuration..."

# First, extract Replica vocabulary
echo "Extracting Replica vocabulary..."
python -c "
import sys
sys.path.append('.')
from evaluation.replica_loader import ReplicaSemanticLoader

loader = ReplicaSemanticLoader('$REPLICA_SCENE')
vocab = loader.get_vocabulary_for_zero_shot()

print('\\nReplica vocabulary for zero-shot evaluation:')
print('semantic_classes:')
for name in vocab:
    print(f'  - \"{name}\"')

# Create a new config file
import yaml
import os

# Load existing config
with open('config/semantic_slam.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Update semantic classes
config['semantic_slam']['semantic_classes'] = vocab

# Save to new config
os.makedirs('$OUTPUT_DIR', exist_ok=True)
with open('$OUTPUT_DIR/semantic_slam_replica.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print(f'\\nUpdated config saved to: $OUTPUT_DIR/semantic_slam_replica.yaml')
"

# Step 3: Run semantic SLAM on generated trajectories
echo -e "\n[Step 3] Running semantic SLAM..."
echo "Note: This step requires the semantic SLAM system to be configured to read Habitat-Sim output format"

# For each trajectory type
for TRAJ_TYPE in circular exploration; do
    echo -e "\nProcessing $TRAJ_TYPE trajectory..."
    
    # Create dataset config for this trajectory
    python -c "
import json
import os

# Create a dataset loader config for Habitat-Sim output
dataset_config = {
    'dataset_type': 'replica',
    'base_path': '$TRAJECTORY_DIR/$TRAJ_TYPE',
    'rgb_path': 'rgb',
    'depth_path': 'depth',
    'intrinsics_file': 'intrinsics.json',
    'poses_file': 'poses.txt'
}

os.makedirs('$OUTPUT_DIR', exist_ok=True)
with open('$OUTPUT_DIR/replica_dataset_config_${TRAJ_TYPE}.json', 'w') as f:
    json.dump(dataset_config, f, indent=2)

print(f'Dataset config saved for {TRAJ_TYPE} trajectory')
"
    
    # Run semantic SLAM (this is a placeholder - adjust based on your system)
    echo "Running semantic SLAM on $TRAJ_TYPE trajectory..."
    # python main_semantic.py \
    #     --config "$OUTPUT_DIR/semantic_slam_replica.yaml" \
    #     --dataset_config "$OUTPUT_DIR/replica_dataset_config_${TRAJ_TYPE}.json" \
    #     --output_dir "$SLAM_OUTPUT_DIR/$TRAJ_TYPE"
done

# Step 4: Evaluate results
echo -e "\n[Step 4] Evaluating semantic segmentation results..."

# Assuming the semantic SLAM outputs are in the format: scene_name_semantic_dense.ply
for TRAJ_TYPE in circular exploration; do
    PRED_PLY="$SLAM_OUTPUT_DIR/$TRAJ_TYPE/replica_semantic_dense.ply"
    
    if [ -f "$PRED_PLY" ]; then
        echo -e "\nEvaluating $TRAJ_TYPE trajectory results..."
        python evaluation/evaluate_semantic_3d.py \
            --pred_ply "$PRED_PLY" \
            --replica_scene "$REPLICA_SCENE" \
            --output_dir "$EVAL_RESULTS_DIR/$TRAJ_TYPE" \
            --distance_threshold 0.05
    else
        echo "Warning: Predicted PLY not found at $PRED_PLY"
        echo "Please run semantic SLAM first to generate predictions"
    fi
done

echo -e "\n=== Evaluation Pipeline Complete ==="
echo "Results saved to: $EVAL_RESULTS_DIR"

# Generate summary report
echo -e "\n[Step 5] Generating summary report..."
python -c "
import json
import os
from pathlib import Path

eval_dir = Path('$EVAL_RESULTS_DIR')
if eval_dir.exists():
    print('\\n=== Evaluation Summary ===')
    for traj_dir in eval_dir.iterdir():
        if traj_dir.is_dir():
            metrics_file = traj_dir / 'metrics.json'
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                print(f'\\n{traj_dir.name} trajectory:')
                print(f'  Mean IoU: {metrics.get(\"mean_iou\", 0):.3f}')
                print(f'  Overall Accuracy: {metrics.get(\"overall_accuracy\", 0):.3f}')
                print(f'  Coverage: {metrics.get(\"coverage\", 0):.3f}')
"