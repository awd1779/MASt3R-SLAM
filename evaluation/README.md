# 3D Semantic Segmentation Evaluation for Replica Dataset

This module provides tools to evaluate open-vocabulary 3D semantic segmentation on the Replica dataset.

## Overview

The evaluation pipeline consists of:

1. **Trajectory Generation**: Use Habitat-Sim to generate camera trajectories through Replica scenes
2. **Semantic SLAM Execution**: Run your semantic SLAM system on the generated trajectories
3. **3D Evaluation**: Compare the reconstructed 3D semantic point clouds against Replica ground truth

## Installation

### Prerequisites

```bash
# Install Habitat-Sim (for trajectory generation)
conda install habitat-sim -c conda-forge -c aihabitat

# Install other dependencies
pip install numpy scipy scikit-learn open3d plyfile quaternion
```

## Usage

### 1. Generate Camera Trajectories

```bash
python evaluation/generate_replica_trajectory.py \
    --scene_path /path/to/replica_dataset/apartment_0 \
    --output_dir ./replica_trajectories \
    --num_frames 100 \
    --trajectory_type both
```

This generates:
- RGB images (`rgb/000000.png`, etc.)
- Depth maps (`depth/000000.png`, etc.)
- Semantic ground truth (`semantic/000000.png`, etc.)
- Camera poses (`poses.txt` in TUM format)
- Camera intrinsics (`intrinsics.json`)

### 2. Configure Semantic SLAM for Zero-Shot Evaluation

Extract Replica's vocabulary and update your semantic SLAM config:

```python
from evaluation.replica_loader import ReplicaSemanticLoader

loader = ReplicaSemanticLoader('/path/to/replica_dataset/apartment_0')
vocabulary = loader.get_vocabulary_for_zero_shot()

# Update your config/semantic_slam.yaml with this vocabulary
```

### 3. Run Semantic SLAM

Run your semantic SLAM system on the generated trajectories. The system should output:
- 3D semantic point cloud in PLY format with `label` property
- Label mapping file (`.txt`) with class names

### 4. Evaluate 3D Semantic Segmentation

```bash
python evaluation/evaluate_semantic_3d.py \
    --pred_ply /path/to/your/semantic_dense.ply \
    --replica_scene /path/to/replica_dataset/apartment_0 \
    --output_dir ./evaluation_results \
    --distance_threshold 0.05
```

## Complete Pipeline

Run the entire evaluation pipeline:

```bash
# Edit the script to set your Replica dataset path
vim evaluation/run_replica_evaluation.sh

# Run the pipeline
./evaluation/run_replica_evaluation.sh
```

## Evaluation Metrics

The evaluation computes:

- **Overall Accuracy**: Percentage of correctly labeled 3D points
- **Mean IoU**: Average Intersection over Union across all classes
- **Per-class IoU**: IoU for each semantic class
- **Coverage**: Percentage of predicted points that match ground truth
- **Confusion Matrix**: Detailed class-wise predictions vs ground truth

## Output Format

Results are saved to the output directory:
- `metrics.json`: All evaluation metrics
- `confusion_matrix.npy`: Numpy array of confusion matrix
- `vocabulary_mapping.json`: Mapping between predicted and GT vocabularies

## Notes

- The evaluation uses nearest-neighbor matching between predicted and ground truth points
- Default distance threshold is 0.05m (5cm) for point matching
- For true zero-shot evaluation, ensure your system uses Replica's vocabulary
- The system handles instance segmentation by removing instance suffixes (e.g., "chair_1" → "chair")

## Troubleshooting

1. **No semantic labels in PLY**: Ensure your PLY file has a 'label' property for each vertex
2. **Vocabulary mismatch**: For zero-shot evaluation, use Replica's exact class names
3. **Low coverage**: Check if coordinate systems match between prediction and ground truth
4. **Memory issues**: Reduce number of sampled points or process in chunks