# 3D Semantic Segmentation Evaluation Progress Summary

## Overview
This document summarizes the work done to create an evaluation framework for open-vocabulary 3D semantic segmentation using the Replica dataset.

## Completed Components

### 1. Evaluation Framework Structure
Created a complete evaluation module in `/evaluation/` directory:
```
evaluation/
├── __init__.py
├── generate_replica_trajectory.py    # Habitat-Sim trajectory generator
├── replica_loader.py                 # Replica ground truth loader  
├── semantic_3d_metrics.py            # 3D evaluation metrics
├── evaluate_semantic_3d.py           # Main evaluation script
├── run_replica_evaluation.sh         # Full pipeline script
├── generate_natural_trajectories.sh  # Natural trajectory generation
└── README.md                         # Documentation
```

### 2. Trajectory Generation (`generate_replica_trajectory.py`)
- **Purpose**: Generate realistic camera trajectories through Replica scenes using Habitat-Sim
- **Features**:
  - Navigation mesh-based path planning (collision-free)
  - Two trajectory types:
    - **Room Tour**: Visits different areas, pauses to look around
    - **Smooth Exploration**: Natural movement between waypoints
  - Realistic camera effects:
    - Human eye-level height (1.6m)
    - Walking bobbing motion
    - Natural head sway
    - Smooth rotation transitions
  - Outputs:
    - RGB images (640x480)
    - Depth maps (16-bit PNG)
    - Semantic ground truth
    - Camera poses (TUM format)
    - Camera intrinsics

### 3. Ground Truth Loader (`replica_loader.py`)
- **Purpose**: Load Replica's semantic ground truth data
- **Features**:
  - Loads semantic mesh (`mesh_semantic.ply`)
  - Parses class definitions (`info_semantic.json`)
  - Extracts vocabulary for zero-shot evaluation
  - Maps instance IDs to class names
  - Samples points from mesh with labels

### 4. 3D Evaluation Metrics (`semantic_3d_metrics.py`)
- **Purpose**: Compute 3D semantic segmentation metrics
- **Metrics Implemented**:
  - 3D IoU (Intersection over Union) per class
  - Mean IoU across all classes
  - Point-wise accuracy
  - Class-wise accuracy
  - Coverage (percentage of matched points)
  - Confusion matrix
- **Method**: Nearest-neighbor matching between predicted and GT points

### 5. Main Evaluation Script (`evaluate_semantic_3d.py`)
- **Purpose**: Evaluate predicted 3D semantic point clouds
- **Process**:
  1. Load predicted semantic PLY file
  2. Load Replica ground truth
  3. Create vocabulary mapping
  4. Match points and compute metrics
  5. Save results (JSON, confusion matrix, etc.)

## Zero-Shot Evaluation Approach

We chose the **zero-shot evaluation** approach:
1. Extract Replica's vocabulary (49 classes like "chair", "table", "wall")
2. Configure semantic SLAM to use this exact vocabulary
3. Run SLAM and compare outputs directly
4. No manual mapping needed between vocabularies

## Replica Vocabulary (for apartment_0)
```
base-cabinet, basket, bathtub, bed, blinds, book, bottle, cabinet, ceiling, 
chair, comforter, cooktop, countertop, cup, curtain, cushion, desk, door, 
faucet, floor, handrail, indoor-plant, lamp, major-appliance, mat, nightstand, 
panel, picture, pillow, plant-stand, plate, pot, rack, refrigerator, rug, 
shower-stall, sink, sofa, stair, stool, switch, table, toilet, vase, vent, 
wall, wall-cabinet, wall-plug, window
```

## Current Status

### Completed ✓
1. Created complete evaluation framework
2. Implemented trajectory generation with natural motion
3. Created ground truth loading from Replica
4. Implemented comprehensive 3D metrics
5. Built evaluation pipeline scripts
6. Fixed Habitat-Sim API compatibility issues

### Next Steps
1. **Update semantic SLAM config** with Replica vocabulary
2. **Run semantic SLAM** on generated trajectories
3. **Evaluate results** using the evaluation script
4. **Analyze performance** across different object categories

## Usage Workflow

1. **Convert semantic descriptors** (required for Habitat-Sim):
   ```bash
   python evaluation/convert_semantic_info.py --scene apartment_0
   ```

2. **Generate trajectories** (in Habitat environment):
   ```bash
   python evaluation/generate_replica_trajectory_fixed_final.py \
       --scene apartment_0 \
       --output_dir ./trajectories_final \
       --num_frames 300
   ```

3. **Update SLAM config** with Replica vocabulary

4. **Run semantic SLAM** on trajectories

5. **Evaluate results**:
   ```bash
   python evaluation/evaluate_semantic_3d.py \
       --pred_ply output/semantic_dense.ply \
       --replica_scene datasets/replica_dataset/apartment_0 \
       --output_dir ./evaluation_results
   ```

## Technical Notes

- Habitat-Sim should be run in a separate conda environment due to dependencies
- The trajectory generator now uses proper Habitat-Sim API (ShortestPath object)
- Evaluation uses nearest-neighbor matching with 5cm distance threshold
- The system handles instance segmentation (removes "_1", "_2" suffixes)

## Fixed Issues

1. **Semantic Rendering**: Converted `info_semantic.json` to `info_semantic.txt` format that Habitat-Sim expects
2. **Navigation**: Ensured paths are computed using navigation mesh, preventing wall traversal
3. **Camera Positioning**: Fixed sensor positioning to work with Replica's coordinate system
4. **Validation**: Added checks for semantic rendering and trajectory validity

## Usage Notes

- Habitat-Sim requires specific GPU drivers and may have compatibility issues
- Large scenes may require memory management for evaluation
- The semantic descriptor must be in text format for proper loading