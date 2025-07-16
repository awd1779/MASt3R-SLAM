# Enhanced Semantic SLAM Evaluation Framework

## Overview

This enhanced evaluation framework provides comprehensive metrics and tools for evaluating semantic SLAM systems. It extends the basic evaluation with advanced metrics, multi-dataset support, and real-time evaluation capabilities.

## Key Features

### 1. **Enhanced Metrics Modules** (`evaluation/metrics/`)

#### Instance Segmentation Metrics (`instance_metrics.py`)
- **Average Precision (AP)** at multiple IoU thresholds (0.5-0.95)
- **Average Recall (AR)** for different object sizes
- **Per-class AP/AR** breakdown
- **3D IoU computation** for point clouds
- Hungarian matching for instance association

#### Temporal Consistency Metrics (`temporal_metrics.py`)
- **MOTA/MOTP** (Multi-Object Tracking Accuracy/Precision)
- **Track fragmentation** analysis
- **ID switch detection** across frames
- **Label consistency** over time
- **Track purity** metrics

#### Boundary Accuracy Metrics (`boundary_metrics.py`)
- **Boundary F1-score** at multiple distance thresholds
- **Trimap accuracy** for 2D evaluation
- **3D boundary extraction** using voxelization
- **Per-class boundary IoU**
- **Mean/median boundary distances**

#### Efficiency Metrics (`efficiency_metrics.py`)
- **Real-time FPS** tracking with moving averages
- **Component-wise latency** breakdown
- **Memory usage** (CPU and GPU)
- **Throughput metrics** (points/objects per second)
- **Scalability analysis** with scene complexity

### 2. **Multi-Dataset Support** (`evaluation/datasets/`)

#### Unified Dataset Interface (`base_dataset.py`)
- Abstract base class for all datasets
- Standardized frame data structure
- Common evaluation interface
- Automatic validation and metadata export

#### Dataset Implementations
- **Replica Dataset** (`replica_dataset.py`)
  - Habitat-Sim trajectory support
  - Semantic mesh loading
  - Zero-shot vocabulary extraction
  
- **ScanNet Dataset** (`scannet_dataset.py`)
  - NYU40 semantic classes
  - Instance segmentation support
  - Official train/val/test splits
  - Raw and processed depth options

#### Dataset Factory (`dataset_factory.py`)
- Automatic dataset type detection
- Configuration-based loading
- Easy extension for new datasets

### 3. **Real-time Evaluation** (`realtime_evaluator.py`)

- **Online metric computation** during SLAM execution
- **Live plotting** of key metrics
- **Checkpoint saving** for long sequences
- **Multi-threaded evaluation** to avoid blocking
- **Comprehensive logging** and reporting

### 4. **Comprehensive Evaluation Script** (`run_comprehensive_evaluation.py`)

Complete evaluation pipeline that:
- Loads any supported dataset
- Runs all metric modules
- Generates detailed reports
- Supports batch and real-time modes
- Creates visualization outputs

## Usage Examples

### 1. Basic 3D Semantic Evaluation

```bash
python evaluation/evaluate_semantic_3d.py \
    --pred_ply output/semantic_dense.ply \
    --replica_scene /path/to/replica/apartment_0 \
    --output_dir ./eval_results
```

### 2. Comprehensive Evaluation with All Metrics

```bash
python evaluation/run_comprehensive_evaluation.py \
    --dataset_type replica \
    --dataset_path /path/to/replica \
    --scene_name apartment_0 \
    --pred_ply output/semantic_dense.ply \
    --predictions_dir output/predictions \
    --output_dir ./comprehensive_results
```

### 3. Real-time Evaluation During SLAM

```python
from evaluation.realtime_evaluator import RealtimeEvaluator

# Initialize evaluator
evaluator = RealtimeEvaluator(
    output_dir="./realtime_eval",
    semantic_classes=["wall", "floor", "chair", "table"],
    eval_interval=30,
    enable_plotting=True
)

# Start evaluation
evaluator.start()

# During SLAM main loop
for frame in slam_system:
    # Process frame
    results = slam_system.process(frame)
    
    # Add to evaluator
    evaluator.add_frame(
        frame_id=frame.id,
        rgb=frame.rgb,
        depth=frame.depth,
        pred_semantic=results.semantic,
        pred_instance=results.instance,
        gt_semantic=frame.gt_semantic,
        gt_instance=frame.gt_instance,
        pose=results.pose,
        processing_time=results.time
    )

# Stop and save results
evaluator.stop()
```

### 4. Multi-Dataset Evaluation

```python
from evaluation.datasets import DatasetFactory

# Automatically detect dataset type
dataset = DatasetFactory.create_dataset(
    "scannet",  # or "replica", "kitti360"
    dataset_path="/path/to/scannet",
    scene_name="scene0000_00",
    load_semantics=True,
    load_instances=True
)

# Use unified interface
for i in range(len(dataset)):
    frame_data = dataset[i]
    # Process frame...
```

## Metrics Explained

### Instance Segmentation Metrics
- **mAP**: Mean Average Precision across IoU thresholds 0.5-0.95
- **mAP@50**: Mean AP at IoU threshold 0.5
- **mAR**: Mean Average Recall
- **Per-class AP**: Performance breakdown by object category

### Temporal Consistency Metrics
- **MOTA**: Multi-Object Tracking Accuracy (1 - (FP + FN + ID_switches) / GT)
- **MOTP**: Multi-Object Tracking Precision (average IoU of matches)
- **Track Purity**: Percentage of tracks with consistent labels
- **ID Switches**: Number of identity changes in tracking

### Boundary Metrics
- **Boundary F1**: Harmonic mean of boundary precision and recall
- **Trimap IoU**: Accuracy within dilated boundary regions
- **Mean Boundary Distance**: Average distance to nearest GT boundary

### Efficiency Metrics
- **FPS**: Frames processed per second
- **Latency Percentiles**: p50, p95, p99 processing times
- **Memory Usage**: Peak and average RAM/VRAM consumption
- **Scalability**: Performance vs scene complexity correlation

## Output Structure

```
evaluation_results/
├── dataset_metadata.json          # Dataset information
├── semantic_3d/
│   ├── metrics.json              # 3D semantic metrics
│   ├── confusion_matrix.npy      # Class confusion matrix
│   └── vocabulary_mapping.json   # Predicted to GT mapping
├── instance_segmentation/
│   ├── instance_metrics.json     # AP/AR metrics
│   └── per_class_results.json    # Per-category breakdown
├── temporal_consistency/
│   ├── temporal_metrics.json     # MOTA/MOTP results
│   └── tracks_visualization.png  # Track timeline plot
├── efficiency/
│   ├── efficiency_metrics.json   # Performance metrics
│   └── latency_breakdown.json    # Component timing
├── realtime/
│   ├── metric_history.png        # Metrics over time
│   ├── checkpoints/              # Periodic snapshots
│   └── final_results.json        # Complete results
└── evaluation_summary.txt        # Human-readable summary
```

## Extending the Framework

### Adding New Metrics

1. Create new metric class inheriting from `BaseMetrics`
2. Implement required methods: `compute()`, `reset()`
3. Add to metric imports in `__init__.py`

### Adding New Datasets

1. Create dataset class inheriting from `BaseDataset`
2. Implement data loading methods
3. Register with `DatasetFactory`

Example:
```python
from evaluation.datasets import BaseDataset, DatasetType

class MyDataset(BaseDataset):
    def _load_dataset(self):
        # Load your dataset structure
        pass
    
    def load_frame(self, idx):
        # Return frame data
        pass
    
    def load_ground_truth_3d(self):
        # Return 3D GT data
        pass

# Register dataset
DatasetFactory.register_dataset(DatasetType.CUSTOM, MyDataset)
```

## Performance Considerations

- **Batch Processing**: Process multiple frames together for efficiency
- **Parallel Metrics**: Metrics are computed independently and can be parallelized
- **Memory Management**: Large point clouds are processed in chunks
- **Caching**: Results are cached to avoid recomputation

## Dependencies

```bash
# Core dependencies
pip install numpy scipy scikit-learn opencv-python

# Additional dependencies
pip install matplotlib  # For plotting
pip install trimesh    # For mesh loading
pip install psutil     # For system monitoring
pip install GPUtil     # For GPU monitoring (optional)
```

## Future Enhancements

The framework is designed to be extensible. Planned future additions include:
- Uncertainty quantification metrics
- Cross-dataset generalization evaluation
- Robustness testing (noise, occlusions)
- Semantic SLAM-specific metrics (loop closure quality)
- Integration with standard benchmarks (ScanNet, SemanticKITTI)

## Citation

If you use this evaluation framework in your research, please cite:
```
@software{semantic_slam_eval,
  title = {Enhanced Semantic SLAM Evaluation Framework},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/semantic-slam-eval}
}
```