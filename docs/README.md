# MASt3R-SLAM Documentation

## Overview
This documentation provides a comprehensive guide to understanding and working with MASt3R-SLAM, a real-time dense SLAM system that combines classical SLAM techniques with the MASt3R (Matching And Stereo 3D Reconstruction) deep learning model.

## Documentation Structure

### 1. [Architecture Overview](../ARCHITECTURE.md)
High-level system design and component interactions
- System overview and key features
- Process architecture and data flow
- Key algorithms and memory management
- Performance optimizations

### 2. Core Module Documentation

#### [main.py - Entry Point](MAIN_MODULE.md)
- Command-line interface and arguments
- Pipeline architecture and system modes
- Process orchestration
- Error handling and resource management

#### [frame.py - Frame Management](FRAME_MODULE.md)
- Frame and SharedKeyframes data structures
- Pointmap filtering strategies
- Thread-safe shared memory design
- Memory management techniques

#### [tracker.py - Tracking Algorithms](TRACKER_MODULE.md)
- Frame-to-frame tracking pipeline
- Pose optimization (calibrated/uncalibrated)
- Keyframe selection criteria
- Robust estimation techniques

#### [matching.py - Feature Matching](MATCHING_MODULE.md)
- Projection-based matching algorithms
- Iterative refinement with CUDA
- Multi-scale matching
- Cycle consistency checks

#### [global_opt.py - Global Optimization](GLOBAL_OPT_MODULE.md)
- Factor graph structure and operations
- Local and global bundle adjustment
- Loop closure integration
- CUDA-accelerated optimization

### 3. Supporting Modules

#### [dataloader.py - Dataset Handling](DATALOADER_MODULE.md)
- Unified dataset interface
- Support for TUM, 7Scenes, EuRoC, ETH3D
- Live camera integration
- Camera calibration handling

#### [visualization.py - 3D Visualization](VISUALIZATION_MODULE.md)
- Real-time rendering pipeline
- Interactive camera controls
- GUI components and statistics
- Recording and export features

#### [retrieval_database.py - Loop Closure](RETRIEVAL_DATABASE_MODULE.md)
- HOW descriptor system
- ASMK aggregation
- Efficient similarity search
- Geometric verification

### 4. [Configuration Guide](CONFIGURATION_GUIDE.md)
- YAML configuration system
- Parameter descriptions
- Tuning guidelines
- Custom configuration examples

## Quick Start Guide

### Basic Usage
```bash
# Run on TUM dataset
python main.py --scene freiburg1_desk --dataset tum --root ./datasets/TUM

# Run with custom configuration
python main.py --scene test --config config/eval_calib.yaml

# Live camera
python main.py --scene live --cam 0 --K 520 520 320 240
```

### Key Concepts

#### 1. System Modes
- **INIT**: Initial frame processing
- **TRACKING**: Normal operation
- **RELOC**: Relocalization after failure
- **TERMINATED**: Shutdown

#### 2. Optimization Modes
- **Uncalibrated**: Uses ray-to-ray distances (Sim3)
- **Calibrated**: Uses pixel reprojection (SE3)

#### 3. Pointmap Filtering
- **AVG_WEIGHTED**: Weighted average by confidence
- **BEST_SCORE**: Keep highest confidence
- **NEAREST/FARTHEST**: Distance-based selection

## System Requirements

### Hardware
- NVIDIA GPU with CUDA support
- 8GB+ GPU memory recommended
- 16GB+ system RAM

### Software
- Python 3.8+
- PyTorch with CUDA
- MASt3R model checkpoints
- See requirements.txt for full list

## Performance Tips

### For Speed
1. Reduce `target_img_size` (384 or 256)
2. Increase `img_stride` for frame skipping
3. Use fewer refinement iterations
4. Disable visualization

### For Quality
1. Use full resolution (512+)
2. Enable calibrated mode if possible
3. Increase confidence thresholds
4. Use BEST_SCORE filtering

### For Robustness
1. Use robust loss functions (Tukey)
2. Increase RANSAC iterations
3. Conservative loop closure thresholds
4. Larger optimization windows

## Common Issues and Solutions

### Tracking Failures
- Check MASt3R confidence threshold
- Verify camera calibration
- Ensure sufficient texture in scene
- Try different filtering modes

### Memory Issues
- Reduce max_n_keyframes
- Lower image resolution
- Disable feature refinement
- Use single-threaded mode

### Poor Reconstruction
- Increase confidence thresholds
- Use calibrated mode
- Check for motion blur
- Verify depth range settings

## Advanced Usage

### Custom Datasets
1. Inherit from `MonocularDataset`
2. Implement required methods
3. Add to dataset factory

### Custom Configurations
1. Create new YAML inheriting base
2. Override specific parameters
3. Test on small sequences

### Extending the System
- Add new filtering modes in Frame
- Implement custom loss functions
- Create new visualization modes
- Add dataset-specific optimizations

## Development Guidelines

### Code Organization
- Core SLAM logic in `mast3r_slam/`
- C++ backend in `backend/`
- Configuration in `config/`
- Scripts in `scripts/`

### Testing
```bash
# Run evaluation
bash scripts/eval_tum.sh

# Test specific dataset
python main.py --scene test --dataset 7scenes --processes 1
```

### Debugging
- Use `--verbose` flag
- Check visualization for issues
- Monitor confidence values
- Verify pose trajectories

## References

### Key Papers
- MASt3R: [arXiv:2406.09756]
- DROID-SLAM: Inspiration for backend
- HOW: Hierarchical descriptor system

### Related Projects
- dust3r: Predecessor to MASt3R
- COLMAP: Classical SfM reference
- ORB-SLAM: Sparse SLAM comparison

## Contributing

When contributing to the codebase:
1. Follow existing code style
2. Add appropriate documentation
3. Test on multiple datasets
4. Update relevant configs

## License
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)

---

For specific implementation details, refer to the individual module documentation linked above. For usage examples and tutorials, see the scripts directory and configuration files.