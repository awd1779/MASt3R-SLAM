# Configuration Guide - YAML Files

## Overview
MASt3R-SLAM uses YAML configuration files to control various aspects of the system. The configuration system supports inheritance, allowing specialized configurations to override base settings.

## Configuration Hierarchy

```
config/
├── base.yaml              # Default configuration
├── calib.yaml            # Calibrated mode (inherits base)
├── eval_calib.yaml       # Evaluation with calibration
├── eval_no_calib.yaml    # Evaluation without calibration
├── eth3d.yaml            # ETH3D specific settings
└── intrinsics.yaml       # Camera intrinsics template
```

## Base Configuration (base.yaml)

### Overall Settings
```yaml
n_frames: -1              # Number of frames to process (-1 for all)
max_n_keyframes: 5000     # Maximum keyframes in buffer
target_img_size: 512      # Image size for MASt3R processing
```

### MASt3R Model Settings
```yaml
use_mast3r_server: false  # Use external MASt3R server
mast3r_server_url: "http://localhost:8080"  # Server URL if used
```

### Matching Parameters
```yaml
matching:
  match_kp_use_img_feat_dist_thresh: 0.35  # Initial matching threshold
  match_conf_thresh: 1.5                   # MASt3R confidence threshold
  max_pts: 20000                          # Maximum points to match
  feature_extract_max_pts: 40000          # Maximum features to extract
  iterative_proj_n_iters: 10              # Refinement iterations
  iterative_proj_radius: 32               # Search radius (pixels)
  iterative_proj_radius_pow_decay: 0.8    # Radius decay per iteration
  iterative_proj_nms_radius: 1            # Non-max suppression radius
  iterative_proj_lr: 0.02                 # Learning rate for refinement
  iterative_proj_flow_loss: true          # Use optical flow loss
  iterative_proj_warp_img: true           # Warp images for matching
  iterative_proj_cycle_thresh: 2          # Cycle consistency threshold
  iterative_proj_d_max: 10000             # Maximum depth
  iterative_proj_d_min: 0.1               # Minimum depth
  iterative_proj_skip_frame: -1           # Skip every N frames
  iterative_proj_ransac_iters: 10         # RANSAC iterations
  iterative_proj_verbose: false           # Verbose output
  feature_refinement_radius: -1           # Descriptor refinement (-1 disabled)
  descriptor_refinement_radius: -1        # Additional refinement
```

### Tracking Parameters
```yaml
tracking:
  mast3r_conf_thresh: 1.5               # MASt3R confidence for tracking
  tracks_conf_thresh: 0.                # Track confidence threshold
  dilations: 8                          # Mask dilation for edges
  matcher_match_kp_threshold: 0.3       # Keypoint match threshold
  loss_fn: "huber"                      # Robust loss function
  hnr_threshold: 5.                     # Huber δ parameter
  filtering_mode: "AVG_WEIGHTED"        # Pointmap update mode
  # Options: AVG_WEIGHTED, BEST_SCORE, NEAREST, FARTHEST, AVG_CENTER
  
  # Uncalibrated tracking
  uncalibrated_keyframe_match_threshold: 0.3
  
  # Calibrated tracking  
  calibrated_opt_flow_loss: false       # Use optical flow term
  calibrated_opt_reproj_loss: true      # Use reprojection error
  calib_opt_inlier_threshold: 5         # Inlier threshold (pixels)
  calibrated_kf_match_threshold: 0.3    # Keyframe threshold
  calibrated_pose_conf_threshold: 0.1   # Pose confidence threshold
```

### Local Optimization (Tracking)
```yaml
local_optimization:
  σ_p: 2                   # Point measurement noise (pixels)
  σ_d: 0.2                 # Depth measurement noise
  σ_r: 0.05                # Ray direction noise
  σ_i: 1                   # Image measurement noise
  n_iters: 10              # Optimization iterations
  lm_lambda: 0.01          # Levenberg-Marquardt damping
  damp_factor: 10          # Damping increase factor
  terminate: false         # Early termination
  print_timing: false      # Print performance stats
  cuda: true              # Use GPU acceleration
```

### Global Optimization (Backend)
```yaml
optimization:
  # Retrieval settings
  retrieval_interval: 10        # Check loop closure every N keyframes
  retrieval_k: 3               # Number of candidates
  retrieval_thresh: 0.5        # Similarity threshold
  retrieval_kf_match_threshold: 0.3  # Match threshold for verification
  n_local_retrieval_kf: 10     # Local keyframes for retrieval
  
  # Bundle adjustment
  local_match_kps_threshold: 0.3    # Local match threshold
  opt_local_BA_kfs: 20             # Keyframes in optimization window
  non_opt_local_BA_kfs: 40         # Fixed keyframes for constraints
  opt_local_BA_interval: 25        # Run BA every N keyframes
  
  # Optimization parameters
  n_iters: 10                      # Gauss-Newton iterations
  lm_lambda: 1.                    # Initial damping
  confidence_weight: true          # Weight by point confidence
  pixel_measurement_huber_delta: 3  # Huber δ for pixels
  tukey_threshold: 10              # Tukey threshold
  term_criteria_cost_tol: 1e-5     # Cost tolerance
  term_criteria_grad_tol: 1e-7     # Gradient tolerance
  term_criteria_params_tol: 1e-5   # Parameter tolerance
  cuda: true                       # GPU acceleration
```

### Visualization Settings
```yaml
visualization:
  confidence_thresh: 1.5      # Minimum confidence to display
  filter_depth: 5            # Maximum depth to render
  pointcloud_size: 0.02      # Point/surfel size
  render_every: 3            # Update every N frames
  fov: 50                    # Field of view (degrees)
  camera_z: -3               # Initial camera Z position
  camera_y: 2                # Initial camera Y position
  up: [0, -1, 0]            # Up vector
  target_img_size: 3         # Image display size factor
  visualize: true            # Enable visualization
  offscreen: false           # Headless rendering
  use_mask: false           # Use confidence mask
  mode: "surfel"            # Render mode: points/surfel/triangle
```

## Calibrated Mode (calib.yaml)

Inherits from base.yaml with overrides:

```yaml
inherit: "base.yaml"        # Inherit base configuration

# Override tracking for calibrated mode
tracking:
  calibrated_opt_flow_loss: true
  calibrated_opt_reproj_loss: true
  
# Tighter thresholds
matching:
  match_conf_thresh: 2.0
  
# Use projection-based optimization
optimization:
  pixel_measurement_huber_delta: 2  # Stricter inlier threshold
```

## Evaluation Configurations

### eval_calib.yaml
For benchmarking with calibration:

```yaml
inherit: "calib.yaml"

# Single-threaded for deterministic results
processes: 1

# Process every other frame
img_stride: 2

# Disable visualization for speed
visualization:
  visualize: false
```

### eval_no_calib.yaml
For benchmarking without calibration:

```yaml
inherit: "base.yaml"

processes: 1
img_stride: 2

visualization:
  visualize: false
  
# Ensure uncalibrated mode
tracking:
  calibrated_opt_flow_loss: false
  calibrated_opt_reproj_loss: false
```

### eth3d.yaml
ETH3D dataset specific settings:

```yaml
inherit: "calib.yaml"

# ETH3D doesn't use centered principal point
center_principal_point: false

# Adjusted parameters for ETH3D
tracking:
  calib_opt_inlier_threshold: 3
  
optimization:
  pixel_measurement_huber_delta: 1.5
```

## Camera Intrinsics (intrinsics.yaml)

Template for specifying camera parameters:

```yaml
# Camera matrix
fx: 520.0         # Focal length X
fy: 520.0         # Focal length Y  
cx: 320.0         # Principal point X
cy: 240.0         # Principal point Y
width: 640        # Image width
height: 480       # Image height

# Distortion parameters (optional)
distortion_model: "radtan"  # Model type
k1: 0.0          # Radial coefficient 1
k2: 0.0          # Radial coefficient 2
p1: 0.0          # Tangential coefficient 1
p2: 0.0          # Tangential coefficient 2
k3: 0.0          # Radial coefficient 3
```

## Parameter Tuning Guide

### For Speed
```yaml
# Reduce image size
target_img_size: 384

# Fewer refinement iterations
matching:
  iterative_proj_n_iters: 5
  
# Larger strides
img_stride: 3

# Lower quality thresholds
tracking:
  mast3r_conf_thresh: 1.0
  
# Smaller optimization windows
optimization:
  opt_local_BA_kfs: 10
```

### For Quality
```yaml
# Full resolution
target_img_size: 640

# More iterations
matching:
  iterative_proj_n_iters: 20
  feature_refinement_radius: 5
  
# Process all frames
img_stride: 1

# Stricter thresholds
tracking:
  mast3r_conf_thresh: 2.5
  filtering_mode: "BEST_SCORE"
  
# Larger windows
optimization:
  opt_local_BA_kfs: 40
  n_iters: 20
```

### For Robustness
```yaml
# Robust loss functions
tracking:
  loss_fn: "tukey"
  hnr_threshold: 3.0
  
# More RANSAC iterations
matching:
  iterative_proj_ransac_iters: 20
  
# Conservative thresholds
optimization:
  retrieval_thresh: 0.7
  pixel_measurement_huber_delta: 5
```

## Creating Custom Configurations

### Example: Outdoor Scene Config
```yaml
inherit: "base.yaml"

# Handle larger depth range
matching:
  iterative_proj_d_max: 100000
  iterative_proj_d_min: 1.0
  
# Adjust for lighting changes
tracking:
  mast3r_conf_thresh: 1.2
  filtering_mode: "AVG_WEIGHTED"
  
# More aggressive loop closure
optimization:
  retrieval_interval: 5
  retrieval_k: 5
  retrieval_thresh: 0.4
```

### Example: Fast Preview Config
```yaml
inherit: "base.yaml"

# Minimal processing
target_img_size: 256
img_stride: 5

# Fast matching
matching:
  iterative_proj_n_iters: 3
  max_pts: 5000
  
# Simple visualization
visualization:
  mode: "points"
  render_every: 10
  confidence_thresh: 2.0
```

## Command-Line Override

Configuration parameters can be overridden via command line:

```bash
# Override specific parameters
python main.py --scene test --config config/base.yaml \
  --target_img_size 384 \
  --tracking.mast3r_conf_thresh 2.0 \
  --optimization.n_iters 20
```

## Best Practices

1. **Start with Base**: Always inherit from base.yaml for consistency
2. **Incremental Changes**: Make small parameter adjustments
3. **Document Changes**: Comment why parameters were modified
4. **Version Control**: Track configuration changes with git
5. **Validation**: Test configurations on small sequences first

## Troubleshooting

### Tracking Failures
- Reduce `mast3r_conf_thresh`
- Increase `iterative_proj_radius`
- Try different `filtering_mode`

### Poor Reconstruction Quality
- Increase `target_img_size`
- Use stricter confidence thresholds
- Enable calibrated mode if possible

### Slow Performance
- Reduce `max_pts` and `feature_extract_max_pts`
- Increase `img_stride`
- Disable visualization

### Memory Issues
- Reduce `max_n_keyframes`
- Lower `target_img_size`
- Disable `feature_refinement_radius`

This configuration system provides fine-grained control over the SLAM pipeline while maintaining ease of use through inheritance and sensible defaults.