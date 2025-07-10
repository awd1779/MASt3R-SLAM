# tracker.py - Tracking Algorithms and Keyframe Selection

## Overview
The `tracker.py` module implements frame-to-frame tracking algorithms, pose optimization, and keyframe selection logic. It provides both calibrated and uncalibrated tracking modes using different optimization objectives.

## Core Classes

### Tracker Class
Main tracking interface that handles pose estimation and keyframe decisions.

```python
class Tracker:
    def __init__(self, 
                 mast3r_conf_thresh=1.5,
                 tracks_conf_thresh=0.,
                 dilations=8,
                 matcher_match_kp_threshold=0.3,
                 loss_fn="huber",
                 hnr_threshold=5.,
                 calibrated_opt_flow_loss=True,
                 calibrated_opt_reproj_loss=True,
                 calib_opt_inlier_threshold=5,
                 calibrated_kf_match_threshold=0.3,
                 calibrated_pose_conf_threshold=0.1,
                 filtering_mode="AVG_WEIGHTED"):
```

#### Configuration Parameters
- **mast3r_conf_thresh**: Confidence threshold for MASt3R predictions
- **tracks_conf_thresh**: Minimum confidence for valid tracks
- **dilations**: Mask dilation for edge filtering
- **matcher_match_kp_threshold**: Threshold for keypoint matching
- **loss_fn**: Robust loss function ("huber", "tukey", etc.)
- **calibrated_opt_***: Flags for calibrated optimization terms
- **filtering_mode**: Pointmap update strategy

### TrackResult Class
Encapsulates tracking results and metrics.

```python
@dataclass
class TrackResult:
    ref_frame: Frame          # Reference keyframe
    tracked_frame: Frame      # Current tracked frame
    matches_im1: np.ndarray   # Matched points in reference
    matches_im2: np.ndarray   # Matched points in current
    p3d_ref: np.ndarray      # 3D points from reference
    p3d_cur: np.ndarray      # 3D points from current
    match_frac: float        # Fraction of successful matches
    is_keyframe: bool        # Keyframe decision
    matcher_metrics: dict    # Additional matching statistics
```

## Tracking Pipeline

### 1. Main Tracking Function
```python
def track(self, model, ref_frame, cur_frame):
    """Track current frame against reference keyframe"""
```

#### Pipeline Steps:
1. **Feature Matching**: Match features between frames using MASt3R
2. **Pose Optimization**: Estimate relative pose
3. **Pointmap Update**: Merge 3D observations
4. **Keyframe Decision**: Determine if current frame should be keyframe

### 2. Feature Matching Phase

#### MASt3R Matching
```python
# Asymmetric matching for efficiency
pred12 = mast3r_match_asymmetric(
    model, 
    ref_frame, 
    cur_frame,
    subsample=2  # Downsample for speed
)
```

#### Iterative Projection Matching
```python
# Project 3D points and refine matches
matches_im1, matches_im2 = self.matcher.proj_match(
    ref_frame.X_world,     # 3D points in world
    ref_frame.img,         # Reference image
    cur_frame.img,         # Current image
    cur_intrinsics,        # Camera parameters
    T_init                 # Initial pose estimate
)
```

### 3. Pose Optimization

The tracker implements two optimization modes:

#### Uncalibrated Mode
Uses ray-to-ray and point-to-point distances.

```python
def opt_pose_ray_dist_sim3(self, ref_frame, cur_frame, 
                           matches_im1, matches_im2,
                           p3d_cur, K_cur):
    """Optimize Sim3 pose using ray distances"""
    
    # Objective function components:
    # 1. Ray-to-ray distance
    ray1 = normalize(p3d_ref)
    ray2 = K_cur.inv() @ matches_im2
    error_ray = ||cross(ray1, R @ ray2)||
    
    # 2. Point-to-point distance  
    error_point = ||p3d_ref - s*R*p3d_cur - t||
    
    # Combined objective with robust loss
    loss = robust_loss(w_ray * error_ray + w_point * error_point)
```

##### Optimization Details:
- **Transform**: Sim3 (7 DOF: rotation, translation, scale)
- **Method**: Levenberg-Marquardt with adaptive damping
- **Robust Loss**: Huber or Tukey for outlier handling
- **Convergence**: Based on parameter and loss changes

#### Calibrated Mode
Uses pixel reprojection error with known intrinsics.

```python
def opt_pose_calib_sim3(self, ref_frame, cur_frame,
                        matches_im1, matches_im2, 
                        p3d_ref, p3d_cur,
                        K_ref, K_cur):
    """Optimize pose using calibrated projection"""
    
    # Reprojection error
    proj_ref = K_ref @ (R @ p3d_cur + t)
    error_reproj = ||matches_im1 - proj_ref[:2]/proj_ref[2]||
    
    # Optional: Optical flow consistency
    if self.calibrated_opt_flow_loss:
        flow_pred = compute_flow(R, t, p3d, K)
        error_flow = ||flow_obs - flow_pred||
        
    loss = w_reproj * error_reproj + w_flow * error_flow
```

##### Inlier Detection:
```python
# Mark inliers based on reprojection error
reproj_error = compute_reprojection_error(...)
inliers = reproj_error < self.calib_opt_inlier_threshold
```

### 4. Pointmap Update

After pose optimization, the reference keyframe's pointmap is updated:

```python
def update_keyframe_pointmap(ref_frame, cur_frame, T_cur_to_ref):
    """Merge current frame's 3D points into reference"""
    
    # Transform current points to reference frame
    p3d_in_ref = T_cur_to_ref @ cur_frame.get_pointmap()
    
    # Project to reference image plane
    uv_proj = ref_frame.intrinsics @ p3d_in_ref
    
    # Update using selected filtering mode
    ref_frame.update_pointmap(
        p3d_in_ref,
        cur_frame.C,
        filtering_mode=self.filtering_mode
    )
```

### 5. Keyframe Selection

Keyframes are selected based on multiple criteria:

```python
def is_keyframe(self, track_result):
    """Determine if current frame should be keyframe"""
    
    # Criteria 1: Match quality
    if track_result.match_frac < self.kf_match_threshold:
        return True
        
    # Criteria 2: New area coverage
    unique_coverage = compute_unique_coverage(
        track_result.matches_im2,
        existing_keypoints
    )
    if unique_coverage > self.kf_coverage_threshold:
        return True
        
    # Criteria 3: Pose confidence (calibrated mode)
    if self.use_calibration:
        pose_conf = compute_pose_confidence(track_result)
        if pose_conf < self.pose_conf_threshold:
            return True
            
    return False
```

## Advanced Features

### 1. Multi-Scale Tracking
Handles features at different scales:
```python
# Extract multi-scale features
features_ms = extract_multiscale_features(img)

# Match at each scale
for scale in scales:
    matches_scale = match_at_scale(features_ms[scale])
    all_matches.append(matches_scale)
```

### 2. Robust Estimation
Multiple strategies for handling outliers:

#### Huber Loss
```python
def huber_loss(residual, delta=1.0):
    """Robust loss less sensitive to outliers"""
    if abs(residual) <= delta:
        return 0.5 * residual**2
    else:
        return delta * (abs(residual) - 0.5 * delta)
```

#### RANSAC-like Sampling
```python
# Sample subset for initialization
inlier_subset = sample_matches(matches, n_samples=100)
T_init = estimate_pose(inlier_subset)

# Refine with all inliers
inliers = find_inliers(matches, T_init, threshold)
T_refined = estimate_pose(inliers)
```

### 3. Failure Detection
Identifies tracking failures:

```python
def detect_tracking_failure(track_result):
    """Check if tracking has failed"""
    
    # Low match count
    if track_result.match_count < min_matches:
        return True
        
    # High residual error
    if track_result.avg_residual > max_residual:
        return True
        
    # Degenerate configuration
    if is_degenerate(track_result.matches):
        return True
        
    return False
```

## Configuration Examples

### High-Quality Tracking
```yaml
tracking:
  mast3r_conf_thresh: 2.0      # Stricter confidence
  dilations: 16                # More edge filtering
  loss_fn: "tukey"             # More robust to outliers
  filtering_mode: "BEST_SCORE" # Keep best observations
```

### Fast Tracking
```yaml
tracking:
  mast3r_conf_thresh: 1.0      # Relaxed threshold
  dilations: 4                 # Less filtering
  matcher_iterations: 1        # Fewer refinement iterations
  filtering_mode: "AVG_WEIGHTED" # Simple averaging
```

### Calibrated Mode
```yaml
tracking:
  calibrated_opt_flow_loss: true
  calibrated_opt_reproj_loss: true
  calib_opt_inlier_threshold: 3  # Pixels
  calibrated_kf_match_threshold: 0.4
```

## Integration with SLAM Pipeline

### Frame Processing Flow
```
New Frame → Feature Extraction → Matching → Pose Optimization
    ↓                                              ↓
Keyframe?                                   Update Pointmap
    ↓                                              ↓
Add to Map ←────────────────────────────────────┘
```

### Communication with Backend
When a keyframe is selected:
1. Added to `SharedKeyframes` buffer
2. Backend notified via task queue
3. Triggers global optimization

### Handling Tracking Failures
On failure, the system:
1. Switches to `RELOC` mode
2. Queries retrieval database
3. Attempts tracking against retrieved keyframes
4. Falls back to reinitialization if needed

## Performance Optimization

### GPU Utilization
- MASt3R inference on GPU
- CUDA kernels for projection matching
- Batch processing of matches

### Computational Efficiency
- Asymmetric matching (only compute features for new frame)
- Downsampling for initial matching
- Early termination on convergence

### Memory Efficiency
- Reuse allocated buffers
- In-place pointmap updates
- Sparse storage for matches

## Debugging and Visualization

### Tracking Metrics
Available in `TrackResult.matcher_metrics`:
- Number of initial matches
- Inlier ratio
- Average reprojection error
- Optimization iterations
- Convergence status

### Visualization Helpers
```python
# Visualize matches
draw_matches(ref_frame.img, cur_frame.img, 
             matches_im1, matches_im2)

# Plot residuals
plot_residual_distribution(residuals)

# Show inlier mask
visualize_inliers(img, matches, inlier_mask)
```

This module is crucial for maintaining accurate camera tracking throughout the SLAM pipeline, balancing accuracy with real-time performance requirements.