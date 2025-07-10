# global_opt.py - Factor Graph and Global Optimization

## Overview
The `global_opt.py` module implements the backend optimization system using factor graphs. It maintains global consistency by jointly optimizing all keyframe poses while incorporating loop closure constraints.

## Core Components

### FactorGraph Class
Central data structure managing pose optimization.

```python
class FactorGraph:
    def __init__(self,
                 retrieval_interval=10,
                 retrieval_k=3,
                 retrieval_thresh=0.5,
                 retrieval_kf_match_threshold=0.3,
                 n_local_retrieval_kf=10,
                 local_match_kps_threshold=0.3,
                 opt_local_BA_kfs=20,
                 non_opt_local_BA_kfs=40,
                 opt_local_BA_interval=25,
                 gn_iters=10,
                 lam=1.,
                 confidence_weight=True,
                 pixel_measurement_huber_delta=3,
                 tukey_threshold=10,
                 shared_keyframes=None,
                 K=None,
                 cuda=True):
```

#### Key Parameters
- **retrieval_interval**: Frequency of loop closure detection
- **retrieval_k**: Number of candidate loop closures
- **opt_local_BA_kfs**: Keyframes in optimization window
- **gn_iters**: Gauss-Newton iterations
- **lam**: Levenberg-Marquardt damping
- **confidence_weight**: Use point confidence in optimization

### Graph Structure

#### Nodes
Each node represents a keyframe pose:
```python
class PoseNode:
    pose: Sim3/SE3     # 7-DOF or 6-DOF transformation
    fixed: bool        # Whether pose is fixed
    frame_idx: int     # Keyframe index
```

#### Edges
Edges represent geometric constraints:
```python
class Factor:
    type: str          # "sequential", "loop", "retrieval"
    i: int            # Source keyframe index
    j: int            # Target keyframe index
    weight: float     # Factor importance
    matches: dict     # Point correspondences
```

## Factor Graph Operations

### 1. Adding Keyframes
```python
def add_keyframe(self, keyframe_idx):
    """Add new keyframe to factor graph"""
    
    # Add sequential factor to previous keyframe
    if self.n_keyframes > 0:
        self.add_sequential_factor(
            self.n_keyframes - 1, 
            keyframe_idx
        )
    
    # Check for loop closures
    if keyframe_idx % self.retrieval_interval == 0:
        self.detect_loop_closures(keyframe_idx)
    
    self.n_keyframes += 1
```

### 2. Factor Creation

#### Sequential Factors
Connect consecutive keyframes:
```python
def add_sequential_factor(self, i, j):
    """Add factor between consecutive keyframes"""
    
    # Get keyframes
    kf_i = self.shared_keyframes.get(i)
    kf_j = self.shared_keyframes.get(j)
    
    # Match features
    matches = self.match_keyframes(kf_i, kf_j)
    
    # Create factor
    factor = {
        'type': 'sequential',
        'i': i,
        'j': j,
        'matches': matches,
        'weight': 1.0
    }
    
    self.factors.append(factor)
```

#### Loop Closure Factors
Connect revisited locations:
```python
def detect_loop_closures(self, query_idx):
    """Find and add loop closure constraints"""
    
    query_kf = self.shared_keyframes.get(query_idx)
    
    # Query retrieval database
    candidates = self.retrieval_db.query(
        query_kf,
        k=self.retrieval_k,
        threshold=self.retrieval_thresh
    )
    
    for candidate_idx in candidates:
        # Geometric verification
        matches = self.match_and_verify(query_kf, candidate_kf)
        
        if len(matches) > min_matches:
            self.add_loop_factor(query_idx, candidate_idx, matches)
```

### 3. Optimization Process

#### Local Bundle Adjustment
Optimizes a sliding window of keyframes:

```python
def local_BA(self, center_idx):
    """Local bundle adjustment around center keyframe"""
    
    # Select keyframes in window
    opt_kfs = self.select_local_keyframes(
        center_idx, 
        self.opt_local_BA_kfs
    )
    
    # Fixed keyframes for constraints
    fixed_kfs = self.select_local_keyframes(
        center_idx,
        self.non_opt_local_BA_kfs,
        exclude=opt_kfs
    )
    
    # Run optimization
    self.optimize_poses(opt_kfs, fixed_kfs)
```

#### Global Optimization
Full factor graph optimization:

```python
def optimize(self):
    """Global pose optimization using Gauss-Newton"""
    
    for iter in range(self.gn_iters):
        # Build linear system
        H, b = self.build_linear_system()
        
        # Add damping (Levenberg-Marquardt)
        H_diag = H.diagonal()
        H += self.lam * torch.diag(H_diag)
        
        # Solve H * delta = -b
        delta = self.solve_linear_system(H, b)
        
        # Update poses
        self.update_poses(delta)
        
        # Check convergence
        if self.has_converged(delta):
            break
```

## Optimization Objectives

### 1. Uncalibrated Mode (Ray-based)
Minimizes ray-to-ray distances:

```python
def compute_ray_factors(self, factor):
    """Compute ray-based reprojection error"""
    
    # Get matched 3D points
    p3d_i = factor['matches']['p3d_i']
    p3d_j = factor['matches']['p3d_j']
    
    # Transform to camera frames
    p_i_in_cam_i = T_i.inverse() @ p3d_i
    p_j_in_cam_j = T_j.inverse() @ p3d_j
    
    # Compute rays
    ray_i = normalize(p_i_in_cam_i)
    ray_j = normalize(p_j_in_cam_j)
    
    # Ray distance error
    T_ij = T_i.inverse() @ T_j
    ray_j_in_i = T_ij.rotate(ray_j)
    error = cross_product_norm(ray_i, ray_j_in_i)
    
    return error
```

### 2. Calibrated Mode (Projection-based)
Minimizes pixel reprojection error:

```python
def compute_projection_factors(self, factor):
    """Compute calibrated reprojection error"""
    
    # Project 3D points
    uv_i_proj = self.K @ (T_i.inverse() @ p3d_world)
    uv_i_proj = uv_i_proj[:2] / uv_i_proj[2]
    
    # Observed pixels
    uv_i_obs = factor['matches']['pixels_i']
    
    # Reprojection error
    error = uv_i_obs - uv_i_proj
    
    # Apply robust loss
    if self.pixel_measurement_huber_delta > 0:
        error = huber_loss(error, self.pixel_measurement_huber_delta)
    
    return error
```

### 3. Confidence Weighting
Incorporates point confidence:

```python
def apply_confidence_weighting(self, errors, confidences):
    """Weight errors by point confidence"""
    
    if self.confidence_weight:
        # Normalize confidences
        conf_normalized = confidences / confidences.max()
        # Apply weighting
        weighted_errors = errors * conf_normalized
        return weighted_errors
    
    return errors
```

## CUDA Acceleration

The module uses custom CUDA kernels for efficiency:

### Factor Computation
```python
# Python wrapper
def compute_factors_cuda(self, poses, factors):
    """GPU-accelerated factor computation"""
    
    if self.K is None:
        # Ray-based factors
        errors = cuda_extension.compute_ray_factors(
            poses, 
            factors,
            self.tukey_threshold
        )
    else:
        # Projection-based factors
        errors = cuda_extension.compute_proj_factors(
            poses,
            factors, 
            self.K,
            self.pixel_measurement_huber_delta
        )
    
    return errors
```

### Linear System Construction
```python
# CUDA kernel for building H and b
cuda_extension.build_linear_system(
    jacobians,    # [n_factors, n_residuals, n_params]
    residuals,    # [n_factors, n_residuals]
    weights,      # [n_factors]
    H,           # [n_params, n_params] - output
    b            # [n_params] - output
)
```

## Loop Closure Integration

### Retrieval-based Detection
```python
def process_retrieval_candidates(self, query_idx):
    """Process loop closure candidates from retrieval"""
    
    # Get query descriptor
    query_desc = self.compute_global_descriptor(query_idx)
    
    # Find similar keyframes
    candidates = self.retrieval_db.search(
        query_desc,
        k=self.retrieval_k
    )
    
    # Filter by time difference
    candidates = [c for c in candidates 
                  if abs(c.idx - query_idx) > min_time_diff]
    
    return candidates
```

### Geometric Verification
```python
def verify_loop_closure(self, kf_i, kf_j):
    """Verify loop closure with geometric constraints"""
    
    # Initial matching
    matches = self.match_keyframes(kf_i, kf_j)
    
    # RANSAC for robust pose estimation
    pose, inliers = self.ransac_pose_estimation(
        matches,
        threshold=self.pixel_measurement_huber_delta
    )
    
    # Check inlier ratio
    inlier_ratio = len(inliers) / len(matches)
    
    return inlier_ratio > self.retrieval_kf_match_threshold
```

## Optimization Strategies

### 1. Hierarchical Optimization
Optimize at multiple scales:
```python
# Coarse optimization
self.optimize(subsample=4, iterations=5)
# Fine optimization  
self.optimize(subsample=1, iterations=10)
```

### 2. Robust Loss Functions
Handle outliers in measurements:

```python
def robust_loss(residual, type="huber", threshold=1.0):
    """Apply robust loss function"""
    
    if type == "huber":
        return huber_loss(residual, threshold)
    elif type == "tukey":
        return tukey_loss(residual, threshold)
    elif type == "cauchy":
        return cauchy_loss(residual, threshold)
```

### 3. Adaptive Damping
Levenberg-Marquardt with adaptive λ:
```python
if loss_decreased:
    self.lam *= 0.5  # Reduce damping
else:
    self.lam *= 2.0  # Increase damping
```

## Memory Management

### Sparse Matrix Storage
Efficient storage for large systems:
```python
# Use sparse matrices for H
H_sparse = torch.sparse_coo_tensor(
    indices,
    values,
    size=(n_params, n_params)
)
```

### Factor Pruning
Remove redundant factors:
```python
def prune_factors(self):
    """Remove redundant or weak factors"""
    
    # Remove factors with low weight
    self.factors = [f for f in self.factors 
                    if f['weight'] > min_weight]
    
    # Limit factors per edge
    self.limit_factors_per_edge(max_factors=5)
```

## Integration with SLAM Pipeline

### Communication Flow
```
Frontend → Add Keyframe → Task Queue → Backend
                                          ↓
                                    Factor Graph
                                          ↓
                                    Optimization
                                          ↓
                                  Update SharedKeyframes
```

### Synchronization
- Asynchronous optimization doesn't block tracking
- Pose updates applied atomically
- Dirty flags trigger visualization updates

## Configuration Examples

### Aggressive Loop Closure
```yaml
optimization:
  retrieval_interval: 5      # Check frequently
  retrieval_k: 5            # More candidates
  retrieval_thresh: 0.3     # Lower threshold
  gn_iters: 20             # More iterations
```

### Conservative Settings
```yaml
optimization:
  retrieval_interval: 20    # Less frequent
  retrieval_k: 2           # Fewer candidates  
  retrieval_thresh: 0.7    # Higher threshold
  opt_local_BA_kfs: 10     # Smaller window
```

## Debugging Tools

### Visualization
```python
# Visualize factor graph
plot_factor_graph(self.factors, self.poses)

# Show optimization convergence
plot_convergence(self.loss_history)

# Display covariance matrix
visualize_covariance(H.inverse())
```

### Metrics
- Optimization loss per iteration
- Factor counts by type
- Inlier ratios for loop closures
- Pose update magnitudes

This module ensures global consistency in the SLAM system by continuously refining all poses based on geometric constraints, enabling accurate long-term mapping.