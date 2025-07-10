# matching.py - Feature Matching Algorithms

## Overview
The `matching.py` module implements efficient feature matching algorithms using iterative projection and CUDA acceleration. It provides the core matching functionality for tracking and loop closure detection.

## Core Classes

### Matcher Class
Main interface for feature matching operations.

```python
class Matcher:
    def __init__(self,
                 max_pts,
                 radius=32,
                 nms_radius=1,
                 conf_thresh=0.0,
                 n_iters=10,
                 match_threshold=0.2,
                 lr=0.02,
                 cycle_thresh=2,
                 d_max=10000,
                 d_min=0.1,
                 device='cuda',
                 verbose=False,
                 feature_refinement_radius=-1,
                 descriptor_refinement_radius=-1):
```

#### Key Parameters
- **max_pts**: Maximum points to match
- **radius**: Search radius for matching
- **nms_radius**: Non-maximum suppression radius
- **n_iters**: Refinement iterations
- **match_threshold**: Minimum correlation threshold
- **lr**: Learning rate for refinement
- **cycle_thresh**: Cycle consistency threshold

## Matching Pipeline

### 1. Projection-based Matching
Main matching function that projects 3D points and finds correspondences.

```python
def proj_match(self, X_w, img0, img1, K1, T10):
    """Match by projecting 3D points"""
    
    # Project 3D points to image 1
    X_cam1 = T10 @ X_w  # Transform to camera 1
    uv1_proj = K1 @ X_cam1  # Project to pixels
    uv1_proj = uv1_proj[:2] / uv1_proj[2]  # Normalize
    
    # Find correspondences
    matches0, matches1 = self.find_correspondences(
        img0, img1, 
        uv0_source, uv1_proj,
        radius=self.radius
    )
    
    return matches0, matches1
```

### 2. Iterative Refinement
Refines matches using image gradients.

```python
def refine_matches(self, img0, img1, uv0, uv1_init):
    """Iteratively refine match positions"""
    
    uv1 = uv1_init.clone()
    
    for iter in range(self.n_iters):
        # Compute image patches
        patches0 = sample_patches(img0, uv0)
        patches1 = sample_patches(img1, uv1)
        
        # Compute correlation
        corr = compute_correlation(patches0, patches1)
        
        # Compute gradients
        grad = compute_image_gradient(img1, uv1)
        
        # Update positions
        delta = self.lr * correlation_gradient(corr, grad)
        uv1 += delta
        
        # Check convergence
        if delta.norm() < 1e-3:
            break
    
    return uv1
```

## CUDA Implementation

### Kernel Functions
Low-level CUDA kernels for performance.

#### Projection Kernel
```cuda
__global__ void proj_match_kernel(
    const float* X_world,      // 3D points
    const float* K,            // Intrinsics
    const float* T,            // Pose
    const float* img0,         // Reference image
    const float* img1,         // Target image
    float* matches0,           // Output matches
    float* matches1,
    int H, int W,
    int radius
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Transform and project
    float3 X_cam = transform_point(X_world[idx], T);
    float2 uv_proj = project_point(X_cam, K);
    
    // Search window
    float best_corr = -1;
    float2 best_match;
    
    for(int dy = -radius; dy <= radius; dy++) {
        for(int dx = -radius; dx <= radius; dx++) {
            float2 uv_test = uv_proj + make_float2(dx, dy);
            
            // Compute correlation
            float corr = patch_correlation(
                img0, img1,
                make_float2(u0, v0), uv_test
            );
            
            if(corr > best_corr) {
                best_corr = corr;
                best_match = uv_test;
            }
        }
    }
    
    // Store result
    matches1[idx] = best_match;
}
```

#### Refinement Kernel
```cuda
__global__ void refine_matches_kernel(
    const float* img0,
    const float* img1,
    const float* grad_x,       // Image gradients
    const float* grad_y,
    float* uv0,
    float* uv1,
    int n_iters,
    float lr
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    for(int iter = 0; iter < n_iters; iter++) {
        // Sample patches
        float patch0[PATCH_SIZE];
        float patch1[PATCH_SIZE];
        sample_patch(img0, uv0[idx], patch0);
        sample_patch(img1, uv1[idx], patch1);
        
        // Compute residual
        float residual[PATCH_SIZE];
        for(int i = 0; i < PATCH_SIZE; i++) {
            residual[i] = patch1[i] - patch0[i];
        }
        
        // Compute update using gradients
        float2 grad = sample_gradient(grad_x, grad_y, uv1[idx]);
        float2 update = compute_gauss_newton_update(
            residual, grad
        );
        
        // Update position
        uv1[idx] += lr * update;
    }
}
```

## Advanced Features

### 1. Multi-Scale Matching
Matches at multiple image scales for robustness.

```python
def multiscale_match(self, img0, img1, scales=[1, 2, 4]):
    """Match at multiple scales"""
    
    all_matches = []
    
    for scale in scales:
        # Downsample images
        img0_s = downsample(img0, scale)
        img1_s = downsample(img1, scale)
        
        # Match at scale
        matches0_s, matches1_s = self.match(img0_s, img1_s)
        
        # Upscale matches
        matches0 = matches0_s * scale
        matches1 = matches1_s * scale
        
        # Refine at original scale
        matches1 = self.refine_matches(
            img0, img1, matches0, matches1
        )
        
        all_matches.append((matches0, matches1))
    
    # Merge matches from all scales
    return self.merge_multiscale_matches(all_matches)
```

### 2. Descriptor-based Refinement
Optional refinement using feature descriptors.

```python
def descriptor_refinement(self, features0, features1, 
                         uv0, uv1, radius):
    """Refine matches using descriptors"""
    
    if self.descriptor_refinement_radius <= 0:
        return uv1
    
    refined_uv1 = []
    
    for i, (u0, v0) in enumerate(uv0):
        # Get descriptor for source point
        desc0 = features0[int(v0), int(u0)]
        
        # Search in neighborhood
        best_sim = -1
        best_pos = uv1[i]
        
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                u1 = uv1[i][0] + dx
                v1 = uv1[i][1] + dy
                
                # Get descriptor
                desc1 = features1[int(v1), int(u1)]
                
                # Compute similarity
                sim = cosine_similarity(desc0, desc1)
                
                if sim > best_sim:
                    best_sim = sim
                    best_pos = [u1, v1]
        
        refined_uv1.append(best_pos)
    
    return np.array(refined_uv1)
```

### 3. Cycle Consistency Check
Ensures bidirectional consistency of matches.

```python
def cycle_consistency_check(self, matches01, matches10):
    """Filter matches by cycle consistency"""
    
    consistent_matches = []
    
    for i, (u0, v0) in enumerate(matches01[0]):
        # Forward match
        u1, v1 = matches01[1][i]
        
        # Find reverse match
        reverse_idx = find_nearest(matches10[1], (u0, v0))
        u0_rev, v0_rev = matches10[0][reverse_idx]
        
        # Check consistency
        dist = np.linalg.norm([u1 - u0_rev, v1 - v0_rev])
        
        if dist < self.cycle_thresh:
            consistent_matches.append(i)
    
    return consistent_matches
```

### 4. Non-Maximum Suppression
Removes redundant matches in dense regions.

```python
def nms(self, matches, scores, radius):
    """Non-maximum suppression for matches"""
    
    # Sort by score
    indices = np.argsort(scores)[::-1]
    
    keep = []
    suppressed = set()
    
    for idx in indices:
        if idx in suppressed:
            continue
            
        keep.append(idx)
        
        # Suppress neighbors
        pos = matches[idx]
        for other_idx in indices:
            if other_idx != idx:
                dist = np.linalg.norm(matches[other_idx] - pos)
                if dist < radius:
                    suppressed.add(other_idx)
    
    return keep
```

## Integration with Tracking

### Frame-to-Frame Matching
Used in the tracking pipeline:

```python
# In tracker.py
def track_frame(ref_frame, cur_frame):
    # Get 3D points from reference
    X_world = ref_frame.get_pointmap()
    
    # Initial pose from previous frame
    T_init = cur_frame.pose
    
    # Match using projection
    matches_ref, matches_cur = matcher.proj_match(
        X_world,
        ref_frame.img,
        cur_frame.img,
        cur_frame.intrinsics.K,
        T_init
    )
    
    # Refine pose using matches
    T_refined = optimize_pose(matches_ref, matches_cur, X_world)
```

### Loop Closure Matching
For detecting revisited locations:

```python
# In global_opt.py
def detect_loop_closure(query_kf, candidate_kf):
    # Match keyframes
    matches_q, matches_c = matcher.match_keyframes(
        query_kf, candidate_kf
    )
    
    # Geometric verification
    if len(matches_q) > min_matches:
        # Estimate relative pose
        T_rel, inliers = estimate_pose(
            matches_q, matches_c,
            query_kf.intrinsics,
            candidate_kf.intrinsics
        )
        
        return T_rel, inliers
```

## Performance Optimization

### 1. GPU Memory Management
Efficient allocation and reuse:

```python
class MatcherMemoryPool:
    def __init__(self, max_points):
        # Pre-allocate GPU buffers
        self.uv_buffer = torch.zeros(max_points, 2).cuda()
        self.corr_buffer = torch.zeros(max_points).cuda()
        self.grad_buffer = torch.zeros(max_points, 2).cuda()
    
    def get_buffers(self, n_points):
        # Return views of pre-allocated memory
        return (
            self.uv_buffer[:n_points],
            self.corr_buffer[:n_points],
            self.grad_buffer[:n_points]
        )
```

### 2. Batch Processing
Process multiple matches simultaneously:

```python
def batch_match(self, batch_X, batch_img0, batch_img1, batch_K, batch_T):
    """Match multiple frame pairs"""
    
    # Stack inputs
    X_stacked = torch.cat(batch_X)
    
    # Single kernel call
    all_matches = self.cuda_module.batch_proj_match(
        X_stacked, batch_img0, batch_img1, 
        batch_K, batch_T
    )
    
    # Split results
    return split_batch_results(all_matches, batch_sizes)
```

### 3. Early Termination
Stop refinement when converged:

```python
def adaptive_refinement(self, uv0, uv1, max_iters=10):
    """Refine with early stopping"""
    
    prev_uv1 = uv1.clone()
    
    for iter in range(max_iters):
        uv1 = self.refine_step(uv0, uv1)
        
        # Check convergence
        movement = (uv1 - prev_uv1).norm(dim=1)
        if movement.max() < 0.1:  # pixels
            break
            
        prev_uv1 = uv1.clone()
    
    return uv1, iter
```

## Configuration

### High Accuracy Settings
```yaml
matching:
  radius: 64                    # Large search window
  n_iters: 20                  # More refinement
  match_threshold: 0.3         # Stricter threshold
  feature_refinement_radius: 5 # Use descriptors
  nms_radius: 3               # Remove close matches
```

### Fast Settings
```yaml
matching:
  radius: 16                   # Small search window
  n_iters: 5                  # Less refinement
  match_threshold: 0.1        # Relaxed threshold
  lr: 0.05                    # Larger steps
```

## Debugging Tools

### Match Visualization
```python
def visualize_matches(img0, img1, matches0, matches1):
    """Draw matches between images"""
    
    # Create side-by-side image
    vis = np.hstack([img0, img1])
    
    # Draw lines
    for (u0, v0), (u1, v1) in zip(matches0, matches1):
        # Offset second image coordinates
        u1_vis = u1 + img0.shape[1]
        
        cv2.line(vis, (u0, v0), (u1_vis, v1), (0, 255, 0), 1)
        cv2.circle(vis, (u0, v0), 3, (255, 0, 0), -1)
        cv2.circle(vis, (u1_vis, v1), 3, (0, 0, 255), -1)
    
    return vis
```

### Match Statistics
```python
def compute_match_stats(matches0, matches1, X_world, K, T):
    """Compute matching statistics"""
    
    stats = {
        'n_matches': len(matches0),
        'mean_reproj_error': compute_reprojection_error(
            matches1, X_world, K, T
        ),
        'inlier_ratio': compute_inlier_ratio(
            matches0, matches1, threshold=3.0
        ),
        'distribution': analyze_spatial_distribution(matches0)
    }
    
    return stats
```

This module provides the foundation for accurate feature matching in the SLAM system, balancing speed and accuracy through GPU acceleration and iterative refinement.