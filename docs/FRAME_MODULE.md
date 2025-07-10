# frame.py - Frame Management and Data Structures

## Overview
The `frame.py` module defines the core data structures for representing camera frames and managing keyframes in the SLAM system. It provides thread-safe shared memory structures for inter-process communication.

## Core Classes

### Frame Class
The fundamental data structure representing a camera frame with its associated 3D reconstruction.

```python
class Frame:
    def __init__(self, intrinsics, pose, img, canonical_pointmap, 
                 mast3r_features, mast3r_scales, conf, frame_idx, 
                 timestamp, feat_residuals):
```

#### Attributes
- **intrinsics**: `Intrinsics` - Camera calibration parameters
- **pose**: `Sim3` - Camera pose (7-DOF similarity transform)
- **img**: `np.ndarray` - RGB image data
- **X_canon**: `np.ndarray` - 3D pointmap in canonical coordinates
- **mast3r_features**: `torch.Tensor` - Encoded features from MASt3R
- **mast3r_scales**: `np.ndarray` - Feature scale information
- **C**: `np.ndarray` - Point confidence scores
- **frame_idx**: `int` - Frame index in dataset
- **timestamp**: `float` - Frame timestamp
- **feat_residuals**: `float` - Feature matching residuals

#### Key Methods

##### get_pointmap()
```python
def get_pointmap(self, ret_canon=False):
    """Get 3D pointmap in world or canonical coordinates"""
    if ret_canon:
        return self.X_canon
    else:
        # Transform to world coordinates
        return self.pose @ self.X_canon
```

##### update_pointmap()
Implements various filtering strategies for pointmap updates:

```python
def update_pointmap(self, X_new, C_new, filtering_mode):
    """Update pointmap with new observations"""
```

**Filtering Modes**:
1. **AVG_WEIGHTED**: Weighted average based on confidence
   ```python
   X = (X_old * C_old + X_new * C_new) / (C_old + C_new)
   ```

2. **BEST_SCORE**: Keep points with highest confidence
   ```python
   mask = C_new > C_old
   X[mask] = X_new[mask]
   ```

3. **NEAREST**: Keep points closest to camera
   ```python
   dist_old = ||X_old||
   dist_new = ||X_new||
   X[dist_new < dist_old] = X_new[dist_new < dist_old]
   ```

4. **FARTHEST**: Keep points farthest from camera

5. **AVG_CENTER**: Average when close to image center

##### apply_mask()
```python
def apply_mask(self, mask):
    """Apply binary mask to filter points"""
    self.X_canon[~mask] = nan
    self.C[~mask] = 0
```

##### resize()
```python
def resize(self, target_size):
    """Resize frame data to target resolution"""
    # Resizes image, pointmap, and confidence
    # Updates intrinsics accordingly
```

### SharedKeyframes Class
Thread-safe container for keyframes shared across processes.

```python
class SharedKeyframes:
    def __init__(self, max_keyframes, 
                 canonical_pointmap_shape,
                 img_shape, 
                 feat_shape,
                 feat_scales_shape):
```

#### Design Principles
- **Pre-allocated Memory**: Fixed-size buffer for efficiency
- **Shared Memory**: Zero-copy access across processes
- **Lock-based Synchronization**: Thread-safe operations
- **Dirty Tracking**: Marks modified frames for visualization

#### Key Attributes
- **buffer**: Pre-allocated numpy arrays in shared memory
- **max_keyframes**: Maximum capacity
- **n_keyframes**: Current keyframe count
- **latest_idx**: Index of most recent keyframe
- **dirty**: Boolean flags for visualization updates

#### Memory Layout
```
SharedKeyframes Memory Buffer:
┌─────────────────────────────────────┐
│ Metadata (counts, indices, flags)   │
├─────────────────────────────────────┤
│ Poses (7 * max_keyframes)           │
├─────────────────────────────────────┤
│ Images (H * W * 3 * max_keyframes)  │
├─────────────────────────────────────┤
│ Pointmaps (H * W * 3 * max_kf)      │
├─────────────────────────────────────┤
│ Confidence (H * W * max_keyframes)  │
├─────────────────────────────────────┤
│ Features (N * D * max_keyframes)    │
└─────────────────────────────────────┘
```

#### Methods

##### add()
```python
def add(self, frame):
    """Add new keyframe to buffer"""
    with self.lock:
        idx = self.n_keyframes.value % self.max_keyframes
        # Copy frame data to shared memory
        self._copy_to_buffer(frame, idx)
        self.n_keyframes.value += 1
        self.latest_idx.value = idx
        self.dirty[idx] = True
```

##### get()
```python
def get(self, idx):
    """Retrieve keyframe by index"""
    with self.lock:
        if idx >= self.n_keyframes.value:
            return None
        buffer_idx = idx % self.max_keyframes
        return self._create_frame_from_buffer(buffer_idx)
```

##### get_dirty_keyframes()
```python
def get_dirty_keyframes(self):
    """Get keyframes marked for visualization update"""
    with self.lock:
        dirty_indices = np.where(self.dirty)[0]
        frames = [self.get(idx) for idx in dirty_indices]
        # Clear dirty flags
        self.dirty[:] = False
        return frames
```

### SharedStates Class
Manages system-wide state information.

```python
class SharedStates:
    def __init__(self, canonical_pointmap_shape, img_shape):
```

#### State Information
- **mode**: Current system mode (INIT, TRACKING, RELOC, TERMINATED)
- **latest_frame**: Most recent processed frame
- **n_frames**: Total frames processed
- **n_keyframes**: Total keyframes added
- **n_failed_tracks**: Tracking failure count

#### Methods

##### set_mode()
```python
def set_mode(self, mode):
    """Update system mode with thread safety"""
    with self.lock:
        self.mode.value = mode.value
```

##### update_latest_frame()
```python
def update_latest_frame(self, frame):
    """Update latest frame data in shared memory"""
    with self.lock:
        # Copy frame data to shared buffers
        self._copy_frame_data(frame)
        self.n_frames.value += 1
```

## Pointmap Filtering Strategies

The system implements sophisticated filtering for merging 3D observations:

### 1. Weighted Averaging (AVG_WEIGHTED)
Best for smooth surfaces and noise reduction:
```python
# Weighted by confidence scores
X_merged = (X1 * C1 + X2 * C2) / (C1 + C2)
C_merged = max(C1, C2)
```

### 2. Best Score Selection (BEST_SCORE)
Keeps most confident observations:
```python
mask = C_new > C_old
X_merged[mask] = X_new[mask]
C_merged = max(C_old, C_new)
```

### 3. Distance-based Selection
- **NEAREST**: Prefers closer points (better for foreground)
- **FARTHEST**: Prefers distant points (better for background)

### 4. Region-based Averaging (AVG_CENTER)
Averages points near image center, keeps best elsewhere:
```python
center_mask = (distance_to_center < threshold)
X_merged[center_mask] = average(X_old, X_new)
X_merged[~center_mask] = best_score_selection()
```

## Memory Management

### Shared Memory Allocation
Uses `multiprocessing.shared_memory` for zero-copy access:
```python
# Example: Allocating shared image buffer
img_buffer = np.ndarray(
    shape=(max_keyframes, H, W, 3),
    dtype=np.uint8,
    buffer=shared_memory.buf
)
```

### Memory Efficiency
- **Pre-allocation**: Avoids dynamic allocation during runtime
- **Circular Buffer**: Reuses memory for old keyframes
- **Lazy Loading**: Images loaded only when needed
- **Type Optimization**: Uses appropriate dtypes (uint8 for images, float32 for coords)

## Thread Safety

### Locking Strategy
- **Coarse-grained Locks**: One lock per shared structure
- **Read-Write Patterns**: Multiple readers, single writer
- **Lock Duration**: Minimize critical sections

### Example Thread-Safe Operation
```python
def update_keyframe_pose(self, idx, new_pose):
    with self.lock:
        # Critical section
        buffer_idx = idx % self.max_keyframes
        self.poses[buffer_idx] = new_pose.matrix()
        self.dirty[buffer_idx] = True
```

## Usage Examples

### Creating a Frame
```python
# From MASt3R predictions
frame = Frame(
    intrinsics=camera_intrinsics,
    pose=initial_pose,
    img=rgb_image,
    canonical_pointmap=mast3r_pointmap,
    mast3r_features=encoded_features,
    conf=confidence_map,
    frame_idx=0,
    timestamp=time.time()
)
```

### Managing Keyframes
```python
# Initialize shared keyframes
shared_kf = SharedKeyframes(
    max_keyframes=5000,
    canonical_pointmap_shape=(384, 512, 3),
    img_shape=(384, 512, 3),
    feat_shape=(768, 256)
)

# Add keyframe
shared_kf.add(frame)

# Retrieve keyframe
kf = shared_kf.get(idx)

# Get frames for visualization
dirty_frames = shared_kf.get_dirty_keyframes()
```

### Pointmap Updates
```python
# Update with new observations
frame.update_pointmap(
    X_new=new_pointmap,
    C_new=new_confidence,
    filtering_mode="AVG_WEIGHTED"
)

# Apply confidence threshold
mask = frame.C > confidence_threshold
frame.apply_mask(mask)
```

## Performance Considerations

### Memory Access Patterns
- **Sequential Access**: Optimize for cache efficiency
- **Batched Operations**: Process multiple points together
- **SIMD-friendly**: Data layout supports vectorization

### Optimization Techniques
1. **Numpy Vectorization**: Avoid Python loops
2. **In-place Operations**: Minimize memory allocation
3. **View Creation**: Use array views instead of copies
4. **Type Consistency**: Maintain consistent dtypes

This module provides the foundation for efficient frame management in the SLAM system, enabling real-time performance through careful memory management and thread-safe design.