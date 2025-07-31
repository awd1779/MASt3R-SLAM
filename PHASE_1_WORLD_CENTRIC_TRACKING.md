# Phase 1: World-Centric Object Tracking Implementation

## Overview
Transform object tracking from camera-centric to world-centric coordinates using MASt3R-SLAM's camera poses (T_WC). This will dramatically improve tracking consistency by leveraging the static world assumption.

## Current State
- **Camera-Centric**: Objects tracked in camera coordinates (X_canon)
- **Problem**: Same object gets different track IDs as camera moves
- **Available but Unused**: T_WC (world-to-camera transform) from MASt3R-SLAM

## Implementation Plan

### 1. Data Structure Updates

#### 1.1 Extend `TrackedObject3D` Class
**File**: `mast3r_slam/geometric_3d_tracker.py`

Add world coordinate storage:
```python
@dataclass
class TrackedObject3D:
    # Existing fields
    track_id: int
    label: str
    first_seen_frame: int
    last_seen_frame: int
    
    # Camera coordinates (existing, rename for clarity)
    bbox_3d_cam: Optional[Dict] = None  # Renamed from bbox_3d
    last_centroid_cam: Optional[torch.Tensor] = None  # Renamed
    
    # NEW: World coordinates
    centroid_world: Optional[torch.Tensor] = None
    bbox_3d_world: Optional[Dict] = None
    position_variance_world: float = 0.0  # Track stability metric
    
    # NEW: Tracking state
    last_T_WC: Optional[torch.Tensor] = None  # Last camera pose
    world_observations: List[torch.Tensor] = field(default_factory=list)  # History
```

### 2. Core Algorithm Changes

#### 2.1 Add World Transform Method
**File**: `mast3r_slam/geometric_3d_tracker.py`

```python
def _transform_to_world(self, bbox_cam: Dict, T_WC) -> Dict:
    """Transform 3D bounding box from camera to world coordinates.
    
    Args:
        bbox_cam: Bounding box in camera coordinates
        T_WC: Camera-to-world transformation matrix (4x4)
    
    Returns:
        bbox_world: Bounding box in world coordinates
    """
    # Extract T_WC matrix if it's a lietorch object
    if hasattr(T_WC, 'matrix'):
        T_WC_matrix = T_WC.matrix().squeeze()
    else:
        T_WC_matrix = T_WC
    
    # Transform center point
    center_cam = bbox_cam['center']
    center_homo = torch.cat([center_cam, torch.tensor([1.0], device=center_cam.device)])
    center_world = (T_WC_matrix @ center_homo)[:3]
    
    # Transform all 8 corners
    corners_cam = bbox_cam['corners']  # 8x3
    corners_homo = torch.cat([corners_cam, torch.ones((8, 1), device=corners_cam.device)], dim=1)
    corners_world = (T_WC_matrix @ corners_homo.T).T[:, :3]
    
    # Create world bbox
    bbox_world = {
        'center': center_world,
        'corners': corners_world,
        'dimensions': bbox_cam['dimensions'],  # Size unchanged
        'volume': bbox_cam['volume'],
        'confidence': bbox_cam['confidence'],
        'type': bbox_cam['type']
    }
    
    return bbox_world
```

#### 2.2 Modify Detection Extraction
**File**: `mast3r_slam/geometric_3d_tracker.py`

Update `_extract_3d_bboxes` to include T_WC:
```python
def _extract_3d_bboxes(self, keyframe, semantic_data: Dict) -> Dict[int, Dict]:
    """Extract 3D bounding boxes in both camera and world coordinates."""
    
    detections = {}
    h, w = keyframe.img_shape[0, 0].item(), keyframe.img_shape[0, 1].item()
    
    # Get camera pose
    T_WC = keyframe.T_WC
    
    for instance_id, mask_rle in semantic_data.get('masks_rle', {}).items():
        label = semantic_data.get('labels', {}).get(instance_id, 'unknown')
        
        # Decode mask
        mask = decode_rle(mask_rle, h, w)
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        # Create 3D bbox in camera coordinates
        bbox_cam = create_generic_3d_bbox(
            keyframe, mask, label,
            config=self.bbox_computer.config,
            estimator=self.bbox_estimator
        )
        
        if bbox_cam is not None and bbox_cam['size_valid']:
            # Transform to world coordinates
            bbox_world = self._transform_to_world(bbox_cam, T_WC)
            
            detections[instance_id] = {
                'bbox_cam': bbox_cam,
                'bbox_world': bbox_world,
                'label': label,
                'instance_id': instance_id
            }
            
            logger.debug(f"Extracted 3D bbox for {label}: "
                        f"cam_center={bbox_cam['center'].cpu().numpy()}, "
                        f"world_center={bbox_world['center'].cpu().numpy()}")
    
    return detections
```

#### 2.3 Implement World-Space Matching
**File**: `mast3r_slam/geometric_3d_tracker.py`

Replace matching logic with world-centric approach:
```python
def _match_detections_to_tracks(self, detections: Dict[int, Dict], frame_id: int) -> Dict[int, int]:
    """Match detections to tracks using world coordinates."""
    
    assignments = {}
    used_tracks = set()
    
    # Sort by confidence for stable matching
    sorted_detections = sorted(
        detections.items(),
        key=lambda x: x[1]['bbox_cam']['confidence'],
        reverse=True
    )
    
    for instance_id, detection in sorted_detections:
        best_track_id = None
        best_score = float('inf')  # Lower is better for distance
        
        # Search existing tracks with same label
        for track_id, track in self.tracked_objects.items():
            if track_id in used_tracks:
                continue
                
            if track.label != detection['label']:
                continue
                
            # Skip if lost too long
            if track.lost_frames > self.max_lost_frames:
                continue
            
            # WORLD SPACE MATCHING
            if track.centroid_world is not None:
                # Distance in world coordinates
                world_distance = torch.norm(
                    detection['bbox_world']['center'] - track.centroid_world
                ).item()
                
                # Expected distance threshold based on object type
                if detection['label'] in ['wall', 'floor', 'ceiling']:
                    distance_threshold = 1.0  # 1m for large objects
                elif detection['label'] in ['chair', 'table', 'sofa']:
                    distance_threshold = 0.5  # 50cm for furniture
                else:
                    distance_threshold = 0.3  # 30cm for small objects
                
                if world_distance < distance_threshold and world_distance < best_score:
                    best_score = world_distance
                    best_track_id = track_id
        
        # Assign or create new track
        if best_track_id is not None:
            assignments[instance_id] = best_track_id
            used_tracks.add(best_track_id)
            
            # Update track with new observation
            track = self.tracked_objects[best_track_id]
            self._update_track_world(track, detection, frame_id)
            
            logger.info(f"Matched {detection['label']} to track {best_track_id} "
                       f"(world_dist: {best_score:.3f}m)")
        else:
            # Create new track
            new_track_id = self._create_new_track(detection, frame_id)
            assignments[instance_id] = new_track_id
            
            logger.info(f"Created new track {new_track_id} for {detection['label']} "
                       f"at world pos {detection['bbox_world']['center'].cpu().numpy()}")
    
    return assignments

def _update_track_world(self, track: TrackedObject3D, detection: Dict, frame_id: int):
    """Update track with new world observation."""
    
    # Update world position
    old_world = track.centroid_world
    new_world = detection['bbox_world']['center']
    
    # Track position stability
    if old_world is not None:
        position_change = torch.norm(new_world - old_world).item()
        track.position_variance_world = max(track.position_variance_world, position_change)
        
        # Log if significant movement (potential issue)
        if position_change > 0.1:  # 10cm
            logger.warning(f"Track {track.track_id} ({track.label}) moved {position_change:.3f}m in world space!")
    
    # Update track state
    track.centroid_world = new_world
    track.bbox_3d_world = detection['bbox_world']
    track.bbox_3d_cam = detection['bbox_cam']
    track.last_centroid_cam = detection['bbox_cam']['center']
    track.last_seen_frame = frame_id
    track.lost_frames = 0
    track.total_observations += 1
    
    # Add to observation history
    track.world_observations.append(new_world.clone())
    if len(track.world_observations) > track.max_history:
        track.world_observations.pop(0)

def _create_new_track(self, detection: Dict, frame_id: int) -> int:
    """Create new track with world coordinates."""
    
    new_track_id = self.next_track_id
    self.next_track_id += 1
    
    new_track = TrackedObject3D(
        track_id=new_track_id,
        label=detection['label'],
        first_seen_frame=frame_id,
        last_seen_frame=frame_id,
        # World coordinates
        centroid_world=detection['bbox_world']['center'].clone(),
        bbox_3d_world=detection['bbox_world'],
        # Camera coordinates
        bbox_3d_cam=detection['bbox_cam'],
        last_centroid_cam=detection['bbox_cam']['center'].clone()
    )
    
    self.tracked_objects[new_track_id] = new_track
    return new_track_id
```

### 3. Debug and Validation

#### 3.1 Add World Position Logging
**File**: `mast3r_slam/geometric_3d_tracker.py`

```python
def _log_tracking_stats(self):
    """Enhanced logging with world position information."""
    
    logger.info("=== Tracking Statistics ===")
    logger.info(f"Total tracks: {len(self.tracked_objects)}")
    logger.info(f"Active tracks: {sum(1 for t in self.tracked_objects.values() if t.lost_frames == 0)}")
    
    # World position report
    logger.info("=== World Positions ===")
    for track_id, track in sorted(self.tracked_objects.items()):
        if track.centroid_world is not None:
            pos = track.centroid_world.cpu().numpy()
            logger.info(f"Track {track_id} ({track.label}): "
                       f"pos=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}], "
                       f"variance={track.position_variance_world:.3f}m, "
                       f"observations={track.total_observations}")
```

#### 3.2 Configuration Updates
**File**: `config/replica_semantic_auto_tracked_3d_global_improved_filtered.yaml`

Add world-centric parameters:
```yaml
object_tracking:
  # Enable world-centric tracking
  use_world_coordinates: true
  
  # World-space thresholds (meters)
  world_distance_thresholds:
    default: 0.3
    furniture: 0.5  # chair, table, sofa
    large: 1.0      # wall, floor, ceiling
    small: 0.2      # bottle, cup, mouse
  
  # Position stability monitoring
  max_position_variance: 0.2  # Flag tracks that move >20cm
  log_world_positions: true
  world_position_interval: 10  # Log every N frames
```

### 4. Testing Strategy

#### 4.1 Validation Script
Create `test_world_tracking.py`:
```python
def analyze_world_stability(tracking_log):
    """Analyze how stable objects are in world coordinates."""
    
    tracks = parse_tracking_log(tracking_log)
    
    for track_id, positions in tracks.items():
        positions_array = np.array(positions)
        variance = np.std(positions_array, axis=0)
        max_variance = np.max(variance)
        
        print(f"Track {track_id}: max_variance={max_variance:.3f}m")
        
        if max_variance > 0.1:
            print(f"  WARNING: High variance detected!")
```

#### 4.2 Test Scenarios
1. **Circle Test**: Camera circles around static object
2. **Return Test**: Look away and return to same view
3. **Rapid Motion**: Fast camera movements
4. **Occlusion Test**: Object temporarily hidden

### 5. Expected Results

#### 5.1 Immediate Improvements
- Static objects maintain position within 10cm variance
- Track IDs remain consistent across different viewpoints
- Reduced track fragmentation

#### 5.2 Debug Output Example
```
Track 1 (chair): pos=[2.15, 0.50, 1.20], variance=0.023m, observations=15
Track 2 (table): pos=[0.00, 0.75, 2.10], variance=0.018m, observations=20
Track 3 (monitor): pos=[0.10, 1.20, 2.00], variance=0.031m, observations=12
```

### 6. Implementation Steps

1. **Hour 1-2**: Update data structures
2. **Hour 3-4**: Implement transform methods
3. **Hour 5-6**: Replace matching logic
4. **Hour 7-8**: Testing and debugging

### 7. Success Criteria

- [ ] World position variance < 0.1m for static objects
- [ ] Track consistency > 80% across keyframes
- [ ] No performance regression (still 30+ FPS)
- [ ] Clear improvement in visual consistency

## Next Steps
After Phase 1 completion, we'll add incremental voxel grid (Phase 2) to eliminate duplicate points.