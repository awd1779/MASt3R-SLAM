# Methods for Detecting Point Cloud Drift

## Detection Approaches

### 1. **Object Boundary Analysis**
```python
def detect_drift_by_object_spread(instance_points):
    # Compute bounding box or convex hull
    bbox = compute_bounding_box(instance_points)
    volume = bbox.width * bbox.height * bbox.depth
    
    # Check against expected object size
    if object_type == "chair":
        expected_volume = 0.5 * 0.5 * 1.0  # typical chair
        if volume > 2.0 * expected_volume:
            return True, "Object too spread out"
```

### 2. **Point Density Analysis**
```python
def detect_drift_by_density(instance_points):
    # Points should form dense clusters
    kdtree = build_kdtree(instance_points)
    
    for point in instance_points:
        # Find k nearest neighbors with same label
        neighbors = kdtree.find_k_nearest(point, k=10)
        avg_distance = compute_average_distance(point, neighbors)
        
        if avg_distance > density_threshold:
            # Point is isolated from others
            mark_as_drifted(point)
```

### 3. **Temporal Consistency Check**
```python
def detect_drift_across_frames(instance_t1, instance_t2):
    # Same object in consecutive keyframes
    centroid_t1 = compute_centroid(instance_t1)
    centroid_t2 = compute_centroid(instance_t2)
    
    motion = distance(centroid_t1, centroid_t2)
    if motion > max_plausible_motion:
        return True, "Object moved too fast"
```

### 4. **Multi-View Consistency**
```python
def detect_drift_by_reprojection(point_3d, cameras_observing):
    labels_observed = []
    
    for camera in cameras_observing:
        pixel = project_3d_to_2d(point_3d, camera)
        label = semantic_mask[camera.id][pixel]
        labels_observed.append(label)
    
    # If same 3D point projects to different labels
    if len(set(labels_observed)) > 1:
        return True, "Inconsistent semantic labels"
```

### 5. **Statistical Outlier Detection**
```python
def detect_drift_by_statistics(instance_points):
    centroid = compute_centroid(instance_points)
    distances = [distance(p, centroid) for p in instance_points]
    
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    
    outliers = []
    for i, d in enumerate(distances):
        if d > mean_dist + 3 * std_dist:  # 3-sigma rule
            outliers.append(instance_points[i])
    
    return outliers
```

## Practical Drift Indicators

### Visual Cues (What You See)
1. **Object Fragmentation**: Single object appears as multiple pieces
2. **Ghost Objects**: Duplicate objects at different locations
3. **Stretched Objects**: Objects appear elongated in one direction
4. **Floating Points**: Labeled points in empty space

### Quantitative Metrics
```python
# Compute drift score for each object
drift_metrics = {
    'spatial_spread': bbox_volume / expected_volume,
    'density_ratio': actual_density / expected_density,
    'outlier_percentage': num_outliers / total_points,
    'label_consistency': consistent_labels / total_observations
}

# Overall drift score
drift_score = weighted_sum(drift_metrics)
if drift_score > threshold:
    trigger_drift_correction()
```

## Real-Time Drift Detection Pipeline

```python
def drift_detection_pipeline(current_keyframe, semantic_data):
    drift_detected = False
    
    for instance in semantic_data.instances:
        # 1. Check spatial compactness
        if is_too_spread(instance.points):
            drift_detected = True
            
        # 2. Check temporal consistency
        if has_previous_observation(instance.id):
            if moved_too_much(instance):
                drift_detected = True
                
        # 3. Check semantic consistency
        if has_inconsistent_labels(instance):
            drift_detected = True
    
    if drift_detected:
        # Trigger corrective action
        queue_semantic_bundle_adjustment()
        adjust_optimization_weights()
```

## Drift Severity Classification

```python
def classify_drift_severity(instance):
    metrics = compute_drift_metrics(instance)
    
    if metrics.outlier_ratio < 0.1:
        return "NONE"
    elif metrics.outlier_ratio < 0.2:
        return "MINOR"  # Can be corrected by local optimization
    elif metrics.outlier_ratio < 0.4:
        return "MODERATE"  # Needs semantic constraints
    else:
        return "SEVERE"  # May need full reprocessing
```

## Key Thresholds to Set

- **Spatial spread threshold**: 2x expected object size
- **Density threshold**: 0.5x normal point density  
- **Temporal motion threshold**: 0.5m between consecutive frames
- **Outlier threshold**: 3 standard deviations from centroid
- **Label consistency threshold**: 80% same label across views