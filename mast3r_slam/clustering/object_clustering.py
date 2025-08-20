"""
Object Clustering Implementation for Semantic SLAM

This module provides data structures and core classes for clustering object instances
across keyframes to create unified global object representations.

The clustering approach combines spatial proximity with temporal consistency to handle:
- Multiple detections of the same physical object
- Objects appearing/disappearing due to occlusion
- Multiple identical objects in different locations
- Robot revisiting the same location

Author: Generated for MASt3R SLAM semantic extension
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from sklearn.cluster import DBSCAN

# Configure logging for the clustering module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class ObjectInstance:
    """
    Represents a single object detection in a keyframe.
    
    This class encapsulates all information about an individual object detection,
    including its spatial location, semantic information, and relationship to
    the keyframe where it was detected.
    
    Attributes:
        global_id: Current global ID (kf_idx * 10000 + local_id)
        local_id: Local instance ID within keyframe (1, 2, 3, ...)
        keyframe_idx: Which keyframe this detection belongs to
        label: Semantic label (e.g., "a table")
        confidence: Detection confidence score (0.0 to 1.0)
        centroid_3d: 3D centroid coordinates [x, y, z] in world coordinates
        num_points: Number of 3D points in this object instance
        point_indices: Indices of points belonging to this object in the point cloud
    """
    global_id: int
    local_id: int
    keyframe_idx: int
    label: str
    confidence: float
    centroid_3d: np.ndarray
    num_points: int
    point_indices: np.ndarray
    
    def __post_init__(self):
        """Validate object instance data after initialization."""
        if self.confidence < 0.0 or self.confidence > 1.0:
            logger.warning(f"ObjectInstance {self.global_id}: Invalid confidence {self.confidence}")
            self.confidence = np.clip(self.confidence, 0.0, 1.0)
        
        if len(self.centroid_3d) != 3:
            raise ValueError(f"ObjectInstance {self.global_id}: centroid_3d must be 3D coordinates")
            
        if self.num_points != len(self.point_indices):
            logger.warning(f"ObjectInstance {self.global_id}: num_points ({self.num_points}) "
                         f"doesn't match point_indices length ({len(self.point_indices)})")
            self.num_points = len(self.point_indices)


@dataclass
class ObjectCluster:
    """
    Represents a cluster of object instances that correspond to the same physical object.
    
    This class aggregates multiple ObjectInstance detections that have been determined
    to represent the same physical object across different keyframes.
    
    Attributes:
        cluster_id: New unified global object ID
        instances: All detections of this object across keyframes
        label: Clean label (e.g., "table_obj_001")
        avg_centroid: Average centroid across all instances
        keyframes: List of keyframes containing this object
        total_points: Total points across all instances
    """
    cluster_id: int
    instances: List[ObjectInstance]
    label: str
    avg_centroid: np.ndarray
    keyframes: List[int]
    total_points: int
    
    def __post_init__(self):
        """Compute cluster properties after initialization."""
        if len(self.instances) == 0:
            raise ValueError("ObjectCluster cannot be empty")
            
        # Validate and recompute cluster properties
        self._update_cluster_properties()
    
    def _update_cluster_properties(self):
        """Update cluster properties based on current instances."""
        if not self.instances:
            return
            
        # Compute average centroid
        centroids = np.array([inst.centroid_3d for inst in self.instances])
        self.avg_centroid = np.mean(centroids, axis=0)
        
        # Update keyframes list
        self.keyframes = sorted(list(set(inst.keyframe_idx for inst in self.instances)))
        
        # Update total points
        self.total_points = sum(inst.num_points for inst in self.instances)
    
    def add_instance(self, instance: ObjectInstance):
        """Add a new instance to this cluster and update properties."""
        self.instances.append(instance)
        self._update_cluster_properties()
    
    def get_temporal_span(self) -> Tuple[int, int]:
        """Return the temporal span (min_keyframe, max_keyframe) of this cluster."""
        if not self.keyframes:
            return (0, 0)
        return (min(self.keyframes), max(self.keyframes))
    
    def get_spatial_extent(self) -> float:
        """Return the maximum spatial extent (distance between furthest instances)."""
        if len(self.instances) < 2:
            return 0.0
            
        from .utils import compute_spatial_extent_from_centroids
        centroids = np.array([inst.centroid_3d for inst in self.instances])
        return compute_spatial_extent_from_centroids(centroids)



def create_object_cluster(instances: List[ObjectInstance], cluster_id: int = 0) -> ObjectCluster:
    """
    Create an ObjectCluster from a list of instances.
    
    Args:
        instances: List of object instances to cluster
        cluster_id: Unique cluster identifier
        
    Returns:
        ObjectCluster with computed properties
    """
    if not instances:
        raise ValueError("Cannot create cluster from empty instance list")
    
    # Use the first instance's base label for the cluster
    base_label = instances[0].label.split('_kf')[0] if '_kf' in instances[0].label else instances[0].label
    cluster_label = f"{base_label}_obj_{cluster_id:03d}" if cluster_id > 0 else base_label
    
    # Compute average centroid using consolidated function
    from .utils import compute_instance_centroid
    avg_centroid = compute_instance_centroid(instances)
    
    # Collect keyframes and total points
    keyframes = sorted(list(set(inst.keyframe_idx for inst in instances)))
    total_points = sum(inst.num_points for inst in instances)
    
    cluster = ObjectCluster(
        cluster_id=cluster_id,
        instances=instances,
        label=cluster_label,
        avg_centroid=avg_centroid,
        keyframes=keyframes,
        total_points=total_points
    )
    
    logger.debug(f"Created cluster {cluster_id} with {len(instances)} instances "
                f"spanning keyframes {min(keyframes)}-{max(keyframes)}")
    
    return cluster


def validate_temporal_consistency(cluster: ObjectCluster, temporal_threshold: int) -> List[ObjectCluster]:
    """
    Check if cluster makes sense temporally and split if needed.
    
    Args:
        cluster: ObjectCluster to validate
        temporal_threshold: Maximum keyframe gap to allow
        
    Returns:
        List of validated clusters (may split original cluster)
    """
    keyframes = sorted([inst.keyframe_idx for inst in cluster.instances])
    
    # Find large temporal gaps
    gaps = []
    for i in range(1, len(keyframes)):
        gap = keyframes[i] - keyframes[i-1]
        if gap > temporal_threshold:
            gaps.append(i)
    
    if not gaps:
        logger.debug(f"Cluster {cluster.cluster_id}: No temporal gaps found")
        return [cluster]  # No large gaps, keep as single cluster
    
    logger.info(f"Cluster {cluster.cluster_id}: Splitting at {len(gaps)} temporal gaps")
    
    # Sort instances by keyframe index for splitting
    sorted_instances = sorted(cluster.instances, key=lambda x: x.keyframe_idx)
    
    # Split cluster at temporal gaps
    split_points = [0] + gaps + [len(sorted_instances)]
    subclusters = []
    
    for i in range(len(split_points) - 1):
        start, end = split_points[i], split_points[i+1]
        subcluster_instances = sorted_instances[start:end]
        
        if subcluster_instances:  # Ensure we have instances
            subcluster = create_object_cluster(subcluster_instances, cluster.cluster_id * 100 + i)
            subclusters.append(subcluster)
    
    return subclusters


def detect_object_movement(cluster: ObjectCluster, movement_threshold: float) -> List[ObjectCluster]:
    """
    Detect if clustered object actually moved between keyframes and split if needed.
    
    Args:
        cluster: ObjectCluster to analyze for movement
        movement_threshold: Maximum movement distance (meters) to consider same object
        
    Returns:
        List of clusters (may split if significant movement detected)
    """
    if len(cluster.instances) <= 1:
        return [cluster]
    
    positions_by_kf = {}
    for instance in cluster.instances:
        positions_by_kf[instance.keyframe_idx] = instance.centroid_3d
    
    keyframes = sorted(positions_by_kf.keys())
    max_movement = 0.0
    
    # Check maximum movement between consecutive keyframes
    for i in range(1, len(keyframes)):
        movement = np.linalg.norm(
            positions_by_kf[keyframes[i]] - positions_by_kf[keyframes[i-1]]
        )
        max_movement = max(max_movement, movement)
    
    if max_movement > movement_threshold:
        logger.info(f"Cluster {cluster.cluster_id}: Significant movement detected "
                   f"({max_movement:.2f}m > {movement_threshold}m threshold)")
        # For now, split by temporal gaps when movement is detected
        # This is a conservative approach - could be enhanced with trajectory analysis
        return validate_temporal_consistency(cluster, temporal_threshold=5)
    
    logger.debug(f"Cluster {cluster.cluster_id}: Movement within threshold "
                f"({max_movement:.2f}m <= {movement_threshold}m)")
    return [cluster]  # Keep as single object


def dbscan_spatial_cluster(instances: List[ObjectInstance], 
                          spatial_threshold: float,
                          min_samples: int) -> List[ObjectCluster]:
    """
    Apply DBSCAN clustering based on spatial proximity.
    
    Args:
        instances: List of object instances to cluster spatially
        spatial_threshold: Maximum distance for DBSCAN clustering (eps parameter)
        min_samples: Minimum samples for core points in DBSCAN
        
    Returns:
        List of spatial clusters
    """
    if len(instances) == 0:
        return []
    
    if len(instances) == 1:
        return [create_object_cluster(instances, 0)]
    
    # Extract centroids for clustering
    centroids = np.array([inst.centroid_3d for inst in instances])
    
    logger.debug(f"Applying DBSCAN with eps={spatial_threshold}, min_samples={min_samples} "
                f"to {len(instances)} instances")
    
    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=spatial_threshold, min_samples=min_samples).fit(centroids)
    
    # Group instances by cluster labels
    clusters = []
    unique_labels = set(clustering.labels_)
    
    logger.debug(f"DBSCAN found {len(unique_labels)} spatial clusters")
    
    for cluster_label in unique_labels:
        if cluster_label == -1:  # Noise points become individual clusters
            noise_indices = np.where(clustering.labels_ == -1)[0]
            logger.debug(f"Creating {len(noise_indices)} individual clusters from noise points")
            
            for idx in noise_indices:
                cluster = create_object_cluster([instances[idx]], 0)
                clusters.append(cluster)
        else:
            # Regular cluster
            cluster_indices = np.where(clustering.labels_ == cluster_label)[0]
            cluster_instances = [instances[i] for i in cluster_indices]
            
            logger.debug(f"Creating spatial cluster with {len(cluster_instances)} instances")
            
            cluster = create_object_cluster(cluster_instances, 0)
            clusters.append(cluster)
    
    return clusters