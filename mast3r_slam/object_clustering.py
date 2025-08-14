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

# Default clustering configuration parameters
HYBRID_CLUSTERING_CONFIG = {
    "spatial_threshold": 0.6,        # Max distance (meters) for spatial clustering
    "temporal_threshold": 15,        # Max keyframe gap to consider same object
    "movement_threshold": 0.5,       # Max movement (meters) to consider same object
    "min_samples": 1,               # Minimum instances for DBSCAN cluster
    "confidence_threshold": 0.3,     # Minimum detection confidence to consider
}


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
            
        centroids = np.array([inst.centroid_3d for inst in self.instances])
        max_dist = 0.0
        
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                dist = np.linalg.norm(centroids[i] - centroids[j])
                max_dist = max(max_dist, dist)
                
        return max_dist


def compute_object_centroid(points_3d: np.ndarray, labels: np.ndarray, object_id: int) -> np.ndarray:
    """
    Compute 3D centroid of an object instance from its point cloud.
    
    Args:
        points_3d: (N, 3) array of 3D world coordinates
        labels: (N,) array of semantic labels per point
        object_id: Target object ID to compute centroid for
    
    Returns:
        centroid_3d: (3,) array [x, y, z] centroid coordinates
    """
    try:
        # Extract points belonging to this object
        object_mask = labels == object_id
        object_points = points_3d[object_mask]
        
        if len(object_points) == 0:
            logger.warning(f"No points found for object ID {object_id}")
            return np.array([0.0, 0.0, 0.0])
        
        # Compute centroid as mean of all object points
        centroid = np.mean(object_points, axis=0)
        
        logger.debug(f"Computed centroid for object {object_id}: {centroid} "
                    f"({len(object_points)} points)")
        
        return centroid
        
    except Exception as e:
        logger.error(f"Error computing centroid for object {object_id}: {e}")
        return np.array([0.0, 0.0, 0.0])


def compute_object_bbox_center(points_3d: np.ndarray, labels: np.ndarray, object_id: int) -> np.ndarray:
    """
    Alternative centroid computation using bounding box center.
    
    More robust to outliers but less precise for irregular objects.
    
    Args:
        points_3d: (N, 3) array of 3D world coordinates
        labels: (N,) array of semantic labels per point
        object_id: Target object ID to compute bounding box center for
        
    Returns:
        bbox_center: (3,) array [x, y, z] bounding box center coordinates
    """
    try:
        object_mask = labels == object_id
        object_points = points_3d[object_mask]
        
        if len(object_points) == 0:
            logger.warning(f"No points found for object ID {object_id}")
            return np.array([0.0, 0.0, 0.0])
        
        # Compute axis-aligned bounding box
        min_coords = np.min(object_points, axis=0)
        max_coords = np.max(object_points, axis=0)
        bbox_center = (min_coords + max_coords) / 2.0
        
        return bbox_center
        
    except Exception as e:
        logger.error(f"Error computing bbox center for object {object_id}: {e}")
        return np.array([0.0, 0.0, 0.0])


def spatial_similarity(centroid1: np.ndarray, centroid2: np.ndarray, max_distance: float = 1.0) -> float:
    """
    Compute spatial similarity based on 3D centroid distance.
    
    Args:
        centroid1: First object centroid [x, y, z]
        centroid2: Second object centroid [x, y, z]
        max_distance: Maximum distance to consider similar (meters)
    
    Returns:
        similarity: 1.0 if distance = 0, 0.0 if distance >= max_distance
    """
    try:
        distance = np.linalg.norm(centroid1 - centroid2)
        similarity = max(0.0, 1.0 - distance / max_distance)
        return similarity
        
    except Exception as e:
        logger.error(f"Error computing spatial similarity: {e}")
        return 0.0


def semantic_similarity(label1: str, label2: str) -> float:
    """
    Compute semantic similarity between object labels.
    
    Args:
        label1: First object label (may include keyframe info)
        label2: Second object label (may include keyframe info)
    
    Returns:
        1.0 if same object class, 0.0 otherwise
    """
    try:
        # Extract base label (remove keyframe/instance info)
        base_label1 = label1.split('_kf')[0] if '_kf' in label1 else label1
        base_label2 = label2.split('_kf')[0] if '_kf' in label2 else label2
        
        return 1.0 if base_label1 == base_label2 else 0.0
        
    except Exception as e:
        logger.error(f"Error computing semantic similarity: {e}")
        return 0.0


def temporal_continuity_score(instance1: ObjectInstance, instance2: ObjectInstance) -> float:
    """
    Score based on keyframe proximity and expected object persistence.
    
    Args:
        instance1: First object instance
        instance2: Second object instance
        
    Returns:
        Temporal continuity score between 0.0 and 1.0
    """
    kf_gap = abs(instance1.keyframe_idx - instance2.keyframe_idx)
    
    if kf_gap == 0:
        return 0.0  # Same keyframe - different objects
    elif kf_gap <= 3:
        return 1.0  # Adjacent keyframes - likely same object
    elif kf_gap <= 10:
        return 0.5  # Nearby keyframes - possible same object
    else:
        return 0.1  # Distant keyframes - unlikely same object


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
    
    # Compute average centroid
    centroids = np.array([inst.centroid_3d for inst in instances])
    avg_centroid = np.mean(centroids, axis=0)
    
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


def validate_temporal_consistency(cluster: ObjectCluster, temporal_threshold: int = 15) -> List[ObjectCluster]:
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


def detect_object_movement(cluster: ObjectCluster, movement_threshold: float = 0.5) -> List[ObjectCluster]:
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


def hybrid_cluster_objects(instances: List[ObjectInstance], 
                          spatial_threshold: float = 0.6,
                          temporal_threshold: int = 15,
                          movement_threshold: float = 0.5,
                          min_samples: int = 1) -> List[ObjectCluster]:
    """
    Hybrid clustering combining DBSCAN spatial clustering with temporal validation.
    
    This is the main clustering function that implements the complete hybrid approach:
    1. Group instances by semantic label
    2. Apply spatial clustering within each semantic group
    3. Validate clusters with temporal constraints
    4. Check for object movement and split if necessary
    
    Args:
        instances: List of all object instances across keyframes
        spatial_threshold: Maximum distance between instances for spatial clustering (meters)
        temporal_threshold: Maximum keyframe gap to consider same object
        movement_threshold: Maximum movement to consider same object (meters)
        min_samples: Minimum instances required to form a DBSCAN cluster
        
    Returns:
        clusters: List of ObjectCluster representing unified objects
    """
    if len(instances) == 0:
        logger.info("No instances to cluster")
        return []
    
    logger.info(f"Starting hybrid clustering of {len(instances)} object instances")
    logger.info(f"Parameters: spatial_threshold={spatial_threshold}m, "
               f"temporal_threshold={temporal_threshold}kf, "
               f"movement_threshold={movement_threshold}m")
    
    # Group instances by semantic label first
    label_groups = {}
    for instance in instances:
        base_label = instance.label.split('_kf')[0] if '_kf' in instance.label else instance.label
        if base_label not in label_groups:
            label_groups[base_label] = []
        label_groups[base_label].append(instance)
    
    logger.info(f"Grouped instances into {len(label_groups)} semantic classes: "
               f"{list(label_groups.keys())}")
    
    all_clusters = []
    cluster_id_counter = 1
    
    # Process each semantic group separately
    for base_label, label_instances in label_groups.items():
        logger.info(f"Processing {len(label_instances)} instances of '{base_label}'")
        
        # Step 1: Apply DBSCAN for spatial clustering
        spatial_clusters = dbscan_spatial_cluster(label_instances, spatial_threshold, min_samples)
        logger.debug(f"  Spatial clustering created {len(spatial_clusters)} clusters")
        
        # Step 2: Validate each spatial cluster with temporal constraints
        for spatial_cluster in spatial_clusters:
            # Check temporal consistency and split if needed
            temporal_validated = validate_temporal_consistency(spatial_cluster, temporal_threshold)
            
            # Step 3: Check for object movement within each temporal cluster
            for validated_cluster in temporal_validated:
                movement_validated = detect_object_movement(validated_cluster, movement_threshold)
                
                # Assign cluster IDs and labels
                for final_cluster in movement_validated:
                    final_cluster.cluster_id = cluster_id_counter
                    final_cluster.label = f"{base_label}_obj_{cluster_id_counter:03d}"
                    all_clusters.append(final_cluster)
                    cluster_id_counter += 1
    
    logger.info(f"Hybrid clustering complete: {len(instances)} instances → {len(all_clusters)} clusters")
    
    # Log clustering summary
    for cluster in all_clusters:
        kf_span = f"{min(cluster.keyframes)}-{max(cluster.keyframes)}" if cluster.keyframes else "empty"
        logger.info(f"  {cluster.label}: {len(cluster.instances)} detections, "
                   f"keyframes {kf_span}, "
                   f"extent {cluster.get_spatial_extent():.2f}m")
    
    return all_clusters


# Configuration and utility functions for integration
def get_clustering_config() -> Dict:
    """Return default clustering configuration."""
    return HYBRID_CLUSTERING_CONFIG.copy()


def set_clustering_config(config: Dict):
    """Update global clustering configuration."""
    global HYBRID_CLUSTERING_CONFIG
    HYBRID_CLUSTERING_CONFIG.update(config)
    logger.info(f"Updated clustering configuration: {HYBRID_CLUSTERING_CONFIG}")


if __name__ == "__main__":
    # Basic testing and demonstration
    logger.setLevel(logging.DEBUG)
    
    # Create some sample instances for testing
    sample_instances = [
        ObjectInstance(
            global_id=1, local_id=1, keyframe_idx=0, label="table",
            confidence=0.9, centroid_3d=np.array([1.0, 2.0, 0.5]),
            num_points=100, point_indices=np.arange(100)
        ),
        ObjectInstance(
            global_id=10001, local_id=1, keyframe_idx=1, label="table",
            confidence=0.8, centroid_3d=np.array([1.1, 2.1, 0.5]),
            num_points=120, point_indices=np.arange(120)
        ),
        ObjectInstance(
            global_id=20001, local_id=1, keyframe_idx=2, label="chair",
            confidence=0.7, centroid_3d=np.array([3.0, 1.0, 0.4]),
            num_points=80, point_indices=np.arange(80)
        ),
    ]
    
    # Test clustering
    clusters = hybrid_cluster_objects(sample_instances)
    
    print(f"\nClustering Results:")
    print(f"Input: {len(sample_instances)} instances")
    print(f"Output: {len(clusters)} clusters")
    
    for cluster in clusters:
        print(f"  Cluster {cluster.cluster_id} ({cluster.label}): "
              f"{len(cluster.instances)} instances, "
              f"keyframes {cluster.keyframes}")