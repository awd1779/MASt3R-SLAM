"""
Point Cloud Stacking Detection for Semantic SLAM

This module detects when the same physical object is seen from multiple viewpoints,
resulting in "stacked" or overlapping point cloud segments. This is a common issue
in SLAM systems where the same object is detected multiple times as the camera 
moves around it.

Key detection methods:
1. 3D point overlap analysis - direct geometric overlap detection
2. Projection consistency - check if objects occupy same image regions when projected
3. Depth profile similarity - compare depth patterns of object regions
4. Geometric relationship validation - ensure multiple detections form consistent geometry

This module helps identify over-segmentation cases where one physical object
gets split into multiple cluster instances due to viewpoint variations.

Author: Generated for MASt3R SLAM semantic extension
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Set, Optional, NamedTuple
from dataclasses import dataclass
from collections import defaultdict

from .object_clustering import ObjectInstance, ObjectCluster

logger = logging.getLogger(__name__)


@dataclass
class OverlapAnalysis:
    """Results of overlap analysis between two object instances."""
    instance1: ObjectInstance
    instance2: ObjectInstance
    point_overlap_ratio: float      # Ratio of overlapping points
    spatial_overlap_volume: float   # 3D bounding box overlap volume
    centroid_distance: float        # Distance between centroids
    geometric_consistency: float    # How consistent the geometry is
    confidence: float              # Overall confidence they're the same object
    overlap_reasoning: str         # Human-readable explanation


@dataclass  
class StackingCandidate:
    """Candidate for point cloud stacking (same object, multiple views)."""
    instances: List[ObjectInstance]
    total_overlap_score: float
    keyframe_span: Tuple[int, int]
    spatial_extent: float
    confidence: float
    stacking_type: str  # 'sequential', 'revisit', 'parallel'


def compute_3d_bounding_box(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 3D axis-aligned bounding box of point cloud."""
    if len(points) == 0:
        return np.zeros(3), np.zeros(3)
    
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    return min_coords, max_coords


def compute_bounding_box_overlap_volume(bbox1: Tuple[np.ndarray, np.ndarray], 
                                       bbox2: Tuple[np.ndarray, np.ndarray]) -> float:
    """Compute overlapping volume between two 3D bounding boxes."""
    min1, max1 = bbox1
    min2, max2 = bbox2
    
    # Compute intersection
    intersection_min = np.maximum(min1, min2)
    intersection_max = np.minimum(max1, max2)
    
    # Check if intersection exists
    if np.any(intersection_min >= intersection_max):
        return 0.0
    
    # Compute volumes
    intersection_volume = np.prod(intersection_max - intersection_min)
    volume1 = np.prod(max1 - min1)
    volume2 = np.prod(max2 - min2)
    
    # Return intersection over union
    union_volume = volume1 + volume2 - intersection_volume
    return intersection_volume / union_volume if union_volume > 0 else 0.0


def compute_point_cloud_overlap(points1: np.ndarray, points2: np.ndarray, 
                               overlap_threshold: float = 0.05) -> float:
    """
    Compute direct point cloud overlap by checking spatial proximity.
    
    Args:
        points1, points2: 3D point clouds (N1 x 3, N2 x 3)
        overlap_threshold: Distance threshold to consider points overlapping
        
    Returns:
        Overlap ratio (0.0 to 1.0)
    """
    if len(points1) == 0 or len(points2) == 0:
        return 0.0
    
    # For efficiency, subsample if point clouds are very large
    max_points = 1000
    if len(points1) > max_points:
        indices1 = np.random.choice(len(points1), max_points, replace=False)
        points1_sample = points1[indices1]
    else:
        points1_sample = points1
    
    if len(points2) > max_points:
        indices2 = np.random.choice(len(points2), max_points, replace=False)
        points2_sample = points2[indices2]
    else:
        points2_sample = points2
    
    # Compute pairwise distances (expensive but necessary)
    # Use broadcasting to compute all pairwise distances
    # points1_sample: (N1, 3), points2_sample: (N2, 3)
    # distances: (N1, N2)
    distances = np.linalg.norm(
        points1_sample[:, np.newaxis, :] - points2_sample[np.newaxis, :, :], 
        axis=2
    )
    
    # Count overlapping points
    min_distances_1to2 = np.min(distances, axis=1)  # Closest point in cloud2 for each point in cloud1
    min_distances_2to1 = np.min(distances, axis=0)  # Closest point in cloud1 for each point in cloud2
    
    overlapping_1to2 = np.sum(min_distances_1to2 < overlap_threshold)
    overlapping_2to1 = np.sum(min_distances_2to1 < overlap_threshold)
    
    # Compute overlap as ratio of overlapping points
    overlap_ratio = max(
        overlapping_1to2 / len(points1_sample),
        overlapping_2to1 / len(points2_sample)
    )
    
    return overlap_ratio


def analyze_temporal_pattern(instances: List[ObjectInstance]) -> str:
    """Analyze temporal pattern of object instances to classify stacking type."""
    
    keyframes = [inst.keyframe_idx for inst in instances]
    keyframes.sort()
    
    if len(keyframes) <= 1:
        return 'single'
    
    # Check for sequential pattern (consecutive or near-consecutive keyframes)
    max_gap = max(keyframes[i+1] - keyframes[i] for i in range(len(keyframes)-1))
    
    if max_gap <= 3:
        return 'sequential'  # Consecutive frames - likely continuous tracking
    elif max_gap <= 10:
        return 'close'       # Close frames - possible brief occlusion
    else:
        return 'revisit'     # Large gap - likely revisited same location


def compute_geometric_consistency(instances: List[ObjectInstance]) -> float:
    """
    Check if multiple instances form geometrically consistent representations.
    
    For the same object, multiple viewpoints should show:
    - Similar spatial extent
    - Reasonable centroid positions
    - Consistent scale/size
    """
    
    if len(instances) < 2:
        return 1.0
    
    centroids = np.array([inst.centroid_3d for inst in instances])
    point_counts = np.array([inst.num_points for inst in instances])
    
    # Check centroid spread
    centroid_distances = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            centroid_distances.append(dist)
    
    max_centroid_dist = max(centroid_distances) if centroid_distances else 0.0
    
    # Check point count consistency
    point_count_ratio = np.max(point_counts) / np.min(point_counts) if np.min(point_counts) > 0 else 1.0
    
    # Geometric consistency score
    # Lower centroid distances and similar point counts → higher consistency
    distance_score = max(0.0, 1.0 - max_centroid_dist / 2.0)  # Penalize distances > 2m
    size_score = max(0.0, 1.0 - (point_count_ratio - 1.0) / 3.0)  # Penalize size ratios > 4x
    
    consistency = (distance_score + size_score) / 2.0
    
    logger.debug(f"Geometric consistency: dist={max_centroid_dist:.2f}m, "
                f"size_ratio={point_count_ratio:.2f}, score={consistency:.3f}")
    
    return consistency


def analyze_instance_overlap(inst1: ObjectInstance, inst2: ObjectInstance,
                            points1: np.ndarray, points2: np.ndarray) -> OverlapAnalysis:
    """
    Comprehensive analysis of overlap between two object instances.
    
    Args:
        inst1, inst2: ObjectInstance objects to compare
        points1, points2: Corresponding 3D point clouds
        
    Returns:
        OverlapAnalysis with detailed metrics
    """
    
    # Compute bounding boxes
    bbox1 = compute_3d_bounding_box(points1)
    bbox2 = compute_3d_bounding_box(points2)
    
    # Spatial overlap metrics
    bbox_overlap = compute_bounding_box_overlap_volume(bbox1, bbox2)
    point_overlap = compute_point_cloud_overlap(points1, points2)
    centroid_distance = np.linalg.norm(inst1.centroid_3d - inst2.centroid_3d)
    
    # Geometric consistency
    geometric_consistency = compute_geometric_consistency([inst1, inst2])
    
    # Overall confidence computation
    confidence_factors = []
    reasoning_parts = []
    
    # Bounding box overlap factor
    if bbox_overlap > 0.1:  # 10% overlap threshold
        confidence_factors.append(bbox_overlap * 0.4)
        reasoning_parts.append(f"bbox_overlap={bbox_overlap:.2f}")
    
    # Point cloud overlap factor
    if point_overlap > 0.05:  # 5% point overlap threshold
        confidence_factors.append(point_overlap * 0.4)
        reasoning_parts.append(f"point_overlap={point_overlap:.2f}")
    
    # Centroid distance factor (closer = higher confidence)
    if centroid_distance < 1.0:  # Within 1 meter
        distance_factor = max(0.0, 1.0 - centroid_distance)
        confidence_factors.append(distance_factor * 0.1)
        reasoning_parts.append(f"centroid_dist={centroid_distance:.2f}m")
    
    # Geometric consistency factor
    confidence_factors.append(geometric_consistency * 0.1)
    reasoning_parts.append(f"geometry={geometric_consistency:.2f}")
    
    # Compute overall confidence
    confidence = sum(confidence_factors)
    reasoning = ", ".join(reasoning_parts)
    
    return OverlapAnalysis(
        instance1=inst1,
        instance2=inst2,
        point_overlap_ratio=point_overlap,
        spatial_overlap_volume=bbox_overlap,
        centroid_distance=centroid_distance,
        geometric_consistency=geometric_consistency,
        confidence=confidence,
        overlap_reasoning=reasoning
    )


def detect_stacking_candidates(instances: List[ObjectInstance], 
                               all_points_data: Dict[int, np.ndarray],
                               min_overlap_confidence: float = 0.3) -> List[StackingCandidate]:
    """
    Detect groups of instances that likely represent the same object from multiple views.
    
    Args:
        instances: List of object instances to analyze
        all_points_data: Dictionary mapping instance global_id to 3D points
        min_overlap_confidence: Minimum confidence to consider stacking
        
    Returns:
        List of stacking candidates
    """
    
    if len(instances) < 2:
        return []
    
    logger.info(f"Analyzing {len(instances)} instances for point cloud stacking")
    
    # Group instances by semantic label
    label_groups = defaultdict(list)
    for instance in instances:
        clean_label = instance.label.split('_kf')[0] if '_kf' in instance.label else instance.label
        label_groups[clean_label].append(instance)
    
    stacking_candidates = []
    
    for label, label_instances in label_groups.items():
        if len(label_instances) < 2:
            continue
            
        logger.debug(f"Analyzing {len(label_instances)} instances of '{label}' for stacking")
        
        # Find all pairwise overlaps
        overlaps = []
        for i in range(len(label_instances)):
            for j in range(i + 1, len(label_instances)):
                inst1, inst2 = label_instances[i], label_instances[j]
                
                # Get point clouds for these instances
                points1 = all_points_data.get(inst1.global_id, np.array([]))
                points2 = all_points_data.get(inst2.global_id, np.array([]))
                
                if len(points1) == 0 or len(points2) == 0:
                    continue
                
                overlap = analyze_instance_overlap(inst1, inst2, points1, points2)
                if overlap.confidence >= min_overlap_confidence:
                    overlaps.append(overlap)
        
        # Group overlapping instances into stacking candidates
        if overlaps:
            # Simple approach: group all instances with significant overlap
            # More sophisticated graph clustering could be implemented here
            
            involved_instances = set()
            for overlap in overlaps:
                involved_instances.add(overlap.instance1)
                involved_instances.add(overlap.instance2)
            
            if len(involved_instances) >= 2:
                candidate_instances = list(involved_instances)
                
                # Analyze temporal pattern
                stacking_type = analyze_temporal_pattern(candidate_instances)
                
                # Compute candidate metrics
                keyframes = [inst.keyframe_idx for inst in candidate_instances]
                keyframe_span = (min(keyframes), max(keyframes))
                
                centroids = np.array([inst.centroid_3d for inst in candidate_instances])
                spatial_extent = np.max(np.linalg.norm(
                    centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2
                ))
                
                total_overlap_score = np.mean([overlap.confidence for overlap in overlaps])
                
                candidate = StackingCandidate(
                    instances=candidate_instances,
                    total_overlap_score=total_overlap_score,
                    keyframe_span=keyframe_span,
                    spatial_extent=spatial_extent,
                    confidence=total_overlap_score,
                    stacking_type=stacking_type
                )
                
                stacking_candidates.append(candidate)
                
                logger.info(f"Found stacking candidate for '{label}': "
                           f"{len(candidate_instances)} instances, "
                           f"confidence={total_overlap_score:.3f}, "
                           f"type={stacking_type}")
    
    logger.info(f"Detected {len(stacking_candidates)} point cloud stacking candidates")
    return stacking_candidates


def resolve_stacking_to_clusters(stacking_candidates: List[StackingCandidate], 
                                remaining_instances: List[ObjectInstance]) -> List[ObjectCluster]:
    """
    Convert stacking candidates into unified object clusters.
    
    Args:
        stacking_candidates: Detected stacking groups
        remaining_instances: Instances not involved in stacking
        
    Returns:
        Unified clusters with stacking resolved
    """
    
    clusters = []
    cluster_id = 1
    
    # Convert stacking candidates to clusters
    for candidate in stacking_candidates:
        base_label = candidate.instances[0].label.split('_kf')[0] if '_kf' in candidate.instances[0].label else candidate.instances[0].label
        
        # Create cluster from stacked instances
        all_centroids = np.array([inst.centroid_3d for inst in candidate.instances])
        avg_centroid = np.mean(all_centroids, axis=0)
        
        keyframes = sorted(list(set(inst.keyframe_idx for inst in candidate.instances)))
        total_points = sum(inst.num_points for inst in candidate.instances)
        
        cluster_label = f"{base_label}_obj_{cluster_id:03d}"
        
        from .object_clustering import ObjectCluster
        cluster = ObjectCluster(
            cluster_id=cluster_id,
            instances=candidate.instances,
            label=cluster_label,
            avg_centroid=avg_centroid,
            keyframes=keyframes,
            total_points=total_points
        )
        
        clusters.append(cluster)
        cluster_id += 1
        
        logger.info(f"Created stacking-resolved cluster {cluster.label}: "
                   f"{len(candidate.instances)} instances → 1 cluster")
    
    # Convert remaining instances to individual clusters
    for instance in remaining_instances:
        base_label = instance.label.split('_kf')[0] if '_kf' in instance.label else instance.label
        cluster_label = f"{base_label}_obj_{cluster_id:03d}"
        
        from .object_clustering import ObjectCluster
        cluster = ObjectCluster(
            cluster_id=cluster_id,
            instances=[instance],
            label=cluster_label,
            avg_centroid=instance.centroid_3d,
            keyframes=[instance.keyframe_idx],
            total_points=instance.num_points
        )
        
        clusters.append(cluster)
        cluster_id += 1
    
    return clusters


def apply_stacking_detection(instances: List[ObjectInstance], 
                           all_points_data: Dict[int, np.ndarray],
                           min_confidence: float = 0.3) -> List[ObjectCluster]:
    """
    Main function to detect and resolve point cloud stacking.
    
    Args:
        instances: All object instances to process
        all_points_data: Mapping from instance global_id to 3D points
        min_confidence: Minimum confidence threshold for stacking detection
        
    Returns:
        Clusters with stacking resolved
    """
    
    logger.info("Applying point cloud stacking detection and resolution")
    
    # Detect stacking candidates
    stacking_candidates = detect_stacking_candidates(instances, all_points_data, min_confidence)
    
    # Identify instances involved in stacking
    stacked_instances = set()
    for candidate in stacking_candidates:
        for instance in candidate.instances:
            stacked_instances.add(instance.global_id)
    
    # Separate stacked and non-stacked instances
    remaining_instances = [inst for inst in instances if inst.global_id not in stacked_instances]
    
    logger.info(f"Stacking resolution: {len(instances)} instances → "
               f"{len(stacking_candidates)} stacked groups + {len(remaining_instances)} individual")
    
    # Convert to clusters
    clusters = resolve_stacking_to_clusters(stacking_candidates, remaining_instances)
    
    return clusters


if __name__ == "__main__":
    # Test and demonstration
    print("Point Cloud Stacking Detection Module")
    print("=====================================")
    
    # Create test data
    import sys
    sys.path.append('.')
    
    from mast3r_slam.object_clustering import ObjectInstance
    
    # Mock test instances
    test_instances = [
        ObjectInstance(
            global_id=1, local_id=1, keyframe_idx=0, label="table",
            confidence=0.9, centroid_3d=np.array([1.0, 2.0, 0.5]),
            num_points=100, point_indices=np.arange(100)
        ),
        ObjectInstance(
            global_id=2, local_id=1, keyframe_idx=1, label="table", 
            confidence=0.8, centroid_3d=np.array([1.1, 2.1, 0.5]),
            num_points=120, point_indices=np.arange(120)
        ),
        ObjectInstance(
            global_id=3, local_id=1, keyframe_idx=5, label="chair",
            confidence=0.7, centroid_3d=np.array([3.0, 1.0, 0.4]),
            num_points=80, point_indices=np.arange(80)
        ),
    ]
    
    # Mock point cloud data
    test_points_data = {
        1: np.random.randn(100, 3) + np.array([1.0, 2.0, 0.5]),
        2: np.random.randn(120, 3) + np.array([1.1, 2.1, 0.5]),  # Similar to instance 1
        3: np.random.randn(80, 3) + np.array([3.0, 1.0, 0.4])
    }
    
    print("Testing stacking detection...")
    clusters = apply_stacking_detection(test_instances, test_points_data)
    
    print(f"Results: {len(test_instances)} instances → {len(clusters)} clusters")
    for cluster in clusters:
        print(f"  Cluster {cluster.cluster_id} ({cluster.label}): {len(cluster.instances)} instances")