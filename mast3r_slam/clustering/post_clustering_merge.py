"""
Post-Clustering Merge Algorithm for Semantic SLAM

This module implements advanced merging logic to combine over-segmented clusters
that represent the same physical object. It addresses cases where the initial
hybrid clustering creates multiple clusters for a single object due to:
- Detection gaps across keyframes  
- Spatial discontinuities in large objects
- Viewpoint-dependent detection variations
- Conservative clustering parameters

The algorithm uses semantic similarity, spatial analysis, and temporal overlap
to identify and merge clusters that should represent unified objects.

Author: Generated for MASt3R SLAM semantic extension
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
from dataclasses import dataclass

from .object_clustering import ObjectCluster, ObjectInstance
from .utils import (
    compute_3d_bounding_box,
    compute_bounding_box_overlap_volume,
    compute_euclidean_distance,
    extract_base_label
)

logger = logging.getLogger(__name__)

# Import from adaptive parameter engine directly
from .adaptive_parameter_engine import get_adaptive_merge_config


@dataclass
class MergeCandidate:
    """Represents a potential merge between two clusters."""
    cluster1: ObjectCluster
    cluster2: ObjectCluster  
    merge_confidence: float
    spatial_distance: float
    temporal_overlap: float
    centroid_distance: float
    merge_reasoning: str




def compute_temporal_overlap(cluster1: ObjectCluster, cluster2: ObjectCluster) -> float:
    """Compute temporal overlap ratio between two clusters."""
    kf1 = set(cluster1.keyframes)
    kf2 = set(cluster2.keyframes)
    
    if not kf1 or not kf2:
        return 0.0
    
    overlap = len(kf1.intersection(kf2))
    union = len(kf1.union(kf2))
    
    return overlap / union if union > 0 else 0.0


def compute_spatial_extent_overlap(cluster1: ObjectCluster, cluster2: ObjectCluster) -> float:
    """Compute spatial extent overlap for large objects that might span multiple detections."""
    
    # Get all instance centroids from both clusters
    centroids1 = np.array([inst.centroid_3d for inst in cluster1.instances])
    centroids2 = np.array([inst.centroid_3d for inst in cluster2.instances])
    
    if len(centroids1) == 0 or len(centroids2) == 0:
        return 0.0
    
    # Use utility functions for bounding box computation
    bbox1 = compute_3d_bounding_box(centroids1)
    bbox2 = compute_3d_bounding_box(centroids2)
    
    # Use utility function for overlap computation
    return compute_bounding_box_overlap_volume(bbox1, bbox2)


def get_conservative_merge_params(cluster1: ObjectCluster, cluster2: ObjectCluster) -> dict:
    """Extract and consolidate parameter retrieval logic."""
    params1 = get_adaptive_merge_config(cluster1)
    params2 = get_adaptive_merge_config(cluster2)
    
    return {
        'max_centroid_distance': min(params1['max_centroid_distance'], params2['max_centroid_distance']),
        'min_temporal_overlap': max(params1['min_temporal_overlap'], params2['min_temporal_overlap']),
        'max_spatial_distance': min(params1['max_spatial_distance'], params2['max_spatial_distance']),
        'confidence_threshold': max(params1['confidence_threshold'], params2['confidence_threshold'])
    }


def compute_geometric_consistency(cluster1: ObjectCluster, cluster2: ObjectCluster) -> float:
    """Check if two clusters could geometrically belong to the same object using geometry-based analysis."""
    
    # Get consolidated parameters
    params = get_conservative_merge_params(cluster1, cluster2)
    
    # Use the more conservative (smaller) max distance
    max_distance = params['max_centroid_distance']
    
    # Get spatial extents to determine object type
    from .geometry_based_clustering_config import compute_geometric_properties
    
    # Get points from first cluster to estimate object size
    all_points = []
    for instance in cluster1.instances:
        if hasattr(instance, 'points_3d') and instance.points_3d is not None:
            all_points.append(instance.points_3d)
    
    if all_points:
        combined_points = np.vstack(all_points)
        geo_props = compute_geometric_properties(combined_points)
        spatial_extent = geo_props.spatial_extent
    else:
        spatial_extent = 1.0  # Default medium size
    
    # For large objects (spatial_extent > 2.0), check alignment
    if spatial_extent > 2.0:
        centroids1 = np.array([inst.centroid_3d for inst in cluster1.instances])
        centroids2 = np.array([inst.centroid_3d for inst in cluster2.instances])
        
        # Compute principal axes (simplified)
        all_centroids = np.vstack([centroids1, centroids2])
        if len(all_centroids) < 3:
            return 0.8  # Not enough data, assume reasonable
            
        # Check variance in different dimensions
        variance = np.var(all_centroids, axis=0)
        
        # Large objects should have high variance in at least 2 dimensions
        sorted_var = np.sort(variance)
        if sorted_var[-2] > 0.5:  # Reasonable spread
            return 0.9
        else:
            return 0.5
    
    else:
        # For smaller objects, check if centroids are reasonably close
        centroid_dist = compute_euclidean_distance(cluster1.avg_centroid, cluster2.avg_centroid)
        
        if centroid_dist <= max_distance:
            return 1.0 - (centroid_dist / max_distance)
        else:
            return 0.0


def evaluate_merge_candidate(cluster1: ObjectCluster, cluster2: ObjectCluster) -> Optional[MergeCandidate]:
    """Evaluate if two clusters should be merged."""
    
    # Check semantic compatibility using centralized utility
    base_label1 = extract_base_label(cluster1.label)
    base_label2 = extract_base_label(cluster2.label)
    
    if base_label1 != base_label2:
        return None  # Different object types, don't merge
    
    # Get consolidated conservative parameters
    params = get_conservative_merge_params(cluster1, cluster2)
    
    # Compute metrics
    temporal_overlap = compute_temporal_overlap(cluster1, cluster2)
    centroid_distance = compute_euclidean_distance(cluster1.avg_centroid, cluster2.avg_centroid)
    spatial_extent_overlap = compute_spatial_extent_overlap(cluster1, cluster2)
    geometric_consistency = compute_geometric_consistency(cluster1, cluster2)
    
    # Check basic constraints
    if centroid_distance > params['max_centroid_distance']:
        return None
    
    if temporal_overlap < params['min_temporal_overlap'] and spatial_extent_overlap < 0.1:
        return None
    
    # Compute merge confidence
    confidence_factors = []
    reasoning_parts = []
    
    # Temporal factor
    if temporal_overlap > params['min_temporal_overlap']:
        temporal_factor = min(1.0, temporal_overlap / params['min_temporal_overlap'])
        confidence_factors.append(temporal_factor * 0.3)
        reasoning_parts.append(f"temporal_overlap={temporal_overlap:.2f}")
    
    # Spatial factor
    if centroid_distance <= params['max_centroid_distance']:
        spatial_factor = 1.0 - (centroid_distance / params['max_centroid_distance'])
        confidence_factors.append(spatial_factor * 0.4)
        reasoning_parts.append(f"centroid_dist={centroid_distance:.2f}m")
    
    # Spatial extent factor
    if spatial_extent_overlap > 0:
        confidence_factors.append(spatial_extent_overlap * 0.2)
        reasoning_parts.append(f"spatial_overlap={spatial_extent_overlap:.2f}")
    
    # Geometric consistency factor
    confidence_factors.append(geometric_consistency * 0.1)
    reasoning_parts.append(f"geometric_consistency={geometric_consistency:.2f}")
    
    merge_confidence = sum(confidence_factors)
    
    # Check if confidence meets threshold
    if merge_confidence >= params['confidence_threshold']:
        return MergeCandidate(
            cluster1=cluster1,
            cluster2=cluster2,
            merge_confidence=merge_confidence,
            spatial_distance=centroid_distance,
            temporal_overlap=temporal_overlap,
            centroid_distance=centroid_distance,
            merge_reasoning=", ".join(reasoning_parts)
        )
    
    return None


def merge_two_clusters(cluster1: ObjectCluster, cluster2: ObjectCluster, new_cluster_id: int) -> ObjectCluster:
    """Merge two clusters into a single unified cluster."""
    
    # Combine instances
    merged_instances = cluster1.instances + cluster2.instances
    
    # Create new unified label using centralized utility
    base_label = extract_base_label(cluster1.label)
    merged_label = f"{base_label}_obj_{new_cluster_id:03d}"
    
    # Compute new properties
    all_centroids = np.array([inst.centroid_3d for inst in merged_instances])
    avg_centroid = np.mean(all_centroids, axis=0)
    
    keyframes = sorted(list(set(inst.keyframe_idx for inst in merged_instances)))
    total_points = sum(inst.num_points for inst in merged_instances)
    
    merged_cluster = ObjectCluster(
        cluster_id=new_cluster_id,
        instances=merged_instances,
        label=merged_label,
        avg_centroid=avg_centroid,
        keyframes=keyframes,
        total_points=total_points
    )
    
    logger.debug(f"Merged clusters {cluster1.cluster_id} + {cluster2.cluster_id} → "
                f"{new_cluster_id} ({len(merged_instances)} instances, "
                f"{len(keyframes)} keyframes)")
    
    return merged_cluster


def find_merge_candidates(clusters: List[ObjectCluster]) -> List[MergeCandidate]:
    """Find all valid merge candidates among clusters."""
    
    candidates = []
    
    # Group clusters by base label for efficiency using centralized utility
    from .utils import extract_base_label
    
    label_groups = defaultdict(list)
    for cluster in clusters:
        base_label = extract_base_label(cluster.label)
        label_groups[base_label].append(cluster)
    
    # Find merge candidates within each label group
    for base_label, label_clusters in label_groups.items():
        if len(label_clusters) < 2:
            continue
            
        logger.debug(f"Evaluating {len(label_clusters)} clusters for '{base_label}'")
        
        for i in range(len(label_clusters)):
            for j in range(i + 1, len(label_clusters)):
                candidate = evaluate_merge_candidate(label_clusters[i], label_clusters[j])
                if candidate:
                    candidates.append(candidate)
    
    # Sort by merge confidence (highest first)
    candidates.sort(key=lambda x: x.merge_confidence, reverse=True)
    
    logger.info(f"Found {len(candidates)} potential merge candidates")
    return candidates


def execute_merges(clusters: List[ObjectCluster], candidates: List[MergeCandidate]) -> List[ObjectCluster]:
    """Execute merges in order of confidence, handling conflicts."""
    
    if not candidates:
        return clusters
    
    # Create mapping of cluster ID to cluster
    cluster_map = {cluster.cluster_id: cluster for cluster in clusters}
    merged_clusters = set()  # Track which clusters have been merged
    next_cluster_id = max(cluster.cluster_id for cluster in clusters) + 1
    
    successful_merges = 0
    
    for candidate in candidates:
        # Check if either cluster has already been merged
        if (candidate.cluster1.cluster_id in merged_clusters or 
            candidate.cluster2.cluster_id in merged_clusters):
            continue
        
        # Execute merge
        logger.info(f"Merging clusters {candidate.cluster1.cluster_id} + {candidate.cluster2.cluster_id} "
                   f"(confidence={candidate.merge_confidence:.3f})")
        logger.debug(f"  Merge reasoning: {candidate.merge_reasoning}")
        
        merged_cluster = merge_two_clusters(
            candidate.cluster1, 
            candidate.cluster2, 
            next_cluster_id
        )
        
        # Update cluster map
        del cluster_map[candidate.cluster1.cluster_id]
        del cluster_map[candidate.cluster2.cluster_id]
        cluster_map[next_cluster_id] = merged_cluster
        
        # Track merged clusters
        merged_clusters.add(candidate.cluster1.cluster_id)
        merged_clusters.add(candidate.cluster2.cluster_id)
        
        next_cluster_id += 1
        successful_merges += 1
    
    # Return updated cluster list
    final_clusters = list(cluster_map.values())
    
    logger.info(f"Executed {successful_merges} merges: {len(clusters)} → {len(final_clusters)} clusters")
    
    return final_clusters


def post_clustering_merge(clusters: List[ObjectCluster], 
                         max_iterations: int = 3,
                         min_confidence: float = 0.7) -> List[ObjectCluster]:
    """
    Main post-clustering merge function.
    
    Args:
        clusters: List of clusters from initial clustering
        max_iterations: Maximum merge iterations to prevent infinite loops
        min_confidence: Minimum confidence threshold for merges
        
    Returns:
        Merged and optimized cluster list
    """
    
    if not clusters:
        return clusters
    
    logger.info(f"Starting post-clustering merge on {len(clusters)} initial clusters")
    
    current_clusters = clusters
    total_merges = 0
    
    for iteration in range(max_iterations):
        logger.info(f"Merge iteration {iteration + 1}/{max_iterations}")
        
        # Find merge candidates
        candidates = find_merge_candidates(current_clusters)
        
        # Filter by minimum confidence
        candidates = [c for c in candidates if c.merge_confidence >= min_confidence]
        
        if not candidates:
            logger.info(f"No merge candidates found with confidence >= {min_confidence}")
            break
        
        logger.info(f"Found {len(candidates)} candidates with sufficient confidence")
        
        # Execute merges
        initial_count = len(current_clusters)
        current_clusters = execute_merges(current_clusters, candidates)
        iteration_merges = initial_count - len(current_clusters)
        total_merges += iteration_merges
        
        if iteration_merges == 0:
            logger.info("No merges executed, stopping iterations")
            break
    
    # Final statistics
    logger.info(f"Post-clustering merge complete:")
    logger.info(f"  Input clusters: {len(clusters)}")
    logger.info(f"  Output clusters: {len(current_clusters)}")
    logger.info(f"  Total merges: {total_merges}")
    logger.info(f"  Reduction: {(1 - len(current_clusters) / len(clusters)) * 100:.1f}%")
    
    # Log final cluster summary using centralized utility
    label_counts = defaultdict(int)
    for cluster in current_clusters:
        base_label = extract_base_label(cluster.label)
        label_counts[base_label] += 1
    
    logger.info("Final cluster distribution:")
    for label, count in sorted(label_counts.items()):
        logger.info(f"  {label}: {count} clusters")
    
    return current_clusters