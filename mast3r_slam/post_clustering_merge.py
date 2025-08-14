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

logger = logging.getLogger(__name__)

# Object type classifications for adaptive merging
OBJECT_TYPE_CATEGORIES = {
    'large_surfaces': [
        'wall', 'ceiling', 'floor', 'door', 'window', 
        'blinds', 'vent', 'a wall', 'a ceiling', 'a floor', 
        'a door', 'a window', 'a blinds', 'a vent'
    ],
    'furniture': [
        'table', 'chair', 'sofa', 'cabinet', 'stool', 'rug',
        'a table', 'a chair', 'a sofa', 'a cabinet', 'a stool', 'a rug'
    ],
    'small_objects': [
        'book', 'plate', 'vase', 'candle', 'switch', 'wall plug',
        'pot', 'cushion', 'basket', 'lamp', 'a book', 'a plate',
        'a vase', 'a candle', 'a switch', 'a wall plug', 'a pot',
        'a cushion', 'a basket', 'a lamp'
    ],
    'plants': [
        'indoor plant', 'plant stand', 'a indoor plant', 'a plant stand'
    ]
}

# Adaptive merge parameters by object category
MERGE_PARAMETERS = {
    'large_surfaces': {
        'max_spatial_distance': 3.0,      # Large surfaces can be far apart
        'min_temporal_overlap': 0.2,      # Low overlap requirement
        'max_centroid_distance': 4.0,     # Very large centroid distance allowed
        'confidence_threshold': 0.7       # High confidence for merging
    },
    'furniture': {
        'max_spatial_distance': 1.5,      # Medium spatial distance
        'min_temporal_overlap': 0.3,      # Medium overlap requirement  
        'max_centroid_distance': 2.0,     # Medium centroid distance
        'confidence_threshold': 0.8       # High confidence
    },
    'small_objects': {
        'max_spatial_distance': 0.8,      # Small spatial distance
        'min_temporal_overlap': 0.4,      # Higher overlap requirement
        'max_centroid_distance': 1.0,     # Small centroid distance
        'confidence_threshold': 0.9       # Very high confidence
    },
    'plants': {
        'max_spatial_distance': 1.0,      # Medium-small distance
        'min_temporal_overlap': 0.3,      # Medium overlap
        'max_centroid_distance': 1.5,     # Medium distance
        'confidence_threshold': 0.8       # High confidence
    },
    'default': {
        'max_spatial_distance': 1.2,      # Default conservative
        'min_temporal_overlap': 0.3,      # Default overlap
        'max_centroid_distance': 1.8,     # Default distance
        'confidence_threshold': 0.8       # Default confidence
    }
}


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


def get_object_category(label: str) -> str:
    """Determine object category for adaptive merging parameters."""
    clean_label = label.lower().strip()
    
    for category, keywords in OBJECT_TYPE_CATEGORIES.items():
        for keyword in keywords:
            if keyword.lower() in clean_label:
                return category
    
    return 'default'


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
    
    # Compute bounding boxes
    min1, max1 = np.min(centroids1, axis=0), np.max(centroids1, axis=0)
    min2, max2 = np.min(centroids2, axis=0), np.max(centroids2, axis=0)
    
    # Compute intersection volume
    intersection_min = np.maximum(min1, min2)
    intersection_max = np.minimum(max1, max2)
    
    # Check if there's any intersection
    if np.any(intersection_min >= intersection_max):
        return 0.0
    
    # Compute volumes
    intersection_volume = np.prod(intersection_max - intersection_min)
    volume1 = np.prod(max1 - min1)
    volume2 = np.prod(max2 - min2)
    union_volume = volume1 + volume2 - intersection_volume
    
    return intersection_volume / union_volume if union_volume > 0 else 0.0


def compute_geometric_consistency(cluster1: ObjectCluster, cluster2: ObjectCluster) -> float:
    """Check if two clusters could geometrically belong to the same object."""
    
    # For large surfaces, check if centroids form reasonable geometric patterns
    category = get_object_category(cluster1.label.split('_obj_')[0])
    
    if category == 'large_surfaces':
        # For large surfaces, check alignment (walls should be roughly aligned)
        centroids1 = np.array([inst.centroid_3d for inst in cluster1.instances])
        centroids2 = np.array([inst.centroid_3d for inst in cluster2.instances])
        
        # Compute principal axes (very simplified)
        all_centroids = np.vstack([centroids1, centroids2])
        if len(all_centroids) < 3:
            return 0.8  # Not enough data, assume reasonable
            
        # Check variance in different dimensions
        variance = np.var(all_centroids, axis=0)
        
        # Large surfaces should have high variance in at least 2 dimensions
        sorted_var = np.sort(variance)
        if sorted_var[-2] > 0.5:  # Reasonable spread
            return 0.9
        else:
            return 0.5
    
    else:
        # For other objects, check if centroids are reasonably close
        centroid_dist = np.linalg.norm(cluster1.avg_centroid - cluster2.avg_centroid)
        params = MERGE_PARAMETERS.get(category, MERGE_PARAMETERS['default'])
        
        if centroid_dist <= params['max_centroid_distance']:
            return 1.0 - (centroid_dist / params['max_centroid_distance'])
        else:
            return 0.0


def evaluate_merge_candidate(cluster1: ObjectCluster, cluster2: ObjectCluster) -> Optional[MergeCandidate]:
    """Evaluate if two clusters should be merged."""
    
    # Check semantic compatibility
    base_label1 = cluster1.label.split('_obj_')[0]
    base_label2 = cluster2.label.split('_obj_')[0]
    
    if base_label1 != base_label2:
        return None  # Different object types, don't merge
    
    # Get adaptive parameters
    category = get_object_category(base_label1)
    params = MERGE_PARAMETERS.get(category, MERGE_PARAMETERS['default'])
    
    # Compute metrics
    temporal_overlap = compute_temporal_overlap(cluster1, cluster2)
    centroid_distance = np.linalg.norm(cluster1.avg_centroid - cluster2.avg_centroid)
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
            merge_reasoning=f"category={category}, " + ", ".join(reasoning_parts)
        )
    
    return None


def merge_two_clusters(cluster1: ObjectCluster, cluster2: ObjectCluster, new_cluster_id: int) -> ObjectCluster:
    """Merge two clusters into a single unified cluster."""
    
    # Combine instances
    merged_instances = cluster1.instances + cluster2.instances
    
    # Create new unified label
    base_label = cluster1.label.split('_obj_')[0]
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
    
    # Group clusters by base label for efficiency
    label_groups = defaultdict(list)
    for cluster in clusters:
        base_label = cluster.label.split('_obj_')[0]
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
    
    # Log final cluster summary
    label_counts = defaultdict(int)
    for cluster in current_clusters:
        base_label = cluster.label.split('_obj_')[0]
        label_counts[base_label] += 1
    
    logger.info("Final cluster distribution:")
    for label, count in sorted(label_counts.items()):
        logger.info(f"  {label}: {count} clusters")
    
    return current_clusters