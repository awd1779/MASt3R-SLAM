"""
Geometry-Based Adaptive Clustering Configuration

This module provides truly adaptive clustering parameters based on actual
geometric properties of objects rather than hardcoded object lists.

UPDATED: Now delegates to adaptive_parameter_engine.py for unified parameter derivation.

Key principles:
- NO hardcoded object names or categories
- Parameters derived from actual bounding box volume, point density, spatial extent
- Automatic adaptation to any object type
- Unified with merge parameter derivation

Author: Generated for MASt3R SLAM semantic extension
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GeometricProperties:
    """Geometric properties of an object derived from point cloud analysis."""
    volume: float                    # 3D bounding box volume (cubic meters)
    spatial_extent: float           # Maximum dimension (meters)
    point_density: float            # Points per cubic meter
    aspect_ratio: float             # Length/width ratio (elongated vs compact)
    compactness: float              # How concentrated the points are
    estimated_size_category: str    # 'large', 'medium', 'small' (auto-detected)




def compute_geometric_properties(points_3d: np.ndarray) -> GeometricProperties:
    """
    Compute geometric properties from a point cloud.
    
    This is a compatibility wrapper that uses the unified geometric analysis
    from utils.py and converts to the local GeometricProperties format.
    
    Args:
        points_3d: (N, 3) array of 3D points
        
    Returns:
        GeometricProperties with all computed metrics
    """
    from .utils import analyze_point_cloud_geometry
    
    if len(points_3d) == 0:
        logger.warning("Empty point cloud provided for geometric analysis")
        return GeometricProperties(
            volume=0.0, spatial_extent=0.0, point_density=0.0,
            aspect_ratio=1.0, compactness=0.0, estimated_size_category='small'
        )
    
    # Use unified geometric analysis
    geometry_dict = analyze_point_cloud_geometry(points_3d)
    
    # Convert to local format
    return GeometricProperties(
        volume=geometry_dict['volume'],
        spatial_extent=geometry_dict['spatial_extent'],
        point_density=geometry_dict['point_density'],
        aspect_ratio=geometry_dict['aspect_ratio'],
        compactness=geometry_dict['compactness'],
        estimated_size_category=geometry_dict['size_category']
    )




def analyze_clustering_effectiveness(instances, clusters, all_points_data: Dict) -> Dict:
    """
    Analyze clustering effectiveness and provide metrics.
    
    Args:
        instances: Original object instances
        clusters: Resulting object clusters
        all_points_data: Point cloud data for instances
        
    Returns:
        Dictionary with clustering effectiveness metrics
    """
    
    if not clusters:
        return {
            'total_instances': len(instances),
            'total_clusters': 0,
            'clustering_ratio': 0.0,
            'avg_cluster_size': 0.0,
            'temporal_span_avg': 0.0,
            'spatial_extent_avg': 0.0
        }
    
    # Basic metrics
    total_instances = len(instances)
    total_clusters = len(clusters)
    clustering_ratio = total_instances / max(total_clusters, 1)
    
    # Cluster size statistics
    cluster_sizes = [len(cluster.instances) for cluster in clusters]
    avg_cluster_size = np.mean(cluster_sizes)
    
    # Temporal span analysis
    temporal_spans = []
    for cluster in clusters:
        if cluster.keyframes:
            span = max(cluster.keyframes) - min(cluster.keyframes) + 1
            temporal_spans.append(span)
    
    temporal_span_avg = np.mean(temporal_spans) if temporal_spans else 0.0
    
    # Spatial extent analysis  
    spatial_extents = [cluster.get_spatial_extent() for cluster in clusters]
    spatial_extent_avg = np.mean(spatial_extents)
    
    # Quality metrics
    large_clusters = sum(1 for size in cluster_sizes if size >= 3)
    quality_score = large_clusters / max(total_clusters, 1)
    
    # Geometric diversity
    if all_points_data:
        volumes = []
        for cluster in clusters:
            cluster_points = []
            for instance in cluster.instances:
                if instance.global_id in all_points_data:
                    points = all_points_data[instance.global_id]
                    if len(points) > 0:
                        cluster_points.append(points)
            
            if cluster_points:
                combined_points = np.vstack(cluster_points)
                geometry = compute_geometric_properties(combined_points)
                volumes.append(geometry.volume)
        
        volume_diversity = np.std(volumes) if volumes else 0.0
    else:
        volume_diversity = 0.0
    
    effectiveness = {
        'total_instances': total_instances,
        'total_clusters': total_clusters,
        'clustering_ratio': clustering_ratio,
        'avg_cluster_size': avg_cluster_size,
        'temporal_span_avg': temporal_span_avg,
        'spatial_extent_avg': spatial_extent_avg,
        'quality_score': quality_score,
        'volume_diversity': volume_diversity,
        'large_clusters_count': large_clusters
    }
    
    logger.info(f"Clustering effectiveness: {total_instances} instances → {total_clusters} clusters "
               f"(ratio: {clustering_ratio:.1f}, quality: {quality_score:.2f})")
    
    return effectiveness


# Testing and validation
if __name__ == "__main__":
    # Test geometric property computation
    print("Testing Geometry-Based Clustering Configuration")
    print("=" * 50)
    
    # Create test point clouds
    test_cases = [
        ("Large wall-like object", np.random.rand(1000, 3) * [5, 3, 0.1] + [0, 0, 1]),
        ("Medium table-like object", np.random.rand(500, 3) * [1.5, 1, 0.1] + [2, 2, 0.8]),
        ("Small book-like object", np.random.rand(100, 3) * [0.3, 0.2, 0.05] + [1, 1, 1]),
    ]
    
    for name, points in test_cases:
        print(f"\n{name}:")
        geometry = compute_geometric_properties(points)
        params = derive_adaptive_parameters(geometry)
        
        print(f"  Volume: {geometry.volume:.3f}m³")
        print(f"  Spatial extent: {geometry.spatial_extent:.2f}m")
        print(f"  Category: {geometry.estimated_size_category}")
        print(f"  Spatial threshold: {params.spatial_threshold:.2f}m")
        print(f"  Temporal threshold: {params.temporal_threshold}kf")
        print(f"  Movement threshold: {params.movement_threshold:.2f}m")