"""
Consolidated geometric utilities for clustering operations.

This module provides unified implementations of geometric calculations
to eliminate duplication across clustering modules.
"""

import numpy as np
import logging
from typing import Tuple, Optional, List, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def normalize_semantic_label(label: str) -> str:
    """
    Normalize semantic labels for consistent clustering.
    
    This function provides a single source of truth for label normalization,
    eliminating duplication across multiple clustering modules.
    
    Args:
        label: Raw semantic label (e.g., "a table_kf001", "an object")
        
    Returns:
        Normalized label (e.g., "table", "object")
        
    Examples:
        >>> normalize_semantic_label("a table_kf001")
        'table'
        >>> normalize_semantic_label("an object")
        'object'
        >>> normalize_semantic_label("the chair_kf005")
        'chair'
    """
    # Remove keyframe suffixes
    clean_label = label.split('_kf')[0] if '_kf' in label else label
    
    # Remove common articles
    for prefix in ['a ', 'an ', 'the ']:
        if clean_label.startswith(prefix):
            clean_label = clean_label[len(prefix):]
            break
    
    return clean_label.strip()


def extract_base_label(cluster_label: str) -> str:
    """
    Extract base label from cluster label.
    
    Args:
        cluster_label: Cluster label (e.g., "table_obj_001")
        
    Returns:
        Base label (e.g., "table")
    """
    return cluster_label.split('_obj_')[0] if '_obj_' in cluster_label else cluster_label


@dataclass
class BoundingBox3D:
    """3D axis-aligned bounding box representation."""
    min_coords: np.ndarray  # (3,) array [x_min, y_min, z_min]
    max_coords: np.ndarray  # (3,) array [x_max, y_max, z_max]
    
    @property
    def dimensions(self) -> np.ndarray:
        """Get bounding box dimensions [width, height, depth]."""
        return self.max_coords - self.min_coords
    
    @property
    def volume(self) -> float:
        """Get bounding box volume."""
        dims = self.dimensions
        return np.prod(dims) if np.all(dims > 0) else 0.0
    
    @property
    def center(self) -> np.ndarray:
        """Get bounding box center point."""
        return (self.min_coords + self.max_coords) / 2.0
    
    @property
    def diagonal_length(self) -> float:
        """Get length of bounding box diagonal."""
        return np.linalg.norm(self.dimensions)


def compute_3d_bounding_box(points: np.ndarray) -> BoundingBox3D:
    """
    Compute 3D axis-aligned bounding box of point cloud.
    
    Args:
        points: (N, 3) array of 3D points
        
    Returns:
        BoundingBox3D object with min/max coordinates
    """
    if len(points) == 0:
        logger.warning("Empty point cloud provided for bounding box computation")
        return BoundingBox3D(np.zeros(3), np.zeros(3))
    
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    return BoundingBox3D(min_coords, max_coords)


def compute_bounding_box_overlap_volume(bbox1: BoundingBox3D, bbox2: BoundingBox3D) -> float:
    """
    Compute intersection over union (IoU) volume between two 3D bounding boxes.
    
    Args:
        bbox1: First bounding box
        bbox2: Second bounding box
        
    Returns:
        IoU ratio (0.0 to 1.0)
    """
    # Compute intersection
    intersection_min = np.maximum(bbox1.min_coords, bbox2.min_coords)
    intersection_max = np.minimum(bbox1.max_coords, bbox2.max_coords)
    
    # Check if intersection exists
    if np.any(intersection_min >= intersection_max):
        return 0.0
    
    # Compute volumes
    intersection_volume = np.prod(intersection_max - intersection_min)
    union_volume = bbox1.volume + bbox2.volume - intersection_volume
    
    return intersection_volume / union_volume if union_volume > 0 else 0.0


def compute_point_cloud_overlap(points1: np.ndarray, points2: np.ndarray, 
                               proximity_threshold: float = 0.05) -> float:
    """
    Compute overlap ratio between two point clouds based on proximity.
    
    Args:
        points1: (N1, 3) array of first point cloud
        points2: (N2, 3) array of second point cloud  
        proximity_threshold: Distance threshold for considering points overlapping
        
    Returns:
        Overlap ratio (0.0 to 1.0)
    """
    if len(points1) == 0 or len(points2) == 0:
        return 0.0
    
    # For efficiency, use a subset if point clouds are very large
    max_points = 1000
    if len(points1) > max_points:
        indices = np.random.choice(len(points1), max_points, replace=False)
        points1 = points1[indices]
    if len(points2) > max_points:
        indices = np.random.choice(len(points2), max_points, replace=False)
        points2 = points2[indices]
    
    # Compute pairwise distances using numpy (avoid scipy dependency)
    # Use broadcasting to compute all pairwise distances
    diff = points1[:, np.newaxis, :] - points2[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)
    
    # Count overlapping points
    min_distances = np.min(distances, axis=1)
    overlapping_points = np.sum(min_distances <= proximity_threshold)
    
    return overlapping_points / len(points1)


def compute_object_centroid(points_3d: np.ndarray, labels: np.ndarray, object_id: int) -> np.ndarray:
    """
    Compute 3D centroid of a specific object from labeled point cloud.
    
    Args:
        points_3d: (N, 3) array of 3D world coordinates
        labels: (N,) array of object labels per point
        object_id: Target object ID to compute centroid for
        
    Returns:
        centroid: (3,) array [x, y, z] centroid coordinates
    """
    try:
        object_mask = labels == object_id
        object_points = points_3d[object_mask]
        
        if len(object_points) == 0:
            logger.warning(f"No points found for object ID {object_id}")
            return np.array([0.0, 0.0, 0.0])
        
        centroid = np.mean(object_points, axis=0)
        return centroid
        
    except Exception as e:
        logger.error(f"Error computing centroid for object {object_id}: {e}")
        return np.array([0.0, 0.0, 0.0])


def compute_instance_centroid(instances: List) -> np.ndarray:
    """
    Compute average centroid from a list of object instances.
    
    Args:
        instances: List of ObjectInstance objects with centroid_3d attributes
        
    Returns:
        avg_centroid: (3,) array average centroid coordinates
    """
    if not instances:
        return np.array([0.0, 0.0, 0.0])
    
    centroids = np.array([inst.centroid_3d for inst in instances])
    return np.mean(centroids, axis=0)


def compute_euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two 3D points.
    
    Args:
        point1: (3,) array first point coordinates
        point2: (3,) array second point coordinates
        
    Returns:
        distance: Euclidean distance
    """
    return np.linalg.norm(point1 - point2)


def compute_spatial_extent_from_points(points: np.ndarray) -> float:
    """
    Compute maximum spatial extent of a point cloud.
    
    Args:
        points: (N, 3) array of 3D points
        
    Returns:
        extent: Maximum distance between any two points
    """
    if len(points) < 2:
        return 0.0
    
    bbox = compute_3d_bounding_box(points)
    return bbox.diagonal_length


def compute_spatial_extent_from_centroids(centroids: np.ndarray) -> float:
    """
    Compute maximum spatial extent from a set of centroids.
    
    Args:
        centroids: (N, 3) array of centroid coordinates
        
    Returns:
        extent: Maximum distance between any two centroids
    """
    if len(centroids) < 2:
        return 0.0
        
    max_dist = 0.0
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            dist = compute_euclidean_distance(centroids[i], centroids[j])
            max_dist = max(max_dist, dist)
            
    return max_dist


def compute_volume_from_dimensions(dimensions: np.ndarray) -> float:
    """
    Compute volume from bounding box dimensions.
    
    Args:
        dimensions: (3,) array [width, height, depth]
        
    Returns:
        volume: Bounding box volume
    """
    return np.prod(dimensions) if np.all(dimensions > 0) else 0.0


def compute_point_density(num_points: int, volume: float) -> float:
    """
    Compute point density (points per unit volume).
    
    Args:
        num_points: Number of points
        volume: Volume of space containing points
        
    Returns:
        density: Points per unit volume
    """
    return num_points / max(volume, 1e-6)  # Avoid division by zero


def compute_aspect_ratio(dimensions: np.ndarray) -> float:
    """
    Compute aspect ratio (longest dimension / shortest dimension).
    
    Args:
        dimensions: (3,) array [width, height, depth]
        
    Returns:
        aspect_ratio: Ratio of longest to shortest dimension
    """
    if np.all(dimensions > 0):
        sorted_dims = np.sort(dimensions)
        return sorted_dims[2] / sorted_dims[0]  # longest / shortest
    else:
        return 1.0


def compute_compactness(points: np.ndarray, bbox_center: np.ndarray, volume: float) -> float:
    """
    Compute compactness measure (how concentrated points are relative to bounding box).
    
    Args:
        points: (N, 3) array of 3D points
        bbox_center: (3,) array bounding box center
        volume: Bounding box volume
        
    Returns:
        compactness: Compactness measure (higher = more compact)
    """
    if volume <= 0 or len(points) == 0:
        return 0.0
    
    distances_to_center = np.linalg.norm(points - bbox_center, axis=1)
    avg_distance_to_center = np.mean(distances_to_center)
    
    # Normalize by theoretical average distance for uniform distribution
    theoretical_avg = 0.5 * np.cbrt(volume)  # Rough estimate
    compactness = theoretical_avg / max(avg_distance_to_center, 1e-6)
    
    return min(compactness, 1.0)  # Cap at 1.0


# Convenience function that combines multiple geometric analyses
def analyze_point_cloud_geometry(points: np.ndarray) -> dict:
    """
    Comprehensive geometric analysis of a point cloud.
    
    Args:
        points: (N, 3) array of 3D points
        
    Returns:
        Dict with geometric properties: bbox, volume, extent, density, etc.
    """
    if len(points) == 0:
        return {
            'bbox': BoundingBox3D(np.zeros(3), np.zeros(3)),
            'volume': 0.0,
            'spatial_extent': 0.0,
            'point_density': 0.0,
            'aspect_ratio': 1.0,
            'compactness': 0.0,
            'num_points': 0
        }
    
    bbox = compute_3d_bounding_box(points)
    volume = bbox.volume
    spatial_extent = bbox.diagonal_length
    point_density = compute_point_density(len(points), volume)
    aspect_ratio = compute_aspect_ratio(bbox.dimensions)
    compactness = compute_compactness(points, bbox.center, volume)
    
    return {
        'bbox': bbox,
        'volume': volume,
        'spatial_extent': spatial_extent,
        'point_density': point_density,
        'aspect_ratio': aspect_ratio,
        'compactness': compactness,
        'num_points': len(points)
    }