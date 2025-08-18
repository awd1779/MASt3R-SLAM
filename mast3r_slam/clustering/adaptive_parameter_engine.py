"""
Adaptive Parameter Engine for Clustering Operations

This module provides unified parameter derivation for both initial clustering 
and post-clustering merge phases, eliminating duplication and ensuring consistency.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from .utils import analyze_point_cloud_geometry

logger = logging.getLogger(__name__)


@dataclass  
class ClusteringParams:
    """Parameters for initial clustering phase."""
    spatial_threshold: float      # Maximum distance for spatial clustering (meters)
    temporal_threshold: int       # Maximum keyframe gap to consider same object
    movement_threshold: float     # Maximum movement to consider same object (meters)
    min_samples: int             # Minimum instances for DBSCAN cluster
    confidence_weight: float     # Confidence weighting factor


@dataclass
class MergeParams:
    """Parameters for post-clustering merge phase."""
    max_spatial_distance: float     # Maximum spatial distance for merging
    min_temporal_overlap: float     # Minimum temporal overlap requirement
    max_centroid_distance: float    # Maximum centroid distance for merging
    confidence_threshold: float     # Confidence threshold for merge decisions


@dataclass
class UnifiedParams:
    """Combined parameters for both clustering phases."""
    clustering: ClusteringParams
    merge: MergeParams
    geometry: 'GeometricProperties'
    size_category: str


class AdaptiveParameterEngine:
    """
    Unified engine for deriving adaptive clustering and merge parameters.
    
    This engine eliminates duplication between geometry_based_clustering_config
    and post_clustering_merge by providing a single source of truth for
    parameter derivation based on object geometry.
    """
    
    def __init__(self):
        self.geometry_cache = {}  # Cache geometric analysis results
        
        # Base parameter values (optimized from previous implementations)
        self.base_params = {
            'spatial_base': 0.4,      # Base spatial threshold (40cm)
            'temporal_base': 10,      # Base temporal threshold (10 keyframes)
            'movement_base': 0.3,     # Base movement threshold (30cm)
            'merge_base': 0.8,        # Base merge distance multiplier
        }
        
        # Size category thresholds (unified from both modules)
        self.size_thresholds = {
            'large': {'volume': 2.0, 'extent': 2.0},    # > 2m³ or > 2m extent
            'medium': {'volume': 0.1, 'extent': 0.8},   # > 0.1m³ or > 0.8m extent
            # Everything else is 'small'
        }

    def analyze_object_geometry(self, points_3d: np.ndarray, cache_key: Optional[str] = None) -> 'GeometricProperties':
        """
        Single source of truth for geometric analysis with optional caching.
        
        Args:
            points_3d: (N, 3) array of 3D points
            cache_key: Optional cache key to avoid recomputation
            
        Returns:
            GeometricProperties object with comprehensive analysis
        """
        if cache_key and cache_key in self.geometry_cache:
            return self.geometry_cache[cache_key]
        
        # Use consolidated geometric analysis from geometric_utils
        geometry_dict = analyze_point_cloud_geometry(points_3d)
        
        # Convert to GeometricProperties format for compatibility
        from .geometry_based_clustering_config import GeometricProperties
        geometry = GeometricProperties(
            volume=geometry_dict['volume'],
            spatial_extent=geometry_dict['spatial_extent'],
            point_density=geometry_dict['point_density'],
            aspect_ratio=geometry_dict['aspect_ratio'],
            compactness=geometry_dict['compactness'],
            estimated_size_category=self._determine_size_category(geometry_dict)
        )
        
        if cache_key:
            self.geometry_cache[cache_key] = geometry
            
        return geometry

    def _determine_size_category(self, geometry_dict: dict) -> str:
        """Unified size categorization logic."""
        volume = geometry_dict['volume']
        extent = geometry_dict['spatial_extent']
        
        if (volume > self.size_thresholds['large']['volume'] or 
            extent > self.size_thresholds['large']['extent']):
            return 'large'
        elif (volume > self.size_thresholds['medium']['volume'] or 
              extent > self.size_thresholds['medium']['extent']):
            return 'medium'
        else:
            return 'small'

    def derive_clustering_parameters(self, geometry: 'GeometricProperties') -> ClusteringParams:
        """
        Derive parameters for initial clustering phase.
        
        Optimized combination of volume-based and extent-based scaling
        from the original geometry_based_clustering_config logic.
        """
        base_spatial = self.base_params['spatial_base']
        base_temporal = self.base_params['temporal_base']
        base_movement = self.base_params['movement_base']
        
        # Enhanced spatial threshold scaling (combines volume and extent factors)
        volume_factor = min(3.0, max(0.5, np.log10(max(geometry.volume, 0.01)) + 2.0))
        extent_factor = min(3.0, max(0.5, geometry.spatial_extent / 1.0))
        spatial_threshold = base_spatial * max(volume_factor, extent_factor)
        
        # Improved temporal threshold (considers both compactness and size)
        compactness_factor = 2.0 - geometry.compactness  # [1.0, 2.0] range
        size_factor = 1.0 + (geometry.spatial_extent / 2.0)
        temporal_threshold = int(base_temporal * compactness_factor * min(size_factor, 2.0))
        
        # Optimized movement threshold (scales with object size)
        movement_threshold = base_movement * max(1.0, geometry.spatial_extent / 1.0)
        
        # Adaptive confidence weighting
        if geometry.estimated_size_category == 'large':
            confidence_weight = 0.6  # More permissive for large objects
        elif geometry.estimated_size_category == 'small':
            confidence_weight = 0.9  # Stricter for small objects
        else:
            confidence_weight = 0.8  # Balanced for medium objects
        
        # Apply optimized bounds
        spatial_threshold = max(0.2, min(spatial_threshold, 5.0))
        temporal_threshold = max(5, min(temporal_threshold, 50))
        movement_threshold = max(0.2, min(movement_threshold, 2.0))
        confidence_weight = max(0.4, min(confidence_weight, 1.0))
        
        return ClusteringParams(
            spatial_threshold=spatial_threshold,
            temporal_threshold=temporal_threshold,
            movement_threshold=movement_threshold,
            min_samples=1,  # Keep flexible
            confidence_weight=confidence_weight
        )

    def derive_merge_parameters(self, geometry: 'GeometricProperties') -> MergeParams:
        """
        Derive parameters for post-clustering merge phase.
        
        Optimized version of the post_clustering_merge logic with smoother
        scaling and better coordination with clustering parameters.
        """
        extent = geometry.spatial_extent
        base_merge = self.base_params['merge_base']
        
        # Smooth scaling instead of hard category boundaries
        if extent <= 0.8:  # Small objects
            scale_factor = 1.0 + extent  # [1.0, 1.8]
            base_confidence = 0.6
            base_temporal_overlap = 0.4
        elif extent <= 2.0:  # Medium objects  
            scale_factor = 1.8 + (extent - 0.8) * 0.5  # [1.8, 2.4]
            base_confidence = 0.5
            base_temporal_overlap = 0.3
        else:  # Large objects
            scale_factor = 2.4 + min(extent - 2.0, 2.0) * 0.3  # [2.4, 3.0]
            base_confidence = 0.4
            base_temporal_overlap = 0.2
        
        # Coordinated parameter derivation
        max_spatial_distance = base_merge * scale_factor
        max_centroid_distance = max_spatial_distance * 1.5  # Centroid can be further
        
        # Volume-based confidence adjustment
        volume_adjustment = min(0.1, geometry.volume / 10.0)  # Up to 0.1 reduction
        confidence_threshold = max(0.3, base_confidence - volume_adjustment)
        
        # Compactness-based temporal overlap adjustment
        compactness_adjustment = (1.0 - geometry.compactness) * 0.1  # Up to 0.1 reduction
        min_temporal_overlap = max(0.1, base_temporal_overlap - compactness_adjustment)
        
        return MergeParams(
            max_spatial_distance=max_spatial_distance,
            min_temporal_overlap=min_temporal_overlap,
            max_centroid_distance=max_centroid_distance,
            confidence_threshold=confidence_threshold
        )

    def get_unified_parameters(self, points_or_cluster, cache_key: Optional[str] = None) -> UnifiedParams:
        """
        Get both clustering and merge parameters for an object.
        
        Args:
            points_or_cluster: Either np.ndarray of points or ObjectCluster
            cache_key: Optional cache key for geometry analysis
            
        Returns:
            UnifiedParams with both clustering and merge parameters
        """
        # Extract points from cluster if needed
        if hasattr(points_or_cluster, 'instances'):
            # It's an ObjectCluster
            all_points = []
            for instance in points_or_cluster.instances:
                if hasattr(instance, 'points_3d') and instance.points_3d is not None:
                    all_points.append(instance.points_3d)
            
            if all_points:
                points_3d = np.vstack(all_points)
            else:
                # Fallback: use centroids as proxy points
                points_3d = np.array([inst.centroid_3d for inst in points_or_cluster.instances])
        else:
            # It's already a point cloud
            points_3d = points_or_cluster
        
        # Analyze geometry
        geometry = self.analyze_object_geometry(points_3d, cache_key)
        
        # Derive both parameter sets
        clustering_params = self.derive_clustering_parameters(geometry)
        merge_params = self.derive_merge_parameters(geometry)
        
        return UnifiedParams(
            clustering=clustering_params,
            merge=merge_params,
            geometry=geometry,
            size_category=geometry.estimated_size_category
        )

    def get_clustering_config_dict(self, points_or_cluster, cache_key: Optional[str] = None) -> Dict:
        """
        Get clustering parameters in dictionary format for backward compatibility.
        
        This replaces get_geometry_based_clustering_config() from the old module.
        """
        unified_params = self.get_unified_parameters(points_or_cluster, cache_key)
        clustering = unified_params.clustering
        
        return {
            'spatial_threshold': clustering.spatial_threshold,
            'temporal_threshold': clustering.temporal_threshold,
            'movement_threshold': clustering.movement_threshold,
            'min_samples': clustering.min_samples,
            'confidence_weight': clustering.confidence_weight
        }

    def get_merge_config_dict(self, cluster) -> Dict:
        """
        Get merge parameters in dictionary format for backward compatibility.
        
        This replaces get_geometry_based_merge_params() from the old module.
        """
        unified_params = self.get_unified_parameters(cluster)
        merge = unified_params.merge
        
        return {
            'max_spatial_distance': merge.max_spatial_distance,
            'min_temporal_overlap': merge.min_temporal_overlap,
            'max_centroid_distance': merge.max_centroid_distance,
            'confidence_threshold': merge.confidence_threshold
        }

    def clear_cache(self):
        """Clear the geometry analysis cache."""
        self.geometry_cache.clear()

    def get_cache_stats(self) -> Dict:
        """Get cache statistics for debugging."""
        return {
            'cache_size': len(self.geometry_cache),
            'cached_keys': list(self.geometry_cache.keys())
        }


# Global instance for shared use across modules
_parameter_engine = AdaptiveParameterEngine()


def get_parameter_engine() -> AdaptiveParameterEngine:
    """Get the global parameter engine instance."""
    return _parameter_engine


# Convenience functions for backward compatibility
def get_adaptive_clustering_config(instances, all_points_data: Dict) -> Dict:
    """
    Replacement for geometry_based_clustering_config.get_geometry_based_clustering_config()
    
    This function maintains the same interface but uses the unified parameter engine.
    """
    engine = get_parameter_engine()
    
    # Combine all points for analysis (same logic as original)
    all_points = []
    for instance in instances:
        instance_points = all_points_data.get(instance.global_id)
        if instance_points is not None and len(instance_points) > 0:
            all_points.append(instance_points)
    
    if not all_points:
        # Fallback to default config
        logger.warning("No point cloud data available for adaptive clustering")
        return {
            'spatial_threshold': 1.0,
            'temporal_threshold': 25,
            'movement_threshold': 0.8,
            'min_samples': 1,
            'confidence_weight': 0.7
        }
    
    combined_points = np.vstack(all_points)
    return engine.get_clustering_config_dict(combined_points, cache_key="global_clustering")


def get_adaptive_merge_config(cluster) -> Dict:
    """
    Replacement for post_clustering_merge.get_geometry_based_merge_params()
    
    This function maintains the same interface but uses the unified parameter engine.
    """
    engine = get_parameter_engine()
    return engine.get_merge_config_dict(cluster)