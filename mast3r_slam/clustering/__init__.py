"""
Object Clustering Subsystem

This package provides advanced object clustering for semantic SLAM,
including spatial clustering, temporal validation, and post-processing merge operations.

Key Components:
- ObjectInstance, ObjectCluster: Core data structures
- enhanced_hybrid_cluster_objects: Main clustering pipeline
- adaptive_parameter_engine: Unified parameter derivation
- Post-clustering merge: Over-segmentation handling
"""

# Core data structures
from .object_clustering import ObjectInstance, ObjectCluster, create_object_cluster

# Main clustering pipeline
from .enhanced_object_clustering import enhanced_hybrid_cluster_objects

# Parameter engine
from .adaptive_parameter_engine import get_parameter_engine, get_adaptive_clustering_config

# Configuration interface
from .geometry_based_clustering_config import compute_geometric_properties

# Geometric utilities (key functions)
from .utils import BoundingBox3D, compute_3d_bounding_box, analyze_point_cloud_geometry, normalize_semantic_label, extract_base_label

__all__ = [
    'ObjectInstance',
    'ObjectCluster', 
    'create_object_cluster',
    'enhanced_hybrid_cluster_objects',
    'get_parameter_engine',
    'get_adaptive_clustering_config',
    'compute_geometric_properties',
    'BoundingBox3D',
    'compute_3d_bounding_box',
    'analyze_point_cloud_geometry',
    'normalize_semantic_label',
    'extract_base_label',
]