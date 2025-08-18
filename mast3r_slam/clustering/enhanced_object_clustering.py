"""
Enhanced Object Clustering with Advanced Over-Segmentation Solutions

EXECUTION FLOW DOCUMENTATION:
============================

This module is the MAIN ENTRY POINT for object clustering in the semantic SLAM pipeline:

1. main_semantic_tracked_3d.py
   └── calls create_dense_semantic_reconstruction_tracked()

2. dense_semantic_reconstruction_tracked_v2.py  
   └── calls enhanced_hybrid_cluster_objects() [THIS MODULE - PRIMARY FUNCTION]

3. enhanced_hybrid_cluster_objects() orchestrates the full clustering pipeline:
   ├── geometry_based_clustering_config.py 
   │   └── get_geometry_based_clustering_config() [ADAPTIVE PARAMETERS]
   │
   ├── object_clustering.py [CORE BUILDING BLOCKS]
   │   ├── dbscan_spatial_cluster() [SPATIAL CLUSTERING]
   │   ├── validate_temporal_consistency() [TEMPORAL VALIDATION]
   │   └── detect_object_movement() [MOVEMENT DETECTION]
   │
   ├── point_cloud_stacking_detection.py
   │   └── apply_stacking_detection() [MULTI-VIEW DETECTION]
   │
   └── post_clustering_merge.py
       └── post_clustering_merge() [OVER-SEGMENTATION FIXES - NOW GEOMETRY-BASED]

KEY FEATURES:
- Geometry-based adaptive parameters (NO hardcoded object types)
- Multi-viewpoint stacking detection 
- Post-processing merge using geometric properties instead of hardcoded categories
- Full integration with semantic point cloud generation

IMPORTANT: This is the ONLY clustering entry point used in production.
The basic hybrid_cluster_objects() in object_clustering.py was REMOVED (unused).

Author: Generated for MASt3R SLAM semantic extension  
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict

from .object_clustering import ObjectInstance, ObjectCluster, create_object_cluster
from .geometry_based_clustering_config import (
    analyze_clustering_effectiveness,
    compute_geometric_properties
)
from .adaptive_parameter_engine import get_adaptive_clustering_config
from .post_clustering_merge import post_clustering_merge
from .point_cloud_stacking_detection import apply_stacking_detection

logger = logging.getLogger(__name__)


def enhanced_hybrid_cluster_objects(instances: List[ObjectInstance],
                                   all_points_data: Optional[Dict[int, np.ndarray]] = None,
                                   global_config: Optional[Dict] = None,
                                   use_adaptive_params: bool = True,
                                   use_stacking_detection: bool = True,
                                   use_post_merge: bool = True,
                                   debug: bool = False) -> List[ObjectCluster]:
    """
    Enhanced hybrid clustering with advanced over-segmentation solutions.
    
    This function implements a comprehensive clustering pipeline:
    1. Adaptive parameter selection based on object types
    2. Optional point cloud stacking detection  
    3. Initial hybrid clustering (spatial + temporal)
    4. Post-clustering merge for over-segmented objects
    5. Final validation and optimization
    
    Args:
        instances: List of all object instances across keyframes
        all_points_data: Optional mapping from instance global_id to 3D point arrays
        global_config: Override clustering parameters (legacy compatibility)
        use_adaptive_params: Enable object-type specific parameters
        use_stacking_detection: Enable multi-viewpoint stacking detection
        use_post_merge: Enable post-clustering merge algorithm
        debug: Enable detailed logging and reports
        
    Returns:
        clusters: List of ObjectCluster representing unified objects
    """
    
    if len(instances) == 0:
        logger.info("No instances to cluster")
        return []
    
    logger.info(f"Starting enhanced clustering of {len(instances)} object instances")
    logger.info(f"Pipeline: adaptive_params={use_adaptive_params}, "
               f"stacking_detection={use_stacking_detection}, "
               f"post_merge={use_post_merge}")
    
    # Step 1: Determine clustering configuration
    if use_adaptive_params and all_points_data:
        # Use geometry-based adaptive configuration (no hardcoded object lists!)
        config = get_adaptive_clustering_config(instances, all_points_data)
        
        if debug:
            logger.info("GEOMETRY-BASED ADAPTIVE CLUSTERING CONFIGURATION:")
            logger.info(f"  Derived from actual point cloud geometry of {len(instances)} objects")
            logger.info(f"  No hardcoded object categories - fully adaptive!")
            
    else:
        # Use provided config or defaults
        config = global_config or {
            'spatial_threshold': 1.0,
            'temporal_threshold': 25, 
            'movement_threshold': 0.8,
            'min_samples': 1
        }
        
        if use_adaptive_params and not all_points_data:
            logger.warning("Adaptive params requested but no point cloud data available - using defaults")
    
    logger.info(f"Using clustering config: spatial={config['spatial_threshold']:.2f}m, "
               f"temporal={config['temporal_threshold']}kf, "
               f"movement={config['movement_threshold']:.2f}m")
    
    # Step 2: Optional stacking detection (early stage)
    if use_stacking_detection and all_points_data:
        logger.info("Applying point cloud stacking detection...")
        
        try:
            stacking_clusters = apply_stacking_detection(
                instances, 
                all_points_data, 
                min_confidence=0.3
            )
            
            if stacking_clusters:
                logger.info(f"Stacking detection: {len(instances)} instances → "
                           f"{len(stacking_clusters)} clusters")
                
                # If stacking detection found significant clustering, use those results
                initial_clusters = stacking_clusters
                skip_hybrid = True
            else:
                initial_clusters = []
                skip_hybrid = False
                
        except Exception as e:
            logger.warning(f"Stacking detection failed: {e}, falling back to hybrid clustering")
            initial_clusters = []
            skip_hybrid = False
    else:
        initial_clusters = []
        skip_hybrid = False
    
    # Step 3: Apply hybrid clustering (if not skipped by stacking detection)
    if not skip_hybrid:
        logger.info("Applying hybrid clustering...")
        
        # Import the original clustering functions
        from .object_clustering import (
            dbscan_spatial_cluster,
            validate_temporal_consistency,
            detect_object_movement
        )
        
        # Group instances by normalized semantic label using centralized utility
        from .utils import normalize_semantic_label
        
        label_groups = defaultdict(list)
        for instance in instances:
            normalized_label = normalize_semantic_label(instance.label)
            label_groups[normalized_label].append(instance)
        
        logger.info(f"Grouped instances into {len(label_groups)} semantic classes: "
                   f"{list(label_groups.keys())}")
        
        all_clusters = []
        cluster_id_counter = 1
        
        # Process each semantic group with adaptive parameters
        for base_label, label_instances in label_groups.items():
            logger.info(f"Processing {len(label_instances)} instances of '{base_label}'")
            
            # Use unified geometry-based parameters (no per-object hardcoded values)
            spatial_thresh = config['spatial_threshold']
            temporal_thresh = config['temporal_threshold']
            movement_thresh = config['movement_threshold']
            min_samples = config['min_samples']
            
            if use_adaptive_params:
                logger.debug(f"  Geometry-based params for '{base_label}': "
                            f"spatial={spatial_thresh:.2f}m, temporal={temporal_thresh}kf")
            
            # Apply spatial clustering
            spatial_clusters = dbscan_spatial_cluster(label_instances, spatial_thresh, min_samples)
            logger.debug(f"  Spatial clustering created {len(spatial_clusters)} clusters")
            
            # Validate each spatial cluster with temporal constraints
            for spatial_cluster in spatial_clusters:
                # Check temporal consistency and split if needed
                temporal_validated = validate_temporal_consistency(spatial_cluster, temporal_thresh)
                
                # Check for object movement within each temporal cluster
                for validated_cluster in temporal_validated:
                    movement_validated = detect_object_movement(validated_cluster, movement_thresh)
                    
                    # Assign cluster IDs and labels
                    for final_cluster in movement_validated:
                        final_cluster.cluster_id = cluster_id_counter
                        final_cluster.label = f"{base_label}_obj_{cluster_id_counter:03d}"
                        all_clusters.append(final_cluster)
                        cluster_id_counter += 1
        
        initial_clusters = all_clusters
    
    logger.info(f"Initial clustering complete: {len(instances)} instances → "
               f"{len(initial_clusters)} clusters")
    
    # Step 4: Post-clustering merge (if enabled)
    if use_post_merge and initial_clusters:
        logger.info("Applying post-clustering merge...")
        
        try:
            merged_clusters = post_clustering_merge(
                initial_clusters,
                max_iterations=3,
                min_confidence=0.5  # Lowered from 0.7 to be more permissive
            )
            
            merge_reduction = len(initial_clusters) - len(merged_clusters)
            if merge_reduction > 0:
                logger.info(f"Post-merge reduced clusters by {merge_reduction}: "
                           f"{len(initial_clusters)} → {len(merged_clusters)}")
            else:
                logger.info("Post-merge found no additional merge opportunities")
            
            final_clusters = merged_clusters
            
        except Exception as e:
            logger.warning(f"Post-clustering merge failed: {e}, using initial clusters")
            final_clusters = initial_clusters
    else:
        final_clusters = initial_clusters
    
    # Step 5: Final validation and renumbering
    logger.info("Finalizing cluster IDs and labels...")
    
    # Renumber clusters for consistency using centralized utility
    from .utils import extract_base_label
    
    for i, cluster in enumerate(final_clusters, 1):
        base_label = extract_base_label(cluster.label)
        cluster.cluster_id = i
        cluster.label = f"{base_label}_obj_{i:03d}"
    
    # Generate final statistics
    logger.info(f"Enhanced clustering complete:")
    logger.info(f"  Input instances: {len(instances)}")
    logger.info(f"  Output clusters: {len(final_clusters)}")
    logger.info(f"  Reduction ratio: {len(instances) / len(final_clusters):.2f}:1")
    
    if debug:
        # Detailed cluster analysis
        label_stats = defaultdict(int)
        instance_stats = defaultdict(int)
        
        for cluster in final_clusters:
            base_label = extract_base_label(cluster.label)
            label_stats[base_label] += 1
            instance_stats[base_label] += len(cluster.instances)
        
        logger.info("Final cluster distribution:")
        for label in sorted(label_stats.keys()):
            cluster_count = label_stats[label]
            instance_count = instance_stats[label]
            avg_instances = instance_count / cluster_count if cluster_count > 0 else 0
            
            logger.info(f"  {label}: {cluster_count} clusters "
                       f"({instance_count} instances, avg {avg_instances:.1f} per cluster)")
        
        # Geometry-based clustering effectiveness analysis
        if use_adaptive_params and all_points_data:
            try:
                analysis = analyze_clustering_effectiveness(instances, final_clusters, all_points_data)
                logger.info("Geometry-based clustering effectiveness:")
                logger.info(f"  Size distribution: {analysis['size_distribution']}")
                logger.info(f"  Avg instances per cluster: {analysis['avg_instances_per_cluster']:.1f}")
            except Exception as e:
                logger.warning(f"Could not perform clustering effectiveness analysis: {e}")
    
    # Log cluster details
    for cluster in final_clusters:
        kf_span = f"{min(cluster.keyframes)}-{max(cluster.keyframes)}" if cluster.keyframes else "empty"
        logger.debug(f"  {cluster.label}: {len(cluster.instances)} detections, "
                    f"keyframes {kf_span}, "
                    f"extent {cluster.get_spatial_extent():.2f}m")
    
    return final_clusters