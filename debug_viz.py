#!/usr/bin/env python3
"""Debug visualization issues"""

import numpy as np
import sys
from pathlib import Path
import traceback

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from mast3r_slam.object_clustering import ObjectInstance, hybrid_cluster_objects

def main():
    print("Creating minimal test data...")
    
    instances = []
    for i in range(3):
        num_points = 100 + i*10
        instances.append(ObjectInstance(
            global_id=i+1,
            local_id=1,
            keyframe_idx=i,
            label="table",
            confidence=0.8,
            centroid_3d=np.array([1.0+i*0.1, 2.0, 0.5]),
            num_points=num_points,
            point_indices=np.arange(num_points)
        ))
    
    print(f"Created {len(instances)} instances")
    
    # Test clustering
    print("Testing clustering...")
    try:
        clusters = hybrid_cluster_objects(instances)
        print(f"Clustering successful: {len(clusters)} clusters")
    except Exception as e:
        print(f"Clustering failed: {e}")
        return
    
    # Test each visualization function individually
    print("Testing individual visualization functions...")
    
    try:
        from mast3r_slam.clustering_visualization import visualize_clusters_3d
        print("Testing 3D visualization...")
        result = visualize_clusters_3d(clusters, save_dir="debug_viz/", show_interactive=False)
        print("3D visualization successful")
    except Exception as e:
        print(f"3D visualization failed: {e}")
        traceback.print_exc()
    
    try:
        from mast3r_slam.clustering_visualization import plot_temporal_distribution
        print("Testing temporal distribution...")
        result = plot_temporal_distribution(clusters, save_dir="debug_viz/")
        print("Temporal distribution successful")
    except Exception as e:
        print(f"Temporal distribution failed: {e}")
        traceback.print_exc()
    
    try:
        from mast3r_slam.clustering_visualization import plot_spatial_distribution
        print("Testing spatial distribution...")
        result = plot_spatial_distribution(clusters, save_dir="debug_viz/")
        print("Spatial distribution successful")
    except Exception as e:
        print(f"Spatial distribution failed: {e}")
        traceback.print_exc()
    
    try:
        from mast3r_slam.clustering_visualization import plot_size_distributions
        print("Testing size distributions...")
        result = plot_size_distributions(clusters, save_dir="debug_viz/")
        print("Size distributions successful")
    except Exception as e:
        print(f"Size distributions failed: {e}")
        traceback.print_exc()
    
    try:
        from mast3r_slam.clustering_visualization import plot_semantic_breakdown
        print("Testing semantic breakdown...")
        result = plot_semantic_breakdown(clusters, save_dir="debug_viz/")
        print("Semantic breakdown successful")
    except Exception as e:
        print(f"Semantic breakdown failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()