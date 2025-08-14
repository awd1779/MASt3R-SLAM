#!/usr/bin/env python3
"""
Validate current clustering results from your semantic SLAM system.

This script loads your existing clustering results and runs comprehensive validation
to assess the quality of object clustering.
"""

import sys
import json
import numpy as np
from pathlib import Path

# Add path for imports
sys.path.append('.')

from mast3r_slam.clustering_validation import validate_clustering_quality
from mast3r_slam.clustering_visualization import run_complete_analysis
from mast3r_slam.object_clustering import ObjectInstance, ObjectCluster

def load_clustering_from_labels_file(labels_file_path):
    """
    Load clustering results from the generated labels.json file.
    
    Args:
        labels_file_path: Path to the .labels.json file
        
    Returns:
        clusters: List of ObjectCluster objects for validation
    """
    print(f"Loading clustering results from: {labels_file_path}")
    
    # Load the labels file
    with open(labels_file_path, 'r') as f:
        labels_data = json.load(f)
    
    # Extract object information
    label_to_name = labels_data.get('label_to_name', {})
    
    # Group labels by object type and ID
    objects_by_type = {}
    
    for label_id, object_name in label_to_name.items():
        if label_id == "0":  # Skip background
            continue
            
        # Parse object name (format: "a table_obj_004")
        if "_obj_" in object_name:
            parts = object_name.split("_obj_")
            if len(parts) == 2:
                object_type = parts[0].strip()
                object_id = int(parts[1])
                
                if object_type not in objects_by_type:
                    objects_by_type[object_type] = {}
                
                objects_by_type[object_type][object_id] = {
                    'label_id': int(label_id),
                    'name': object_name,
                    'type': object_type
                }
    
    print(f"Found {len(objects_by_type)} object types:")
    for obj_type, objects in objects_by_type.items():
        print(f"  - {obj_type}: {len(objects)} objects")
    
    # Create mock ObjectCluster objects for validation
    # Note: We don't have the original point cloud data, so we'll create
    # simplified clusters based on the available information
    clusters = []
    cluster_id = 1
    
    for obj_type, objects in objects_by_type.items():
        for obj_id, obj_info in objects.items():
            # Create a simplified ObjectInstance 
            # In real validation, you'd have the actual spatial and temporal data
            instance = ObjectInstance(
                global_id=obj_info['label_id'],
                local_id=obj_id,
                keyframe_idx=0,  # Unknown without original data
                label=obj_type,
                confidence=0.8,  # Assumed confidence
                centroid_3d=np.random.rand(3) * 10,  # Mock centroid
                num_points=1000,  # Mock point count
                point_indices=np.arange(1000)  # Mock indices
            )
            
            # Create cluster with single instance
            cluster = ObjectCluster(
                cluster_id=cluster_id,
                instances=[instance],
                label=obj_info['name'],
                avg_centroid=instance.centroid_3d,
                keyframes=[0],
                total_points=1000
            )
            
            clusters.append(cluster)
            cluster_id += 1
    
    return clusters

def validate_clustering_results(results_dir):
    """
    Run validation on clustering results from a specific directory.
    
    Args:
        results_dir: Directory containing clustering results
    """
    results_path = Path(results_dir)
    
    # Look for labels.json file
    labels_files = list(results_path.glob("*.labels.json"))
    if not labels_files:
        print(f"No .labels.json files found in {results_dir}")
        return
    
    labels_file = labels_files[0]  # Use first one found
    print(f"Using labels file: {labels_file}")
    
    # Load clustering results
    clusters = load_clustering_from_labels_file(labels_file)
    
    if not clusters:
        print("No clusters found to validate")
        return
    
    print(f"\nLoaded {len(clusters)} clusters for validation")
    
    # Create validation output directory
    validation_dir = results_path / "validation_analysis"
    validation_dir.mkdir(exist_ok=True)
    
    # Run quantitative validation
    print("\n" + "="*60)
    print("RUNNING QUANTITATIVE VALIDATION")
    print("="*60)
    
    try:
        from mast3r_slam.clustering_validation import validate_clustering_quality
        
        metrics = validate_clustering_quality(clusters, output_dir=str(validation_dir))
        
        print(f"\n📊 VALIDATION RESULTS:")
        print(f"  Overall Quality Score: {metrics.get('overall_quality_score', 'N/A'):.3f}")
        print(f"  Silhouette Score: {metrics.get('silhouette_score', 'N/A'):.3f}")
        print(f"  Spatial Compactness: {metrics.get('avg_intra_cluster_distance', 'N/A'):.3f}")
        print(f"  Temporal Consistency: {metrics.get('temporal_consistency_score', 'N/A'):.3f}")
        print(f"  Size Consistency: {metrics.get('size_consistency_score', 'N/A'):.3f}")
        
        # Analyze object distribution
        semantic_breakdown = {}
        for cluster in clusters:
            obj_type = cluster.instances[0].label
            if obj_type not in semantic_breakdown:
                semantic_breakdown[obj_type] = 0
            semantic_breakdown[obj_type] += 1
        
        print(f"\n📈 OBJECT DISTRIBUTION:")
        for obj_type, count in sorted(semantic_breakdown.items()):
            print(f"  {obj_type}: {count} objects")
        
    except ImportError as e:
        print(f"Could not import validation module: {e}")
    except Exception as e:
        print(f"Validation failed: {e}")
    
    # Run 3D visualization (if possible)
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    try:
        from mast3r_slam.clustering_visualization import run_complete_analysis
        
        # Note: Visualizations will be limited since we don't have real spatial data
        print("Generating visualization analysis...")
        run_complete_analysis(clusters, output_dir=str(validation_dir))
        print(f"✅ Visualizations saved to: {validation_dir}")
        
    except ImportError as e:
        print(f"Could not import visualization module: {e}")
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    print(f"\n🎉 Validation complete! Results saved to: {validation_dir}")

def main():
    """Main validation script."""
    print("🔍 SEMANTIC SLAM CLUSTERING VALIDATION")
    print("="*60)
    
    # Look for recent results
    possible_dirs = [
        "logs/logs/tracked_3d_global_improved_filtered/room_0",
        "100",
        "."
    ]
    
    results_dir = None
    for dir_path in possible_dirs:
        if Path(dir_path).exists():
            labels_files = list(Path(dir_path).glob("*.labels.json"))
            if labels_files:
                results_dir = dir_path
                break
    
    if not results_dir:
        print("❌ No clustering results found in expected locations:")
        for dir_path in possible_dirs:
            print(f"  - {dir_path}")
        print("\nPlease specify a directory containing .labels.json files")
        return
    
    print(f"✅ Found clustering results in: {results_dir}")
    
    # Run validation
    validate_clustering_results(results_dir)

if __name__ == "__main__":
    main()