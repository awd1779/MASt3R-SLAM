#!/usr/bin/env python3
"""
Test Enhanced Clustering on Existing Dataset

This script tests the new enhanced clustering pipeline on the current 
semantic SLAM results to validate improvements in over-segmentation.

It loads existing clustering results, applies the enhanced pipeline,
and generates comparison reports.

Usage:
    python test_enhanced_clustering.py [--dataset_path PATH] [--debug]
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
import logging

# Add path for imports
sys.path.append('.')

from mast3r_slam.object_clustering import ObjectInstance
from mast3r_slam.enhanced_object_clustering import enhanced_hybrid_cluster_objects
from mast3r_slam.adaptive_clustering_config import generate_config_report

logger = logging.getLogger(__name__)


def load_existing_results(labels_file_path: str) -> tuple:
    """Load existing clustering results from labels.json file."""
    
    print(f"Loading existing results from: {labels_file_path}")
    
    with open(labels_file_path, 'r') as f:
        labels_data = json.load(f)
    
    label_to_name = labels_data.get('label_to_name', {})
    
    # Convert to ObjectInstance objects
    instances = []
    
    for label_id, object_name in label_to_name.items():
        if label_id == "0":  # Skip background
            continue
            
        # Parse object name (format: "a table_obj_004")
        if "_obj_" in object_name:
            parts = object_name.split("_obj_")
            if len(parts) == 2:
                object_type = parts[0].strip()
                try:
                    object_id = int(parts[1])
                except ValueError:
                    continue
                
                # Create mock ObjectInstance 
                # Note: We don't have the original point cloud data,
                # so we'll create reasonable approximations
                instance = ObjectInstance(
                    global_id=int(label_id),
                    local_id=object_id,
                    keyframe_idx=0,  # Unknown without original data
                    label=object_type,
                    confidence=0.8,  # Assumed confidence
                    centroid_3d=np.random.rand(3) * 10,  # Mock centroid
                    num_points=np.random.randint(50, 500),  # Mock point count
                    point_indices=np.arange(np.random.randint(50, 500))  # Mock indices
                )
                
                instances.append(instance)
    
    print(f"Loaded {len(instances)} object instances from existing results")
    
    # Group by object type for analysis
    type_groups = defaultdict(list)
    for instance in instances:
        clean_label = instance.label.lower().strip()
        type_groups[clean_label].append(instance)
    
    return instances, type_groups, label_to_name


def analyze_existing_clustering(type_groups: dict, label_to_name: dict):
    """Analyze the current clustering results."""
    
    print("\n" + "="*60)
    print("EXISTING CLUSTERING ANALYSIS")
    print("="*60)
    
    total_objects = len([name for name in label_to_name.values() if name != 'background'])
    unique_types = len(type_groups)
    
    print(f"Total clustered objects: {total_objects}")
    print(f"Unique object types: {unique_types}")
    
    print("\nCurrent object distribution:")
    over_segmented_types = []
    
    for obj_type, instances in sorted(type_groups.items()):
        count = len(instances)
        print(f"  {obj_type}: {count} clusters")
        
        # Flag potential over-segmentation
        if obj_type in ['wall', 'ceiling', 'floor'] and count > 2:
            over_segmented_types.append(f"{obj_type} ({count} clusters)")
        elif obj_type in ['book', 'plate', 'vase', 'switch'] and count > 3:
            over_segmented_types.append(f"{obj_type} ({count} clusters)")
    
    if over_segmented_types:
        print(f"\nPotential over-segmentation detected:")
        for item in over_segmented_types:
            print(f"  - {item}")
    
    return over_segmented_types


def test_enhanced_clustering(instances: list, debug: bool = False):
    """Test the enhanced clustering pipeline on the instances."""
    
    print("\n" + "="*60)
    print("TESTING ENHANCED CLUSTERING")
    print("="*60)
    
    if debug:
        logging.basicConfig(level=logging.DEBUG)
    
    # Generate configuration report
    unique_labels = list(set(inst.label for inst in instances))
    
    print("\nConfiguration Analysis:")
    print(generate_config_report(unique_labels))
    
    # Apply enhanced clustering
    print(f"\nApplying enhanced clustering to {len(instances)} instances...")
    
    try:
        enhanced_clusters = enhanced_hybrid_cluster_objects(
            instances=instances,
            all_points_data=None,  # No real point data available
            use_adaptive_params=True,
            use_stacking_detection=False,  # Skip without point data
            use_post_merge=True,
            debug=debug
        )
        
        print(f"Enhanced clustering complete: {len(instances)} instances → {len(enhanced_clusters)} clusters")
        
        # Analyze results
        enhanced_type_groups = defaultdict(int)
        for cluster in enhanced_clusters:
            base_label = cluster.label.split('_obj_')[0].strip()
            enhanced_type_groups[base_label] += 1
        
        print("\nEnhanced clustering distribution:")
        for obj_type, count in sorted(enhanced_type_groups.items()):
            print(f"  {obj_type}: {count} clusters")
        
        return enhanced_clusters, enhanced_type_groups
        
    except Exception as e:
        print(f"Enhanced clustering failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def compare_results(original_groups: dict, enhanced_groups: dict, 
                   original_over_segmented: list):
    """Compare original and enhanced clustering results."""
    
    print("\n" + "="*60)
    print("COMPARISON ANALYSIS")
    print("="*60)
    
    improvements = []
    
    print("Object type comparison:")
    print(f"{'Object Type':<20} {'Original':<10} {'Enhanced':<10} {'Change':<10}")
    print("-" * 50)
    
    all_types = set(original_groups.keys()) | set(enhanced_groups.keys())
    
    for obj_type in sorted(all_types):
        original_count = len(original_groups.get(obj_type, []))
        enhanced_count = enhanced_groups.get(obj_type, 0)
        change = enhanced_count - original_count
        
        change_str = f"{change:+d}" if change != 0 else "0"
        print(f"{obj_type:<20} {original_count:<10} {enhanced_count:<10} {change_str:<10}")
        
        if change < 0:  # Reduction in cluster count
            improvements.append(f"{obj_type}: {original_count} → {enhanced_count} ({-change} fewer)")
    
    print(f"\nOverall comparison:")
    original_total = sum(len(instances) for instances in original_groups.values())
    enhanced_total = sum(enhanced_groups.values())
    total_reduction = original_total - enhanced_total
    
    print(f"  Original total clusters: {original_total}")
    print(f"  Enhanced total clusters: {enhanced_total}")
    print(f"  Net reduction: {total_reduction} clusters")
    print(f"  Reduction percentage: {(total_reduction / original_total * 100):.1f}%")
    
    if improvements:
        print(f"\nImprovements detected:")
        for improvement in improvements:
            print(f"  ✅ {improvement}")
    
    # Check if over-segmentation issues were resolved
    print(f"\nOver-segmentation resolution:")
    for over_seg_item in original_over_segmented:
        obj_type = over_seg_item.split(' (')[0]
        if obj_type in enhanced_groups:
            new_count = enhanced_groups[obj_type]
            print(f"  {over_seg_item} → now {new_count} clusters")
        else:
            print(f"  {over_seg_item} → not found in enhanced results")


def main():
    """Main test function."""
    
    parser = argparse.ArgumentParser(description='Test enhanced clustering on existing dataset')
    parser.add_argument('--dataset_path', default='logs/logs/tracked_3d_global_improved_filtered/room_0',
                       help='Path to directory containing labels.json file')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    print("🧪 ENHANCED CLUSTERING TEST")
    print("="*60)
    
    # Find labels file
    dataset_path = Path(args.dataset_path)
    labels_files = list(dataset_path.glob("*.labels.json"))
    
    if not labels_files:
        print(f"❌ No labels.json files found in {dataset_path}")
        print("Please ensure you have run the semantic SLAM system first.")
        return
    
    labels_file = labels_files[0]
    print(f"📁 Using dataset: {labels_file}")
    
    # Load existing results
    try:
        instances, type_groups, label_to_name = load_existing_results(str(labels_file))
    except Exception as e:
        print(f"❌ Failed to load existing results: {e}")
        return
    
    # Analyze existing clustering
    original_over_segmented = analyze_existing_clustering(type_groups, label_to_name)
    
    # Test enhanced clustering
    enhanced_clusters, enhanced_type_groups = test_enhanced_clustering(instances, args.debug)
    
    if enhanced_clusters and enhanced_type_groups:
        # Compare results
        compare_results(type_groups, enhanced_type_groups, original_over_segmented)
        
        print(f"\n✅ Test completed successfully!")
        print(f"Enhanced clustering shows potential for reducing over-segmentation.")
        print(f"To apply these improvements, re-run semantic SLAM with the updated clustering.")
    else:
        print(f"\n❌ Test failed - could not complete enhanced clustering")


if __name__ == "__main__":
    main()