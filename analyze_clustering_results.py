#!/usr/bin/env python3
"""
Simple analysis of clustering results without complex validation.
Focuses on practical assessment of clustering quality.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

def analyze_object_distribution(labels_file):
    """Analyze the distribution of clustered objects."""
    
    with open(labels_file, 'r') as f:
        data = json.load(f)
    
    label_to_name = data.get('label_to_name', {})
    
    # Group objects by type
    object_types = defaultdict(list)
    
    for label_id, object_name in label_to_name.items():
        if label_id == "0":  # Skip background
            continue
            
        if "_obj_" in object_name:
            # Extract object type (everything before _obj_)
            object_type = object_name.split("_obj_")[0].strip()
            object_id = object_name.split("_obj_")[1]
            
            object_types[object_type].append({
                'label_id': label_id,
                'name': object_name,
                'obj_id': object_id
            })
    
    return object_types

def assess_clustering_quality(object_types):
    """Assess clustering quality based on object distribution."""
    
    print("🔍 CLUSTERING QUALITY ASSESSMENT")
    print("="*60)
    
    total_objects = sum(len(objects) for objects in object_types.values())
    print(f"📊 Total clustered objects: {total_objects}")
    print(f"📊 Unique object types: {len(object_types)}")
    
    # Analyze each object type
    print(f"\n📈 OBJECT TYPE ANALYSIS:")
    
    large_surface_types = ['wall', 'ceiling', 'floor', 'door', 'window']
    furniture_types = ['table', 'chair', 'sofa', 'cabinet', 'stool']
    small_objects = ['book', 'plate', 'vase', 'candle', 'switch', 'wall plug']
    
    issues_found = []
    
    for obj_type, objects in sorted(object_types.items()):
        count = len(objects)
        
        # Assess if clustering makes sense
        assessment = "✅ Good"
        issue = None
        
        # Check for potential over-segmentation
        obj_type_clean = obj_type.replace('a ', '').replace('an ', '')
        
        if obj_type_clean in large_surface_types and count > 2:
            assessment = "⚠️  Possibly over-segmented"
            issue = f"Large surfaces ({obj_type_clean}) shouldn't have {count} separate instances"
            issues_found.append(issue)
        
        elif obj_type_clean in furniture_types and count > 4:
            assessment = "⚠️  Check if legitimate"
            issue = f"Many {obj_type_clean} instances ({count}) - verify if scene has multiple items"
        
        elif obj_type_clean in small_objects and count > 3:
            assessment = "⚠️  Possibly over-segmented"
            issue = f"Small objects ({obj_type_clean}) often get over-segmented ({count} instances)"
            issues_found.append(issue)
        
        print(f"  {obj_type}: {count} objects {assessment}")
        if issue:
            print(f"    → {issue}")
    
    # Overall assessment
    print(f"\n🎯 OVERALL ASSESSMENT:")
    
    if len(issues_found) == 0:
        print("✅ Clustering appears to be working well!")
        print("   All object counts seem reasonable for the scene.")
    
    elif len(issues_found) <= 3:
        print("⚠️  Minor clustering issues detected:")
        for issue in issues_found:
            print(f"   • {issue}")
        print("\n💡 RECOMMENDATIONS:")
        print("   • Try increasing spatial_threshold to 0.8-1.0m")
        print("   • Consider increasing temporal_threshold to 20-25 keyframes")
    
    else:
        print("🔴 Significant over-segmentation detected:")
        for issue in issues_found:
            print(f"   • {issue}")
        print("\n💡 RECOMMENDATIONS:")
        print("   • Increase spatial_threshold from 0.6m to 1.0-1.5m")
        print("   • Increase temporal_threshold from 15 to 25-30 keyframes")
        print("   • Consider increasing movement_threshold to 1.0m")
    
    return issues_found

def suggest_parameter_improvements(issues_found):
    """Suggest specific parameter improvements based on issues."""
    
    if not issues_found:
        return
    
    print(f"\n⚙️  PARAMETER TUNING SUGGESTIONS:")
    print("="*60)
    
    print("Current clustering config:")
    print("""
clustering_config = {
    "spatial_threshold": 0.6,    # Current: 60cm
    "temporal_threshold": 15,    # Current: 15 keyframes
    "movement_threshold": 0.5,   # Current: 50cm
    "min_samples": 1
}""")
    
    print("\nRecommended improvements:")
    print("""
clustering_config = {
    "spatial_threshold": 1.0,    # Increase to 100cm (less strict)
    "temporal_threshold": 25,    # Increase to 25 keyframes (more continuity)
    "movement_threshold": 0.8,   # Increase to 80cm (allow more movement)
    "min_samples": 1
}""")
    
    print("\n📝 How to apply changes:")
    print("1. Edit main_semantic_tracked_3d.py")
    print("2. Update the clustering_config dictionary (around line 682)")
    print("3. Re-run semantic SLAM to generate new clustering")
    print("4. Compare results with this analysis")

def main():
    """Main analysis function."""
    
    # Find labels file
    possible_paths = [
        "logs/logs/tracked_3d_global_improved_filtered/room_0/room_0_semantic_dense_tracked_3d.labels.json",
        "100/semantic_dense_tracked_3d.labels.json"
    ]
    
    labels_file = None
    for path in possible_paths:
        if Path(path).exists():
            labels_file = path
            break
    
    if not labels_file:
        print("❌ No labels.json file found in expected locations")
        return
    
    print(f"📁 Analyzing: {labels_file}")
    
    # Analyze object distribution
    object_types = analyze_object_distribution(labels_file)
    
    # Assess clustering quality
    issues_found = assess_clustering_quality(object_types)
    
    # Suggest improvements
    suggest_parameter_improvements(issues_found)
    
    print(f"\n✅ Analysis complete!")

if __name__ == "__main__":
    main()