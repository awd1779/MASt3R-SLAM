#!/usr/bin/env python3
"""
Robust object extraction from semantic point cloud.
Handles typos, exact matching, and better point extraction.
"""

import numpy as np
import open3d as o3d
import json
import argparse
from pathlib import Path
import sys
from collections import defaultdict
from difflib import get_close_matches


def find_best_match(query, available_objects):
    """Find best matching object name, handling typos."""
    query_lower = query.lower()
    
    # First try exact match
    for obj in available_objects:
        if query_lower in obj.lower():
            return [obj]
    
    # Try fuzzy matching for typos
    matches = get_close_matches(query_lower, 
                               [obj.lower() for obj in available_objects], 
                               n=3, cutoff=0.6)
    
    if matches:
        # Return the original case versions
        return [obj for obj in available_objects if obj.lower() in matches]
    
    return []


def get_track_color_mapping(semantic_pcd, tracking_info):
    """Create a more accurate mapping between tracks and colors."""
    colors = np.asarray(semantic_pcd.colors)
    points = np.asarray(semantic_pcd.points)
    
    # Get label to name mapping
    label_to_name = tracking_info['label_to_name']
    track_to_label = tracking_info['track_to_label']
    
    # Group points by color
    color_to_indices = defaultdict(list)
    for i, color in enumerate(colors):
        color_key = tuple((color * 255).astype(int))
        color_to_indices[color_key].append(i)
    
    # Sort by size
    sorted_colors = sorted(color_to_indices.items(), 
                          key=lambda x: len(x[1]), 
                          reverse=True)
    
    # Create mapping based on track coloring algorithm
    track_color_map = {}
    
    # The coloring uses golden ratio hashing
    golden_ratio = 0.618033988749895
    
    for track_id in track_to_label.keys():
        # Calculate expected hue
        hue = (int(track_id) * golden_ratio) % 1.0
        
        # Convert HSV to RGB (matching the algorithm in the code)
        saturation = 0.8
        value = 0.9
        
        c = value * saturation
        x = c * (1 - abs((hue * 6) % 2 - 1))
        m = value - c
        
        if hue < 1/6:
            r, g, b = c, x, 0
        elif hue < 2/6:
            r, g, b = x, c, 0
        elif hue < 3/6:
            r, g, b = 0, c, x
        elif hue < 4/6:
            r, g, b = 0, x, c
        elif hue < 5/6:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        # Convert to 0-255
        expected_color = (
            int((r + m) * 255),
            int((g + m) * 255),
            int((b + m) * 255)
        )
        
        # Find closest color in our point cloud
        min_dist = float('inf')
        best_match = None
        
        for color_tuple, indices in sorted_colors[:50]:  # Check top 50 colors
            dist = np.sqrt(sum((a - b)**2 for a, b in zip(color_tuple, expected_color)))
            if dist < min_dist and len(indices) > 100:  # Minimum points threshold
                min_dist = dist
                best_match = color_tuple
        
        if best_match and min_dist < 50:  # Color distance threshold
            track_color_map[int(track_id)] = best_match
    
    return track_color_map, color_to_indices


def extract_object_robust(query, scene_path="logs/logs/tracked_3d_global_improved_filtered_FIXED/room_0"):
    """Extract object with robust matching and color mapping."""
    
    scene_dir = Path(scene_path)
    
    # Load data
    semantic_ply = scene_dir / f"{scene_dir.name}_semantic_dense_tracked_3d.ply"
    print(f"Loading: {semantic_ply}")
    semantic_pcd = o3d.io.read_point_cloud(str(semantic_ply))
    
    tracking_json = scene_dir / f"{scene_dir.name}_semantic_dense_tracked_3d.tracking.json"
    with open(tracking_json, 'r') as f:
        tracking_info = json.load(f)
    
    points = np.asarray(semantic_pcd.points)
    colors = np.asarray(semantic_pcd.colors)
    
    print(f"Total points: {len(points):,}")
    
    # Get available objects
    track_to_label = tracking_info['track_to_label']
    available_objects = sorted(set(track_to_label.values()))
    
    # Find best match for query
    matches = find_best_match(query, available_objects)
    
    if not matches:
        print(f"\nNo objects found matching '{query}'")
        print("\nDid you mean one of these?")
        # Show suggestions
        suggestions = get_close_matches(query.lower(), 
                                      [obj.lower() for obj in available_objects], 
                                      n=5, cutoff=0.4)
        for sugg in suggestions:
            orig = [obj for obj in available_objects if obj.lower() == sugg][0]
            print(f"  - {orig}")
        
        print("\nAll available objects:")
        for obj in available_objects:
            print(f"  - {obj}")
        return None
    
    # If multiple matches (e.g., "a cushion" and "cushion"), merge them
    all_tracks = []
    for match in matches:
        for track_id, label in track_to_label.items():
            if label == match:
                all_tracks.append((int(track_id), label))
    
    print(f"\nFound {len(all_tracks)} instance(s) matching '{query}':")
    for track_id, label in all_tracks:
        print(f"  - Track {track_id}: {label}")
    
    # Get color mapping
    print("\nBuilding color-track mapping...")
    track_color_map, color_to_indices = get_track_color_mapping(semantic_pcd, tracking_info)
    
    # Extract points
    extracted_indices = []
    tracks_found = 0
    
    for track_id, label in all_tracks:
        if track_id in track_color_map:
            color = track_color_map[track_id]
            if color in color_to_indices:
                indices = color_to_indices[color]
                extracted_indices.extend(indices)
                tracks_found += 1
                print(f"  Extracted track {track_id}: {len(indices):,} points")
    
    if not extracted_indices:
        print("\nWarning: Couldn't map tracks to colors accurately.")
        print("Falling back to heuristic method...")
        
        # Fallback: extract likely candidates based on size
        sorted_colors = sorted(color_to_indices.items(), 
                             key=lambda x: len(x[1]), 
                             reverse=True)
        
        # Skip very large groups (background)
        candidates = []
        threshold = len(points) * 0.15
        
        for color, indices in sorted_colors:
            if 100 < len(indices) < threshold:
                candidates.append((color, indices))
        
        # Take as many candidates as we have tracks
        for i in range(min(len(all_tracks), len(candidates))):
            color, indices = candidates[i]
            extracted_indices.extend(indices)
            print(f"  Extracted color group {i+1}: {len(indices):,} points")
    
    if not extracted_indices:
        print("\nNo points could be extracted!")
        return None
    
    # Remove duplicates
    extracted_indices = list(set(extracted_indices))
    
    # Create extracted point cloud
    extracted_points = points[extracted_indices]
    extracted_colors = colors[extracted_indices]
    
    extracted_pcd = o3d.geometry.PointCloud()
    extracted_pcd.points = o3d.utility.Vector3dVector(extracted_points)
    extracted_pcd.colors = o3d.utility.Vector3dVector(extracted_colors)
    
    # Save
    output_dir = scene_dir / "extracted_objects"
    output_dir.mkdir(exist_ok=True)
    
    # Clean filename
    clean_query = query.replace(' ', '_').replace('/', '_')
    output_file = output_dir / f"{clean_query}.ply"
    o3d.io.write_point_cloud(str(output_file), extracted_pcd)
    
    print(f"\n✓ Saved {len(extracted_points):,} points to: {output_file}")
    
    return str(output_file)


def main():
    parser = argparse.ArgumentParser(
        description="Robust object extraction from semantic point cloud",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_object_robust.py chair
  python extract_object_robust.py sofa  
  python extract_object_robust.py cusion     # handles typos
  python extract_object_robust.py cushion    # finds both "a cushion" and "cushion"
  python extract_object_robust.py --list
        """)
    
    parser.add_argument("query", type=str, nargs='?', 
                       help="Object to extract")
    parser.add_argument("--scene", type=str, 
                       default="logs/logs/tracked_3d_global_improved_filtered_FIXED/room_0",
                       help="Path to scene directory")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List available objects")
    
    args = parser.parse_args()
    
    # List mode
    if args.list or not args.query:
        scene_dir = Path(args.scene)
        tracking_json = scene_dir / f"{scene_dir.name}_semantic_dense_tracked_3d.tracking.json"
        
        with open(tracking_json, 'r') as f:
            tracking_info = json.load(f)
        
        track_to_label = tracking_info['track_to_label']
        unique_objects = sorted(set(track_to_label.values()))
        
        print("\nAvailable objects:")
        
        # Group by base name to show duplicates
        base_names = defaultdict(list)
        for obj in unique_objects:
            base = obj.replace('a ', '').strip()
            base_names[base].append(obj)
        
        for base, variants in sorted(base_names.items()):
            if len(variants) > 1:
                print(f"  - {base} (appears as: {', '.join(variants)})")
            else:
                print(f"  - {variants[0]}")
        
        if not args.query:
            print("\nUsage: python extract_object_robust.py <object_name>")
        sys.exit(0)
    
    # Extract object
    result = extract_object_robust(args.query, args.scene)
    
    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()