#!/usr/bin/env python3
"""
Create object overlay following the same approach as semantic_overlay.py
Keeps SLAM geometry but updates colors for the queried object.
"""

import numpy as np
import argparse
from pathlib import Path
import sys
from collections import defaultdict
from difflib import get_close_matches
import json
from plyfile import PlyData, PlyElement
from scipy.spatial import KDTree


def load_ply(filename):
    """Load PLY file and extract points and colors."""
    plydata = PlyData.read(filename)
    vertex = plydata['vertex']
    
    # Extract points
    points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
    
    # Extract colors
    colors = np.vstack([vertex['red'], vertex['green'], vertex['blue']]).T
    
    return {
        'points': points,
        'colors': colors,
        'vertex_data': vertex
    }


def save_ply(filename, points, colors):
    """Save point cloud as PLY file."""
    # Ensure colors are uint8
    colors = colors.astype(np.uint8)
    
    # Create structured array
    vertex_data = np.zeros(
        len(points),
        dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ]
    )
    
    vertex_data['x'] = points[:, 0]
    vertex_data['y'] = points[:, 1]
    vertex_data['z'] = points[:, 2]
    vertex_data['red'] = colors[:, 0]
    vertex_data['green'] = colors[:, 1]
    vertex_data['blue'] = colors[:, 2]
    
    # Create PLY element and save
    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    PlyData([vertex_element]).write(filename)


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


def create_object_overlay(
    query,
    scene_path="logs/logs/tracked_3d_global_improved_filtered_FIXED/room_0",
    distance_threshold=0.005,
    highlight_color=None
):
    """
    Create overlay with queried object highlighted, following semantic_overlay.py approach.
    
    Args:
        query: Object to highlight
        scene_path: Path to scene directory
        distance_threshold: Maximum distance for point matching (meters)
        highlight_color: Optional RGB color for object (0-1 range)
    """
    scene_dir = Path(scene_path)
    
    # First extract the object if needed
    extracted_dir = scene_dir / "extracted_objects"
    clean_query = query.replace(' ', '_').replace('/', '_')
    extracted_ply = extracted_dir / f"{clean_query}.ply"
    
    if not extracted_ply.exists():
        print(f"Extracted PLY not found. Running extraction first...")
        from extract_object_robust import extract_object_robust
        result = extract_object_robust(query, str(scene_dir))
        if result is None:
            return None
    
    # Load PLY files
    slam_ply_path = scene_dir / f"{scene_dir.name}.ply"
    print(f"Loading SLAM PLY: {slam_ply_path}")
    slam_data = load_ply(str(slam_ply_path))
    
    print(f"Loading extracted object: {extracted_ply}")
    object_data = load_ply(str(extracted_ply))
    
    slam_points = slam_data['points']
    slam_colors = slam_data['colors']
    object_points = object_data['points']
    object_colors = object_data['colors']
    
    print(f"SLAM points: {len(slam_points):,}")
    print(f"Object points: {len(object_points):,}")
    
    # Build KDTree from object points
    print("Building KDTree for object points...")
    object_tree = KDTree(object_points)
    
    # Start with original SLAM colors
    final_colors = slam_colors.copy()
    
    # Find matches and update colors
    print("Matching points and updating colors...")
    distances, indices = object_tree.query(slam_points, k=1)
    
    # Update colors where matches are found
    mask = distances < distance_threshold
    matches = np.sum(mask)
    
    if highlight_color is not None:
        # Use custom highlight color
        highlight_rgb = (np.array(highlight_color) * 255).astype(np.uint8)
        final_colors[mask] = highlight_rgb
    else:
        # Use semantic colors
        final_colors[mask] = object_colors[indices[mask]]
    
    print(f"Matched {matches:,} points ({100*matches/len(slam_points):.1f}%)")
    
    # Save result
    output_dir = scene_dir / "object_overlays"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{clean_query}_overlay.ply"
    
    save_ply(str(output_path), slam_points, final_colors)
    print(f"\n✓ Saved overlay to: {output_path}")
    
    return {
        'output_path': output_path,
        'total_points': len(slam_points),
        'matched_points': matches,
        'match_percentage': 100 * matches / len(slam_points)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Create object overlay following semantic_overlay.py approach",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Creates an overlay with SLAM geometry but object colors, exactly like semantic_overlay.py

Examples:
  python create_object_overlay.py chair
  python create_object_overlay.py sofa
  python create_object_overlay.py cushion
  python create_object_overlay.py chair --color 1 0 0       # Red highlight
  python create_object_overlay.py table --threshold 0.01    # Larger match radius
  python create_object_overlay.py --list
        """)
    
    parser.add_argument("query", type=str, nargs='?', 
                       help="Object to highlight")
    parser.add_argument("--scene", type=str, 
                       default="logs/logs/tracked_3d_global_improved_filtered_FIXED/room_0",
                       help="Path to scene directory")
    parser.add_argument("--color", type=float, nargs=3, metavar=('R', 'G', 'B'),
                       help="Highlight color in 0-1 range (default: use semantic colors)")
    parser.add_argument("--threshold", type=float, default=0.005,
                       help="Distance threshold for matching (default: 0.005m)")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List available objects")
    
    args = parser.parse_args()
    
    # List mode
    if args.list or not args.query:
        scene_dir = Path(args.scene)
        tracking_json = scene_dir / f"{scene_dir.name}_semantic_dense_tracked_3d.tracking.json"
        
        if not tracking_json.exists():
            print(f"Error: {tracking_json} not found!")
            sys.exit(1)
            
        with open(tracking_json, 'r') as f:
            tracking_info = json.load(f)
        
        track_to_label = tracking_info['track_to_label']
        unique_objects = sorted(set(track_to_label.values()))
        
        print("\nAvailable objects:")
        
        # Group by base name
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
            print("\nUsage: python create_object_overlay.py <object_name>")
        sys.exit(0)
    
    # Create overlay
    result = create_object_overlay(
        args.query,
        args.scene,
        args.threshold,
        args.color
    )
    
    if result is None:
        sys.exit(1)
    
    print(f"\nStatistics:")
    print(f"  Total SLAM points: {result['total_points']:,}")
    print(f"  Matched points: {result['matched_points']:,}")
    print(f"  Match percentage: {result['match_percentage']:.1f}%")


if __name__ == "__main__":
    main()