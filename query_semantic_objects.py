#!/usr/bin/env python3
"""
Semantic Object Query Tool for MASt3R-SLAM
Query and extract specific objects from semantic point clouds
"""

import argparse
import json
import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class SemanticObjectQuery:
    def __init__(self, ply_path: str, semantic_map_path: Optional[str] = None):
        """
        Initialize the semantic object query tool.
        
        Args:
            ply_path: Path to the PLY file
            semantic_map_path: Path to the semantic map JSON
        """
        self.ply_path = Path(ply_path)
        self.ply_data = PlyData.read(ply_path)
        self.vertices = self.ply_data['vertex']
        
        # Extract arrays
        self.points = np.vstack([
            self.vertices['x'],
            self.vertices['y'],
            self.vertices['z']
        ]).T
        
        self.colors = np.vstack([
            self.vertices['red'],
            self.vertices['green'],
            self.vertices['blue']
        ]).T
        
        # Extract semantic IDs
        self.semantic_ids = None
        vertex_dtype = self.vertices.data.dtype
        for prop in ['semantic_id', 'quality']:
            if prop in vertex_dtype.names:
                self.semantic_ids = self.vertices.data[prop]
                break
        
        if self.semantic_ids is None:
            raise ValueError("No semantic labels found in PLY file")
        
        # Load semantic map
        self.semantic_map = self._load_semantic_map(semantic_map_path)
        
        # Build reverse map (name -> ids)
        self.name_to_ids = {}
        for label_id, name in self.semantic_map.items():
            if name not in self.name_to_ids:
                self.name_to_ids[name] = []
            self.name_to_ids[name].append(label_id)
    
    def _load_semantic_map(self, semantic_map_path: Optional[str]) -> Dict[int, str]:
        """Load semantic map from JSON or PLY comments."""
        semantic_map = {0: "background"}
        
        # Try JSON first
        if semantic_map_path and Path(semantic_map_path).exists():
            with open(semantic_map_path, 'r') as f:
                json_map = json.load(f)
                semantic_map = {int(k): v for k, v in json_map.items()}
        else:
            # Extract from PLY comments
            if hasattr(self.ply_data, 'comments'):
                in_map = False
                for comment in self.ply_data.comments:
                    if comment == 'label_map_start':
                        in_map = True
                    elif comment == 'label_map_end':
                        in_map = False
                    elif in_map and comment.startswith('label_id'):
                        parts = comment.split(':', 1)
                        if len(parts) == 2:
                            label_id = int(parts[0].split()[-1])
                            semantic_map[label_id] = parts[1].strip()
        
        return semantic_map
    
    def get_object_list(self) -> List[str]:
        """Get list of all object types in the scene."""
        unique_ids = np.unique(self.semantic_ids)
        objects = []
        for label_id in unique_ids:
            if label_id in self.semantic_map:
                objects.append(self.semantic_map[label_id])
        return sorted(set(objects))
    
    def query_by_name(self, object_name: str) -> Dict:
        """
        Query objects by name (supports partial matching).
        
        Returns:
            Dictionary with query results
        """
        # Find matching objects
        matches = []
        for name, ids in self.name_to_ids.items():
            if object_name.lower() in name.lower():
                matches.append((name, ids))
        
        if not matches:
            return {
                'found': False,
                'query': object_name,
                'message': f"No objects found matching '{object_name}'"
            }
        
        # Collect all points
        all_mask = np.zeros(len(self.points), dtype=bool)
        instance_info = []
        
        for name, label_ids in matches:
            for label_id in label_ids:
                mask = self.semantic_ids == label_id
                if mask.any():
                    all_mask |= mask
                    points = self.points[mask]
                    
                    # Calculate bounding box
                    bbox_min = points.min(axis=0)
                    bbox_max = points.max(axis=0)
                    center = (bbox_min + bbox_max) / 2
                    size = bbox_max - bbox_min
                    
                    instance_info.append({
                        'id': label_id,
                        'name': name,
                        'num_points': mask.sum(),
                        'center': center.tolist(),
                        'size': size.tolist(),
                        'bbox_min': bbox_min.tolist(),
                        'bbox_max': bbox_max.tolist()
                    })
        
        return {
            'found': True,
            'query': object_name,
            'num_instances': len(instance_info),
            'total_points': all_mask.sum(),
            'instances': instance_info,
            'mask': all_mask,
            'points': self.points[all_mask],
            'colors': self.colors[all_mask]
        }
    
    def query_by_region(self, center: np.ndarray, radius: float) -> Dict:
        """
        Query objects within a spherical region.
        
        Args:
            center: 3D center point
            radius: Search radius
            
        Returns:
            Dictionary with objects found in region
        """
        # Find points within radius
        distances = np.linalg.norm(self.points - center, axis=1)
        region_mask = distances <= radius
        
        # Find unique objects in region
        region_ids = np.unique(self.semantic_ids[region_mask])
        
        objects = []
        for label_id in region_ids:
            if label_id == 0:  # Skip background
                continue
            
            # Get object mask within region
            object_mask = (self.semantic_ids == label_id) & region_mask
            
            if object_mask.any():
                objects.append({
                    'id': label_id,
                    'name': self.semantic_map.get(label_id, f"unknown_{label_id}"),
                    'num_points': object_mask.sum(),
                    'percentage': object_mask.sum() / region_mask.sum() * 100
                })
        
        return {
            'center': center.tolist(),
            'radius': radius,
            'total_points': region_mask.sum(),
            'num_objects': len(objects),
            'objects': objects,
            'mask': region_mask
        }
    
    def get_instance(self, instance_id: int) -> Dict:
        """Get a specific instance by ID."""
        mask = self.semantic_ids == instance_id
        
        if not mask.any():
            return {
                'found': False,
                'id': instance_id,
                'message': f"No instance with ID {instance_id}"
            }
        
        points = self.points[mask]
        colors = self.colors[mask]
        
        # Calculate properties
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        center = (bbox_min + bbox_max) / 2
        size = bbox_max - bbox_min
        
        return {
            'found': True,
            'id': instance_id,
            'name': self.semantic_map.get(instance_id, f"unknown_{instance_id}"),
            'num_points': mask.sum(),
            'center': center.tolist(),
            'size': size.tolist(),
            'bbox_min': bbox_min.tolist(),
            'bbox_max': bbox_max.tolist(),
            'points': points,
            'colors': colors,
            'mask': mask
        }
    
    def export_query_result(self, result: Dict, output_path: str):
        """Export query result to PLY file."""
        if 'points' not in result or len(result['points']) == 0:
            print("No points to export")
            return
        
        points = result['points']
        colors = result['colors']
        
        # Create PLY data
        vertex = np.zeros(len(points), dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ])
        
        vertex['x'] = points[:, 0]
        vertex['y'] = points[:, 1]
        vertex['z'] = points[:, 2]
        vertex['red'] = colors[:, 0]
        vertex['green'] = colors[:, 1]
        vertex['blue'] = colors[:, 2]
        
        # Save PLY
        el = PlyElement.describe(vertex, 'vertex')
        PlyData([el]).write(output_path)
        print(f"Exported {len(points)} points to {output_path}")
    
    def visualize_query(self, result: Dict, show_bbox: bool = True):
        """Visualize query result with matplotlib."""
        if 'points' not in result or len(result['points']) == 0:
            print("No points to visualize")
            return
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        points = result['points']
        colors = result['colors'] / 255.0
        
        # Plot points
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                  c=colors, s=1, alpha=0.6)
        
        # Plot bounding boxes if available
        if show_bbox and 'instances' in result:
            for instance in result['instances']:
                self._plot_bbox(ax, instance['bbox_min'], instance['bbox_max'])
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"Query Result: {result.get('query', 'Region')}")
        
        plt.show()
    
    def _plot_bbox(self, ax, bbox_min, bbox_max):
        """Plot 3D bounding box."""
        # Define the 8 corners of the box
        corners = [
            [bbox_min[0], bbox_min[1], bbox_min[2]],
            [bbox_max[0], bbox_min[1], bbox_min[2]],
            [bbox_max[0], bbox_max[1], bbox_min[2]],
            [bbox_min[0], bbox_max[1], bbox_min[2]],
            [bbox_min[0], bbox_min[1], bbox_max[2]],
            [bbox_max[0], bbox_min[1], bbox_max[2]],
            [bbox_max[0], bbox_max[1], bbox_max[2]],
            [bbox_min[0], bbox_max[1], bbox_max[2]]
        ]
        
        # Define the 12 edges of the box
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
        ]
        
        # Plot edges
        for edge in edges:
            points = [corners[edge[0]], corners[edge[1]]]
            ax.plot3D(*zip(*points), 'r-', linewidth=2)


def main():
    parser = argparse.ArgumentParser(description="Query semantic objects in MASt3R-SLAM point clouds")
    parser.add_argument("ply_file", help="Path to PLY file")
    parser.add_argument("--semantic-map", help="Path to semantic map JSON")
    
    # Query modes
    parser.add_argument("--list", action="store_true", help="List all object types")
    parser.add_argument("--query", help="Query by object name (e.g., 'chair')")
    parser.add_argument("--instance", type=int, help="Get specific instance by ID")
    parser.add_argument("--region", nargs=4, type=float, metavar=('X', 'Y', 'Z', 'R'),
                       help="Query region (center xyz and radius)")
    
    # Output options
    parser.add_argument("--export", help="Export query result to PLY file")
    parser.add_argument("--visualize", action="store_true", help="Visualize query result")
    parser.add_argument("--json", help="Save query result as JSON")
    
    args = parser.parse_args()
    
    # Create query tool
    query_tool = SemanticObjectQuery(args.ply_file, args.semantic_map)
    
    # Handle different query modes
    result = None
    
    if args.list:
        objects = query_tool.get_object_list()
        print("\n=== Objects in Scene ===")
        for obj in objects:
            count = sum(1 for name, _ in query_tool.name_to_ids.items() if obj == name)
            print(f"  - {obj} ({count} instance{'s' if count > 1 else ''})")
        return
    
    elif args.query:
        result = query_tool.query_by_name(args.query)
        if result['found']:
            print(f"\nFound {result['num_instances']} instance(s) of '{args.query}'")
            print(f"Total points: {result['total_points']:,}")
            for inst in result['instances']:
                print(f"\n  Instance ID {inst['id']} ({inst['name']}):")
                print(f"    Points: {inst['num_points']:,}")
                print(f"    Center: ({inst['center'][0]:.2f}, {inst['center'][1]:.2f}, {inst['center'][2]:.2f})")
                print(f"    Size: ({inst['size'][0]:.2f}, {inst['size'][1]:.2f}, {inst['size'][2]:.2f})")
        else:
            print(result['message'])
    
    elif args.instance is not None:
        result = query_tool.get_instance(args.instance)
        if result['found']:
            print(f"\nInstance ID {result['id']} ({result['name']}):")
            print(f"  Points: {result['num_points']:,}")
            print(f"  Center: ({result['center'][0]:.2f}, {result['center'][1]:.2f}, {result['center'][2]:.2f})")
            print(f"  Size: ({result['size'][0]:.2f}, {result['size'][1]:.2f}, {result['size'][2]:.2f})")
        else:
            print(result['message'])
    
    elif args.region:
        center = np.array(args.region[:3])
        radius = args.region[3]
        result = query_tool.query_by_region(center, radius)
        print(f"\nRegion query at ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}) with radius {radius:.2f}:")
        print(f"  Total points: {result['total_points']:,}")
        print(f"  Objects found: {result['num_objects']}")
        for obj in result['objects']:
            print(f"    - {obj['name']} (ID {obj['id']}): {obj['num_points']:,} points ({obj['percentage']:.1f}%)")
    
    # Handle output options
    if result:
        if args.export:
            query_tool.export_query_result(result, args.export)
        
        if args.visualize:
            query_tool.visualize_query(result)
        
        if args.json:
            # Remove non-serializable arrays for JSON
            json_result = {k: v for k, v in result.items() 
                          if k not in ['points', 'colors', 'mask']}
            with open(args.json, 'w') as f:
                json.dump(json_result, f, indent=2)
            print(f"Saved query result to {args.json}")


if __name__ == "__main__":
    main()