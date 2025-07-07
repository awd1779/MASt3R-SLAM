#!/usr/bin/env python3
"""
Simple Semantic PLY Viewer for MASt3R-SLAM (No Open3D required)
Uses matplotlib for basic visualization
"""

import argparse
import json
import numpy as np
from pathlib import Path
from plyfile import PlyData
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import colorsys


class SimpleSemanticViewer:
    def __init__(self, ply_path, semantic_map_path=None):
        """Initialize the simple semantic viewer."""
        self.ply_path = Path(ply_path)
        self.ply_data = PlyData.read(ply_path)
        self.vertices = self.ply_data['vertex']
        
        # Extract point cloud data
        self.points = np.vstack([
            self.vertices['x'],
            self.vertices['y'],
            self.vertices['z']
        ]).T
        
        self.rgb_colors = np.vstack([
            self.vertices['red'],
            self.vertices['green'],
            self.vertices['blue']
        ]).T / 255.0
        
        # Extract semantic labels
        self.semantic_ids = None
        vertex_dtype = self.vertices.data.dtype
        for prop in ['semantic_id', 'quality']:
            if prop in vertex_dtype.names:
                self.semantic_ids = self.vertices.data[prop]
                print(f"Found semantic labels in property '{prop}'")
                break
        
        if self.semantic_ids is None:
            print("Warning: No semantic labels found in PLY file")
            self.semantic_ids = np.zeros(len(self.points), dtype=np.int32)
        
        # Load semantic map
        self.semantic_map = self._load_semantic_map(semantic_map_path)
        
        # Generate semantic colors
        self.semantic_colors = self._generate_semantic_colors()
        
        print(f"Loaded {len(self.points):,} points")
        unique_ids = np.unique(self.semantic_ids)
        print(f"Found {len(unique_ids)} unique semantic classes")
    
    def _load_semantic_map(self, semantic_map_path):
        """Load semantic map from JSON file or PLY comments."""
        semantic_map = {0: "background"}
        
        # Try loading from JSON file
        if semantic_map_path and Path(semantic_map_path).exists():
            with open(semantic_map_path, 'r') as f:
                json_map = json.load(f)
                semantic_map = {int(k): v for k, v in json_map.items()}
                print(f"Loaded semantic map from {semantic_map_path}")
        else:
            # Try extracting from PLY comments
            if hasattr(self.ply_data, 'comments'):
                in_label_map = False
                for comment in self.ply_data.comments:
                    if comment == 'label_map_start':
                        in_label_map = True
                    elif comment == 'label_map_end':
                        in_label_map = False
                    elif in_label_map and comment.startswith('label_id'):
                        parts = comment.split(':', 1)
                        if len(parts) == 2:
                            label_id = int(parts[0].split()[-1])
                            label_name = parts[1].strip()
                            semantic_map[label_id] = label_name
                
                if len(semantic_map) > 1:
                    print("Loaded semantic map from PLY comments")
        
        return semantic_map
    
    def _generate_semantic_colors(self):
        """Generate distinct colors for each semantic class."""
        unique_ids = np.unique(self.semantic_ids)
        colors = np.zeros((len(self.points), 3))
        
        # Generate distinct colors using HSV
        for i, label_id in enumerate(unique_ids):
            if label_id == 0:
                # Background is dark gray
                color = [0.2, 0.2, 0.2]
            else:
                # Generate distinct color
                hue = (i - 1) * 360.0 / max(1, len(unique_ids) - 1)
                color = colorsys.hsv_to_rgb(hue/360.0, 0.8, 0.9)
            
            mask = self.semantic_ids == label_id
            colors[mask] = color
        
        return colors
    
    def show_statistics(self):
        """Print statistics about objects in the scene."""
        unique_ids, counts = np.unique(self.semantic_ids, return_counts=True)
        
        print("\n=== Object Statistics ===")
        print(f"Total points: {len(self.points):,}")
        print(f"Unique objects: {len(unique_ids)}")
        print("\nObject counts:")
        
        # Sort by count
        sorted_indices = np.argsort(counts)[::-1]
        
        for idx in sorted_indices:
            label_id = unique_ids[idx]
            count = counts[idx]
            label_name = self.semantic_map.get(label_id, f"unknown_{label_id}")
            percentage = count / len(self.points) * 100
            print(f"  {label_id:3d}: {label_name:20s} - {count:8,} points ({percentage:5.1f}%)")
    
    def visualize(self, show_semantic=False, subsample=None, figsize=(12, 9)):
        """
        Visualize the point cloud with matplotlib.
        
        Args:
            show_semantic: Show semantic colors instead of RGB
            subsample: Subsample points for performance (e.g., 10000)
            figsize: Figure size
        """
        # Subsample if requested
        if subsample and len(self.points) > subsample:
            indices = np.random.choice(len(self.points), subsample, replace=False)
            points = self.points[indices]
            colors = self.semantic_colors[indices] if show_semantic else self.rgb_colors[indices]
            print(f"Subsampled to {subsample:,} points for visualization")
        else:
            points = self.points
            colors = self.semantic_colors if show_semantic else self.rgb_colors
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot points
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  c=colors, s=1, alpha=0.6)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        title = "Semantic Segmentation" if show_semantic else "RGB Colors"
        ax.set_title(f"MASt3R-SLAM Point Cloud - {title}")
        
        # Add legend if showing semantic colors
        if show_semantic:
            unique_ids = np.unique(self.semantic_ids)
            if len(unique_ids) <= 20:  # Only show legend for reasonable number of classes
                from matplotlib.patches import Patch
                legend_elements = []
                for label_id in unique_ids:
                    if label_id in self.semantic_map:
                        color_idx = np.where(unique_ids == label_id)[0][0]
                        if label_id == 0:
                            color = [0.2, 0.2, 0.2]
                        else:
                            hue = (color_idx - 1) * 360.0 / max(1, len(unique_ids) - 1)
                            color = colorsys.hsv_to_rgb(hue/360.0, 0.8, 0.9)
                        
                        legend_elements.append(
                            Patch(facecolor=color, label=self.semantic_map[label_id])
                        )
                
                ax.legend(handles=legend_elements, loc='upper right', 
                         bbox_to_anchor=(1.15, 1), prop={'size': 8})
        
        plt.tight_layout()
        plt.show()
    
    def visualize_class(self, class_name, subsample=None):
        """Visualize only points belonging to a specific class."""
        # Find matching label IDs
        label_ids = []
        for label_id, name in self.semantic_map.items():
            if class_name.lower() in name.lower():
                label_ids.append(label_id)
        
        if not label_ids:
            print(f"No objects found matching '{class_name}'")
            return
        
        # Get mask for these labels
        mask = np.isin(self.semantic_ids, label_ids)
        filtered_points = self.points[mask]
        filtered_colors = self.rgb_colors[mask]
        
        print(f"Found {len(filtered_points):,} points matching '{class_name}'")
        
        # Subsample if needed
        if subsample and len(filtered_points) > subsample:
            indices = np.random.choice(len(filtered_points), subsample, replace=False)
            filtered_points = filtered_points[indices]
            filtered_colors = filtered_colors[indices]
        
        # Visualize
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(filtered_points[:, 0], filtered_points[:, 1], filtered_points[:, 2],
                  c=filtered_colors, s=2, alpha=0.8)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"Points matching '{class_name}'")
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Simple visualization for MASt3R-SLAM semantic PLY files")
    parser.add_argument("ply_file", help="Path to PLY file")
    parser.add_argument("--semantic-map", help="Path to semantic map JSON file")
    parser.add_argument("--semantic", action="store_true", help="Show semantic colors")
    parser.add_argument("--stats", action="store_true", help="Show object statistics")
    parser.add_argument("--subsample", type=int, help="Subsample points for visualization")
    parser.add_argument("--filter", help="Show only specific class (e.g., 'chair')")
    
    args = parser.parse_args()
    
    # Create viewer
    viewer = SimpleSemanticViewer(args.ply_file, args.semantic_map)
    
    # Show statistics if requested
    if args.stats:
        viewer.show_statistics()
    
    # Visualize
    if args.filter:
        viewer.visualize_class(args.filter, subsample=args.subsample)
    else:
        viewer.visualize(show_semantic=args.semantic, subsample=args.subsample)


if __name__ == "__main__":
    main()