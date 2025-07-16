#!/usr/bin/env python3
"""Visualize semantic images from generated trajectories."""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from evaluation.replica_loader import ReplicaSemanticLoader


def create_color_map(num_instances: int = 500):
    """Create a color map for instance IDs."""
    np.random.seed(42)  # For consistent colors
    colors = np.random.randint(0, 255, size=(num_instances, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Background is black
    return colors


def visualize_semantic_trajectory(trajectory_dir: str, replica_scene: str, 
                                 num_frames: int = 5):
    """Visualize semantic images from a trajectory.
    
    Args:
        trajectory_dir: Path to trajectory directory
        replica_scene: Path to Replica scene for loading instance mappings
        num_frames: Number of frames to visualize
    """
    traj_path = Path(trajectory_dir)
    
    # Load Replica semantic info
    print(f"Loading Replica semantic info from {replica_scene}")
    loader = ReplicaSemanticLoader(replica_scene)
    
    # Create color map
    color_map = create_color_map()
    
    # Load and visualize frames
    semantic_dir = traj_path / 'semantic'
    rgb_dir = traj_path / 'rgb'
    
    frames = sorted(list(semantic_dir.glob('*.png')))[:num_frames]
    
    fig, axes = plt.subplots(num_frames, 3, figsize=(12, 4*num_frames))
    if num_frames == 1:
        axes = axes.reshape(1, -1)
    
    for i, frame_path in enumerate(frames):
        # Load semantic image
        semantic = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
        
        # Load corresponding RGB
        rgb_path = rgb_dir / frame_path.name
        rgb = cv2.imread(str(rgb_path))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # Create colored semantic visualization
        semantic_colored = color_map[semantic.astype(int) % len(color_map)]
        
        # Get unique instances in this frame
        unique_instances = np.unique(semantic)
        unique_instances = unique_instances[unique_instances > 0]  # Skip background
        
        # Map to class names
        class_names = []
        for inst_id in unique_instances[:10]:  # Show first 10
            class_id, class_name = loader.map_instance_to_class(int(inst_id))
            class_names.append(f"{inst_id}: {class_name}")
        
        # Display
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f'RGB Frame {i}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(semantic_colored)
        axes[i, 1].set_title(f'Semantic (Colored)')
        axes[i, 1].axis('off')
        
        axes[i, 2].text(0.1, 0.5, '\n'.join(class_names), 
                       transform=axes[i, 2].transAxes,
                       fontsize=8, verticalalignment='center')
        axes[i, 2].set_title('Instance IDs → Classes')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    output_path = traj_path.parent / f'{traj_path.name}_semantic_vis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    
    # Print statistics
    print(f"\nSemantic statistics for {trajectory_dir}:")
    all_instances = set()
    for frame_path in frames:
        semantic = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
        unique = np.unique(semantic)
        all_instances.update(unique[unique > 0])
    
    print(f"Total unique instances across {len(frames)} frames: {len(all_instances)}")
    
    # Map to classes and count
    class_counts = {}
    for inst_id in all_instances:
        _, class_name = loader.map_instance_to_class(int(inst_id))
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print("\nClass distribution:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {class_name}: {count} instances")


def main():
    parser = argparse.ArgumentParser(description='Visualize semantic trajectory data')
    parser.add_argument('trajectory_dir', type=str,
                       help='Path to trajectory directory')
    parser.add_argument('--replica_scene', type=str, 
                       default='datasets/replica_dataset/apartment_0',
                       help='Path to Replica scene')
    parser.add_argument('--num_frames', type=int, default=5,
                       help='Number of frames to visualize')
    
    args = parser.parse_args()
    
    visualize_semantic_trajectory(args.trajectory_dir, args.replica_scene, args.num_frames)


if __name__ == "__main__":
    main()