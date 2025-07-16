#!/usr/bin/env python3
"""Check and visualize trajectory smoothness."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_trajectory(poses_file: str):
    """Load trajectory from TUM format poses file."""
    poses = []
    with open(poses_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8:
                # timestamp tx ty tz qx qy qz qw
                timestamp = float(parts[0])
                position = [float(parts[1]), float(parts[2]), float(parts[3])]
                poses.append({
                    'timestamp': timestamp,
                    'position': position
                })
    return poses


def analyze_trajectory_smoothness(trajectory_dir: str):
    """Analyze and visualize trajectory smoothness."""
    traj_path = Path(trajectory_dir)
    poses_file = traj_path / 'poses.txt'
    
    if not poses_file.exists():
        print(f"Error: {poses_file} not found")
        return
    
    # Load poses
    poses = load_trajectory(poses_file)
    
    if len(poses) < 2:
        print("Not enough poses to analyze")
        return
    
    # Extract positions
    positions = np.array([p['position'] for p in poses])
    
    # Calculate frame-to-frame distances
    distances = []
    for i in range(1, len(positions)):
        dist = np.linalg.norm(positions[i] - positions[i-1])
        distances.append(dist)
    
    distances = np.array(distances)
    
    # Calculate velocities (distance per frame)
    velocities = distances  # Since we have 1 frame timestep
    
    # Calculate accelerations
    accelerations = []
    for i in range(1, len(velocities)):
        acc = velocities[i] - velocities[i-1]
        accelerations.append(acc)
    
    accelerations = np.array(accelerations)
    
    # Statistics
    print(f"\nTrajectory Analysis for: {trajectory_dir}")
    print(f"Total frames: {len(poses)}")
    print(f"Total distance: {distances.sum():.2f} meters")
    print(f"\nVelocity statistics:")
    print(f"  Mean: {velocities.mean():.3f} m/frame")
    print(f"  Std: {velocities.std():.3f}")
    print(f"  Max: {velocities.max():.3f}")
    print(f"  Min: {velocities.min():.3f}")
    
    # Check for teleportation (sudden jumps)
    teleport_threshold = 0.5  # meters per frame
    teleports = np.where(distances > teleport_threshold)[0]
    print(f"\nTeleportations (>{teleport_threshold}m): {len(teleports)}")
    if len(teleports) > 0:
        print(f"  At frames: {teleports[:10]}...")  # Show first 10
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Top-down trajectory
    ax = axes[0, 0]
    ax.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.7, linewidth=1)
    ax.scatter(positions[0, 0], positions[0, 1], c='green', s=100, label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], c='red', s=100, label='End')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Top-down View')
    ax.legend()
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    # 2. Height profile
    ax = axes[0, 1]
    ax.plot(positions[:, 2], 'g-', alpha=0.7)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Height (m)')
    ax.set_title('Height Profile')
    ax.grid(True, alpha=0.3)
    
    # 3. Velocity profile
    ax = axes[1, 0]
    ax.plot(velocities, 'r-', alpha=0.7)
    ax.axhline(y=teleport_threshold, color='k', linestyle='--', label='Teleport threshold')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Velocity (m/frame)')
    ax.set_title('Frame-to-Frame Velocity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Acceleration profile
    ax = axes[1, 1]
    ax.plot(accelerations, 'purple', alpha=0.7)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Acceleration (m/frame²)')
    ax.set_title('Frame-to-Frame Acceleration')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Trajectory Smoothness Analysis: {traj_path.name}')
    plt.tight_layout()
    
    output_path = traj_path.parent / f'{traj_path.name}_smoothness.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    # Additional smoothness metrics
    smoothness_score = 1.0 / (1.0 + velocities.std())  # Higher is smoother
    print(f"\nSmoothness score: {smoothness_score:.3f} (1.0 = perfectly smooth)")


def main():
    parser = argparse.ArgumentParser(description='Analyze trajectory smoothness')
    parser.add_argument('trajectory_dir', type=str,
                       help='Path to trajectory directory')
    
    args = parser.parse_args()
    analyze_trajectory_smoothness(args.trajectory_dir)


if __name__ == "__main__":
    main()