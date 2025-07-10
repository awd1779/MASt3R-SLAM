"""Create semantic overlay visualization by combining SLAM reconstruction with semantic labels."""

import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement
from scipy.spatial import KDTree
import logging

logger = logging.getLogger('mast3r_slam.semantic_overlay')


def load_ply(filename: str):
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


def save_ply(filename: str, points: np.ndarray, colors: np.ndarray):
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


def create_semantic_overlay(
    slam_ply_path: str,
    semantic_ply_path: str,
    output_path: str,
    distance_threshold: float = 0.005
):
    """
    Create overlay visualization combining SLAM reconstruction with semantic colors.
    
    Args:
        slam_ply_path: Path to original SLAM PLY (my.ply)
        semantic_ply_path: Path to semantic dense PLY
        output_path: Path for output overlay PLY
        distance_threshold: Maximum distance for point matching (meters)
    """
    logger.info(f"Creating semantic overlay visualization...")
    logger.info(f"  SLAM PLY: {slam_ply_path}")
    logger.info(f"  Semantic PLY: {semantic_ply_path}")
    
    # Load both PLY files
    slam_data = load_ply(slam_ply_path)
    semantic_data = load_ply(semantic_ply_path)
    
    slam_points = slam_data['points']
    slam_colors = slam_data['colors']
    semantic_points = semantic_data['points']
    semantic_colors = semantic_data['colors']
    
    logger.info(f"  SLAM points: {len(slam_points):,}")
    logger.info(f"  Semantic points: {len(semantic_points):,}")
    
    # Build KDTree from semantic points for efficient nearest neighbor search
    logger.info("Building KDTree for semantic points...")
    semantic_tree = KDTree(semantic_points)
    
    # Start with original SLAM colors
    final_colors = slam_colors.copy()
    
    # Find matches and update colors
    logger.info("Matching points and updating colors...")
    matches = 0
    
    # Query all SLAM points at once for efficiency
    distances, indices = semantic_tree.query(slam_points, k=1)
    
    # Update colors where matches are found
    mask = distances < distance_threshold
    matches = np.sum(mask)
    final_colors[mask] = semantic_colors[indices[mask]]
    
    logger.info(f"  Matched {matches:,} points ({100*matches/len(slam_points):.1f}%)")
    
    # Save combined PLY
    save_ply(output_path, slam_points, final_colors)
    logger.info(f"Saved overlay to {output_path}")
    
    # Return statistics
    return {
        'total_points': len(slam_points),
        'matched_points': matches,
        'match_percentage': 100 * matches / len(slam_points)
    }