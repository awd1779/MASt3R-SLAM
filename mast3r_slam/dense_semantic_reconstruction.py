"""Dense semantic reconstruction using MAST3R depth maps and semantic masks."""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import cv2
from plyfile import PlyData, PlyElement
import matplotlib.cm as cm
from mast3r_slam.semantic_frame import decode_rle


class DenseSemanticReconstructor:
    """Create dense semantic point clouds by projecting ALL segmented pixels."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.label_colormap = self._create_colormap()
        
    def _create_colormap(self):
        """Create a colormap for semantic labels."""
        cmap = cm.get_cmap('tab20')
        colors = {}
        # Special colors for common objects
        colors[0] = np.array([128, 128, 128])  # Background - gray
        colors[1] = np.array([255, 0, 0])      # Person - red
        colors[2] = np.array([0, 255, 0])      # Chair - green
        colors[3] = np.array([0, 0, 255])      # Table - blue
        colors[4] = np.array([255, 255, 0])    # Bottle - yellow
        colors[5] = np.array([255, 0, 255])    # Car - magenta
        
        # Generate colors for other labels
        for i in range(6, 30):
            color = (np.array(cmap((i-6) % 20)[:3]) * 255).astype(np.uint8)
            colors[i] = color
            
        return colors
    
    def project_semantic_keyframe(self, 
                                keyframe,
                                semantic_data: Dict,
                                label_mapping: Dict[str, int],
                                confidence_threshold: float = 0.3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Project ALL pixels from a semantic keyframe into 3D space.
        
        Returns:
            points_3d: (N, 3) array of 3D points in world coordinates
            colors: (N, 3) array of RGB colors
            labels: (N,) array of semantic labels
        """
        # Get depth map from keyframe
        # X_canon contains 3D points in camera coordinates
        X_cam = keyframe.X_canon.cpu().numpy()  # Shape: (H*W, 3)
        
        # Get image dimensions
        h, w = keyframe.img_shape[0, 0].item(), keyframe.img_shape[0, 1].item()
        
        # Get RGB image
        img_rgb = (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8)
        img_rgb_flat = img_rgb.reshape(-1, 3)
        
        # Create semantic mask for this keyframe
        semantic_mask = np.zeros((h, w), dtype=np.int32)
        instance_to_label = {}
        
        # Decode all semantic masks
        if semantic_data and 'masks_rle' in semantic_data:
            for instance_id, rle in semantic_data['masks_rle'].items():
                if 'size' in rle:
                    size = rle['size']
                    if len(size) == 3:
                        _, mask_h, mask_w = size
                    else:
                        mask_h, mask_w = size
                    
                    # Decode mask
                    mask = decode_rle(rle, mask_h, mask_w)
                    
                    # Resize if needed
                    if (mask_h, mask_w) != (h, w):
                        mask = cv2.resize(
                            mask.astype(np.uint8), 
                            (w, h), 
                            interpolation=cv2.INTER_NEAREST
                        ).astype(bool)
                    
                    # Get label for this instance
                    label_name = semantic_data.get('labels', {}).get(instance_id, 'unknown')
                    label_id = label_mapping.get(label_name, 0)
                    confidence = semantic_data.get('confidences', {}).get(instance_id, 1.0)
                    
                    # Only use high-confidence detections
                    if confidence >= confidence_threshold:
                        semantic_mask[mask] = label_id
                        instance_to_label[instance_id] = label_id
        
        # Flatten semantic mask
        semantic_mask_flat = semantic_mask.reshape(-1)
        
        # Get camera-to-world transformation
        T_WC = keyframe.T_WC
        
        # Transform points to world coordinates
        # Add homogeneous coordinate
        X_cam_homo = np.concatenate([X_cam, np.ones((X_cam.shape[0], 1))], axis=1)
        
        # Convert T_WC to numpy if it's a torch tensor
        if hasattr(T_WC, 'matrix'):
            T_WC_matrix = T_WC.matrix().cpu().numpy().squeeze()
        else:
            T_WC_matrix = T_WC.cpu().numpy()
        
        # Transform to world coordinates
        X_world = (T_WC_matrix @ X_cam_homo.T).T[:, :3]
        
        # Filter out invalid points (too far, behind camera, etc.)
        # Get depth values
        depths = X_cam[:, 2]
        
        # Create validity mask
        valid_mask = (
            (depths > 0.1) &  # Not too close
            (depths < 50.0) &  # Not too far
            (np.isfinite(X_world).all(axis=1)) &  # No inf/nan
            (semantic_mask_flat > 0)  # Only labeled pixels
        )
        
        # Apply validity mask
        points_3d = X_world[valid_mask]
        colors = img_rgb_flat[valid_mask]
        labels = semantic_mask_flat[valid_mask]
        
        print(f"  Keyframe {keyframe.frame_id}: {len(points_3d)} labeled points out of {np.sum(semantic_mask > 0)} segmented pixels")
        
        return points_3d, colors, labels
    
    def create_dense_semantic_pointcloud(self,
                                       keyframes,
                                       semantic_keyframes,
                                       semantic_backend,
                                       confidence_threshold: float = 0.3,
                                       use_semantic_colors: bool = True) -> Dict:
        """
        Create a dense semantic point cloud from all keyframes.
        
        Returns:
            Dictionary with points, colors, labels, and statistics
        """
        all_points = []
        all_colors = []
        all_labels = []
        
        # Create label mapping from track manager
        label_mapping = {'unknown': 0}
        label_id = 1
        
        if hasattr(semantic_backend, 'track_manager'):
            tracks = semantic_backend.track_manager.get_all_tracks()
            for track_id, track_info in tracks.items():
                if track_id > 0:  # Skip background
                    label_name = track_info.get('label', 'unknown')
                    if label_name not in label_mapping:
                        label_mapping[label_name] = label_id
                        label_id += 1
        
        print(f"\nCreating dense semantic reconstruction...")
        print(f"Label mapping: {label_mapping}")
        
        # Process each keyframe
        processed_keyframes = 0
        total_segmented_pixels = 0
        
        for kf_idx in range(len(keyframes)):
            keyframe = keyframes[kf_idx]
            if keyframe is None or keyframe.X_canon is None:
                continue
            
            # Get semantic data
            semantic_data = semantic_keyframes.get_semantics(kf_idx)
            if semantic_data is None or not semantic_data.get('masks_rle'):
                continue
            
            # Project this keyframe
            points, colors, labels = self.project_semantic_keyframe(
                keyframe, 
                semantic_data,
                label_mapping,
                confidence_threshold
            )
            
            if len(points) > 0:
                all_points.append(points)
                all_colors.append(colors)
                all_labels.append(labels)
                processed_keyframes += 1
                total_segmented_pixels += len(points)
        
        if len(all_points) == 0:
            print("No semantic points found!")
            return {}
        
        # Concatenate all points
        all_points = np.concatenate(all_points, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        print(f"\nProcessed {processed_keyframes} keyframes")
        print(f"Total dense semantic points: {len(all_points)}")
        
        # Apply semantic colors if requested
        if use_semantic_colors:
            semantic_colors = np.zeros_like(all_colors)
            for label_id in np.unique(all_labels):
                mask = all_labels == label_id
                color = self.label_colormap.get(label_id, np.array([200, 200, 200]))
                semantic_colors[mask] = color
            all_colors = semantic_colors
        
        # Compute statistics
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        label_stats = {}
        
        reverse_mapping = {v: k for k, v in label_mapping.items()}
        for label_id, count in zip(unique_labels, counts):
            label_name = reverse_mapping.get(label_id, f"label_{label_id}")
            percentage = (count / len(all_labels)) * 100
            label_stats[label_name] = {
                'count': int(count),
                'percentage': float(percentage)
            }
        
        return {
            'points': all_points,
            'colors': all_colors,
            'labels': all_labels,
            'num_points': len(all_points),
            'num_keyframes': processed_keyframes,
            'label_mapping': label_mapping,
            'label_stats': label_stats
        }
    
    def save_dense_semantic_ply(self, 
                              filename: str,
                              points: np.ndarray,
                              colors: np.ndarray,
                              labels: np.ndarray):
        """Save dense semantic point cloud to PLY file."""
        # Create structured array
        num_points = len(points)
        vertex_data = np.zeros(
            num_points,
            dtype=[
                ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                ('label', 'u4')
            ]
        )
        
        vertex_data['x'] = points[:, 0]
        vertex_data['y'] = points[:, 1]
        vertex_data['z'] = points[:, 2]
        vertex_data['red'] = colors[:, 0]
        vertex_data['green'] = colors[:, 1]
        vertex_data['blue'] = colors[:, 2]
        vertex_data['label'] = labels
        
        # Create PLY element
        vertex_element = PlyElement.describe(vertex_data, 'vertex')
        
        # Write PLY file
        ply_data = PlyData([vertex_element], text=False)
        ply_data.write(filename)
        print(f"Saved dense semantic point cloud to {filename}")


def create_dense_semantic_reconstruction(keyframes, 
                                       semantic_keyframes,
                                       semantic_backend,
                                       output_path: str,
                                       confidence_threshold: float = 0.3,
                                       use_semantic_colors: bool = True):
    """High-level function to create and save dense semantic reconstruction."""
    
    reconstructor = DenseSemanticReconstructor()
    
    # Create dense point cloud
    result = reconstructor.create_dense_semantic_pointcloud(
        keyframes,
        semantic_keyframes,
        semantic_backend,
        confidence_threshold,
        use_semantic_colors
    )
    
    if not result:
        print("Failed to create dense semantic reconstruction")
        return None
    
    # Save PLY file
    reconstructor.save_dense_semantic_ply(
        output_path,
        result['points'],
        result['colors'],
        result['labels']
    )
    
    # Save statistics
    stats_path = Path(output_path).with_suffix('.txt')
    with open(stats_path, 'w') as f:
        f.write("Dense Semantic Reconstruction Statistics\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total points: {result['num_points']:,}\n")
        f.write(f"Keyframes used: {result['num_keyframes']}\n")
        f.write(f"\nLabel distribution:\n")
        
        for label_name, stats in sorted(result['label_stats'].items()):
            f.write(f"  {label_name}: {stats['count']:,} points ({stats['percentage']:.1f}%)\n")
    
    print(f"\nSaved statistics to {stats_path}")
    
    return result