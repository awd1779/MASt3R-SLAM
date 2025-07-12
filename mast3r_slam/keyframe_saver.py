"""Save keyframe data for dense reconstruction."""

import numpy as np
import torch
from pathlib import Path
import pickle
import cv2
import logging

logger = logging.getLogger('mast3r_slam.keyframe_saver')


class KeyframeSaver:
    """Save keyframe depth maps and semantic masks for dense reconstruction."""
    
    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.depth_dir = self.save_dir / "depth_maps"
        self.semantic_dir = self.save_dir / "semantic_masks"
        self.pose_dir = self.save_dir / "poses"
        
        # Create directories
        self.depth_dir.mkdir(exist_ok=True, parents=True)
        self.semantic_dir.mkdir(exist_ok=True, parents=True)
        self.pose_dir.mkdir(exist_ok=True, parents=True)
        
        # Data to save
        self.keyframe_data = []
        
    def save_keyframe(self, kf_idx: int, keyframe, semantic_data=None):
        """Save keyframe data for dense reconstruction."""
        
        # 1. Save depth map (3D points in camera coordinates)
        X_cam = keyframe.X_canon.cpu().numpy()  # Shape: (H*W, 3)
        h, w = keyframe.img_shape[0, 0].item(), keyframe.img_shape[0, 1].item()
        
        # Reshape to image format
        depth_map = X_cam[:, 2].reshape(h, w)  # Z-coordinate is depth
        
        # Save as numpy array
        depth_file = self.depth_dir / f"depth_{kf_idx:04d}.npy"
        np.save(depth_file, X_cam.reshape(h, w, 3))  # Save full 3D coords
        
        # 2. Save RGB image
        img_rgb = (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8)
        img_file = self.depth_dir / f"rgb_{kf_idx:04d}.png"
        cv2.imwrite(str(img_file), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        
        # 3. Save camera pose
        if hasattr(keyframe.T_WC, 'matrix'):
            T_WC = keyframe.T_WC.matrix().cpu().numpy().squeeze()
        else:
            T_WC = keyframe.T_WC.cpu().numpy()
        
        pose_file = self.pose_dir / f"pose_{kf_idx:04d}.npy"
        np.save(pose_file, T_WC)
        
        # 4. Save semantic masks if available
        if semantic_data and 'masks_rle' in semantic_data:
            semantic_file = self.semantic_dir / f"semantic_{kf_idx:04d}.pkl"
            with open(semantic_file, 'wb') as f:
                pickle.dump(semantic_data, f)
        
        # Store metadata
        self.keyframe_data.append({
            'kf_idx': kf_idx,
            'frame_id': keyframe.frame_id,
            'img_shape': (h, w),
            'has_semantics': semantic_data is not None
        })
        
        logger.debug(f"Saved keyframe {kf_idx}: depth {depth_map.shape}, "
                    f"pose {T_WC.shape}, semantics: {semantic_data is not None}")
    
    def save_metadata(self):
        """Save metadata about all keyframes."""
        metadata_file = self.save_dir / "keyframe_metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump({
                'keyframes': self.keyframe_data,
                'num_keyframes': len(self.keyframe_data)
            }, f)
        logger.info(f"Saved metadata for {len(self.keyframe_data)} keyframes")


class DenseSemanticBuilder:
    """Build dense semantic point cloud from saved keyframe data."""
    
    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.depth_dir = self.save_dir / "depth_maps"
        self.semantic_dir = self.save_dir / "semantic_masks"
        self.pose_dir = self.save_dir / "poses"
        
        # Load metadata
        metadata_file = self.save_dir / "keyframe_metadata.pkl"
        with open(metadata_file, 'rb') as f:
            self.metadata = pickle.load(f)
            
    def build_dense_cloud(self, output_path: str, use_semantic_colors: bool = True):
        """Build dense semantic point cloud from saved data."""
        from mast3r_slam.semantic_frame import decode_rle
        import matplotlib.cm as cm
        
        all_points = []
        all_colors = []
        all_labels = []
        
        # Create colormap
        cmap = cm.get_cmap('tab20')
        label_colors = {
            0: np.array([128, 128, 128]),  # Background
            1: np.array([255, 0, 0]),      # Person
            2: np.array([0, 255, 0]),      # Chair  
            3: np.array([0, 0, 255]),      # Table
            4: np.array([255, 255, 0]),    # Bottle
        }
        
        # Label name to ID mapping
        label_mapping = {
            'person': 1,
            'chair': 2,
            'table': 3,
            'bottle': 4,
            'car': 5
        }
        
        print(f"\nBuilding dense semantic point cloud from {len(self.metadata['keyframes'])} keyframes...")
        
        for kf_data in self.metadata['keyframes']:
            kf_idx = kf_data['kf_idx']
            h, w = kf_data['img_shape']
            
            # Load 3D points
            depth_file = self.depth_dir / f"depth_{kf_idx:04d}.npy"
            X_cam = np.load(depth_file).reshape(-1, 3)  # (H*W, 3)
            
            # Load RGB
            img_file = self.depth_dir / f"rgb_{kf_idx:04d}.png"
            img_bgr = cv2.imread(str(img_file))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            colors_flat = img_rgb.reshape(-1, 3)
            
            # Load pose
            pose_file = self.pose_dir / f"pose_{kf_idx:04d}.npy"
            T_WC = np.load(pose_file)
            
            # Load semantic masks
            semantic_mask = np.zeros((h, w), dtype=np.int32)
            
            if kf_data['has_semantics']:
                semantic_file = self.semantic_dir / f"semantic_{kf_idx:04d}.pkl"
                with open(semantic_file, 'rb') as f:
                    semantic_data = pickle.load(f)
                
                # Decode masks
                for instance_id, rle in semantic_data.get('masks_rle', {}).items():
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
                        
                        # Get label
                        label_name = semantic_data.get('labels', {}).get(instance_id, 'unknown')
                        label_id = label_mapping.get(label_name, 0)
                        semantic_mask[mask] = label_id
            
            # Flatten semantic mask
            labels_flat = semantic_mask.reshape(-1)
            
            # Transform to world coordinates
            X_cam_homo = np.concatenate([X_cam, np.ones((X_cam.shape[0], 1))], axis=1)
            X_world = (T_WC @ X_cam_homo.T).T[:, :3]
            
            # Filter valid points (only labeled, not too far)
            depths = X_cam[:, 2]
            valid_mask = (
                (depths > 0.1) &
                (depths < 50.0) &
                (np.isfinite(X_world).all(axis=1)) &
                (labels_flat > 0)  # Only labeled points
            )
            
            # Apply mask
            valid_points = X_world[valid_mask]
            valid_colors = colors_flat[valid_mask]
            valid_labels = labels_flat[valid_mask]
            
            all_points.append(valid_points)
            all_colors.append(valid_colors)
            all_labels.append(valid_labels)
            
            print(f"  Keyframe {kf_idx}: {len(valid_points)} points")
        
        # Combine all
        all_points = np.concatenate(all_points, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Apply semantic colors if requested
        if use_semantic_colors:
            semantic_colors = np.zeros_like(all_colors)
            for label_id in np.unique(all_labels):
                mask = all_labels == label_id
                color = label_colors.get(label_id, np.array([200, 200, 200]))
                semantic_colors[mask] = color
            all_colors = semantic_colors
        
        # Save PLY
        from plyfile import PlyData, PlyElement
        
        vertex_data = np.zeros(
            len(all_points),
            dtype=[
                ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                ('label', 'u4')
            ]
        )
        
        vertex_data['x'] = all_points[:, 0]
        vertex_data['y'] = all_points[:, 1]
        vertex_data['z'] = all_points[:, 2]
        vertex_data['red'] = all_colors[:, 0]
        vertex_data['green'] = all_colors[:, 1]
        vertex_data['blue'] = all_colors[:, 2]
        vertex_data['label'] = all_labels
        
        vertex_element = PlyElement.describe(vertex_data, 'vertex')
        ply_data = PlyData([vertex_element], text=False)
        ply_data.write(output_path)
        
        print(f"\nSaved dense semantic cloud: {output_path}")
        print(f"Total points: {len(all_points):,}")
        
        # Statistics
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        print("\nLabel distribution:")
        for label_id, count in zip(unique_labels, counts):
            percentage = (count / len(all_labels)) * 100
            print(f"  Label {label_id}: {count:,} points ({percentage:.1f}%)")
        
        return len(all_points)