"""Fixed dense semantic reconstruction with proper alignment."""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import cv2
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mast3r_slam.semantic_frame import decode_rle


class DenseSemanticReconstructorV2:
    """Create dense semantic point clouds with proper mask-depth alignment."""
    
    def __init__(self, device: str = "cuda", debug: bool = True):
        self.device = device
        self.debug = debug
        self.label_colormap = self._create_colormap()
        
        # Debug directory
        if self.debug:
            self.debug_dir = Path("logs/debug_semantic")
            self.debug_dir.mkdir(exist_ok=True, parents=True)
        
    def _create_colormap(self):
        """Create a colormap for semantic labels."""
        # Use distinct colors for common objects
        colors = {
            0: np.array([128, 128, 128]),  # Background - gray
            'person': np.array([255, 0, 0]),      # Red
            'chair': np.array([0, 255, 0]),       # Green
            'table': np.array([0, 0, 255]),       # Blue
            'bottle': np.array([255, 255, 0]),    # Yellow
            'car': np.array([255, 0, 255]),       # Magenta
            'unknown': np.array([200, 200, 200])  # Light gray
        }
        return colors
    
    def visualize_semantic_alignment(self, 
                                   keyframe,
                                   semantic_mask: np.ndarray,
                                   kf_idx: int,
                                   label_names: Dict[int, str]):
        """Create clean side-by-side visualization of keyframe and segmentation."""
        if not self.debug:
            return
            
        # Get RGB image
        img_rgb = (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8)
        h, w = img_rgb.shape[:2]
        
        # Create segmentation visualization with solid colors
        seg_viz = np.zeros_like(img_rgb)
        
        # Apply colors for each semantic label
        unique_labels = np.unique(semantic_mask)
        for label_id in unique_labels:
            if label_id == 0:  # Skip background
                continue
            
            mask = semantic_mask == label_id
            label_name = label_names.get(label_id, 'unknown')
            color = self.label_colormap.get(label_name, self.label_colormap['unknown'])
            seg_viz[mask] = color
        
        # Create overlay (semi-transparent)
        overlay = img_rgb.copy()
        mask_any = semantic_mask > 0
        overlay[mask_any] = (0.6 * seg_viz[mask_any] + 0.4 * img_rgb[mask_any]).astype(np.uint8)
        
        # Create side-by-side image
        side_by_side = np.hstack([img_rgb, overlay])
        
        # Add titles
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(side_by_side, "Original", (10, 25), font, 0.7, (255, 255, 255), 2)
        cv2.putText(side_by_side, "Semantic Segmentation", (w + 10, 25), font, 0.7, (255, 255, 255), 2)
        
        # Add legend on the segmentation side
        y_offset = 50
        x_offset = w + 10
        for label_id in sorted(unique_labels):
            if label_id == 0:
                continue
            label_name = label_names.get(label_id, f'label_{label_id}')
            percentage = (np.sum(semantic_mask == label_id) / (h * w)) * 100
            
            # Draw color box
            color = self.label_colormap.get(label_name, self.label_colormap['unknown'])
            # Convert numpy array to tuple for OpenCV
            color_bgr = tuple(int(c) for c in color[::-1])  # RGB to BGR
            cv2.rectangle(side_by_side, (x_offset, y_offset - 12), (x_offset + 25, y_offset + 3), color_bgr, -1)
            
            # Add text
            text = f"{label_name}: {percentage:.1f}%"
            cv2.putText(side_by_side, text, (x_offset + 30, y_offset), 
                       font, 0.5, (255, 255, 255), 1)
            y_offset += 25
        
        # Save the side-by-side visualization
        output_file = self.debug_dir / f"keyframe_{kf_idx:04d}_semantic.png"
        cv2.imwrite(str(output_file), cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))
        print(f"  Saved visualization: {output_file.name}")
    
    def project_semantic_keyframe_v2(self, 
                                   keyframe,
                                   semantic_data: Dict,
                                   kf_idx: int,
                                   confidence_threshold: float = 0.3) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Project ALL segmented pixels from a keyframe into 3D space with proper alignment.
        
        Returns:
            points_3d: (N, 3) array of 3D points in world coordinates
            colors: (N, 3) array of RGB colors
            labels: (N,) array of semantic label IDs
            label_names: Dict mapping label IDs to names
        """
        # Get keyframe dimensions
        h, w = keyframe.img_shape[0, 0].item(), keyframe.img_shape[0, 1].item()
        print(f"\n  Processing keyframe {kf_idx}: {w}x{h}")
        
        # Get 3D points from MAST3R (camera coordinates)
        X_cam = keyframe.X_canon.cpu().numpy()  # Shape: (H*W, 3)
        
        # Verify dimensions match
        expected_size = h * w
        if X_cam.shape[0] != expected_size:
            print(f"  WARNING: X_cam size {X_cam.shape[0]} != expected {expected_size}")
            return np.array([]), np.array([]), np.array([]), {}
        
        # Get RGB image
        img_rgb = (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8)
        img_rgb_flat = img_rgb.reshape(-1, 3)
        
        # Create semantic mask with proper alignment
        semantic_mask = np.zeros((h, w), dtype=np.int32)
        label_id_to_name = {0: 'background'}
        next_label_id = 1
        
        # Process semantic data
        if semantic_data and 'masks_rle' in semantic_data:
            print(f"  Found {len(semantic_data['masks_rle'])} instance masks")
            
            for instance_id, rle in semantic_data['masks_rle'].items():
                if 'size' in rle:
                    size = rle['size']
                    if len(size) == 3:
                        _, mask_h, mask_w = size
                    else:
                        mask_h, mask_w = size
                    
                    print(f"    Instance {instance_id}: mask size {mask_w}x{mask_h}")
                    
                    # CRITICAL: Verify mask dimensions match keyframe
                    if (mask_h, mask_w) != (h, w):
                        print(f"    ERROR: Mask size {mask_w}x{mask_h} != keyframe {w}x{h}")
                        print(f"    Skipping this mask to avoid misalignment!")
                        continue
                    
                    # Decode mask at original size
                    mask = decode_rle(rle, mask_h, mask_w)
                    
                    # Get metadata
                    label_name = semantic_data.get('labels', {}).get(instance_id, 'unknown')
                    confidence = semantic_data.get('confidences', {}).get(instance_id, 1.0)
                    
                    print(f"    Label: {label_name}, Confidence: {confidence:.3f}")
                    
                    # Only use high-confidence detections
                    if confidence >= confidence_threshold:
                        # Assign label ID
                        if label_name not in label_id_to_name.values():
                            label_id_to_name[next_label_id] = label_name
                            label_id = next_label_id
                            next_label_id += 1
                        else:
                            # Find existing label ID
                            label_id = [k for k, v in label_id_to_name.items() if v == label_name][0]
                        
                        # Apply mask
                        semantic_mask[mask] = label_id
                        # Convert mask to numpy if it's a tensor
                        if isinstance(mask, torch.Tensor):
                            pixels_labeled = mask.sum().item()
                        else:
                            pixels_labeled = np.sum(mask)
                        print(f"    Applied label {label_id} ({label_name}) to {pixels_labeled} pixels")
                    else:
                        print(f"    Skipped due to low confidence")
        
        # Visualize for debugging
        self.visualize_semantic_alignment(keyframe, semantic_mask, kf_idx, label_id_to_name)
        
        # Flatten semantic mask
        semantic_mask_flat = semantic_mask.reshape(-1)
        
        # Get camera-to-world transformation
        T_WC = keyframe.T_WC
        if hasattr(T_WC, 'matrix'):
            T_WC_matrix = T_WC.matrix().cpu().numpy().squeeze()
        else:
            T_WC_matrix = T_WC.cpu().numpy()
        
        # Transform to world coordinates
        X_cam_homo = np.concatenate([X_cam, np.ones((X_cam.shape[0], 1))], axis=1)
        X_world = (T_WC_matrix @ X_cam_homo.T).T[:, :3]
        
        # Filter points
        depths = X_cam[:, 2]
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
        
        print(f"  Result: {len(points_3d)} labeled 3D points")
        
        # Print label distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label_id, count in zip(unique_labels, counts):
            label_name = label_id_to_name.get(label_id, f'unknown_{label_id}')
            print(f"    {label_name}: {count} points")
        
        return points_3d, colors, labels, label_id_to_name
    
    def create_dense_semantic_pointcloud_v2(self,
                                          keyframes,
                                          semantic_keyframes,
                                          confidence_threshold: float = 0.3,
                                          use_semantic_colors: bool = True) -> Dict:
        """
        Create a dense semantic point cloud from all keyframes with proper alignment.
        """
        all_points = []
        all_colors = []
        all_labels = []
        global_label_mapping = {0: 'background'}
        next_global_id = 1
        
        print(f"\nCreating dense semantic reconstruction V2...")
        print(f"Processing {len(keyframes)} keyframes")
        
        # Process each keyframe
        processed_keyframes = 0
        total_labeled_points = 0
        
        for kf_idx in range(len(keyframes)):
            keyframe = keyframes[kf_idx]
            if keyframe is None or keyframe.X_canon is None:
                print(f"  Skipping keyframe {kf_idx}: No data")
                continue
            
            # Get semantic data
            semantic_data = semantic_keyframes.get_semantics(kf_idx)
            if semantic_data is None or not semantic_data.get('masks_rle'):
                print(f"  Skipping keyframe {kf_idx}: No semantic data")
                continue
            
            # Project this keyframe
            points, colors, labels, label_names = self.project_semantic_keyframe_v2(
                keyframe, 
                semantic_data,
                kf_idx,
                confidence_threshold
            )
            
            if len(points) > 0:
                # Remap local labels to global IDs
                global_labels = np.zeros_like(labels)
                for local_id, name in label_names.items():
                    if local_id == 0:  # Skip background
                        continue
                    
                    # Find or create global ID for this label name
                    if name not in global_label_mapping.values():
                        global_label_mapping[next_global_id] = name
                        global_id = next_global_id
                        next_global_id += 1
                    else:
                        global_id = [k for k, v in global_label_mapping.items() if v == name][0]
                    
                    # Remap
                    mask = labels == local_id
                    global_labels[mask] = global_id
                
                all_points.append(points)
                all_colors.append(colors)
                all_labels.append(global_labels)
                processed_keyframes += 1
                total_labeled_points += len(points)
        
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
            for label_id, label_name in global_label_mapping.items():
                if label_id == 0:  # Skip background
                    continue
                mask = all_labels == label_id
                color = self.label_colormap.get(label_name, self.label_colormap['unknown'])
                semantic_colors[mask] = color
            all_colors = semantic_colors
        
        # Compute statistics
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        label_stats = {}
        
        print("\nFinal label distribution:")
        for label_id, count in zip(unique_labels, counts):
            label_name = global_label_mapping.get(label_id, f"unknown_{label_id}")
            percentage = (count / len(all_labels)) * 100
            label_stats[label_name] = {
                'count': int(count),
                'percentage': float(percentage)
            }
            print(f"  {label_name}: {count} points ({percentage:.1f}%)")
        
        return {
            'points': all_points,
            'colors': all_colors,
            'labels': all_labels,
            'num_points': len(all_points),
            'num_keyframes': processed_keyframes,
            'label_mapping': global_label_mapping,
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


def create_dense_semantic_reconstruction_v2(keyframes, 
                                          semantic_keyframes,
                                          output_path: str,
                                          confidence_threshold: float = 0.3,
                                          use_semantic_colors: bool = True,
                                          debug: bool = True):
    """High-level function to create and save dense semantic reconstruction with fixes."""
    
    reconstructor = DenseSemanticReconstructorV2(debug=debug)
    
    # Create dense point cloud
    result = reconstructor.create_dense_semantic_pointcloud_v2(
        keyframes,
        semantic_keyframes,
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
        f.write("Dense Semantic Reconstruction V2 Statistics\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total points: {result['num_points']:,}\n")
        f.write(f"Keyframes used: {result['num_keyframes']}\n")
        f.write(f"\nLabel distribution:\n")
        
        for label_name, stats in sorted(result['label_stats'].items()):
            f.write(f"  {label_name}: {stats['count']:,} points ({stats['percentage']:.1f}%)\n")
    
    print(f"\nSaved statistics to {stats_path}")
    
    return result