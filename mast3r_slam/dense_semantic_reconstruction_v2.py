"""Dense semantic reconstruction with proper alignment - Reduced version."""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import cv2
from plyfile import PlyData, PlyElement
from mast3r_slam.semantic_frame import decode_rle
import logging
import json

logger = logging.getLogger('mast3r_slam.dense_reconstruction')


class DenseSemanticReconstructorV2:
    """Create dense semantic point clouds with proper mask-depth alignment."""
    
    def __init__(self, device: str = "cuda", debug: bool = False, semantic_backend=None, 
                 mask_merge_strategy: str = "confidence"):
        self.device = device
        self.debug = debug
        self.semantic_backend = semantic_backend
        self.mask_merge_strategy = mask_merge_strategy  # "confidence" or "multilabel"
    
    def _get_instance_color(self, category_name: str) -> np.ndarray:
        """Get a consistent, visually distinct color for each semantic category."""
        import hashlib
        
        # Normalize the category name to ensure consistency
        # Remove common articles and convert to lowercase
        normalized = category_name.lower().strip()
        # Remove common prefixes
        for prefix in ['a ', 'an ', 'the ', 'some ']:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
        # Take the first word if multiple words (e.g., "wooden chair" -> "wooden")
        # But keep the full phrase for now to see what's happening
        
        # Use SHA256 for better distribution
        hash_value = int(hashlib.sha256(normalized.encode()).hexdigest()[:8], 16)
        
        # Use golden ratio for better color distribution
        golden_ratio = 0.618033988749895
        hue = (hash_value * golden_ratio) % 1.0
        
        # Convert HSV to RGB for better color distribution
        # High saturation and value for vibrant colors
        saturation = 0.7 + (hash_value % 30) / 100.0  # 0.7-1.0
        value = 0.8 + (hash_value % 20) / 100.0       # 0.8-1.0
        
        # HSV to RGB conversion
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
        
        # Convert to 0-255 range
        r = int((r + m) * 255)
        g = int((g + m) * 255)
        b = int((b + m) * 255)
        
        return np.array([r, g, b], dtype=np.uint8)
    
    def project_semantic_keyframe_v2(self, 
                                   keyframe,
                                   semantic_data: Dict,
                                   kf_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Project segmented pixels from a keyframe into 3D space."""
        # Get keyframe dimensions
        h, w = keyframe.img_shape[0, 0].item(), keyframe.img_shape[0, 1].item()
        
        # Get 3D points and RGB
        X_cam = keyframe.X_canon.cpu().numpy()  # (H*W, 3)
        img_rgb = (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8)
        img_rgb_flat = img_rgb.reshape(-1, 3)
        
        # Debug: Check if we have valid 3D points
        logger.info(f"  3D point statistics:")
        logger.info(f"    Total pixels: {h*w}")
        logger.info(f"    X_cam shape: {X_cam.shape}")
        logger.info(f"    Valid (non-zero) 3D points: {np.sum(np.any(X_cam != 0, axis=1))}")
        
        # Create semantic mask
        semantic_mask = np.zeros((h, w), dtype=np.int32)
        confidence_mask = np.zeros((h, w), dtype=np.float32)
        label_id_to_name = {0: 'background'}
        next_label_id = 1
        
        # Process semantic data
        if semantic_data and 'masks_rle' in semantic_data:
            logger.info(f"Keyframe {kf_idx}: Processing {len(semantic_data['masks_rle'])} semantic masks")
            logger.info(f"  Labels present: {list(semantic_data.get('labels', {}).values())}")
            
            # Debug: Print detailed label mapping
            logger.info("  Instance ID -> Label mapping:")
            for inst_id, label in semantic_data.get('labels', {}).items():
                conf = semantic_data.get('confidences', {}).get(inst_id, 0.0)
                logger.info(f"    Instance {inst_id}: '{label}' (conf={conf:.3f})")
            
            # Debug: Save mask overlap visualization if requested
            if self.debug:
                import json
                from pathlib import Path
                debug_dir = Path("debug_semantic_pipeline") / f"keyframe_{kf_idx:06d}"
                debug_dir.mkdir(parents=True, exist_ok=True)
                
                # Save all mask info before processing
                mask_info = {
                    'keyframe_idx': kf_idx,
                    'keyframe_shape': (h, w),
                    'instances': {}
                }
                for instance_id, rle in semantic_data['masks_rle'].items():
                    mask_info['instances'][str(instance_id)] = {
                        'label': semantic_data.get('labels', {}).get(instance_id, 'unknown'),
                        'confidence': semantic_data.get('confidences', {}).get(instance_id, 0.0),
                        'rle_size': rle.get('size', []),
                        'rle_counts_length': len(rle.get('counts', []))
                    }
                with open(debug_dir / "mask_info.json", 'w') as f:
                    json.dump(mask_info, f, indent=2)
            
            # Sort instances by a combined score
            instances = []
            size_mismatches = []
            
            for instance_id, rle in semantic_data['masks_rle'].items():
                if 'size' in rle:
                    size = rle['size']
                    if len(size) == 3:
                        _, mask_h, mask_w = size
                    else:
                        mask_h, mask_w = size
                    
                    label_name = semantic_data.get('labels', {}).get(instance_id, 'unknown')
                    confidence = semantic_data.get('confidences', {}).get(instance_id, 0.5)
                    
                    if (mask_h, mask_w) == (h, w):
                        # Decode mask to get its size
                        temp_mask = decode_rle(rle, mask_h, mask_w)
                        if isinstance(temp_mask, torch.Tensor):
                            temp_mask = temp_mask.cpu().numpy()
                        mask_size_ratio = np.sum(temp_mask) / (h * w)
                        
                        # Combined score: higher confidence is better, smaller masks get bonus
                        # This naturally prioritizes small, high-confidence objects
                        size_weight = 1.0 / (1.0 + mask_size_ratio * 10)  # Smaller masks get higher weight
                        combined_score = confidence * (0.5 + 0.5 * size_weight)
                        
                        instances.append((instance_id, combined_score, confidence, mask_size_ratio))
                    else:
                        size_mismatches.append({
                            'instance_id': instance_id,
                            'label': label_name,
                            'mask_size': (mask_w, mask_h),
                            'keyframe_size': (w, h),
                            'confidence': confidence
                        })
            
            if size_mismatches:
                logger.warning(f"Keyframe {kf_idx}: {len(size_mismatches)} masks have size mismatch:")
                for mismatch in size_mismatches:
                    logger.warning(f"  - {mismatch['label']}: mask {mismatch['mask_size']} != keyframe {mismatch['keyframe_size']}")
            
            # Sort by mask size (largest first)
            # This way, small objects will overwrite large background objects
            instances.sort(key=lambda x: -x[3])  # Sort by mask_size_ratio (descending)
            logger.info(f"Keyframe {kf_idx}: {len(instances)} masks match keyframe dimensions")
            
            # Debug: show sorting order
            if self.debug:
                logger.info("  Processing order (largest to smallest):")
                for instance_id, combined_score, conf, size_ratio in instances[:10]:  # Show top 10
                    label = semantic_data.get('labels', {}).get(instance_id, 'unknown')
                    logger.info(f"    {label}: size={size_ratio*100:.1f}% (conf={conf:.3f})")
            
            # Process each instance
            for instance_id, _, _, _ in instances:
                rle = semantic_data['masks_rle'][instance_id]
                size = rle['size']
                if len(size) == 3:
                    _, mask_h, mask_w = size
                else:
                    mask_h, mask_w = size
                
                if (mask_h, mask_w) != (h, w):
                    continue
                
                # Decode mask
                mask = decode_rle(rle, mask_h, mask_w)
                label_name = semantic_data.get('labels', {}).get(instance_id, 'unknown')
                confidence = semantic_data.get('confidences', {}).get(instance_id, 1.0)
                
                # Process all masks without filtering
                # Get or create label ID
                if label_name not in label_id_to_name.values():
                    label_id_to_name[next_label_id] = label_name
                    label_id = next_label_id
                    if self.debug:
                        logger.info(f"    Assigning new label_id {label_id} to '{label_name}'")
                    next_label_id += 1
                else:
                    label_id = [k for k, v in label_id_to_name.items() if v == label_name][0]
                    if self.debug:
                        logger.info(f"    Reusing label_id {label_id} for '{label_name}'")
                
                # Convert mask to boolean if needed
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                mask_bool = mask.astype(bool) if mask.dtype != bool else mask
                
                # Calculate mask statistics
                pixels_in_mask = np.sum(mask_bool)
                mask_size_ratio = pixels_in_mask / (h * w)
                
                # Simple overwrite strategy - since we process large to small,
                # small objects will naturally overwrite large background objects
                semantic_mask[mask_bool] = label_id
                confidence_mask[mask_bool] = confidence
                
                # Log application result
                logger.info(f"  Applied {label_name} (conf={confidence:.3f}) to {pixels_in_mask} pixels ({mask_size_ratio*100:.1f}% of image)")
                
                # Debug large masks
                if mask_size_ratio > 0.3:  # More than 30% of image
                    logger.warning(f"    NOTE: Large mask - {label_name} covers {pixels_in_mask/(h*w)*100:.1f}% of image")
        
        # Flatten semantic mask
        semantic_mask_flat = semantic_mask.reshape(-1)
        
        # Transform to world coordinates
        T_WC = keyframe.T_WC
        T_WC_matrix = T_WC.matrix().cpu().numpy().squeeze() if hasattr(T_WC, 'matrix') else T_WC.cpu().numpy()
        
        X_cam_homo = np.concatenate([X_cam, np.ones((X_cam.shape[0], 1))], axis=1)
        X_world = (T_WC_matrix @ X_cam_homo.T).T[:, :3]
        
        # Filter valid points
        depths = X_cam[:, 2]
        
        # Track filtering statistics
        total_points = len(depths)
        depth_near = depths > 0.1
        depth_far = depths < 50.0
        finite_points = np.isfinite(X_world).all(axis=1)
        has_label = semantic_mask_flat > 0
        
        # Apply basic 3D quality filters
        valid_mask = depth_near & depth_far & finite_points & has_label
        
        # Log filtering statistics
        logger.info(f"Keyframe {kf_idx} point filtering:")
        logger.info(f"  Total points: {total_points}")
        logger.info(f"  After depth > 0.1m: {np.sum(depth_near)} ({np.sum(depth_near)/total_points*100:.1f}%)")
        logger.info(f"  After depth < 50m: {np.sum(depth_near & depth_far)} ({np.sum(depth_near & depth_far)/total_points*100:.1f}%)")
        logger.info(f"  After finite check: {np.sum(depth_near & depth_far & finite_points)} ({np.sum(depth_near & depth_far & finite_points)/total_points*100:.1f}%)")
        logger.info(f"  After semantic label: {np.sum(valid_mask)} ({np.sum(valid_mask)/total_points*100:.1f}%)")
        
        # Check which labels are being filtered
        labeled_but_filtered = has_label & ~valid_mask
        if np.any(labeled_but_filtered):
            filtered_labels = semantic_mask_flat[labeled_but_filtered]
            unique_filtered, counts = np.unique(filtered_labels, return_counts=True)
            logger.warning(f"  Labels filtered due to depth/finite constraints:")
            for label_id, count in zip(unique_filtered, counts):
                if label_id > 0:
                    label_name = label_id_to_name.get(label_id, f'unknown_{label_id}')
                    logger.warning(f"    - {label_name}: {count} points")
        
        # Debug: Show what happened to each label
        logger.info("  Label survival after filtering:")
        semantic_labels_after_filter = semantic_mask_flat[valid_mask]
        for label_id, label_name in label_id_to_name.items():
            if label_id == 0:
                continue
            total_labeled = np.sum(semantic_mask_flat == label_id)
            survived = np.sum(semantic_labels_after_filter == label_id)
            if total_labeled > 0:
                survival_rate = (survived / total_labeled) * 100
                logger.info(f"    - {label_name}: {total_labeled} labeled -> {survived} survived ({survival_rate:.1f}%)")
        
        points_3d = X_world[valid_mask]
        colors = img_rgb_flat[valid_mask]
        labels = semantic_mask_flat[valid_mask]
        
        return points_3d, colors, labels, label_id_to_name
    
    def create_dense_semantic_pointcloud_v2(self,
                                          keyframes,
                                          semantic_keyframes,
                                          use_semantic_colors: bool = True) -> Dict:
        """Create a dense semantic point cloud from all keyframes."""
        all_points = []
        all_colors = []
        all_labels = []
        global_label_mapping = {0: 'background'}
        next_global_id = 1
        
        processed_keyframes = 0
        
        # Process each keyframe
        for kf_idx in range(len(keyframes)):
            keyframe = keyframes[kf_idx]
            if keyframe is None or keyframe.X_canon is None:
                continue
            
            semantic_data = semantic_keyframes.get_semantics(kf_idx)
            if semantic_data is None or not semantic_data.get('masks_rle'):
                continue
            
            # Project keyframe
            points, colors, labels, label_names = self.project_semantic_keyframe_v2(
                keyframe, semantic_data, kf_idx
            )
            
            if len(points) > 0:
                # Remap local labels to global IDs
                global_labels = np.zeros_like(labels)
                
                for local_id, name in label_names.items():
                    if local_id == 0:
                        continue
                    
                    # Find or create global ID
                    if name not in global_label_mapping.values():
                        global_label_mapping[next_global_id] = name
                        global_id = next_global_id
                        next_global_id += 1
                    else:
                        global_id = [k for k, v in global_label_mapping.items() if v == name][0]
                    
                    mask = labels == local_id
                    global_labels[mask] = global_id
                
                all_points.append(points)
                all_colors.append(colors)
                all_labels.append(global_labels)
                processed_keyframes += 1
        
        if len(all_points) == 0:
            logger.warning("No semantic points found!")
            return {}
        
        # Concatenate all points
        all_points = np.concatenate(all_points, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Apply semantic colors if requested
        if use_semantic_colors:
            semantic_colors = np.zeros_like(all_colors)
            
            # Debug: Print global label mapping
            logger.info("Global label mapping for coloring:")
            for label_id, label_name in global_label_mapping.items():
                logger.info(f"  Label ID {label_id} -> '{label_name}'")
            
            for label_id in np.unique(all_labels):
                mask = all_labels == label_id
                if label_id > 0:
                    # Get category name for consistent color
                    label_name = global_label_mapping.get(label_id, f"unknown_{label_id}")
                    color = self._get_instance_color(label_name)
                    num_points = np.sum(mask)
                    # Log both original and normalized names for debugging
                    normalized = label_name.lower().strip()
                    for prefix in ['a ', 'an ', 'the ', 'some ']:
                        if normalized.startswith(prefix):
                            normalized = normalized[len(prefix):]
                            break
                    logger.info(f"  Coloring {num_points} points with label_id={label_id} ('{label_name}' -> normalized: '{normalized}') -> RGB{tuple(color)}")
                else:
                    # Background/unlabeled pixels
                    color = np.array([128, 128, 128])
                semantic_colors[mask] = color
            all_colors = semantic_colors
        
        # Compute statistics
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        label_stats = {}
        
        for label_id, count in zip(unique_labels, counts):
            label_name = global_label_mapping.get(label_id, f"unknown_{label_id}")
            percentage = (count / len(all_labels)) * 100
            label_stats[label_name] = {
                'count': int(count),
                'percentage': float(percentage)
            }
            logger.info(f"{label_name}: {count} points ({percentage:.1f}%)")
        
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
                              labels: np.ndarray,
                              label_mapping: Dict[int, str] = None):
        """Save dense semantic point cloud to PLY file."""
        # Create structured array
        vertex_data = np.zeros(
            len(points),
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
        
        # Write PLY
        vertex_element = PlyElement.describe(vertex_data, 'vertex')
        ply_data = PlyData([vertex_element], text=False)
        ply_data.write(filename)
        logger.info(f"Saved dense semantic point cloud to {filename}")
        
        # Save label mapping
        if label_mapping:
            mapping_file = Path(filename).with_suffix('.labels.json')
            with open(mapping_file, 'w') as f:
                json.dump({'label_to_name': label_mapping}, f, indent=2)


def create_dense_semantic_reconstruction_v2(keyframes, 
                                          semantic_keyframes,
                                          output_path: str,
                                          use_semantic_colors: bool = True,
                                          debug: bool = False,
                                          semantic_backend=None,
                                          mask_merge_strategy: str = "confidence"):
    """Create and save dense semantic reconstruction."""
    
    reconstructor = DenseSemanticReconstructorV2(debug=debug, semantic_backend=semantic_backend,
                                                 mask_merge_strategy=mask_merge_strategy)
    
    result = reconstructor.create_dense_semantic_pointcloud_v2(
        keyframes, semantic_keyframes, use_semantic_colors
    )
    
    if not result:
        return None
    
    # Save PLY file
    reconstructor.save_dense_semantic_ply(
        output_path,
        result['points'],
        result['colors'],
        result['labels'],
        result.get('label_mapping', {})
    )
    
    # Save basic statistics
    stats_path = Path(output_path).with_suffix('.txt')
    with open(stats_path, 'w') as f:
        f.write(f"Total points: {result['num_points']:,}\n")
        f.write(f"Keyframes used: {result['num_keyframes']}\n")
        f.write("\nLabel distribution:\n")
        for label_name, stats in sorted(result['label_stats'].items()):
            f.write(f"  {label_name}: {stats['count']:,} ({stats['percentage']:.1f}%)\n")
    
    return result