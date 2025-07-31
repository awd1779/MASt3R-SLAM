"""Dense semantic reconstruction with track ID support - Fixed version."""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import cv2
from plyfile import PlyData, PlyElement
from mast3r_slam.semantic_frame import decode_rle
import logging
import json

logger = logging.getLogger('mast3r_slam.dense_reconstruction_tracked')


class DenseSemanticReconstructorTrackedV2:
    """Create dense semantic point clouds with track ID coloring."""
    
    def __init__(self, device: str = "cuda", debug: bool = False, semantic_backend=None,
                 min_depth: float = 0.1, max_depth: float = 50.0):
        self.device = device
        self.debug = debug
        self.semantic_backend = semantic_backend
        self.min_depth = min_depth
        self.max_depth = max_depth
    
    def _get_track_color(self, track_id: int) -> np.ndarray:
        """Get a consistent color for each track ID."""
        import hashlib
        
        # Use golden ratio for better color distribution
        golden_ratio = 0.618033988749895
        hue = (track_id * golden_ratio) % 1.0
        
        # Convert HSV to RGB
        saturation = 0.8
        value = 0.9
        
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
            
        r = int((r + m) * 255)
        g = int((g + m) * 255)
        b = int((b + m) * 255)
        
        return np.array([r, g, b], dtype=np.uint8)
    
    def project_semantic_keyframe_tracked(self, keyframe, semantic_data, kf_idx):
        """Project a single keyframe to 3D with track ID support."""
        
        # Get keyframe dimensions
        h, w = keyframe.img_shape[0, 0].item(), keyframe.img_shape[0, 1].item()
        
        # Get 3D points and RGB
        X_cam = keyframe.X_canon.cpu().numpy()  # Already (H*W, 3)
        img_rgb = (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8)  # (3, H, W)
        img_rgb_flat = img_rgb.reshape(3, -1).T  # Reshape to (H*W, 3)
        
        # Get track IDs if available
        track_ids = semantic_data.get('track_ids', {})
        has_tracks = len(track_ids) > 0
        
        if has_tracks:
            logger.info(f"Keyframe {kf_idx}: Using track IDs for {len(track_ids)} instances")
        
        # Create semantic mask
        semantic_mask = np.zeros((h, w), dtype=np.int32)
        priority_mask = np.zeros((h, w), dtype=np.float32)  # Track pixel priorities
        label_id_to_name = {0: 'background'}
        track_id_to_label = {}  # Map track IDs to labels
        
        # Process masks with priority system
        if semantic_data and 'masks_rle' in semantic_data:
            # First, collect all valid masks and calculate their priorities
            mask_instances = []
            
            for instance_id, rle in semantic_data['masks_rle'].items():
                # Get mask dimensions from RLE
                if 'size' in rle:
                    size = rle['size']
                    if len(size) == 3:
                        _, mask_h, mask_w = size
                    else:
                        mask_h, mask_w = size
                else:
                    mask_h, mask_w = h, w
                
                # Only process masks that match keyframe dimensions
                if (mask_h, mask_w) != (h, w):
                    logger.warning(f"Skipping mask {instance_id} with size {mask_h}x{mask_w} != {h}x{w}")
                    continue
                
                # Decode mask
                mask = decode_rle(rle, mask_h, mask_w)
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                
                # Calculate mask statistics
                mask_bool = mask.astype(bool) if mask.dtype != bool else mask
                pixels_in_mask = np.sum(mask_bool)
                if pixels_in_mask == 0:
                    continue
                    
                mask_size_ratio = pixels_in_mask / (h * w)
                label_name = semantic_data.get('labels', {}).get(instance_id, 'unknown')
                confidence = semantic_data.get('confidences', {}).get(instance_id, 0.5)
                
                # Calculate priority (smaller objects get higher priority)
                size_priority = 1.0 - mask_size_ratio  # Smaller = higher priority
                label_priority = 0.5 if label_name != 'unknown' and label_name != 'background' else 0.0
                confidence_priority = confidence * 0.3
                combined_priority = size_priority + label_priority + confidence_priority
                
                mask_instances.append({
                    'instance_id': instance_id,
                    'mask': mask_bool,
                    'label_name': label_name,
                    'priority': combined_priority,
                    'size_ratio': mask_size_ratio,
                    'confidence': confidence
                })
            
            # Sort by priority (highest first)
            mask_instances.sort(key=lambda x: -x['priority'])
            
            logger.info(f"Keyframe {kf_idx}: Processing {len(mask_instances)} masks by priority")
            if len(mask_instances) > 0:
                logger.info("  Top 5 masks by priority:")
                for i, inst in enumerate(mask_instances[:5]):
                    logger.info(f"    {i+1}. {inst['label_name']}: priority={inst['priority']:.3f}, size={inst['size_ratio']*100:.1f}%")
            
            # Apply masks in priority order
            for inst in mask_instances:
                instance_id = inst['instance_id']
                mask_bool = inst['mask']
                label_name = inst['label_name']
                priority = inst['priority']
                
                # Only overwrite pixels if we have higher priority
                overwrite_mask = mask_bool & (priority > priority_mask)
                pixels_to_write = np.sum(overwrite_mask)
                
                if pixels_to_write > 0:
                    # Use track ID if available, otherwise use instance ID
                    if has_tracks and instance_id in track_ids:
                        track_id = track_ids[instance_id]
                        semantic_mask[overwrite_mask] = track_id
                        label_id_to_name[track_id] = f"{label_name}_track{track_id}"
                        track_id_to_label[track_id] = label_name
                    else:
                        semantic_mask[overwrite_mask] = instance_id
                        label_id_to_name[instance_id] = label_name
                    
                    priority_mask[overwrite_mask] = priority
                    
                    pixels_total = np.sum(mask_bool)
                    logger.debug(f"  Applied {label_name}: {pixels_to_write}/{pixels_total} pixels ({pixels_to_write/pixels_total*100:.1f}% kept)")
                else:
                    logger.debug(f"  Skipped {label_name}: all pixels have higher priority")
        
        # Flatten semantic mask
        semantic_mask_flat = semantic_mask.reshape(-1)
        
        # Transform to world coordinates
        T_WC = keyframe.T_WC
        T_WC_matrix = T_WC.matrix().cpu().numpy().squeeze() if hasattr(T_WC, 'matrix') else T_WC.cpu().numpy()
        
        X_cam_homo = np.concatenate([X_cam, np.ones((X_cam.shape[0], 1))], axis=1)
        X_world = (T_WC_matrix @ X_cam_homo.T).T[:, :3]
        
        # Filter valid points
        depths = X_cam[:, 2]
        
        # Apply basic 3D quality filters
        depth_near = depths > self.min_depth
        depth_far = depths < self.max_depth
        finite_points = np.isfinite(X_world).all(axis=1)
        has_label = semantic_mask_flat > 0
        
        valid_mask = depth_near & depth_far & finite_points & has_label
        
        points_3d = X_world[valid_mask]
        colors = img_rgb_flat[valid_mask]
        labels = semantic_mask_flat[valid_mask]
        
        return points_3d, colors, labels, label_id_to_name, track_id_to_label
    
    def create_dense_semantic_pointcloud_tracked(self,
                                                keyframes, 
                                                semantic_keyframes,
                                                use_semantic_colors=True):
        """Create dense point cloud with track ID support."""
        
        all_points = []
        all_colors = []
        all_labels = []
        global_label_mapping = {0: 'background'}
        global_track_mapping = {}  # Track which track IDs we've seen
        processed_keyframes = 0
        
        logger.info("Creating dense semantic reconstruction with track IDs...")
        
        for kf_idx in range(len(keyframes)):
            keyframe = keyframes[kf_idx]
            if keyframe is None or keyframe.X_canon is None:
                continue
                
            if not semantic_keyframes.has_semantic_data(kf_idx):
                continue
                
            semantic_data = semantic_keyframes.get_semantics(kf_idx)
            if semantic_data is None:
                continue
            
            # Project keyframe with track support
            points, colors, labels, label_names, track_to_label = self.project_semantic_keyframe_tracked(
                keyframe, semantic_data, kf_idx
            )
            
            if len(points) > 0:
                # Update global mappings
                for local_id, name in label_names.items():
                    if local_id not in global_label_mapping:
                        global_label_mapping[local_id] = name
                
                for track_id, label in track_to_label.items():
                    if track_id not in global_track_mapping:
                        global_track_mapping[track_id] = label
                
                all_points.append(points)
                all_colors.append(colors)
                all_labels.append(labels)
                processed_keyframes += 1
        
        if len(all_points) == 0:
            logger.warning("No valid points found in any keyframe!")
            return None
        
        # Concatenate all data
        all_points = np.concatenate(all_points, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Apply semantic colors if requested
        if use_semantic_colors:
            logger.info(f"Applying semantic colors to {len(all_points)} points...")
            semantic_colors = np.copy(all_colors)
            
            # Color by track ID if available
            unique_tracks = set()
            for label_id in np.unique(all_labels):
                mask = all_labels == label_id
                if label_id > 0 and label_id in global_track_mapping:
                    # This is a track ID
                    color = self._get_track_color(label_id)
                    semantic_colors[mask] = color
                    unique_tracks.add(label_id)
                elif label_id > 0:
                    # Fall back to instance coloring
                    label_name = global_label_mapping.get(label_id, f"unknown_{label_id}")
                    # Simple color based on label ID
                    color = self._get_track_color(label_id * 100)  # Multiply to spread colors
                    semantic_colors[mask] = color
            
            logger.info(f"Colored {len(unique_tracks)} unique tracks")
            all_colors = semantic_colors
        
        # Compute statistics
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        label_stats = {}
        track_stats = {}
        combined_stats = {}  # Combined statistics for summary
        
        for label_id, count in zip(unique_labels, counts):
            if label_id in global_track_mapping:
                # Track statistic
                track_label = global_track_mapping[label_id]
                if track_label not in track_stats:
                    track_stats[track_label] = {'count': 0, 'tracks': []}
                track_stats[track_label]['count'] += count
                track_stats[track_label]['tracks'].append(label_id)
                
                # Also add to combined stats
                if track_label not in combined_stats:
                    combined_stats[track_label] = {'count': 0, 'percentage': 0.0}
                combined_stats[track_label]['count'] += count
            else:
                # Regular label statistic
                label_name = global_label_mapping.get(label_id, f"unknown_{label_id}")
                percentage = (count / len(all_labels)) * 100
                label_stats[label_name] = {
                    'count': int(count),
                    'percentage': percentage
                }
                
                # Also add to combined stats
                if label_name not in combined_stats:
                    combined_stats[label_name] = {'count': 0, 'percentage': 0.0}
                combined_stats[label_name]['count'] += int(count)
        
        # Calculate percentages for combined stats
        for label in combined_stats:
            combined_stats[label]['percentage'] = (combined_stats[label]['count'] / len(all_labels)) * 100
        
        # Summarize track statistics
        logger.info("\nTrack Statistics:")
        for label, stats in track_stats.items():
            logger.info(f"  {label}: {len(stats['tracks'])} tracks, {stats['count']} total points")
        
        # Special logging for debugging chair/sofa
        chair_found = False
        sofa_found = False
        for label, stats in track_stats.items():
            if 'chair' in label.lower():
                chair_found = True
                logger.info(f"\n✓ CHAIR FOUND: {stats['count']:,} points across {len(stats['tracks'])} tracks")
            elif 'sofa' in label.lower():
                sofa_found = True
                logger.info(f"✓ SOFA FOUND: {stats['count']:,} points across {len(stats['tracks'])} tracks")
        
        if not chair_found:
            logger.warning("\n✗ CHAIR NOT FOUND in track statistics!")
        if not sofa_found:
            logger.warning("✗ SOFA NOT FOUND in track statistics!")
        
        return {
            'points': all_points,
            'colors': all_colors,
            'labels': all_labels,
            'num_points': len(all_points),
            'num_keyframes': processed_keyframes,
            'label_mapping': global_label_mapping,
            'track_mapping': global_track_mapping,
            'label_stats': combined_stats,  # Use combined stats for summary
            'track_stats': track_stats,
            'detailed_label_stats': label_stats  # Keep original for detailed analysis
        }
    
    def save_ply_tracked(self, filename: str,
                        points: np.ndarray,
                        colors: np.ndarray,
                        labels: np.ndarray,
                        label_mapping: Dict[int, str] = None,
                        track_mapping: Dict[int, str] = None):
        """Save dense semantic point cloud with track information."""
        
        n_points = len(points)
        
        # Create vertex data
        vertex_data = np.zeros(n_points, 
                              dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                                    ('label', 'i4')])
        
        vertex_data['x'] = points[:, 0]
        vertex_data['y'] = points[:, 1]
        vertex_data['z'] = points[:, 2]
        vertex_data['red'] = colors[:, 0]
        vertex_data['green'] = colors[:, 1]
        vertex_data['blue'] = colors[:, 2]
        vertex_data['label'] = labels
        
        # Write PLY
        vertex_element = PlyElement.describe(vertex_data, 'vertex')
        ply_data = PlyData([vertex_element])
        ply_data.write(filename)
        
        logger.info(f"Saved {n_points} points to {filename}")
        
        # Save mappings
        mapping_file = Path(filename).with_suffix('.tracking.json')
        mappings = {
            'label_to_name': label_mapping or {},
            'track_to_label': track_mapping or {}
        }
        with open(mapping_file, 'w') as f:
            json.dump(mappings, f, indent=2)
        logger.info(f"Saved tracking information to {mapping_file}")


def create_dense_semantic_reconstruction_tracked(keyframes,
                                               semantic_keyframes,
                                               output_file: str,
                                               use_semantic_colors: bool = True,
                                               debug: bool = False,
                                               semantic_backend=None,
                                               min_depth: float = 0.1,
                                               max_depth: float = 50.0) -> Optional[Dict]:
    """Create dense semantic reconstruction with track ID support."""
    
    reconstructor = DenseSemanticReconstructorTrackedV2(
        debug=debug,
        semantic_backend=semantic_backend,
        min_depth=min_depth,
        max_depth=max_depth
    )
    
    result = reconstructor.create_dense_semantic_pointcloud_tracked(
        keyframes,
        semantic_keyframes,
        use_semantic_colors=use_semantic_colors
    )
    
    if result is None:
        return None
    
    # Save to file
    reconstructor.save_ply_tracked(
        output_file,
        result['points'],
        result['colors'],
        result['labels'],
        result.get('label_mapping', {}),
        result.get('track_mapping', {})
    )
    
    return result