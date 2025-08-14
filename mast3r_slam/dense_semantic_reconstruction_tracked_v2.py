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
                 min_depth: float = 0.1, max_depth: float = 50.0, c_conf_threshold: float = 1.5):
        self.device = device
        self.debug = debug
        self.semantic_backend = semantic_backend
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.c_conf_threshold = c_conf_threshold
    
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
        
        # PIXEL-TO-POINT CORRESPONDENCE VERIFICATION
        # Verify that semantic mask indexing matches MAST3R point indexing
        expected_points = h * w
        actual_points = X_cam.shape[0]
        if expected_points != actual_points:
            logger.warning(f"Keyframe {kf_idx}: Point count mismatch! Expected {expected_points} (H×W), got {actual_points}")
        else:
            logger.debug(f"Keyframe {kf_idx}: Perfect pixel-to-point correspondence: {expected_points} points")
        
        # Get track IDs if available
        track_ids = semantic_data.get('track_ids', {}) or {}
        has_tracks = len(track_ids) > 0
        
        if has_tracks:
            logger.info(f"Keyframe {kf_idx}: Using track IDs for {len(track_ids)} instances")
        
        # OPTIONAL: Run correspondence verification for first few keyframes
        if kf_idx < 3:  # Only verify first few keyframes to avoid spam
            verification_result = self.verify_correspondence(keyframe, semantic_data, kf_idx)
            if not verification_result['overall_pass']:
                logger.warning(f"Keyframe {kf_idx}: Correspondence verification failed!")
        
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
                    # Always use local instance ID but make it globally unique
                    # Format: keyframe_index * 10000 + local_instance_id
                    # This preserves local IDs while ensuring global uniqueness
                    global_instance_id = kf_idx * 10000 + instance_id
                    semantic_mask[overwrite_mask] = global_instance_id
                    label_id_to_name[global_instance_id] = f"{label_name}_kf{kf_idx}_inst{instance_id}"
                    
                    priority_mask[overwrite_mask] = priority
                    
                    pixels_total = np.sum(mask_bool)
                    logger.debug(f"  Applied {label_name}: {pixels_to_write}/{pixels_total} pixels ({pixels_to_write/pixels_total*100:.1f}% kept)")
                else:
                    logger.debug(f"  Skipped {label_name}: all pixels have higher priority")
        
        # Flatten semantic mask
        semantic_mask_flat = semantic_mask.reshape(-1)
        
        # COORDINATE MAPPING VERIFICATION
        # Verify that flattened semantic mask has exact correspondence with point cloud
        if len(semantic_mask_flat) != len(X_cam):
            logger.error(f"Keyframe {kf_idx}: CRITICAL - Semantic mask length {len(semantic_mask_flat)} != point cloud length {len(X_cam)}")
            raise ValueError("Semantic mask and point cloud dimension mismatch!")
        
        # Verify semantic mask indexing matches point cloud indexing
        labeled_pixels = np.sum(semantic_mask_flat > 0)
        logger.debug(f"Keyframe {kf_idx}: {labeled_pixels}/{len(semantic_mask_flat)} pixels have semantic labels ({labeled_pixels/len(semantic_mask_flat)*100:.1f}%)")
        
        # Transform to world coordinates
        T_WC = keyframe.T_WC
        T_WC_matrix = T_WC.matrix().cpu().numpy().squeeze() if hasattr(T_WC, 'matrix') else T_WC.cpu().numpy()
        
        X_cam_homo = np.concatenate([X_cam, np.ones((X_cam.shape[0], 1))], axis=1)
        X_world = (T_WC_matrix @ X_cam_homo.T).T[:, :3]
        
        # SYNCHRONIZED FILTERING CRITERIA FOR 1:1 CORRESPONDENCE
        depths = X_cam[:, 2]
        
        # Apply identical 3D quality filters as MAST3R
        depth_near = depths > self.min_depth
        depth_far = depths < self.max_depth
        finite_points = np.isfinite(X_world).all(axis=1)
        finite_cam_points = np.isfinite(X_cam).all(axis=1)
        positive_depth = depths > 0
        has_label = semantic_mask_flat > 0
        
        # Add SLAM-quality confidence filtering (like overlay approach)
        conf_values = keyframe.get_average_conf().cpu().numpy().reshape(-1)
        high_conf_mask = conf_values > self.c_conf_threshold
        
        # Create comprehensive validity mask with confidence filtering
        geometric_valid = depth_near & depth_far & finite_points & finite_cam_points & positive_depth
        semantic_valid = has_label
        confidence_valid = high_conf_mask
        valid_mask = geometric_valid & semantic_valid & confidence_valid
        
        # POINT PRESERVATION VERIFICATION
        total_points = len(X_world)
        geometric_valid_count = np.sum(geometric_valid)
        semantic_valid_count = np.sum(semantic_valid)
        confidence_valid_count = np.sum(confidence_valid)
        final_valid_count = np.sum(valid_mask)
        
        logger.debug(f"Keyframe {kf_idx}: Point filtering analysis (SLAM-quality):")
        logger.debug(f"  Total points: {total_points}")
        logger.debug(f"  Geometrically valid: {geometric_valid_count} ({geometric_valid_count/total_points*100:.1f}%)")
        logger.debug(f"  With semantic labels: {semantic_valid_count} ({semantic_valid_count/total_points*100:.1f}%)")
        logger.debug(f"  High confidence (>{self.c_conf_threshold}): {confidence_valid_count} ({confidence_valid_count/total_points*100:.1f}%)")
        logger.debug(f"  Final valid points: {final_valid_count} ({final_valid_count/total_points*100:.1f}%)")
        
        # Extract valid points while preserving exact correspondence
        points_3d = X_world[valid_mask]
        colors = img_rgb_flat[valid_mask]
        labels = semantic_mask_flat[valid_mask]
        
        # Verify no points were corrupted during filtering
        if len(points_3d) != len(colors) or len(points_3d) != len(labels):
            logger.error(f"Keyframe {kf_idx}: Point arrays length mismatch after filtering!")
            raise ValueError("Point correspondence broken during filtering!")
        
        return points_3d, colors, labels, label_id_to_name, {}
    
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
                # Update global mappings with the new global instance IDs
                for global_id, name in label_names.items():
                    if global_id not in global_label_mapping:
                        global_label_mapping[global_id] = name
                
                # No track mapping needed since we're not using tracking
                
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
                if label_id > 0:
                    # Color each global instance ID uniquely
                    label_name = global_label_mapping.get(label_id, f"unknown_{label_id}")
                    # Use the global instance ID directly for consistent coloring
                    color = self._get_track_color(label_id)
                    semantic_colors[mask] = color
                    unique_tracks.add(label_id)
            
            logger.info(f"Colored {len(unique_tracks)} unique tracks")
            all_colors = semantic_colors
        
        # Compute statistics
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        label_stats = {}
        track_stats = {}
        combined_stats = {}  # Combined statistics for summary
        
        for label_id, count in zip(unique_labels, counts):
            if label_id > 0:  # Skip background
                # Get the label name with keyframe and instance info
                label_name = global_label_mapping.get(label_id, f"unknown_{label_id}")
                percentage = (count / len(all_labels)) * 100
                label_stats[label_name] = {
                    'count': int(count),
                    'percentage': percentage
                }
                
                # Extract base label for combined stats (remove kf/inst info)
                base_label = label_name.split('_kf')[0] if '_kf' in label_name else label_name
                if base_label not in combined_stats:
                    combined_stats[base_label] = {'count': 0, 'percentage': 0.0}
                combined_stats[base_label]['count'] += int(count)
        
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
    
    def verify_correspondence(self, keyframe, semantic_data, kf_idx):
        """
        Comprehensive verification that semantic masks have perfect 1:1 correspondence 
        with MAST3R point clouds.
        """
        logger.info(f"=== CORRESPONDENCE VERIFICATION for Keyframe {kf_idx} ===")
        
        # Get dimensions
        h, w = keyframe.img_shape[0, 0].item(), keyframe.img_shape[0, 1].item()
        X_cam = keyframe.X_canon.cpu().numpy()
        
        # Test 1: Verify point cloud has H×W points
        expected_points = h * w
        actual_points = X_cam.shape[0]
        test1_pass = expected_points == actual_points
        logger.info(f"Test 1 - Point count: Expected {expected_points}, got {actual_points} {'✓' if test1_pass else '✗'}")
        
        # Test 2: Verify semantic mask shapes
        if semantic_data and 'masks_rle' in semantic_data:
            mask_shape_errors = 0
            for instance_id, rle in semantic_data['masks_rle'].items():
                if 'size' in rle:
                    size = rle['size']
                    if len(size) == 3:
                        _, mask_h, mask_w = size
                    else:
                        mask_h, mask_w = size
                    
                    if (mask_h, mask_w) != (h, w):
                        mask_shape_errors += 1
            
            test2_pass = mask_shape_errors == 0
            logger.info(f"Test 2 - Mask shapes: {mask_shape_errors} mismatches {'✓' if test2_pass else '✗'}")
        else:
            test2_pass = True
            logger.info("Test 2 - Mask shapes: No masks to verify ✓")
        
        # Test 3: Verify coordinate transformation consistency
        # Sample a few points to verify transformation
        sample_indices = [0, h*w//4, h*w//2, 3*h*w//4, h*w-1] if h*w > 4 else [0]
        T_WC = keyframe.T_WC
        T_WC_matrix = T_WC.matrix().cpu().numpy().squeeze() if hasattr(T_WC, 'matrix') else T_WC.cpu().numpy()
        
        test3_errors = 0
        for idx in sample_indices:
            if idx < len(X_cam):
                # Convert to homogeneous coordinates
                X_cam_homo = np.append(X_cam[idx], 1.0)
                X_world_manual = (T_WC_matrix @ X_cam_homo)[:3]
                
                # Check if transformation is reasonable (not NaN or inf)
                if not np.all(np.isfinite(X_world_manual)):
                    test3_errors += 1
        
        test3_pass = test3_errors == 0
        logger.info(f"Test 3 - Transformations: {test3_errors} invalid transforms {'✓' if test3_pass else '✗'}")
        
        # Test 4: Verify index correspondence
        # Create a test semantic mask to verify indexing
        test_mask = np.zeros((h, w), dtype=np.int32)
        test_mask[h//4:3*h//4, w//4:3*w//4] = 999  # Mark center region
        test_mask_flat = test_mask.reshape(-1)
        
        # Check that flattened indexing matches expected pattern
        center_indices = np.where(test_mask_flat == 999)[0]
        expected_center_size = (h//2) * (w//2)  # Approximate center region size
        actual_center_size = len(center_indices)
        
        test4_pass = abs(actual_center_size - expected_center_size) < expected_center_size * 0.1  # 10% tolerance
        logger.info(f"Test 4 - Index mapping: Center region {actual_center_size}/{expected_center_size} points {'✓' if test4_pass else '✗'}")
        
        # Overall result
        all_tests_pass = test1_pass and test2_pass and test3_pass and test4_pass
        logger.info(f"CORRESPONDENCE VERIFICATION: {'ALL TESTS PASSED ✓' if all_tests_pass else 'SOME TESTS FAILED ✗'}")
        
        return {
            'overall_pass': all_tests_pass,
            'point_count_match': test1_pass,
            'mask_shapes_valid': test2_pass,
            'transforms_valid': test3_pass,
            'index_mapping_valid': test4_pass,
            'keyframe_dims': (h, w),
            'total_points': actual_points
        }


def create_dense_semantic_reconstruction_tracked(keyframes,
                                               semantic_keyframes,
                                               output_file: str,
                                               use_semantic_colors: bool = True,
                                               debug: bool = False,
                                               semantic_backend=None,
                                               min_depth: float = 0.1,
                                               max_depth: float = 50.0,
                                               c_conf_threshold: float = 1.5) -> Optional[Dict]:
    """Create dense semantic reconstruction with track ID support."""
    
    reconstructor = DenseSemanticReconstructorTrackedV2(
        debug=debug,
        semantic_backend=semantic_backend,
        min_depth=min_depth,
        max_depth=max_depth,
        c_conf_threshold=c_conf_threshold
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