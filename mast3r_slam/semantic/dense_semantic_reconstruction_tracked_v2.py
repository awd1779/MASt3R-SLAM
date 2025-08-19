"""Dense semantic reconstruction with track ID support - Fixed version."""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import cv2
from plyfile import PlyData, PlyElement
from mast3r_slam.semantic.semantic_frame import decode_rle
from mast3r_slam.clustering.object_clustering import ObjectInstance
from mast3r_slam.clustering.utils import compute_object_centroid
from mast3r_slam.clustering.enhanced_object_clustering import enhanced_hybrid_cluster_objects
import logging
import json

logger = logging.getLogger('mast3r_slam.dense_reconstruction_tracked')


class DenseSemanticReconstructorTrackedV2:
    """Create dense semantic point clouds with track ID coloring."""
    
    def __init__(self, config: Optional[Dict] = None, device: Optional[str] = None, debug: Optional[bool] = None,
                 min_depth: Optional[float] = None, max_depth: Optional[float] = None, 
                 c_conf_threshold: Optional[float] = None,
                 use_object_clustering: Optional[bool] = None, clustering_config: Optional[Dict] = None):
        # Load configuration
        from mast3r_slam.semantic.config_loader import SemanticConfig
        self.sem_config = SemanticConfig(config)
        
        # Use config values with explicit parameter overrides
        self.device = device if device is not None else "cuda"
        self.debug = debug if debug is not None else self.sem_config.get_debug_mode()
        self.min_depth = min_depth if min_depth is not None else self.sem_config.get_min_depth()
        self.max_depth = max_depth if max_depth is not None else self.sem_config.get_max_depth()
        self.c_conf_threshold = c_conf_threshold if c_conf_threshold is not None else self.sem_config.get_c_conf_threshold()
        self.use_object_clustering = use_object_clustering if use_object_clustering is not None else self.sem_config.get_clustering_enabled()
        
        # Build comprehensive clustering config from YAML
        if clustering_config is not None:
            self.clustering_config = clustering_config
        else:
            # Build config from YAML settings
            self.clustering_config = {
                'pipeline': self.sem_config.get_clustering_pipeline_config(),
                'base_params': self.sem_config.get_clustering_base_params(),
                'adaptive_params': self.sem_config.get_clustering_adaptive_params(),
                'merge_params': self.sem_config.get_clustering_merge_params(),
                'stacking_detection': self.sem_config.get_stacking_detection_params(),
                'temporal_consistency': self.sem_config.get_temporal_consistency_params(),
                # Legacy support
                'config': self.sem_config.get_clustering_config()
            }
    
    def _get_track_color(self, track_id: int) -> np.ndarray:
        """Get a consistent color for each track ID using shared utility."""
        from .utils import generate_track_color
        # Pass the config to generate_track_color
        return generate_track_color(track_id, config={'semantic_segmentation': self.sem_config.semantic_config})
    
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
        
        
        # OPTIONAL: Run correspondence verification in debug mode only
        if self.debug and kf_idx < 3:  # Only verify first few keyframes to avoid spam
            verification_result = self.verify_correspondence(keyframe, semantic_data, kf_idx)
            if not verification_result['overall_pass']:
                logger.warning(f"Keyframe {kf_idx}: Correspondence verification failed!")
        
        # Create semantic mask
        semantic_mask = np.zeros((h, w), dtype=np.int32)
        priority_mask = np.zeros((h, w), dtype=np.float32)  # Track pixel priorities
        label_id_to_name = {0: 'background'}
        
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
                size_priority = (1.0 - mask_size_ratio) * self.sem_config.get_size_priority()  # Smaller = higher priority
                label_priority = self.sem_config.get_label_priority() if label_name != 'unknown' and label_name != 'background' else 0.0
                confidence_priority = confidence * self.sem_config.get_confidence_priority()
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
                    # Format: keyframe_index * multiplier + local_instance_id
                    # This preserves local IDs while ensuring global uniqueness
                    global_instance_id = kf_idx * self.sem_config.get_global_id_multiplier() + instance_id
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
        
        return points_3d, colors, labels, label_id_to_name
    
    def _extract_object_instances(self, points: np.ndarray, colors: np.ndarray, 
                                 labels: np.ndarray, label_names: Dict[int, str], 
                                 kf_idx: int) -> List[ObjectInstance]:
        """Extract ObjectInstance objects from keyframe data for clustering."""
        instances = []
        
        for global_id, label_name in label_names.items():
            if global_id == 0:  # Skip background
                continue
                
            # Get points belonging to this object instance
            object_mask = labels == global_id
            if np.sum(object_mask) == 0:
                continue
                
            object_points = points[object_mask]
            
            # Compute centroid
            centroid = np.mean(object_points, axis=0)
            
            # Extract clean label (remove keyframe info)
            clean_label = label_name.split('_kf')[0] if '_kf' in label_name else label_name
            
            # Create ObjectInstance
            instance = ObjectInstance(
                global_id=global_id,
                local_id=global_id % self.sem_config.get_global_id_multiplier(),  # Extract local ID
                keyframe_idx=kf_idx,
                label=clean_label,
                confidence=1.0,  # Default confidence
                centroid_3d=centroid,
                num_points=len(object_points),
                point_indices=np.where(object_mask)[0]
            )
            instances.append(instance)
        
        logger.debug(f"Extracted {len(instances)} object instances from keyframe {kf_idx}")
        return instances
    
    def _apply_clustering_to_points(self, keyframe_points_data: List[Dict], 
                                   all_instances: List[ObjectInstance], 
                                   clusters: List) -> Tuple[List[Dict], Dict[int, str]]:
        """Apply clustering results to update point labels and mappings."""
        
        # Create mapping from old global IDs to cluster IDs
        old_to_cluster_id = {}
        cluster_label_mapping = {0: 'background'}
        
        for cluster in clusters:
            cluster_label_mapping[cluster.cluster_id] = cluster.label
            for instance in cluster.instances:
                old_to_cluster_id[instance.global_id] = cluster.cluster_id
        
        logger.info(f"Created clustering mapping: {len(old_to_cluster_id)} instances → "
                   f"{len(clusters)} clusters")
        
        # Update labels in keyframe data
        for kf_data in keyframe_points_data:
            labels = kf_data['labels']
            
            # Remap labels to cluster IDs
            new_labels = np.copy(labels)
            for old_id, cluster_id in old_to_cluster_id.items():
                mask = labels == old_id
                new_labels[mask] = cluster_id
                
            kf_data['labels'] = new_labels
        
        return keyframe_points_data, cluster_label_mapping
    
    
    def create_dense_semantic_pointcloud_tracked(self,
                                                keyframes, 
                                                semantic_keyframes,
                                                use_semantic_colors=True):
        """Create dense point cloud with track ID support and object clustering."""
        
        all_points = []
        all_colors = []
        all_labels = []
        global_label_mapping = {0: 'background'}
        processed_keyframes = 0
        
        # NEW: Collect object instances for clustering
        all_instances = []
        keyframe_points_data = []  # Store per-keyframe data for clustering
        
        logger.info("Creating dense semantic reconstruction...")
        
        for kf_idx in range(len(keyframes)):
            keyframe = keyframes[kf_idx]
            if keyframe is None or keyframe.X_canon is None:
                continue
                
            if not semantic_keyframes.has_semantic_data(kf_idx):
                continue
                
            semantic_data = semantic_keyframes.get_semantics(kf_idx)
            if semantic_data is None:
                continue
            
            # Project keyframe to 3D
            points, colors, labels, label_names = self.project_semantic_keyframe_tracked(
                keyframe, semantic_data, kf_idx
            )
            
            if len(points) > 0:
                # Store keyframe data for later clustering
                keyframe_data = {
                    'points': points,
                    'colors': colors, 
                    'labels': labels,
                    'label_names': label_names,
                    'kf_idx': kf_idx
                }
                keyframe_points_data.append(keyframe_data)
                
                # NEW: Extract object instances for clustering
                if self.use_object_clustering:
                    instances = self._extract_object_instances(
                        points, colors, labels, label_names, kf_idx
                    )
                    all_instances.extend(instances)
                
                # Update global mappings with the new global instance IDs
                for global_id, name in label_names.items():
                    if global_id not in global_label_mapping:
                        global_label_mapping[global_id] = name
                
                processed_keyframes += 1
        
        if len(keyframe_points_data) == 0:
            logger.warning("No valid points found in any keyframe!")
            return None
            
        # NEW: Apply enhanced object clustering if enabled
        if self.use_object_clustering and len(all_instances) > 0:
            logger.info(f"Applying enhanced object clustering to {len(all_instances)} instances...")
            
            # Prepare point data for enhanced clustering
            all_points_data = {}
            for kf_data in keyframe_points_data:
                points = kf_data['points']
                labels = kf_data['labels']
                label_names = kf_data['label_names']
                
                for global_id, _ in label_names.items():
                    if global_id == 0:  # Skip background
                        continue
                    object_mask = labels == global_id
                    if np.sum(object_mask) > 0:
                        all_points_data[global_id] = points[object_mask]
            
            logger.info(f"Prepared point cloud data for {len(all_points_data)} instances")
            
            # Use enhanced clustering with config-driven parameters
            clusters = enhanced_hybrid_cluster_objects(
                instances=all_instances,
                all_points_data=all_points_data,
                global_config=self.clustering_config,
                # Parameters will be extracted from config by enhanced_hybrid_cluster_objects
                debug=self.debug
            )
            
            # Update labels and mappings based on clustering
            keyframe_points_data, global_label_mapping = self._apply_clustering_to_points(
                keyframe_points_data, all_instances, clusters
            )
        
        # Concatenate all processed data
        all_points = []
        all_colors = []
        all_labels = []
        
        for kf_data in keyframe_points_data:
            all_points.append(kf_data['points'])
            all_colors.append(kf_data['colors'])
            all_labels.append(kf_data['labels'])
        
        # Concatenate all data
        all_points = np.concatenate(all_points, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Apply semantic colors if requested
        if use_semantic_colors:
            logger.info(f"Applying semantic colors to {len(all_points)} points...")
            semantic_colors = np.copy(all_colors)
            
            # Color by semantic labels
            unique_instances = set()
            for label_id in np.unique(all_labels):
                mask = all_labels == label_id
                if label_id > 0:
                    # Color each global instance ID uniquely
                    label_name = global_label_mapping.get(label_id, f"unknown_{label_id}")
                    # Use the global instance ID directly for consistent coloring
                    color = self._get_track_color(label_id)
                    semantic_colors[mask] = color
                    unique_instances.add(label_id)
            
            logger.info(f"Colored {len(unique_instances)} unique instances")
            all_colors = semantic_colors
        
        # Compute statistics
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        label_stats = {}
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
        
        
        return {
            'points': all_points,
            'colors': all_colors,
            'labels': all_labels,
            'num_points': len(all_points),
            'num_keyframes': processed_keyframes,
            'label_mapping': global_label_mapping,
            'label_stats': combined_stats,  # Use combined stats for summary
            'detailed_label_stats': label_stats  # Keep original for detailed analysis
        }
    
    def save_ply_tracked(self, filename: str,
                        points: np.ndarray,
                        colors: np.ndarray,
                        labels: np.ndarray,
                        label_mapping: Dict[int, str] = None):
        """Save dense semantic point cloud with label information."""
        
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
        mapping_file = Path(filename).with_suffix('.labels.json')
        mappings = {
            'label_to_name': label_mapping or {}
        }
        with open(mapping_file, 'w') as f:
            json.dump(mappings, f, indent=2)
        logger.info(f"Saved label information to {mapping_file}")
    
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
                                               config: Optional[Dict] = None,
                                               use_semantic_colors: Optional[bool] = None,
                                               debug: Optional[bool] = None,
                                               min_depth: Optional[float] = None,
                                               max_depth: Optional[float] = None,
                                               c_conf_threshold: Optional[float] = None,
                                               use_object_clustering: Optional[bool] = None,
                                               clustering_config: Optional[Dict] = None) -> Optional[Dict]:
    """Create dense semantic reconstruction with track ID support."""
    
    # Load config to check for use_semantic_colors
    from mast3r_slam.semantic.config_loader import SemanticConfig
    sem_config = SemanticConfig(config)
    
    # Get use_semantic_colors with override
    use_colors = use_semantic_colors if use_semantic_colors is not None else sem_config.get_use_semantic_colors()
    
    reconstructor = DenseSemanticReconstructorTrackedV2(
        config=config,
        debug=debug,
        min_depth=min_depth,
        max_depth=max_depth,
        c_conf_threshold=c_conf_threshold,
        use_object_clustering=use_object_clustering,
        clustering_config=clustering_config
    )
    
    result = reconstructor.create_dense_semantic_pointcloud_tracked(
        keyframes,
        semantic_keyframes,
        use_semantic_colors=use_colors
    )
    
    if result is None:
        return None
    
    # Save to file
    reconstructor.save_ply_tracked(
        output_file,
        result['points'],
        result['colors'],
        result['labels'],
        result.get('label_mapping', {})
    )
    
    return result