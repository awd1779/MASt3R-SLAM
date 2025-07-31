"""Geometric 3D Object Tracker - Phase 2 Implementation.

This tracker uses 3D bounding boxes and geometric matching to track objects
across frames, providing much better consistency than 2D-only tracking.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
import logging
import time
from dataclasses import dataclass, field

from mast3r_slam.bbox_3d_generic import (
    Generic3DBoundingBox,
    AdaptiveBBoxEstimator,
    BBox3DConfig,
    create_generic_3d_bbox
)
from mast3r_slam.semantic_frame import decode_rle
from mast3r_slam.label_based_tracker import LabelBasedTracker
from mast3r_slam.robust_geometric_matcher import RobustGeometricMatcher

logger = logging.getLogger('mast3r_slam.geometric_3d_tracker')


@dataclass
class TrackedObject3D:
    """Represents a tracked object with 3D geometry."""
    track_id: int
    label: str
    first_seen_frame: int
    last_seen_frame: int
    
    # Camera coordinates (for current view matching)
    bbox_3d_cam: Optional[Dict] = None
    last_centroid_cam: Optional[torch.Tensor] = None
    
    # World coordinates (stable reference)
    centroid_world: Optional[torch.Tensor] = None
    bbox_3d_world: Optional[Dict] = None
    position_variance_world: float = 0.0  # Track stability metric
    
    # Tracking state
    confidence: float = 1.0
    lost_frames: int = 0
    total_observations: int = 0
    last_T_WC: Optional[torch.Tensor] = None  # Last camera pose
    
    # History for smoothing/prediction
    centroid_history_cam: List[torch.Tensor] = field(default_factory=list)
    world_observations: List[torch.Tensor] = field(default_factory=list)
    bbox_history: List[Dict] = field(default_factory=list)
    max_history: int = 10
    
    def update_geometry(self, bbox_cam: Dict, bbox_world: Dict, frame_id: int):
        """Update object's 3D geometry in both camera and world coordinates."""
        self.bbox_3d_cam = bbox_cam
        self.last_centroid_cam = bbox_cam['center']
        self.bbox_3d_world = bbox_world
        self.centroid_world = bbox_world['center']
        self.last_seen_frame = frame_id
        self.lost_frames = 0
        self.total_observations += 1
        
        # Update history
        self.centroid_history_cam.append(bbox_cam['center'].clone())
        self.world_observations.append(bbox_world['center'].clone())
        self.bbox_history.append(bbox_cam)
        
        # Limit history size
        if len(self.centroid_history_cam) > self.max_history:
            self.centroid_history_cam.pop(0)
            self.world_observations.pop(0)
            self.bbox_history.pop(0)
    
    def mark_lost(self):
        """Mark object as lost for current frame."""
        self.lost_frames += 1
        self.confidence *= 0.95  # Decay confidence
    
    def get_predicted_position(self) -> Optional[torch.Tensor]:
        """Simple position prediction based on motion history in camera space."""
        if len(self.centroid_history_cam) < 2:
            return self.last_centroid_cam
        
        # Simple linear prediction in camera space
        velocity = self.centroid_history_cam[-1] - self.centroid_history_cam[-2]
        predicted = self.centroid_history_cam[-1] + velocity
        return predicted
    
    def get_world_position_variance(self) -> float:
        """Calculate variance of world positions to measure stability."""
        if len(self.world_observations) < 2:
            return 0.0
        
        positions = torch.stack(self.world_observations)
        variance = torch.std(positions, dim=0).max().item()
        return variance


class Geometric3DTracker:
    """3D Geometric Object Tracker using adaptive bounding boxes."""
    
    def __init__(self, config: Dict):
        self.enabled = config.get('enabled', True)
        
        # Tracking thresholds
        self.iou_threshold_3d = config.get('geometric_iou_threshold', 0.3)
        self.centroid_distance_threshold = config.get('centroid_distance_threshold', 0.5)
        self.max_lost_frames = config.get('max_lost_frames', 10)
        self.min_observations = config.get('min_observations', 2)
        
        # Matching weights
        self.weight_iou = config.get('weight_iou', 0.6)
        self.weight_centroid = config.get('weight_centroid', 0.3)
        self.weight_size = config.get('weight_size', 0.1)
        
        # 3D bbox system
        bbox_config = BBox3DConfig(
            min_points_ratio=config.get('min_points_ratio', 0.001),
            outlier_std_factor=config.get('outlier_std_factor', 3.0),
            min_depth=config.get('min_depth', 0.1),
            max_depth=config.get('max_depth', 50.0)
        )
        self.bbox_estimator = AdaptiveBBoxEstimator()
        self.bbox_computer = Generic3DBoundingBox(bbox_config, self.bbox_estimator)
        
        # World-centric tracking configuration
        self.use_world_coordinates = config.get('use_world_coordinates', True)
        self.world_thresholds = config.get('world_distance_thresholds', {
            'default': 0.3,
            'furniture': 0.5,
            'large': 1.0,
            'small': 0.2
        })
        self.max_position_variance = config.get('max_position_variance', 0.2)
        self.position_variance_warning = config.get('position_variance_warning', 0.1)
        self.log_world_positions = config.get('log_world_positions', True)
        self.world_position_interval = config.get('world_position_interval', 10)
        
        # Tracking state
        self.tracked_objects: Dict[int, TrackedObject3D] = {}
        self.next_track_id = 1
        self.frame_count = 0
        
        # Performance monitoring
        self.timing_stats = {
            'bbox_extraction': [],
            'matching': [],
            'total': []
        }
        
        # Initialize label-based tracker for unique objects
        # Empty list - no hardcoded assumptions about unique objects
        self.label_tracker = LabelBasedTracker({
            'unique_objects': []
        })
        
        logger.info("Initialized Geometric3DTracker with 3D matching and label-based tracking")
    
    def process_frame(self, 
                     keyframe,
                     semantic_data: Dict,
                     frame_id: int) -> Dict[int, int]:
        """Process a frame and return instance_id -> track_id mapping."""
        
        start_time = time.time()
        self.frame_count = frame_id
        
        logger.info(f"=== Geometric3DTracker processing frame {frame_id} ===")
        logger.info(f"  Current tracks: {len(self.tracked_objects)}")
        logger.info(f"  Semantic instances: {len(semantic_data.get('instance_ids', []))}")
        
        # Log unique object tracking status
        unique_stats = self.label_tracker.get_unique_object_stats()
        if unique_stats['tracked_unique_objects'] > 0:
            logger.info(f"  Tracked unique objects: {unique_stats['unique_object_tracks']}")
        
        # Extract 3D bboxes for all detected objects
        bbox_start = time.time()
        current_detections = self._extract_3d_bboxes(keyframe, semantic_data)
        bbox_time = time.time() - bbox_start
        
        # Match with existing tracks
        match_start = time.time()
        assignments = self._match_detections_to_tracks(current_detections, frame_id)
        match_time = time.time() - match_start
        
        # Update tracking statistics
        total_time = time.time() - start_time
        self._update_timing_stats(bbox_time, match_time, total_time)
        
        # Mark lost tracks
        self._update_lost_tracks(assignments, frame_id)
        
        # Clean up old lost tracks
        self._cleanup_lost_tracks()
        
        # Log statistics periodically
        if frame_id % 30 == 0:
            self._log_tracking_stats()
        
        # Log world positions separately if enabled
        if self.log_world_positions and frame_id % self.world_position_interval == 0:
            self._log_world_positions()
        
        return assignments
    
    def _transform_to_world(self, bbox_cam: Dict, T_WC) -> Dict:
        """Transform 3D bounding box from camera to world coordinates.
        
        Args:
            bbox_cam: Bounding box in camera coordinates
            T_WC: Camera-to-world transformation (lietorch.Sim3 or torch.Tensor)
        
        Returns:
            bbox_world: Bounding box in world coordinates
        """
        # Extract T_WC matrix if it's a lietorch object
        if hasattr(T_WC, 'matrix'):
            T_WC_matrix = T_WC.matrix().squeeze()
        else:
            T_WC_matrix = T_WC
        
        # Debug: log matrix shape
        logger.debug(f"T_WC_matrix shape: {T_WC_matrix.shape}")
        
        # Ensure it's on the same device as bbox data
        device = bbox_cam['center'].device
        if T_WC_matrix.device != device:
            T_WC_matrix = T_WC_matrix.to(device)
        
        # Transform center point
        center_cam = bbox_cam['center']
        # Ensure center_cam is 3D
        if center_cam.shape[-1] == 4:
            center_cam = center_cam[..., :3]
        
        # Handle both 4x4 and 3x4 transformation matrices
        center_homo = torch.cat([center_cam, torch.tensor([1.0], device=device)])
        
        # Always do matrix multiplication first
        center_world = T_WC_matrix @ center_homo
        logger.debug(f"Transform result shape: {center_world.shape}, T_WC shape: {T_WC_matrix.shape}")
        
        # Handle different result shapes
        if center_world.dim() > 1:
            center_world = center_world.squeeze()
            logger.debug(f"After squeeze: {center_world.shape}")
        
        # Always extract first 3 elements (x, y, z) regardless of matrix type
        if center_world.shape[0] >= 3:
            center_world = center_world[:3]
            logger.debug(f"After extraction: {center_world.shape}")
        else:
            raise ValueError(f"Invalid transformation result shape: {center_world.shape}")
        
        # Final safety check
        assert center_world.shape == torch.Size([3]), f"center_world has shape {center_world.shape}, expected torch.Size([3])"
        
        # Transform all 8 corners if available
        if 'corners' in bbox_cam:
            corners_cam = bbox_cam['corners']  # 8x3
            corners_homo = torch.cat([corners_cam, torch.ones((8, 1), device=device)], dim=1)
            
            # Transform all corners
            corners_world = (T_WC_matrix @ corners_homo.T).T
            
            # Always extract first 3 columns (x, y, z)
            if corners_world.shape[1] >= 3:
                corners_world = corners_world[:, :3]
            else:
                raise ValueError(f"Invalid corner transformation result shape: {corners_world.shape}")
        else:
            corners_world = None
        
        # Create world bbox
        bbox_world = {
            'center': center_world,
            'corners': corners_world,
            'dimensions': bbox_cam['dimensions'],  # Size unchanged
            'volume': bbox_cam['volume'],
            'confidence': bbox_cam['confidence'],
            'type': bbox_cam['type']
        }
        
        return bbox_world
    
    def _extract_3d_bboxes(self, keyframe, semantic_data: Dict) -> Dict[int, Dict]:
        """Extract 3D bounding boxes in both camera and world coordinates."""
        
        # Debug logging
        logger.debug(f"_extract_3d_bboxes called with semantic_data keys: {list(semantic_data.keys())}")
        logger.debug(f"masks_rle type: {type(semantic_data.get('masks_rle'))}, content: {semantic_data.get('masks_rle')}")
        
        detections = {}
        
        # Handle img_shape - it might be a tensor or already extracted values
        if hasattr(keyframe.img_shape, 'shape') and len(keyframe.img_shape.shape) > 1:
            h, w = keyframe.img_shape[0, 0].item(), keyframe.img_shape[0, 1].item()
        else:
            # Assume it's already a simple tensor
            h, w = keyframe.img_shape[0].item(), keyframe.img_shape[1].item()
        
        # Get camera pose for world transformation
        T_WC = keyframe.T_WC
        
        # Store current camera pose for tracking updates
        self.current_T_WC = T_WC
        
        # Log camera pose for debugging
        if hasattr(T_WC, 'matrix'):
            T_WC_mat = T_WC.matrix().squeeze()
        else:
            T_WC_mat = T_WC
        
        # Extract translation component for logging
        if T_WC_mat.shape[0] >= 3 and T_WC_mat.shape[1] >= 4:
            translation = T_WC_mat[:3, 3].cpu().numpy()
            logger.info(f"Frame {self.frame_count} camera position in world: [{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}]")
        
        for instance_id, mask_rle in semantic_data.get('masks_rle', {}).items():
            label = semantic_data.get('labels', {}).get(instance_id, 'unknown')
            
            # Decode mask
            mask = decode_rle(mask_rle, h, w)
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            
            # Create 3D bbox in camera coordinates
            bbox_cam = create_generic_3d_bbox(
                keyframe, mask, label,
                config=self.bbox_computer.config,
                estimator=self.bbox_estimator
            )
            
            if bbox_cam is not None and bbox_cam['size_valid']:
                # Transform to world coordinates
                bbox_world = self._transform_to_world(bbox_cam, T_WC)
                
                detections[instance_id] = {
                    'bbox_cam': bbox_cam,
                    'bbox_world': bbox_world,
                    'label': label,
                    'instance_id': instance_id
                }
                
                # Log both camera and world positions
                cam_center = bbox_cam['center'].cpu().numpy()
                world_center = bbox_world['center'].cpu().numpy()
                
                # Ensure arrays are properly shaped
                if cam_center.ndim == 0:
                    cam_center = cam_center.reshape(1)
                if world_center.ndim == 0:
                    world_center = world_center.reshape(1)
                    
                # Extract values safely
                cam_vals = [cam_center.flat[i] if i < cam_center.size else 0.0 for i in range(3)]
                world_vals = [world_center.flat[i] if i < world_center.size else 0.0 for i in range(3)]
                
                logger.debug(f"Extracted 3D bbox for {label} (instance {instance_id}): "
                           f"cam_pos=[{cam_vals[0]:.2f}, {cam_vals[1]:.2f}, {cam_vals[2]:.2f}], "
                           f"world_pos=[{world_vals[0]:.2f}, {world_vals[1]:.2f}, {world_vals[2]:.2f}]")
        
        logger.info(f"Extracted {len(detections)} valid 3D bboxes from "
                   f"{len(semantic_data.get('masks_rle', {}))} detections")
        
        # Log details about extracted bboxes
        for instance_id, det in detections.items():
            bbox_cam = det['bbox_cam']
            logger.debug(f"  {det['label']} (instance {instance_id}): "
                        f"{bbox_cam['extraction_stats']['valid_3d_points']} points, "
                        f"volume={bbox_cam['volume']:.3f}m³")
        
        return detections
    
    def _match_detections_to_tracks(self, 
                                   detections: Dict[int, Dict],
                                   frame_id: int) -> Dict[int, int]:
        """Match detections to tracks using 3D IoU and world coordinates."""
        
        assignments = {}
        used_tracks = set()
        
        # Sort detections by confidence for stable matching
        sorted_detections = sorted(
            detections.items(),
            key=lambda x: x[1]['bbox_cam']['confidence'],
            reverse=True
        )
        
        for instance_id, detection in sorted_detections:
            best_track_id = None
            best_score = 0.0  # Higher is better for matching score
            
            # Check if this is a unique object that should have only one track
            if self.label_tracker.is_unique_object(detection['label']):
                existing_track_id = self.label_tracker.get_track_for_unique_object(detection['label'])
                logger.info(f"Checking unique object '{detection['label']}': existing_track_id={existing_track_id}")
                
                if existing_track_id is not None:
                    if existing_track_id in self.tracked_objects:
                        if existing_track_id not in used_tracks:
                            # Force match to existing track for unique objects
                            logger.info(f"Forcing match for unique object '{detection['label']}' to track {existing_track_id}")
                            best_track_id = existing_track_id
                            best_score = 1.0  # Maximum confidence
                            used_tracks.add(existing_track_id)  # Mark as used immediately
                        else:
                            logger.warning(f"Track {existing_track_id} for '{detection['label']}' already used in this frame")
                    else:
                        logger.warning(f"Track {existing_track_id} for '{detection['label']}' no longer exists")
                else:
                    logger.info(f"No existing track for unique object '{detection['label']}' yet")
            
            # If not forced match, search existing tracks with same label
            if best_track_id is None:
                for track_id, track in self.tracked_objects.items():
                    if track_id in used_tracks:
                        continue
                    
                    if track.label != detection['label']:
                        continue
                    
                    # Skip if lost too long
                    if track.lost_frames > self.max_lost_frames:
                        continue
                    
                    # ROBUST 3D MATCHING
                    if track.bbox_3d_world is not None and track.centroid_world is not None:
                        # Compute matching score using multiple criteria
                        matching_score = self._compute_robust_matching_score(
                            detection, track, frame_id
                        )
                    
                        # Enhanced debug logging for table tracking
                        if detection['label'] == 'a table':
                            det_center = detection['bbox_world']['center'].cpu().numpy()
                            track_center = track.centroid_world.cpu().numpy()
                            dist = np.linalg.norm(det_center - track_center)
                            logger.info(f"TABLE MATCHING - Frame {frame_id}, Track {track_id}:")
                            logger.info(f"  Detection pos: [{det_center[0]:.2f}, {det_center[1]:.2f}, {det_center[2]:.2f}]")
                            logger.info(f"  Track pos: [{track_center[0]:.2f}, {track_center[1]:.2f}, {track_center[2]:.2f}]")
                            logger.info(f"  Distance: {dist:.3f}m")
                            logger.info(f"  Matching score: {matching_score:.3f}")
                            logger.info(f"  Threshold: {self.iou_threshold_3d}")
                        
                        # Debug logging for important objects
                        elif detection['label'] in ['a sofa', 'a wall'] and matching_score > 0:
                            logger.debug(f"Matching {detection['label']} to track {track_id}: "
                                       f"score={matching_score:.3f} (best so far: {best_score:.3f})")
                        
                        # Use adaptive threshold based on matching score quality
                        # Lower threshold for high-quality matches (high containment, good distance)
                        if matching_score > 0.8:
                            threshold = 0.05  # Very confident match
                        elif matching_score > 0.5:
                            threshold = self.iou_threshold_3d * 0.7  # Good match
                        else:
                            threshold = self.iou_threshold_3d  # Standard threshold
                        
                        if matching_score > threshold and matching_score > best_score:
                            best_score = matching_score
                            best_track_id = track_id
            
            # Assign or create new track
            if best_track_id is not None:
                assignments[instance_id] = best_track_id
                if best_track_id not in used_tracks:  # Check if not already added
                    used_tracks.add(best_track_id)
                
                # Update track with new observation
                track = self.tracked_objects[best_track_id]
                self._update_track_world(track, detection, frame_id)
                
                # Check if this was a partial object match
                if track.bbox_3d_world is not None:
                    size_ratio = detection['bbox_world']['volume'] / (track.bbox_3d_world['volume'] + 1e-6)
                    if size_ratio > 2.0 and track.total_observations < 3:
                        logger.info(f"Matched partial→full {detection['label']} to track {best_track_id} "
                                   f"(size grew {size_ratio:.1f}x, score: {best_score:.3f})")
                    else:
                        logger.info(f"Matched {detection['label']} to track {best_track_id} "
                                   f"(score: {best_score:.3f})")
                else:
                    logger.info(f"Matched {detection['label']} to track {best_track_id} "
                               f"(score: {best_score:.3f})")
            else:
                # Create new track
                new_track_id = self._create_new_track(detection, frame_id)
                assignments[instance_id] = new_track_id
                
                # Register unique objects
                if self.label_tracker.is_unique_object(detection['label']):
                    self.label_tracker.assign_track_to_unique_object(detection['label'], new_track_id)
                
                world_pos = detection['bbox_world']['center'].cpu().numpy()
                logger.info(f"Created new track {new_track_id} for {detection['label']} "
                           f"at world pos [{world_pos[0]:.2f}, {world_pos[1]:.2f}, {world_pos[2]:.2f}]")
        
        return assignments
    
    def _get_world_distance_threshold(self, label: str) -> float:
        """Get distance threshold based on object type from config."""
        if hasattr(self, 'world_thresholds'):
            # Check specific object categories
            if label in ['wall', 'floor', 'ceiling']:
                return self.world_thresholds.get('large', 1.0)
            elif label in ['chair', 'table', 'sofa', 'bed', 'desk']:
                return self.world_thresholds.get('furniture', 0.5)
            elif label in ['bottle', 'cup', 'mouse', 'keyboard']:
                return self.world_thresholds.get('small', 0.2)
            else:
                return self.world_thresholds.get('default', 0.3)
        else:
            # Fallback if config not loaded
            if label in ['wall', 'floor', 'ceiling']:
                return 1.0
            elif label in ['chair', 'table', 'sofa', 'bed', 'desk']:
                return 0.5
            elif label in ['bottle', 'cup', 'mouse', 'keyboard']:
                return 0.2
            else:
                return 0.3
    
    def _update_track_world(self, track: TrackedObject3D, detection: Dict, frame_id: int):
        """Update track with new world observation."""
        
        # Track position stability
        old_world = track.centroid_world
        new_world = detection['bbox_world']['center']
        
        if old_world is not None:
            position_change = torch.norm(new_world - old_world).item()
            track.position_variance_world = max(track.position_variance_world, position_change)
            
            # Log if significant movement (potential issue)
            if position_change > self.position_variance_warning:
                logger.warning(f"Track {track.track_id} ({track.label}) moved {position_change:.3f}m in world space!")
        
        # Update track state
        track.update_geometry(detection['bbox_cam'], detection['bbox_world'], frame_id)
        
        # Store camera pose for future predictions (if available)
        if hasattr(self, 'current_T_WC'):
            track.last_T_WC = self.current_T_WC
    
    def _create_new_track(self, detection: Dict, frame_id: int) -> int:
        """Create new track with world coordinates."""
        
        new_track_id = self.next_track_id
        self.next_track_id += 1
        
        new_track = TrackedObject3D(
            track_id=new_track_id,
            label=detection['label'],
            first_seen_frame=frame_id,
            last_seen_frame=frame_id,
            # World coordinates
            centroid_world=detection['bbox_world']['center'].clone(),
            bbox_3d_world=detection['bbox_world'],
            # Camera coordinates
            bbox_3d_cam=detection['bbox_cam'],
            last_centroid_cam=detection['bbox_cam']['center'].clone()
        )
        
        self.tracked_objects[new_track_id] = new_track
        return new_track_id
    
    def _compute_robust_matching_score(self, 
                                      detection: Dict,
                                      track: TrackedObject3D,
                                      frame_id: int) -> float:
        """Compute robust matching score using 3D IoU and other metrics."""
        
        det_bbox = detection['bbox_world']
        track_bbox = track.bbox_3d_world
        
        if track_bbox is None:
            return 0.0
        
        # 1. 3D IoU - Primary matching criterion
        iou_3d = self.bbox_computer.compute_3d_iou(det_bbox, track_bbox)
        
        # Log detailed info for tables
        if detection['label'] == 'a table':
            logger.info(f"  TABLE MATCHING DETAILS:")
            logger.info(f"    3D IoU: {iou_3d:.3f}")
        
        # Calculate volumes for containment check
        det_vol = det_bbox['volume']
        track_vol = track_bbox['volume']
        
        # Calculate intersection volume for containment ratios
        # For AABB boxes, we can compute this directly
        if det_bbox['type'] == 'aabb' and track_bbox['type'] == 'aabb':
            inter_min = torch.max(det_bbox['min'], track_bbox['min'])
            inter_max = torch.min(det_bbox['max'], track_bbox['max'])
            if torch.all(inter_min < inter_max):
                inter_vol = torch.prod(inter_max - inter_min).item()
            else:
                inter_vol = 0.0
        else:
            # For other bbox types, estimate from IoU
            inter_vol = iou_3d * (det_vol + track_vol - iou_3d * (det_vol + track_vol))
        
        # Calculate containment ratios
        containment_det_in_track = inter_vol / det_vol if det_vol > 0 else 0
        containment_track_in_det = inter_vol / track_vol if track_vol > 0 else 0
        
        # Log containment for tables
        if detection['label'] == 'a table':
            logger.info(f"    Detection volume: {det_vol:.3f}m³")
            logger.info(f"    Track volume: {track_vol:.3f}m³")
            logger.info(f"    Containment det in track: {containment_det_in_track:.3f}")
            logger.info(f"    Containment track in det: {containment_track_in_det:.3f}")
        
        # Check for partial object case
        if containment_det_in_track > 0.7 or containment_track_in_det > 0.7:
            # This is likely the same object, just partially visible before
            dist = torch.norm(det_bbox['center'] - track.centroid_world).item()
            norm_dist = dist / (det_bbox['dimensions'].max().item() + 1e-6)
            
            if detection['label'] == 'a table':
                logger.info(f"    PARTIAL OBJECT CASE - distance: {dist:.3f}m, norm_dist: {norm_dist:.3f}")
            else:
                logger.debug(f"  Partial object match: containment_det={containment_det_in_track:.3f}, "
                            f"containment_track={containment_track_in_det:.3f}, dist={dist:.3f}m")
            
            # Return high score based on centroid distance
            return 0.5 + 0.5 * (1.0 / (1.0 + norm_dist))
        
        # If IoU is high enough, it's definitely the same object
        if iou_3d > 0.3:  # Strong overlap
            logger.debug(f"  High IoU match: {iou_3d:.3f}")
            return iou_3d
        
        # 2. For low/no IoU, use distance-based matching with size awareness
        dist = torch.norm(det_bbox['center'] - track.centroid_world).item()
        
        # Get object size for distance normalization
        det_size = det_bbox['dimensions'].max().item()
        track_size = track_bbox['dimensions'].max().item()
        avg_size = (det_size + track_size) / 2.0
        
        # Normalize distance by object size
        norm_dist = dist / (avg_size + 1e-6)
        
        # 3. Size consistency check with tolerance for early observations
        size_ratio = det_bbox['volume'] / (track_bbox['volume'] + 1e-6)
        
        # Allow more size variation for tracks with few observations (partial objects becoming visible)
        if track.total_observations < 3:
            # Early in tracking - allow significant growth (partial to full object)
            size_consistent = 0.2 < size_ratio < 5.0  # Allow 5x growth
            if size_ratio > 2.0:
                logger.debug(f"  Allowing size growth for early track: ratio={size_ratio:.2f}, observations={track.total_observations}")
        else:
            # Established track - normal size variation
            size_consistent = 0.5 < size_ratio < 2.0  # Allow 2x size variation
        
        # 4. Compute combined score with robust handling
        # Always compute base scores
        dist_score = 1.0 / (1.0 + norm_dist)
        size_score = 2.0 * min(size_ratio, 1/size_ratio) / (1 + min(size_ratio, 1/size_ratio))
        
        # Adaptive weighting based on context
        if iou_3d > 0.3:  # Strong overlap
            # High confidence - use IoU primarily
            score = 0.7 * iou_3d + 0.2 * dist_score + 0.1 * size_score
        elif iou_3d > 0.1:  # Some overlap
            # Moderate confidence - balanced approach
            score = 0.5 * iou_3d + 0.3 * dist_score + 0.2 * size_score
        elif containment_det_in_track > 0.5 or containment_track_in_det > 0.5:
            # Partial visibility with containment
            containment_score = max(containment_det_in_track, containment_track_in_det)
            score = 0.3 * containment_score + 0.5 * dist_score + 0.2 * size_score
        else:
            # No overlap - use distance and size if reasonable
            # Adaptive distance threshold based on object size and tracking history
            object_size = det_bbox['dimensions'].max().item()
            
            # More permissive for objects with few observations (might be partial->full)
            if track.total_observations < 3:
                distance_tolerance = 3.0
            else:
                distance_tolerance = 2.0
            
            # Size-aware distance threshold
            max_norm_dist = distance_tolerance * (1.0 + 0.5 * min(object_size, 2.0))
            
            if norm_dist < max_norm_dist and size_consistent:
                # Reasonable distance and size - might be same object
                score = 0.6 * dist_score + 0.3 * size_score + 0.1 * (1.0 - norm_dist/max_norm_dist)
                
                if detection['label'] == 'a table':
                    logger.info(f"    Distance-based match: dist={dist:.3f}m, norm_dist={norm_dist:.3f}, score={score:.3f}")
            else:
                score = 0.0  # Too far or size mismatch
        
        # 5. Boost score for temporal consistency
        if track.total_observations > 1 and score > 0:
            # Use world position history for prediction
            if len(track.world_observations) >= 2:
                # Simple velocity prediction
                pos_prev = track.world_observations[-2]
                pos_curr = track.world_observations[-1]
                velocity = pos_curr - pos_prev
                
                # Predicted position
                predicted_pos = pos_curr + velocity * (frame_id - track.last_seen_frame)
                
                # Check prediction error
                pred_error = torch.norm(det_bbox['center'] - predicted_pos).item()
                pred_error_normalized = pred_error / (avg_size + 1e-6)
                
                # Apply boost for good predictions
                if pred_error_normalized < 1.0:
                    boost = 1.0 + 0.3 * (1.0 - pred_error_normalized)
                    score *= boost
                    
                    if detection['label'] == 'a table':
                        logger.info(f"    Temporal boost: pred_error={pred_error:.3f}m, boost={boost:.2f}")
        
        return score
    
    def _update_lost_tracks(self, assignments: Dict[int, int], frame_id: int):
        """Mark tracks that weren't matched as lost."""
        
        assigned_tracks = set(assignments.values())
        
        for track_id, track in self.tracked_objects.items():
            if track_id not in assigned_tracks and track.last_seen_frame < frame_id:
                track.mark_lost()
                logger.debug(f"Track {track_id} ({track.label}) marked as lost "
                           f"({track.lost_frames} frames)")
    
    def _cleanup_lost_tracks(self):
        """Remove tracks that have been lost too long."""
        
        tracks_to_remove = []
        
        for track_id, track in self.tracked_objects.items():
            # Remove if lost too long or low confidence
            if (track.lost_frames > self.max_lost_frames or 
                (track.confidence < 0.1 and track.total_observations < self.min_observations)):
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            track = self.tracked_objects.pop(track_id)
            logger.info(f"Removed track {track_id} ({track.label}): "
                       f"lost_frames={track.lost_frames}, "
                       f"confidence={track.confidence:.2f}, "
                       f"observations={track.total_observations}")
    
    def _update_timing_stats(self, bbox_time: float, match_time: float, total_time: float):
        """Update timing statistics."""
        self.timing_stats['bbox_extraction'].append(bbox_time * 1000)  # ms
        self.timing_stats['matching'].append(match_time * 1000)
        self.timing_stats['total'].append(total_time * 1000)
        
        # Keep only recent stats
        max_stats = 100
        for key in self.timing_stats:
            if len(self.timing_stats[key]) > max_stats:
                self.timing_stats[key] = self.timing_stats[key][-max_stats:]
    
    def _log_tracking_stats(self):
        """Log tracking statistics."""
        
        # Active tracks by label
        label_counts = {}
        for track in self.tracked_objects.values():
            if track.lost_frames == 0:
                label_counts[track.label] = label_counts.get(track.label, 0) + 1
        
        # Timing stats
        avg_times = {}
        for key, times in self.timing_stats.items():
            if times:
                avg_times[key] = np.mean(times)
        
        logger.info(f"Tracking Stats - Frame {self.frame_count}:")
        logger.info(f"  Active tracks: {sum(label_counts.values())} "
                   f"({', '.join(f'{k}:{v}' for k, v in label_counts.items())})")
        logger.info(f"  Total tracks: {len(self.tracked_objects)}")
        logger.info(f"  Timing (ms): bbox={avg_times.get('bbox_extraction', 0):.1f}, "
                   f"matching={avg_times.get('matching', 0):.1f}, "
                   f"total={avg_times.get('total', 0):.1f}")
    
    def _log_world_positions(self):
        """Log world positions of all active tracks for debugging."""
        logger.info("=== World Position Report ===")
        
        # Sort tracks by ID for consistent output
        sorted_tracks = sorted(self.tracked_objects.items())
        
        for track_id, track in sorted_tracks:
            if track.centroid_world is not None and track.lost_frames == 0:  # Only active tracks
                pos = track.centroid_world.cpu().numpy()
                variance = track.get_world_position_variance()
                
                logger.info(f"Track {track_id:3d} ({track.label:12s}): "
                           f"pos=[{pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f}], "
                           f"var={variance:.3f}m, obs={track.total_observations:3d}")
        
        logger.info("=" * 50)
    
    def get_track_info(self, track_id: int) -> Optional[Dict]:
        """Get information about a specific track."""
        
        track = self.tracked_objects.get(track_id)
        if track is None:
            return None
        
        return {
            'track_id': track_id,
            'label': track.label,
            'first_seen': track.first_seen_frame,
            'last_seen': track.last_seen_frame,
            'lost_frames': track.lost_frames,
            'confidence': track.confidence,
            'observations': track.total_observations,
            'bbox_cam': track.bbox_3d_cam,
            'bbox_world': track.bbox_3d_world,
            'world_position': track.centroid_world.cpu().numpy().tolist() if track.centroid_world is not None else None,
            'position_variance': track.position_variance_world
        }
    
    def save_tracking_decisions(self, output_file: str):
        """Save tracking history for analysis."""
        
        import json
        
        decisions = []
        for track_id, track in self.tracked_objects.items():
            decisions.append({
                'track_id': track_id,
                'label': track.label,
                'frames': [track.first_seen_frame, track.last_seen_frame],
                'observations': track.total_observations,
                'final_confidence': track.confidence
            })
        
        with open(output_file, 'w') as f:
            json.dump(decisions, f, indent=2)
        
        logger.info(f"Saved tracking decisions to {output_file}")