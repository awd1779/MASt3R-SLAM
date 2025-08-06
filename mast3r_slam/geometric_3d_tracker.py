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
from mast3r_slam.tracking_utils import (
    TrackingLogger, ConfigHelper, TrackProcessor, StatisticsCollector
)

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
        
        # Global tracking mode configuration
        self.use_global_tracking = config.get('use_global_tracking', False)
        
        # Sliding window parameters (only used in global mode)
        self.window_size = config.get('sliding_window_size', 5)
        self.global_search_tracks = config.get('global_search_tracks', 20)
        
        # Global tracking state (only used in global mode)
        if self.use_global_tracking:
            self.recent_keyframes = []  # List of (frame_id, detections)
            self.track_last_seen = {}  # track_id -> frame_id
        
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
        
        # Initialize centralized logging
        debug_labels = config.get('debug_labels', [])
        debug_all = config.get('debug_all', False)
        self.tracking_logger = TrackingLogger(debug_labels, debug_all)
        
        tracking_mode = "Global" if self.use_global_tracking else "Standard"
        logger.info(f"Initialized Geometric3DTracker ({tracking_mode} mode) with 3D matching and label-based tracking")
    
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
                
                # Use centralized logging for detection extraction
                self.tracking_logger.log_detection_extraction(detections[instance_id], instance_id)
        
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
        
        # Update track last seen info for global tracking
        if self.use_global_tracking:
            def always_true(track_id, track):
                return True
            
            def update_last_seen(track_id, track):
                self.track_last_seen[track_id] = track.last_seen_frame
                return None
            
            TrackProcessor.process_tracks(self.tracked_objects, always_true, update_last_seen)
        
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
            
            # If not forced match, get candidate tracks
            if best_track_id is None:
                if self.use_global_tracking:
                    # Use global candidate search
                    candidate_tracks = self._get_candidate_tracks(
                        detection['label'], 
                        frame_id, 
                        used_tracks
                    )
                else:
                    # Use standard same-label search
                    def is_candidate_track(track_id, track):
                        return (track_id not in used_tracks and 
                               track.label == detection['label'] and
                               track.lost_frames <= self.max_lost_frames)
                    
                    def get_track_id(track_id, track):
                        return track_id
                    
                    candidate_tracks = TrackProcessor.process_tracks(
                        self.tracked_objects, is_candidate_track, get_track_id, collect_results=True
                    )
                
                # Try to match with each candidate
                for track_id in candidate_tracks:
                    track = self.tracked_objects[track_id]
                    
                    # ROBUST 3D MATCHING
                    if track.bbox_3d_world is not None and track.centroid_world is not None:
                        # Compute matching score using multiple criteria
                        matching_score = self._compute_robust_matching_score(
                            detection, track, frame_id
                        )
                        
                        # Apply recency boost for global tracking
                        if self.use_global_tracking:
                            recency_boost = self._compute_recency_boost(track, frame_id)
                            matching_score *= recency_boost
                    
                        # Enhanced debug logging for special objects
                        if self._should_debug_label(detection['label']):
                            dist = self._compute_3d_distance(detection['bbox_world']['center'], track.centroid_world)
                            scores = {
                                'track_pos': track.centroid_world.cpu().numpy(),
                                'distance': dist,
                                'matching_score': matching_score,
                                'threshold': self.iou_threshold_3d
                            }
                            self.tracking_logger.log_matching_details(
                                detection, track_id, scores, frame_id, "checking"
                            )
                        
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
                
                # Log successful match with compact format
                track = self.tracked_objects[best_track_id]
                if track.bbox_3d_world is not None:
                    size_ratio = self._compute_volume_ratio(detection['bbox_world']['volume'], track.bbox_3d_world['volume'])
                    if size_ratio > 2.0 and track.total_observations < 3:
                        logger.info(f"Matched partial→full {detection['label']} to track {best_track_id} "
                                   f"(size grew {size_ratio:.1f}x, score: {best_score:.3f})")
                    else:
                        scores = {'matching_score': best_score}
                        self.tracking_logger.log_matching_details(
                            detection, best_track_id, scores, frame_id, "matched"
                        )
                else:
                    scores = {'matching_score': best_score}
                    self.tracking_logger.log_matching_details(
                        detection, best_track_id, scores, frame_id, "matched"
                    )
            else:
                # Create new track
                new_track_id = self._create_new_track(detection, frame_id)
                assignments[instance_id] = new_track_id
                
                # Register unique objects
                if self.label_tracker.is_unique_object(detection['label']):
                    self.label_tracker.assign_track_to_unique_object(detection['label'], new_track_id)
                
                world_pos = self.tracking_logger.format_position(
                    detection['bbox_world']['center'], "world"
                )
                logger.info(f"Created new track {new_track_id} for {detection['label']} at {world_pos}")
        
        # Update sliding window for global tracking
        if self.use_global_tracking:
            self._update_sliding_window(frame_id, detections)
        
        return assignments
    
    def _get_world_distance_threshold(self, label: str) -> float:
        """Get distance threshold based on object type from config."""
        world_thresholds = getattr(self, 'world_thresholds', None)
        return ConfigHelper.get_distance_threshold(label, world_thresholds)
    
    def _compute_containment_ratios(self, det_bbox: Dict, track_bbox: Dict) -> Tuple[float, float, float]:
        """Calculate containment ratios and intersection volume."""
        det_vol = det_bbox['volume']
        track_vol = track_bbox['volume']
        
        # Calculate intersection volume for containment ratios
        if det_bbox['type'] == 'aabb' and track_bbox['type'] == 'aabb':
            inter_min = torch.max(det_bbox['min'], track_bbox['min'])
            inter_max = torch.min(det_bbox['max'], track_bbox['max'])
            if torch.all(inter_min < inter_max):
                inter_vol = torch.prod(inter_max - inter_min).item()
            else:
                inter_vol = 0.0
        else:
            # For other bbox types, estimate from IoU
            iou_3d = self.bbox_computer.compute_3d_iou(det_bbox, track_bbox)
            inter_vol = iou_3d * (det_vol + track_vol - iou_3d * (det_vol + track_vol))
        
        # Calculate containment ratios
        containment_det_in_track = inter_vol / det_vol if det_vol > 0 else 0
        containment_track_in_det = inter_vol / track_vol if track_vol > 0 else 0
        
        return containment_det_in_track, containment_track_in_det, inter_vol
    
    def _compute_distance_and_size_scores(self, det_bbox: Dict, track: TrackedObject3D) -> Tuple[float, float, float, bool]:
        """Compute distance and size consistency scores."""
        track_bbox = track.bbox_3d_world
        
        # Distance calculation
        dist = self._compute_3d_distance(det_bbox['center'], track.centroid_world)
        
        # Get object size for distance normalization
        det_size = det_bbox['dimensions'].max().item()
        track_size = track_bbox['dimensions'].max().item()
        avg_size = (det_size + track_size) / 2.0
        
        # Normalize distance by object size
        norm_dist = dist / (avg_size + 1e-6)
        
        # Size consistency check with tolerance for early observations
        size_ratio = self._compute_volume_ratio(det_bbox['volume'], track_bbox['volume'])
        
        # Allow more size variation for tracks with few observations
        if track.total_observations < 3:
            size_consistent = 0.2 < size_ratio < 5.0  # Allow 5x growth
            if size_ratio > 2.0:
                logger.debug(f"  Allowing size growth for early track: ratio={size_ratio:.2f}, observations={track.total_observations}")
        else:
            size_consistent = 0.5 < size_ratio < 2.0  # Allow 2x size variation
        
        # Compute base scores
        dist_score = 1.0 / (1.0 + norm_dist)
        size_score = 2.0 * min(size_ratio, 1/size_ratio) / (1 + min(size_ratio, 1/size_ratio))
        
        return dist_score, size_score, norm_dist, size_consistent
    
    def _compute_3d_distance(self, pos1: torch.Tensor, pos2: torch.Tensor) -> float:
        """Compute 3D distance between two positions."""
        return torch.norm(pos1 - pos2).item()
    
    def _should_debug_label(self, label: str) -> bool:
        """Check if label should have detailed debug logging."""
        return label == 'a table' and self.tracking_logger.debug_all
    
    def _log_table_debug(self, message: str, **kwargs):
        """Log debug message for table matching with formatted values."""
        if kwargs:
            formatted_kwargs = ', '.join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" 
                                        for k, v in kwargs.items())
            logger.info(f"    {message}: {formatted_kwargs}")
        else:
            logger.info(f"    {message}")
    
    def _compute_volume_ratio(self, vol1: float, vol2: float) -> float:
        """Compute volume ratio with epsilon protection."""
        return vol1 / (vol2 + 1e-6)
    
    def _apply_temporal_boost(self, score: float, track: TrackedObject3D, det_bbox: Dict, 
                             frame_id: int, avg_size: float, detection: Dict) -> float:
        """Apply temporal consistency boost based on prediction."""
        if track.total_observations <= 1 or score <= 0:
            return score
            
        # Use world position history for prediction
        if len(track.world_observations) >= 2:
            # Simple velocity prediction
            pos_prev = track.world_observations[-2]
            pos_curr = track.world_observations[-1]
            velocity = pos_curr - pos_prev
            
            # Predicted position
            predicted_pos = pos_curr + velocity * (frame_id - track.last_seen_frame)
            
            # Check prediction error
            pred_error = self._compute_3d_distance(det_bbox['center'], predicted_pos)
            pred_error_normalized = pred_error / (avg_size + 1e-6)
            
            # Apply boost for good predictions
            if pred_error_normalized < 1.0:
                boost = 1.0 + 0.3 * (1.0 - pred_error_normalized)
                score *= boost
                
                if self._should_debug_label(detection['label']):
                    self._log_table_debug("Temporal boost", pred_error=pred_error, boost=boost)
        
        return score
    
    def _update_track_world(self, track: TrackedObject3D, detection: Dict, frame_id: int):
        """Update track with new world observation."""
        
        # Track position stability
        old_world = track.centroid_world
        new_world = detection['bbox_world']['center']
        
        if old_world is not None:
            position_change = self._compute_3d_distance(new_world, old_world)
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
        
        # Log detailed info for debug labels
        if self._should_debug_label(detection['label']):
            logger.info(f"  TABLE MATCHING DETAILS:")
            self._log_table_debug("3D IoU", iou=iou_3d)
        
        # 2. Calculate containment ratios
        containment_det_in_track, containment_track_in_det, _ = self._compute_containment_ratios(det_bbox, track_bbox)
        
        # Log containment for debug labels
        if self._should_debug_label(detection['label']):
            self._log_table_debug("Volumes", detection=det_bbox['volume'], track=track_bbox['volume'])
            self._log_table_debug("Containment", det_in_track=containment_det_in_track, track_in_det=containment_track_in_det)
        
        # 3. Check for partial object case
        if containment_det_in_track > 0.7 or containment_track_in_det > 0.7:
            dist = self._compute_3d_distance(det_bbox['center'], track.centroid_world)
            norm_dist = dist / (det_bbox['dimensions'].max().item() + 1e-6)
            
            if self._should_debug_label(detection['label']):
                self._log_table_debug("PARTIAL OBJECT CASE", distance=dist, norm_dist=norm_dist)
            else:
                logger.debug(f"  Partial object match: containment_det={containment_det_in_track:.3f}, "
                            f"containment_track={containment_track_in_det:.3f}, dist={dist:.3f}m")
            
            return 0.5 + 0.5 * (1.0 / (1.0 + norm_dist))
        
        # 4. If IoU is high enough, it's definitely the same object
        if iou_3d > 0.3:  # Strong overlap
            logger.debug(f"  High IoU match: {iou_3d:.3f}")
            return iou_3d
        
        # 5. For low/no IoU, use distance-based matching with size awareness
        dist_score, size_score, norm_dist, size_consistent = self._compute_distance_and_size_scores(det_bbox, track)
        avg_size = (det_bbox['dimensions'].max().item() + track_bbox['dimensions'].max().item()) / 2.0
        
        # 6. Adaptive weighting based on context
        if iou_3d > 0.1:  # Some overlap
            score = 0.5 * iou_3d + 0.3 * dist_score + 0.2 * size_score
        elif containment_det_in_track > 0.5 or containment_track_in_det > 0.5:
            containment_score = max(containment_det_in_track, containment_track_in_det)
            score = 0.3 * containment_score + 0.5 * dist_score + 0.2 * size_score
        else:
            # No overlap - use distance and size if reasonable
            distance_tolerance = 3.0 if track.total_observations < 3 else 2.0
            object_size = det_bbox['dimensions'].max().item()
            max_norm_dist = distance_tolerance * (1.0 + 0.5 * min(object_size, 2.0))
            
            if norm_dist < max_norm_dist and size_consistent:
                score = 0.6 * dist_score + 0.3 * size_score + 0.1 * (1.0 - norm_dist/max_norm_dist)
                
                if self._should_debug_label(detection['label']):
                    dist = self._compute_3d_distance(det_bbox['center'], track.centroid_world)
                    self._log_table_debug("Distance-based match", dist=dist, norm_dist=norm_dist, score=score)
            else:
                score = 0.0  # Too far or size mismatch
        
        # 7. Apply temporal consistency boost
        score = self._apply_temporal_boost(score, track, det_bbox, frame_id, avg_size, detection)
        
        return score
    
    def _update_lost_tracks(self, assignments: Dict[int, int], frame_id: int):
        """Mark tracks that weren't matched as lost."""
        assigned_tracks = set(assignments.values())
        
        def should_mark_lost(track_id, track):
            return track_id not in assigned_tracks and track.last_seen_frame < frame_id
            
        def mark_track_lost(track_id, track):
            track.mark_lost()
            logger.debug(f"Track {track_id} ({track.label}) marked as lost "
                       f"({track.lost_frames} frames)")
            return None
        
        TrackProcessor.process_tracks(self.tracked_objects, should_mark_lost, mark_track_lost)
    
    def _cleanup_lost_tracks(self):
        """Remove tracks that have been lost too long."""
        
        def should_remove_track(track_id, track):
            # Use different thresholds for global vs standard tracking
            remove_threshold = (self.max_lost_frames * 2 if self.use_global_tracking 
                              else self.max_lost_frames)
            
            return (track.lost_frames > remove_threshold or 
                   (track.confidence < 0.1 and track.total_observations < self.min_observations))
        
        def collect_track_id(track_id, track):
            return track_id
        
        tracks_to_remove = TrackProcessor.process_tracks(
            self.tracked_objects, should_remove_track, collect_track_id, collect_results=True
        )
        
        for track_id in tracks_to_remove:
            track = self.tracked_objects.pop(track_id)
            
            # Clean up global tracking state
            if self.use_global_tracking and track_id in self.track_last_seen:
                del self.track_last_seen[track_id]
            
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
        """Log tracking statistics using centralized collector."""
        summary = StatisticsCollector.generate_track_summary(
            self.tracked_objects, self.timing_stats
        )
        StatisticsCollector.log_tracking_stats(summary, self.frame_count)
    
    def _log_world_positions(self):
        """Log world positions using centralized collector."""
        summary = StatisticsCollector.generate_track_summary(self.tracked_objects)
        StatisticsCollector.log_world_positions(summary)
    
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
    
    # Global tracking specific methods
    def _get_candidate_tracks(self, label: str, frame_id: int, 
                             used_tracks: Set[int]) -> List[int]:
        """Get candidate tracks using global search strategy."""
        if not self.use_global_tracking:
            return []
        
        candidates = []
        
        # 1. All tracks with same label (up to a limit)
        def is_same_label_candidate(track_id, track):
            return track.label == label and track_id not in used_tracks
        
        def get_track_id(track_id, track):
            return track_id
        
        same_label_tracks = TrackProcessor.process_tracks(
            self.tracked_objects, is_same_label_candidate, get_track_id, collect_results=True
        )
        
        # 2. Sort by recency and quality
        track_scores = []
        for track_id in same_label_tracks:
            track = self.tracked_objects[track_id]
            
            # Compute track quality score
            recency = frame_id - track.last_seen_frame
            observations = track.total_observations
            confidence = track.confidence
            
            # Combined score (lower is better for recency)
            score = (1.0 / (recency + 1)) * confidence * np.log(observations + 1)
            track_scores.append((track_id, score))
        
        # Sort by score and take top N
        track_scores.sort(key=lambda x: x[1], reverse=True)
        candidates = [tid for tid, _ in track_scores[:self.global_search_tracks]]
        
        return candidates
    
    def _compute_recency_boost(self, track: TrackedObject3D, frame_id: int) -> float:
        """Boost score for recently seen tracks (global tracking only)."""
        if not self.use_global_tracking:
            return 1.0
            
        frames_since_seen = frame_id - track.last_seen_frame
        
        if frames_since_seen == 0:
            return 1.2  # Currently visible
        elif frames_since_seen <= 2:
            return 1.1  # Very recent
        elif frames_since_seen <= 5:
            return 1.0  # Recent
        elif frames_since_seen <= 10:
            return 0.9  # Getting old
        else:
            return 0.8  # Old track
    
    def _update_sliding_window(self, frame_id: int, detections: Dict):
        """Update sliding window of recent keyframes (global tracking only)."""
        if not self.use_global_tracking:
            return
            
        self.recent_keyframes.append((frame_id, detections))
        
        # Keep only recent keyframes
        if len(self.recent_keyframes) > self.window_size:
            self.recent_keyframes.pop(0)
    
    def get_track_statistics(self) -> Dict:
        """Get detailed statistics about tracking performance."""
        stats = StatisticsCollector.generate_track_summary(self.tracked_objects, self.timing_stats)
        stats['tracking_mode'] = 'Global' if self.use_global_tracking else 'Standard'
        return stats