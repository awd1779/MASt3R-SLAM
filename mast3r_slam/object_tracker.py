"""
Standalone object tracking module for temporal consistency in semantic SLAM.
This module can be enabled/disabled without affecting the core system.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import logging
from collections import defaultdict
import time
from mast3r_slam.tracking_utils import (
    compute_3d_iou, compute_centroid_distance, extract_3d_points_for_mask
)

logger = logging.getLogger('mast3r_slam.object_tracker')


@dataclass
class TrackedObject:
    """Represents a tracked object with its history and properties."""
    track_id: int
    label: str
    first_seen_frame: int
    last_seen_frame: int
    confidence_history: List[float] = field(default_factory=list)
    
    # Geometric properties
    point_cloud: Optional[torch.Tensor] = None  # Nx3 tensor of 3D points
    bbox_3d: Optional[Tuple[torch.Tensor, torch.Tensor]] = None  # (min_coords, max_coords)
    
    # Appearance properties
    appearance_signature: Optional[torch.Tensor] = None  # Normalized feature vector
    signature_confidence: float = 0.0
    
    # Tracking state
    is_active: bool = True
    frames_since_seen: int = 0
    observation_count: int = 0
    
    def update_observation(self, frame_id: int, confidence: float):
        """Update tracking state with new observation."""
        self.last_seen_frame = frame_id
        self.frames_since_seen = 0
        self.observation_count += 1
        self.confidence_history.append(confidence)
        self.is_active = True
        
    def mark_lost(self):
        """Mark object as lost (not seen in current frame)."""
        self.frames_since_seen += 1
        
    def update_geometry(self, new_points: torch.Tensor):
        """Update 3D geometry of the object."""
        if self.point_cloud is None:
            self.point_cloud = new_points.clone()
        else:
            # Simple concatenation for now, could do more sophisticated fusion
            self.point_cloud = torch.cat([self.point_cloud, new_points], dim=0)
            
        # Update bounding box
        min_coords = self.point_cloud.min(dim=0)[0]
        max_coords = self.point_cloud.max(dim=0)[0]
        self.bbox_3d = (min_coords, max_coords)
        
    def update_signature(self, new_signature: torch.Tensor, weight: float = 0.1):
        """Update appearance signature with exponential moving average."""
        if self.appearance_signature is None:
            self.appearance_signature = new_signature.clone()
            self.signature_confidence = 1.0
        else:
            # Weighted average to gradually update signature
            self.appearance_signature = (
                (1 - weight) * self.appearance_signature + weight * new_signature
            )
            # Re-normalize
            self.appearance_signature = torch.nn.functional.normalize(
                self.appearance_signature, dim=0
            )


class ObjectTracker:
    """Main object tracking class that maintains temporal consistency."""
    
    def __init__(self, config: Dict):
        """
        Initialize object tracker with configuration.
        
        Args:
            config: Dictionary with tracking parameters
                - enabled: Whether tracking is enabled
                - geometric_iou_threshold: IoU threshold for geometric matching
                - appearance_threshold: Cosine similarity threshold
                - max_lost_frames: Frames before track is considered dead
                - signature_update_weight: Weight for signature updates
        """
        self.enabled = config.get('enabled', True)
        self.geometric_iou_threshold = config.get('geometric_iou_threshold', 0.4)
        self.appearance_threshold = config.get('appearance_threshold', 0.8)
        self.max_lost_frames = config.get('max_lost_frames', 10)
        self.signature_update_weight = config.get('signature_update_weight', 0.1)
        
        # Tracking state
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.next_track_id = 1
        self.current_frame_id = 0
        
        # Performance tracking
        self.timing_stats = defaultdict(list)
        
        # Debug and visualization
        self.debug_mode = config.get('debug_mode', False)
        self.save_tracking_decisions = config.get('save_tracking_decisions', False)
        self.tracking_decisions = []  # Store matching decisions for analysis
        
        # Callbacks for accessing keyframe data
        self.get_keyframe_data_callback = None
        self.get_3d_points_callback = None
        self.get_features_callback = None
        
        logger.info(f"ObjectTracker initialized with config: {config}")
        
    def set_callbacks(self, 
                     get_keyframe_data=None,
                     get_3d_points=None, 
                     get_features=None):
        """Set callbacks for accessing SLAM data."""
        self.get_keyframe_data_callback = get_keyframe_data
        self.get_3d_points_callback = get_3d_points
        self.get_features_callback = get_features
        
    def process_frame(self, 
                     semantic_results: Dict,
                     frame_id: int,
                     keyframe_idx: Optional[int] = None) -> Dict:
        """
        Main entry point - process semantic results and add tracking.
        
        Args:
            semantic_results: Dictionary from semantic segmentation
            frame_id: Current frame ID
            keyframe_idx: Optional keyframe index for accessing SLAM data
            
        Returns:
            Enhanced semantic results with track_ids
        """
        if not self.enabled:
            return semantic_results
            
        start_time = time.time()
        self.current_frame_id = frame_id
        
        # Mark all existing tracks as potentially lost
        for track in self.tracked_objects.values():
            track.mark_lost()
        
        # Copy results to avoid modifying original
        tracked_results = semantic_results.copy()
        tracked_results['track_ids'] = {}
        
        # Process each detected instance
        instance_ids = semantic_results.get('instance_ids', [])
        
        for instance_id in instance_ids:
            # Get instance properties
            label = semantic_results.get('labels', {}).get(instance_id, 'unknown')
            confidence = semantic_results.get('confidences', {}).get(instance_id, 1.0)
            
            # Phase 2: Try geometric matching first
            track_id = None
            
            # Only attempt geometric matching if we have keyframe data
            if keyframe_idx is not None and self.get_3d_points_callback is not None:
                # Get 3D points for this instance
                mask_rle = semantic_results.get('masks_rle', {}).get(instance_id)
                if mask_rle:
                    track_id = self._try_geometric_association(
                        instance_id, mask_rle, label, keyframe_idx
                    )
            
            # If no geometric match, create new track
            if track_id is None:
                track_id = self._create_new_track(instance_id, label, frame_id, confidence)
                
                # Save decision
                if self.save_tracking_decisions:
                    self.tracking_decisions.append({
                        'frame_id': frame_id,
                        'instance_id': instance_id,
                        'label': label,
                        'decision': 'new_track',
                        'track_id': track_id,
                        'reason': 'no_geometric_match' if keyframe_idx is not None else 'no_3d_data'
                    })
            else:
                # Update existing track
                self.tracked_objects[track_id].update_observation(frame_id, confidence)
            
            tracked_results['track_ids'][instance_id] = track_id
            
        # Clean up dead tracks
        self._cleanup_dead_tracks()
        
        # Log timing
        elapsed = time.time() - start_time
        self.timing_stats['process_frame'].append(elapsed)
        
        if frame_id % 30 == 0:
            avg_time = np.mean(self.timing_stats['process_frame'][-30:])
            logger.info(f"Tracking performance: {avg_time*1000:.1f}ms/frame, "
                       f"Active tracks: {len(self.tracked_objects)}")
        
        return tracked_results
    
    def _create_new_track(self, 
                         instance_id: int,
                         label: str, 
                         frame_id: int,
                         confidence: float) -> int:
        """Create a new tracked object."""
        track_id = self.next_track_id
        self.next_track_id += 1
        
        tracked_obj = TrackedObject(
            track_id=track_id,
            label=label,
            first_seen_frame=frame_id,
            last_seen_frame=frame_id
        )
        tracked_obj.update_observation(frame_id, confidence)
        
        self.tracked_objects[track_id] = tracked_obj
        
        logger.debug(f"Created new track {track_id} for {label} at frame {frame_id}")
        
        # Try to initialize geometry if we have keyframe data
        if self.get_3d_points_callback:
            # This will be set on next observation
            pass
            
        return track_id
    
    def _try_geometric_association(self, 
                                  instance_id: int,
                                  mask_rle: Dict,
                                  label: str,
                                  keyframe_idx: int) -> Optional[int]:
        """Try to associate instance with existing track using 3D geometry."""
        if not self.get_3d_points_callback:
            return None
            
        start_time = time.time()
        
        # Get 3D points for current instance
        points_data = self.get_3d_points_callback(keyframe_idx)
        if points_data is None:
            return None
            
        pointmap, confidence, shape = points_data
        instance_points = extract_3d_points_for_mask(
            mask_rle, pointmap, confidence, shape, 
            conf_threshold=0.1
        )
        
        if instance_points.shape[0] < 10:  # Need minimum points
            return None
            
        # Compute bounding box for instance
        instance_min = instance_points.min(dim=0)[0]
        instance_max = instance_points.max(dim=0)[0]
        instance_bbox = (instance_min, instance_max)
        
        # Find best matching track
        best_match_id = None
        best_iou = 0.0
        
        # Only check recently seen tracks with same label
        candidate_tracks = [
            (tid, track) for tid, track in self.tracked_objects.items()
            if track.label == label and 
               track.frames_since_seen <= 3 and
               track.bbox_3d is not None
        ]
        
        for track_id, track in candidate_tracks:
            # Compute 3D IoU
            iou = compute_3d_iou(instance_bbox, track.bbox_3d)
            
            if iou > best_iou and iou >= self.geometric_iou_threshold:
                best_iou = iou
                best_match_id = track_id
                
        # Update timing stats
        elapsed = time.time() - start_time
        self.timing_stats['geometric_matching'].append(elapsed)
        
        if best_match_id is not None:
            # Update matched track's geometry
            matched_track = self.tracked_objects[best_match_id]
            matched_track.update_geometry(instance_points)
            logger.debug(f"Geometric match: instance {instance_id} -> track {best_match_id} (IoU: {best_iou:.3f})")
            
            # Save decision for analysis
            if self.save_tracking_decisions:
                self.tracking_decisions.append({
                    'frame_id': self.current_frame_id,
                    'instance_id': instance_id,
                    'label': label,
                    'decision': 'geometric_match',
                    'track_id': best_match_id,
                    'iou': best_iou,
                    'candidates_checked': len(candidate_tracks),
                    'num_points': instance_points.shape[0]
                })
        else:
            # Log why matching failed
            if self.debug_mode:
                logger.info(f"No geometric match for instance {instance_id}: "
                          f"checked {len(candidate_tracks)} candidates, "
                          f"best IoU was {best_iou:.3f} < {self.geometric_iou_threshold}")
            
        return best_match_id
    
    def _cleanup_dead_tracks(self):
        """Remove tracks that haven't been seen for too long."""
        dead_tracks = []
        for track_id, track in self.tracked_objects.items():
            if track.frames_since_seen > self.max_lost_frames:
                dead_tracks.append(track_id)
                
        for track_id in dead_tracks:
            del self.tracked_objects[track_id]
            logger.debug(f"Removed dead track {track_id}")
            
    def get_active_tracks(self) -> Dict[int, TrackedObject]:
        """Get all currently active tracks."""
        return {
            tid: track for tid, track in self.tracked_objects.items()
            if track.is_active
        }
        
    def get_tracking_stats(self) -> Dict:
        """Get tracking statistics."""
        active_tracks = self.get_active_tracks()
        
        stats = {
            'total_tracks': len(self.tracked_objects),
            'active_tracks': len(active_tracks),
            'next_track_id': self.next_track_id,
            'avg_track_length': 0,
            'label_distribution': defaultdict(int)
        }
        
        if self.tracked_objects:
            track_lengths = [
                track.observation_count 
                for track in self.tracked_objects.values()
            ]
            stats['avg_track_length'] = np.mean(track_lengths)
            
            for track in self.tracked_objects.values():
                stats['label_distribution'][track.label] += 1
                
        return stats
    
    def save_tracking_decisions_to_file(self, filepath):
        """Save tracking decisions for analysis."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.tracking_decisions, f, indent=2)
        logger.info(f"Saved {len(self.tracking_decisions)} tracking decisions to {filepath}")
    
    def get_recent_decisions(self, n=10):
        """Get recent tracking decisions for debugging."""
        return self.tracking_decisions[-n:] if self.tracking_decisions else []