"""
Simplified object tracking module that works without 3D data passing.
Uses 2D IoU and simple heuristics for frame-to-frame tracking.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import logging
from collections import defaultdict
import time
from mast3r_slam.semantic_frame import decode_rle

logger = logging.getLogger('mast3r_slam.object_tracker_simple')


@dataclass
class SimpleTrackedObject:
    """Simplified tracked object using 2D information."""
    track_id: int
    label: str
    first_seen_frame: int
    last_seen_frame: int
    confidence_history: List[float] = field(default_factory=list)
    
    # 2D tracking properties
    last_bbox_2d: Optional[Tuple[int, int, int, int]] = None  # x1, y1, x2, y2
    last_mask_center: Optional[Tuple[float, float]] = None
    last_mask_area: Optional[int] = None
    
    # Tracking state
    is_active: bool = True
    frames_since_seen: int = 0
    observation_count: int = 0
    
    def update_observation(self, frame_id: int, confidence: float, bbox_2d=None, center=None, area=None):
        """Update tracking state with new observation."""
        self.last_seen_frame = frame_id
        self.frames_since_seen = 0
        self.observation_count += 1
        self.confidence_history.append(confidence)
        self.is_active = True
        
        if bbox_2d is not None:
            self.last_bbox_2d = bbox_2d
        if center is not None:
            self.last_mask_center = center
        if area is not None:
            self.last_mask_area = area
        
    def mark_lost(self):
        """Mark object as lost (not seen in current frame)."""
        self.frames_since_seen += 1


def compute_mask_iou(mask1_rle: Dict, mask2_rle: Dict, shape: Tuple[int, int]) -> float:
    """Compute IoU between two RLE-encoded masks."""
    h, w = shape
    
    # Decode masks
    mask1 = decode_rle(mask1_rle, h, w)
    mask2 = decode_rle(mask2_rle, h, w)
    
    # Compute IoU
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
        
    return intersection / union


def get_mask_properties(mask_rle: Dict, shape: Tuple[int, int]) -> Tuple:
    """Extract properties from mask for tracking."""
    h, w = shape
    mask = decode_rle(mask_rle, h, w)
    
    # Find bounding box
    y_coords, x_coords = np.where(mask)
    if len(y_coords) == 0:
        return None, None, 0
    
    x1, y1 = x_coords.min(), y_coords.min()
    x2, y2 = x_coords.max(), y_coords.max()
    bbox = (x1, y1, x2, y2)
    
    # Compute center
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    center = (cx, cy)
    
    # Compute area
    area = mask.sum()
    
    return bbox, center, area


class SimpleObjectTracker:
    """Simplified object tracker using 2D information only."""
    
    def __init__(self, config: Dict):
        self.enabled = config.get('enabled', True)
        self.iou_threshold_2d = config.get('iou_threshold_2d', 0.3)  # Lower threshold for 2D
        self.max_center_distance = config.get('max_center_distance', 100)  # pixels
        self.max_lost_frames = config.get('max_lost_frames', 10)
        
        # Tracking state
        self.tracked_objects: Dict[int, SimpleTrackedObject] = {}
        self.next_track_id = 1
        self.current_frame_id = 0
        
        # Performance tracking
        self.timing_stats = defaultdict(list)
        
        # Debug
        self.debug_mode = config.get('debug_mode', False)
        self.save_tracking_decisions = config.get('save_tracking_decisions', False)
        self.tracking_decisions = []
        
        logger.info(f"SimpleObjectTracker initialized with config: {config}")
        
    def process_frame(self, 
                     semantic_results: Dict,
                     frame_id: int,
                     keyframe_idx: Optional[int] = None) -> Dict:
        """Process semantic results and add tracking."""
        if not self.enabled:
            return semantic_results
            
        start_time = time.time()
        self.current_frame_id = frame_id
        
        # Mark all existing tracks as potentially lost
        for track in self.tracked_objects.values():
            track.mark_lost()
        
        # Copy results
        tracked_results = semantic_results.copy()
        tracked_results['track_ids'] = {}
        
        # Get image shape from first mask
        shape = None
        if semantic_results.get('masks_rle'):
            first_mask = next(iter(semantic_results['masks_rle'].values()))
            if 'size' in first_mask:
                size = first_mask['size']
                shape = (size[1], size[2]) if len(size) == 3 else size
        
        if shape is None:
            # Can't track without shape info
            logger.warning("No shape information in masks, using simple ID assignment")
            for instance_id in semantic_results.get('instance_ids', []):
                track_id = self._create_new_track(
                    instance_id, 
                    semantic_results.get('labels', {}).get(instance_id, 'unknown'),
                    frame_id,
                    semantic_results.get('confidences', {}).get(instance_id, 1.0)
                )
                tracked_results['track_ids'][instance_id] = track_id
        else:
            # Process each instance with 2D tracking
            for instance_id in semantic_results.get('instance_ids', []):
                label = semantic_results.get('labels', {}).get(instance_id, 'unknown')
                confidence = semantic_results.get('confidences', {}).get(instance_id, 1.0)
                mask_rle = semantic_results.get('masks_rle', {}).get(instance_id)
                
                track_id = None
                
                if mask_rle:
                    # Try to match with existing tracks
                    track_id = self._try_2d_association(instance_id, mask_rle, label, shape)
                
                # Create new track if no match
                if track_id is None:
                    track_id = self._create_new_track(instance_id, label, frame_id, confidence)
                    
                    # Initialize 2D properties
                    if mask_rle:
                        bbox, center, area = get_mask_properties(mask_rle, shape)
                        track = self.tracked_objects[track_id]
                        track.update_observation(frame_id, confidence, bbox, center, area)
                else:
                    # Update existing track
                    if mask_rle:
                        bbox, center, area = get_mask_properties(mask_rle, shape)
                        self.tracked_objects[track_id].update_observation(
                            frame_id, confidence, bbox, center, area
                        )
                
                tracked_results['track_ids'][instance_id] = track_id
        
        # Clean up dead tracks
        self._cleanup_dead_tracks()
        
        # Log timing
        elapsed = time.time() - start_time
        self.timing_stats['process_frame'].append(elapsed)
        
        if frame_id % 30 == 0:
            avg_time = np.mean(self.timing_stats['process_frame'][-30:])
            logger.info(f"Simple tracking performance: {avg_time*1000:.1f}ms/frame, "
                       f"Active tracks: {len(self.tracked_objects)}")
        
        return tracked_results
    
    def _try_2d_association(self, 
                           instance_id: int,
                           mask_rle: Dict,
                           label: str,
                           shape: Tuple[int, int]) -> Optional[int]:
        """Try to associate instance with existing track using 2D information."""
        # Get current mask properties
        bbox, center, area = get_mask_properties(mask_rle, shape)
        if center is None:
            return None
        
        # Find best matching track
        best_match_id = None
        best_score = 0.0
        
        # Only check recently seen tracks with same label
        candidate_tracks = [
            (tid, track) for tid, track in self.tracked_objects.items()
            if track.label == label and 
               track.frames_since_seen <= 3 and
               track.last_mask_center is not None
        ]
        
        for track_id, track in candidate_tracks:
            # Compute distance between centers
            if track.last_mask_center:
                dx = center[0] - track.last_mask_center[0]
                dy = center[1] - track.last_mask_center[1]
                distance = np.sqrt(dx*dx + dy*dy)
                
                if distance < self.max_center_distance:
                    # Use inverse distance as score
                    score = 1.0 - (distance / self.max_center_distance)
                    
                    # Bonus for similar area
                    if track.last_mask_area and area > 0:
                        area_ratio = min(area, track.last_mask_area) / max(area, track.last_mask_area)
                        score += area_ratio * 0.5
                    
                    if score > best_score and score > 0.5:
                        best_score = score
                        best_match_id = track_id
        
        if best_match_id is not None and self.debug_mode:
            logger.info(f"2D match: instance {instance_id} -> track {best_match_id} (score: {best_score:.3f})")
            
        if self.save_tracking_decisions and best_match_id is not None:
            self.tracking_decisions.append({
                'frame_id': self.current_frame_id,
                'instance_id': instance_id,
                'label': label,
                'decision': '2d_match',
                'track_id': best_match_id,
                'score': float(best_score)  # Convert to Python float for JSON serialization
            })
        
        return best_match_id
    
    def _create_new_track(self, 
                         instance_id: int,
                         label: str, 
                         frame_id: int,
                         confidence: float) -> int:
        """Create a new tracked object."""
        track_id = self.next_track_id
        self.next_track_id += 1
        
        tracked_obj = SimpleTrackedObject(
            track_id=track_id,
            label=label,
            first_seen_frame=frame_id,
            last_seen_frame=frame_id
        )
        tracked_obj.update_observation(frame_id, confidence)
        
        self.tracked_objects[track_id] = tracked_obj
        
        if self.debug_mode:
            logger.info(f"Created new track {track_id} for {label} at frame {frame_id}")
            
        if self.save_tracking_decisions:
            self.tracking_decisions.append({
                'frame_id': frame_id,
                'instance_id': instance_id,
                'label': label,
                'decision': 'new_track',
                'track_id': track_id
            })
        
        return track_id
    
    def _cleanup_dead_tracks(self):
        """Remove tracks that haven't been seen for too long."""
        dead_tracks = []
        for track_id, track in self.tracked_objects.items():
            if track.frames_since_seen > self.max_lost_frames:
                dead_tracks.append(track_id)
                
        for track_id in dead_tracks:
            del self.tracked_objects[track_id]
            if self.debug_mode:
                logger.debug(f"Removed dead track {track_id}")
    
    def get_tracking_stats(self) -> Dict:
        """Get tracking statistics."""
        active_tracks = {
            tid: track for tid, track in self.tracked_objects.items()
            if track.is_active
        }
        
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