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

logger = logging.getLogger('mast3r_slam.geometric_3d_tracker')


@dataclass
class TrackedObject3D:
    """Represents a tracked object with 3D geometry."""
    track_id: int
    label: str
    first_seen_frame: int
    last_seen_frame: int
    
    # 3D geometry
    bbox_3d: Optional[Dict] = None
    last_centroid: Optional[torch.Tensor] = None
    
    # Tracking state
    confidence: float = 1.0
    lost_frames: int = 0
    total_observations: int = 0
    
    # History for smoothing/prediction
    centroid_history: List[torch.Tensor] = field(default_factory=list)
    bbox_history: List[Dict] = field(default_factory=list)
    max_history: int = 10
    
    def update_geometry(self, bbox: Dict, frame_id: int):
        """Update object's 3D geometry."""
        self.bbox_3d = bbox
        self.last_centroid = bbox['center']
        self.last_seen_frame = frame_id
        self.lost_frames = 0
        self.total_observations += 1
        
        # Update history
        self.centroid_history.append(bbox['center'].clone())
        self.bbox_history.append(bbox)
        
        # Limit history size
        if len(self.centroid_history) > self.max_history:
            self.centroid_history.pop(0)
            self.bbox_history.pop(0)
    
    def mark_lost(self):
        """Mark object as lost for current frame."""
        self.lost_frames += 1
        self.confidence *= 0.95  # Decay confidence
    
    def get_predicted_position(self) -> Optional[torch.Tensor]:
        """Simple position prediction based on motion history."""
        if len(self.centroid_history) < 2:
            return self.last_centroid
        
        # Simple linear prediction
        velocity = self.centroid_history[-1] - self.centroid_history[-2]
        predicted = self.centroid_history[-1] + velocity
        return predicted


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
        
        logger.info("Initialized Geometric3DTracker with 3D matching")
    
    def process_frame(self, 
                     keyframe,
                     semantic_data: Dict,
                     frame_id: int) -> Dict[int, int]:
        """Process a frame and return instance_id -> track_id mapping."""
        
        start_time = time.time()
        self.frame_count = frame_id
        
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
        
        return assignments
    
    def _extract_3d_bboxes(self, keyframe, semantic_data: Dict) -> Dict[int, Dict]:
        """Extract 3D bounding boxes for all detected objects."""
        
        detections = {}
        h, w = keyframe.img_shape[0, 0].item(), keyframe.img_shape[0, 1].item()
        
        for instance_id, mask_rle in semantic_data.get('masks_rle', {}).items():
            label = semantic_data.get('labels', {}).get(instance_id, 'unknown')
            
            # Decode mask
            mask = decode_rle(mask_rle, h, w)
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            
            # Create 3D bbox
            bbox = create_generic_3d_bbox(
                keyframe, mask, label,
                config=self.bbox_computer.config,
                estimator=self.bbox_estimator
            )
            
            if bbox is not None and bbox['size_valid']:
                detections[instance_id] = {
                    'bbox': bbox,
                    'label': label,
                    'instance_id': instance_id
                }
                logger.debug(f"Extracted 3D bbox for {label} (instance {instance_id}): "
                           f"type={bbox['type']}, volume={bbox['volume']:.3f}")
        
        logger.info(f"Extracted {len(detections)} valid 3D bboxes from "
                   f"{len(semantic_data.get('masks_rle', {}))} detections")
        
        # Log details about extracted bboxes
        for instance_id, det in detections.items():
            bbox = det['bbox']
            logger.debug(f"  {det['label']} (instance {instance_id}): "
                        f"{bbox['extraction_stats']['valid_3d_points']} points, "
                        f"volume={bbox['volume']:.3f}m³")
        
        return detections
    
    def _match_detections_to_tracks(self, 
                                   detections: Dict[int, Dict],
                                   frame_id: int) -> Dict[int, int]:
        """Match current detections to existing tracks using 3D geometry."""
        
        assignments = {}
        used_tracks = set()
        
        # Sort detections by confidence/size for stable matching
        sorted_detections = sorted(
            detections.items(),
            key=lambda x: x[1]['bbox']['confidence'],
            reverse=True
        )
        
        for instance_id, detection in sorted_detections:
            best_track_id = None
            best_score = 0.0
            
            # Find best matching track
            for track_id, track in self.tracked_objects.items():
                if track_id in used_tracks:
                    continue
                
                # Only match same label
                if track.label != detection['label']:
                    continue
                
                # Skip if track has been lost too long
                if track.lost_frames > self.max_lost_frames:
                    continue
                
                # Compute matching score
                score = self._compute_3d_matching_score(
                    detection['bbox'],
                    track,
                    detection['label']
                )
                
                if score > best_score and score > self.iou_threshold_3d:
                    best_score = score
                    best_track_id = track_id
            
            # Assign or create new track
            if best_track_id is not None:
                # Update existing track
                assignments[instance_id] = best_track_id
                used_tracks.add(best_track_id)
                
                track = self.tracked_objects[best_track_id]
                track.update_geometry(detection['bbox'], frame_id)
                
                logger.debug(f"Matched {detection['label']} (instance {instance_id}) "
                           f"to track {best_track_id} with score {best_score:.3f}")
            else:
                # Create new track
                new_track_id = self.next_track_id
                self.next_track_id += 1
                
                new_track = TrackedObject3D(
                    track_id=new_track_id,
                    label=detection['label'],
                    first_seen_frame=frame_id,
                    last_seen_frame=frame_id
                )
                new_track.update_geometry(detection['bbox'], frame_id)
                
                self.tracked_objects[new_track_id] = new_track
                assignments[instance_id] = new_track_id
                
                logger.info(f"Created new track {new_track_id} for {detection['label']} "
                          f"(instance {instance_id})")
        
        return assignments
    
    def _compute_3d_matching_score(self, 
                                  bbox: Dict,
                                  track: TrackedObject3D,
                                  label: str) -> float:
        """Compute matching score between detection and track using 3D geometry."""
        
        if track.bbox_3d is None:
            return 0.0
        
        # 1. 3D IoU
        iou_3d = self.bbox_computer.compute_3d_iou(bbox, track.bbox_3d)
        
        # 2. Centroid distance (normalized)
        dist = torch.norm(bbox['center'] - track.last_centroid)
        expected_size_range = self.bbox_estimator.get_expected_size_range(label)
        if expected_size_range is None:
            expected_size = max(bbox['dimensions'].max().item(), 
                              track.bbox_3d['dimensions'].max().item())
        else:
            expected_size = expected_size_range[1]  # Use max from range
        
        norm_dist = dist.item() / (expected_size + 1e-6)
        dist_score = 1.0 / (1.0 + norm_dist)
        
        # 3. Size consistency
        size_ratio = bbox['volume'] / (track.bbox_3d['volume'] + 1e-6)
        size_score = 1.0 - abs(1.0 - size_ratio) / 2.0  # Penalize size changes
        size_score = max(0.0, size_score)
        
        # 4. Confidence weighting
        conf_weight = (bbox['confidence'] + track.confidence) / 2.0
        
        # Combine scores
        score = (
            self.weight_iou * iou_3d +
            self.weight_centroid * dist_score +
            self.weight_size * size_score
        ) * conf_weight
        
        # Boost if using predicted position
        if track.lost_frames > 0:
            predicted_pos = track.get_predicted_position()
            if predicted_pos is not None:
                pred_dist = torch.norm(bbox['center'] - predicted_pos).item()
                if pred_dist < expected_size * 0.5:  # Within predicted area
                    score *= 1.2  # Boost score
        
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
            'bbox': track.bbox_3d
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