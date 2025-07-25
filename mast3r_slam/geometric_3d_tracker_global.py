"""Enhanced 3D tracker with sliding window global matching for better real-time consistency."""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
import logging
import time
from dataclasses import dataclass, field

from mast3r_slam.geometric_3d_tracker import Geometric3DTracker, TrackedObject3D
from mast3r_slam.bbox_3d_generic import (
    Generic3DBoundingBox,
    AdaptiveBBoxEstimator,
    BBox3DConfig,
    create_generic_3d_bbox
)

logger = logging.getLogger('mast3r_slam.geometric_3d_tracker_global')


class Geometric3DTrackerGlobal(Geometric3DTracker):
    """Enhanced 3D tracker with sliding window global matching."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # Sliding window parameters
        self.window_size = config.get('sliding_window_size', 5)  # Keep last N keyframes
        self.global_search_tracks = config.get('global_search_tracks', 20)  # Search last N tracks
        
        # Track index for faster searching
        self.recent_keyframes = []  # List of (frame_id, detections)
        self.track_last_seen = {}  # track_id -> frame_id
        
        logger.info(f"Initialized Global 3D Tracker with window_size={self.window_size}")
    
    def _match_detections_to_tracks(self, 
                                   detections: Dict[int, Dict],
                                   frame_id: int) -> Dict[int, int]:
        """Enhanced matching that searches globally within sliding window."""
        
        assignments = {}
        used_tracks = set()
        
        # Update track last seen info
        for track_id, track in self.tracked_objects.items():
            self.track_last_seen[track_id] = track.last_seen_frame
        
        # Sort detections by confidence/size
        sorted_detections = sorted(
            detections.items(),
            key=lambda x: x[1]['bbox']['confidence'],
            reverse=True
        )
        
        for instance_id, detection in sorted_detections:
            best_track_id = None
            best_score = 0.0
            
            # Get candidate tracks - both recent and same-label tracks
            candidate_tracks = self._get_candidate_tracks(
                detection['label'], 
                frame_id, 
                used_tracks
            )
            
            # Try to match with each candidate
            for track_id in candidate_tracks:
                track = self.tracked_objects[track_id]
                
                # Compute matching score
                score = self._compute_3d_matching_score(
                    detection['bbox'],
                    track,
                    detection['label']
                )
                
                # Boost score for recently seen tracks
                recency_boost = self._compute_recency_boost(track, frame_id)
                score *= recency_boost
                
                if score > best_score and score > self.iou_threshold_3d:
                    best_score = score
                    best_track_id = track_id
            
            # Assign or create new track
            if best_track_id is not None:
                assignments[instance_id] = best_track_id
                used_tracks.add(best_track_id)
                
                track = self.tracked_objects[best_track_id]
                track.update_geometry(detection['bbox'], frame_id)
                
                logger.debug(f"Matched {detection['label']} to track {best_track_id} "
                           f"(score: {best_score:.3f})")
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
                self.track_last_seen[new_track_id] = frame_id
                
                logger.info(f"Created new track {new_track_id} for {detection['label']}")
        
        # Update recent keyframes window
        self._update_sliding_window(frame_id, detections)
        
        return assignments
    
    def _get_candidate_tracks(self, label: str, frame_id: int, 
                             used_tracks: Set[int]) -> List[int]:
        """Get candidate tracks using global search strategy."""
        candidates = []
        
        # 1. All tracks with same label (up to a limit)
        same_label_tracks = [
            track_id for track_id, track in self.tracked_objects.items()
            if track.label == label and track_id not in used_tracks
        ]
        
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
        """Boost score for recently seen tracks."""
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
        """Update sliding window of recent keyframes."""
        self.recent_keyframes.append((frame_id, detections))
        
        # Keep only recent keyframes
        if len(self.recent_keyframes) > self.window_size:
            self.recent_keyframes.pop(0)
    
    def _cleanup_lost_tracks(self):
        """Enhanced cleanup that's more conservative for global tracking."""
        tracks_to_remove = []
        
        for track_id, track in self.tracked_objects.items():
            # More conservative removal
            remove_threshold = self.max_lost_frames * 2  # Double threshold for global
            
            if (track.lost_frames > remove_threshold or 
                (track.confidence < 0.1 and track.total_observations < self.min_observations)):
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            track = self.tracked_objects.pop(track_id)
            if track_id in self.track_last_seen:
                del self.track_last_seen[track_id]
                
            logger.info(f"Removed track {track_id} ({track.label})")
    
    def get_track_statistics(self) -> Dict:
        """Get detailed statistics about tracking performance."""
        stats = {
            'total_tracks': len(self.tracked_objects),
            'active_tracks': sum(1 for t in self.tracked_objects.values() if t.lost_frames == 0),
            'tracks_by_label': {},
            'avg_observations': 0,
            'avg_confidence': 0
        }
        
        # Group by label
        label_counts = {}
        total_obs = []
        total_conf = []
        
        for track in self.tracked_objects.values():
            label_counts[track.label] = label_counts.get(track.label, 0) + 1
            total_obs.append(track.total_observations)
            total_conf.append(track.confidence)
        
        stats['tracks_by_label'] = label_counts
        stats['avg_observations'] = np.mean(total_obs) if total_obs else 0
        stats['avg_confidence'] = np.mean(total_conf) if total_conf else 0
        
        return stats