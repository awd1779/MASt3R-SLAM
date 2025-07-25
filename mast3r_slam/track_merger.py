"""Post-process tracks to merge similar tracks that should be the same object."""

import numpy as np
import torch
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger('mast3r_slam.track_merger')


class TrackMerger:
    """Merge tracks that likely represent the same object."""
    
    def __init__(self, semantic_keyframes):
        self.semantic_keyframes = semantic_keyframes
        
    def merge_similar_tracks(self, merge_threshold: float = 0.7) -> Dict[int, int]:
        """Analyze all tracks and merge similar ones.
        
        Returns mapping from old track IDs to new merged IDs.
        """
        logger.info("Starting track merging analysis...")
        
        # Collect track information
        track_info = self._collect_track_info()
        
        # Find tracks to merge based on:
        # 1. Same label
        # 2. Spatial overlap or proximity
        # 3. Temporal gaps (one ends where another begins)
        
        merge_groups = []
        processed_tracks = set()
        
        for track_id, info in track_info.items():
            if track_id in processed_tracks:
                continue
                
            # Find all tracks with same label
            same_label_tracks = [
                tid for tid, tinfo in track_info.items()
                if tinfo['label'] == info['label'] and tid != track_id
            ]
            
            # Check each for merging
            merge_group = {track_id}
            processed_tracks.add(track_id)
            
            for other_id in same_label_tracks:
                if other_id in processed_tracks:
                    continue
                    
                other_info = track_info[other_id]
                
                # Check temporal overlap or adjacency
                temporal_score = self._compute_temporal_score(
                    info['frames'], other_info['frames']
                )
                
                # Check spatial proximity
                spatial_score = self._compute_spatial_score(
                    info['centroids'], other_info['centroids'],
                    info['label']
                )
                
                # Combined score
                merge_score = 0.6 * spatial_score + 0.4 * temporal_score
                
                if merge_score > merge_threshold:
                    merge_group.add(other_id)
                    processed_tracks.add(other_id)
                    logger.info(f"Merging track {other_id} into {track_id} "
                              f"(label: {info['label']}, score: {merge_score:.3f})")
            
            if len(merge_group) > 1:
                merge_groups.append(merge_group)
        
        # Create remapping
        track_remapping = {}
        for group in merge_groups:
            min_id = min(group)  # Use lowest ID as representative
            for track_id in group:
                track_remapping[track_id] = min_id
        
        # Add identity mappings for unmerged tracks
        for track_id in track_info.keys():
            if track_id not in track_remapping:
                track_remapping[track_id] = track_id
        
        logger.info(f"Merged {sum(len(g)-1 for g in merge_groups)} tracks into "
                   f"{len(merge_groups)} groups")
        
        return track_remapping
    
    def _collect_track_info(self) -> Dict[int, Dict]:
        """Collect information about each track."""
        track_info = defaultdict(lambda: {
            'label': None,
            'frames': [],
            'centroids': [],
            'sizes': []
        })
        
        # Go through all keyframes
        for kf_idx in range(len(self.semantic_keyframes._data)):
            if not self.semantic_keyframes.has_semantic_data(kf_idx):
                continue
                
            semantic_data = self.semantic_keyframes.get_semantics(kf_idx)
            if 'track_ids' not in semantic_data:
                continue
                
            # Get frame info
            frame_id = semantic_data.get('frame_id', kf_idx)
            
            # Process each tracked instance
            for instance_id, track_id in semantic_data['track_ids'].items():
                label = semantic_data['labels'].get(instance_id, 'unknown')
                
                # Update track info
                if track_info[track_id]['label'] is None:
                    track_info[track_id]['label'] = label
                
                track_info[track_id]['frames'].append(frame_id)
                
                # Get spatial info if available
                # (In a real implementation, we'd extract centroid from the 3D bbox)
                # For now, just use frame number as proxy
                track_info[track_id]['centroids'].append(frame_id)
        
        return dict(track_info)
    
    def _compute_temporal_score(self, frames1: List[int], frames2: List[int]) -> float:
        """Compute temporal relationship score."""
        if not frames1 or not frames2:
            return 0.0
            
        # Check for overlap
        set1 = set(frames1)
        set2 = set(frames2)
        
        if set1 & set2:  # Overlapping frames - shouldn't be same object
            return 0.0
        
        # Check for adjacency
        min_gap = min(abs(f1 - f2) for f1 in frames1 for f2 in frames2)
        
        # Score based on gap
        if min_gap <= 2:  # Adjacent or 1 frame gap
            return 1.0
        elif min_gap <= 5:  # Small gap
            return 0.8
        elif min_gap <= 10:  # Medium gap
            return 0.5
        else:
            return 0.2
    
    def _compute_spatial_score(self, centroids1: List, centroids2: List, 
                              label: str) -> float:
        """Compute spatial proximity score."""
        # Simplified version - in practice would use actual 3D centroids
        # For now, return high score for static objects
        static_objects = {'wall', 'floor', 'ceiling', 'door', 'window'}
        
        if any(obj in label.lower() for obj in static_objects):
            return 0.9  # Static objects likely to be same
        else:
            return 0.6  # Dynamic objects need more careful checking
    
    def apply_remapping(self, track_remapping: Dict[int, int]):
        """Apply track remapping to all semantic keyframes."""
        remapped_count = 0
        
        for kf_idx in range(len(self.semantic_keyframes._data)):
            if not self.semantic_keyframes.has_semantic_data(kf_idx):
                continue
                
            semantic_data = self.semantic_keyframes._data[kf_idx]
            if semantic_data is None or 'track_ids' not in semantic_data:
                continue
            
            # Remap track IDs
            original_track_ids = semantic_data['track_ids'].copy()
            for instance_id, track_id in original_track_ids.items():
                if track_id in track_remapping and track_remapping[track_id] != track_id:
                    semantic_data['track_ids'][instance_id] = track_remapping[track_id]
                    remapped_count += 1
        
        logger.info(f"Remapped {remapped_count} track assignments")


def merge_tracks_post_process(semantic_keyframes, merge_threshold: float = 0.7):
    """Post-process tracks to merge similar ones."""
    merger = TrackMerger(semantic_keyframes)
    
    # Find tracks to merge
    track_remapping = merger.merge_similar_tracks(merge_threshold)
    
    # Apply remapping
    merger.apply_remapping(track_remapping)
    
    return track_remapping