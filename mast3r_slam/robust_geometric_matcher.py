"""Robust geometric matching that handles partial visibility and coordinate drift."""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger('mast3r_slam.robust_geometric_matcher')


class RobustGeometricMatcher:
    """Enhanced geometric matching that handles real-world tracking challenges."""
    
    def __init__(self, config: Dict):
        # Adaptive thresholds based on object characteristics
        self.base_iou_threshold = config.get('base_iou_threshold', 0.1)
        self.distance_decay_factor = config.get('distance_decay_factor', 2.0)
        self.size_tolerance_factor = config.get('size_tolerance_factor', 3.0)
        self.min_overlap_ratio = config.get('min_overlap_ratio', 0.1)
        
        # Appearance consistency parameters
        self.use_appearance = config.get('use_appearance', True)
        self.appearance_weight = config.get('appearance_weight', 0.3)
        
        # Temporal consistency parameters
        self.temporal_window = config.get('temporal_window', 5)
        self.velocity_prediction = config.get('velocity_prediction', True)
        
    def compute_adaptive_matching_score(self,
                                      detection: Dict,
                                      track: 'TrackedObject3D',
                                      frame_id: int) -> Tuple[float, Dict]:
        """Compute matching score with adaptive thresholds based on context."""
        
        det_bbox = detection['bbox_world']
        track_bbox = track.bbox_3d_world
        
        if track_bbox is None:
            return 0.0, {'reason': 'no_track_bbox'}
        
        # 1. Compute basic geometric overlap
        iou_3d = self._compute_3d_iou(det_bbox, track_bbox)
        
        # 2. Compute centroid distance with normalization
        centroid_dist = torch.norm(det_bbox['center'] - track_bbox['center']).item()
        
        # Normalize by object size (use larger object as reference)
        max_size = max(det_bbox['dimensions'].max().item(), 
                      track_bbox['dimensions'].max().item())
        normalized_dist = centroid_dist / (max_size + 1e-6)
        
        # 3. Compute size consistency with tolerance
        size_ratio = self._compute_size_ratio(det_bbox, track_bbox)
        size_consistent = self._is_size_consistent(size_ratio, track, frame_id)
        
        # 4. Compute partial visibility score
        partial_score = self._compute_partial_visibility_score(
            det_bbox, track_bbox, iou_3d
        )
        
        # 5. Adaptive threshold based on context
        adaptive_threshold = self._compute_adaptive_threshold(
            detection, track, frame_id, normalized_dist
        )
        
        # 6. Combine scores with adaptive weighting
        geometric_score = self._combine_geometric_scores(
            iou_3d, normalized_dist, size_ratio, partial_score
        )
        
        # 7. Apply temporal consistency boost
        if self.velocity_prediction and track.total_observations > 1:
            temporal_boost = self._compute_temporal_consistency(
                detection, track, frame_id
            )
            geometric_score *= temporal_boost
        
        # 8. Final score with threshold check
        final_score = geometric_score if geometric_score > adaptive_threshold else 0.0
        
        # Debug information
        debug_info = {
            'iou_3d': iou_3d,
            'centroid_dist': centroid_dist,
            'normalized_dist': normalized_dist,
            'size_ratio': size_ratio,
            'partial_score': partial_score,
            'adaptive_threshold': adaptive_threshold,
            'geometric_score': geometric_score,
            'final_score': final_score
        }
        
        return final_score, debug_info
    
    def _compute_3d_iou(self, bbox1: Dict, bbox2: Dict) -> float:
        """Compute 3D IoU between two bounding boxes."""
        # Implementation depends on bbox type (AABB, OBB, etc.)
        if bbox1['type'] == 'aabb' and bbox2['type'] == 'aabb':
            return self._compute_aabb_iou(bbox1, bbox2)
        else:
            # Fallback to approximate IoU
            return self._compute_approximate_iou(bbox1, bbox2)
    
    def _compute_aabb_iou(self, bbox1: Dict, bbox2: Dict) -> float:
        """Compute IoU for axis-aligned bounding boxes."""
        # Get min/max corners
        min1, max1 = bbox1.get('min'), bbox1.get('max')
        min2, max2 = bbox2.get('min'), bbox2.get('max')
        
        if min1 is None or max1 is None or min2 is None or max2 is None:
            # Compute from center and dimensions
            half_dim1 = bbox1['dimensions'] / 2
            half_dim2 = bbox2['dimensions'] / 2
            min1 = bbox1['center'] - half_dim1
            max1 = bbox1['center'] + half_dim1
            min2 = bbox2['center'] - half_dim2
            max2 = bbox2['center'] + half_dim2
        
        # Compute intersection
        inter_min = torch.max(min1, min2)
        inter_max = torch.min(max1, max2)
        
        # Check if there's actual intersection
        if torch.any(inter_min >= inter_max):
            return 0.0
        
        # Compute volumes
        inter_vol = torch.prod(inter_max - inter_min).item()
        vol1 = torch.prod(max1 - min1).item()
        vol2 = torch.prod(max2 - min2).item()
        
        # IoU
        union_vol = vol1 + vol2 - inter_vol
        iou = inter_vol / (union_vol + 1e-6)
        
        return iou
    
    def _compute_approximate_iou(self, bbox1: Dict, bbox2: Dict) -> float:
        """Approximate IoU using center distance and dimensions."""
        # Simple approximation based on center distance
        dist = torch.norm(bbox1['center'] - bbox2['center']).item()
        avg_size = (bbox1['dimensions'].mean() + bbox2['dimensions'].mean()) / 2
        
        # Convert distance to overlap approximation
        overlap_factor = max(0, 1 - dist / avg_size.item())
        
        # Consider size similarity
        size_similarity = self._compute_size_similarity(bbox1, bbox2)
        
        return overlap_factor * size_similarity * 0.5  # Conservative estimate
    
    def _compute_size_ratio(self, bbox1: Dict, bbox2: Dict) -> float:
        """Compute size ratio between two bboxes."""
        vol1 = bbox1['volume']
        vol2 = bbox2['volume']
        return vol1 / (vol2 + 1e-6)
    
    def _compute_size_similarity(self, bbox1: Dict, bbox2: Dict) -> float:
        """Compute size similarity score [0, 1]."""
        ratio = self._compute_size_ratio(bbox1, bbox2)
        # Convert ratio to similarity (1.0 means identical size)
        return 2.0 * min(ratio, 1/ratio) / (1 + min(ratio, 1/ratio))
    
    def _is_size_consistent(self, size_ratio: float, 
                           track: 'TrackedObject3D', 
                           frame_id: int) -> bool:
        """Check if size change is consistent with expected growth patterns."""
        # Allow more tolerance for new tracks (partial visibility)
        if track.total_observations < 3:
            return 0.1 < size_ratio < 10.0  # Very tolerant
        elif track.total_observations < 5:
            return 0.2 < size_ratio < 5.0   # Moderately tolerant
        else:
            return 0.5 < size_ratio < 2.0   # Normal tolerance
    
    def _compute_partial_visibility_score(self, 
                                        det_bbox: Dict,
                                        track_bbox: Dict,
                                        iou: float) -> float:
        """Score for handling partial object visibility."""
        # Check if one bbox is significantly contained in the other
        det_vol = det_bbox['volume']
        track_vol = track_bbox['volume']
        
        # Estimate containment from volumes and IoU
        if iou > 0:
            # Approximate intersection volume
            inter_vol = iou * min(det_vol, track_vol)
            
            # Containment ratios
            det_in_track = inter_vol / (det_vol + 1e-6)
            track_in_det = inter_vol / (track_vol + 1e-6)
            
            # High containment suggests partial visibility
            max_containment = max(det_in_track, track_in_det)
            
            if max_containment > 0.7:
                return 0.8  # High confidence for partial visibility
            elif max_containment > 0.5:
                return 0.6
            else:
                return 0.3
        
        return 0.0
    
    def _compute_adaptive_threshold(self,
                                   detection: Dict,
                                   track: 'TrackedObject3D',
                                   frame_id: int,
                                   normalized_dist: float) -> float:
        """Compute adaptive matching threshold based on context."""
        base_threshold = self.base_iou_threshold
        
        # 1. Adjust for object size (larger objects can have lower IoU)
        size_factor = detection['bbox_world']['dimensions'].max().item()
        if size_factor > 2.0:  # Large object
            base_threshold *= 0.5
        elif size_factor < 0.5:  # Small object
            base_threshold *= 1.5
        
        # 2. Adjust for track age (be more lenient with new tracks)
        if track.total_observations < 3:
            base_threshold *= 0.5
        
        # 3. Adjust for recent visibility (lost tracks need lower threshold)
        if track.lost_frames > 0:
            base_threshold *= (0.8 ** track.lost_frames)
        
        # 4. Adjust for distance (nearby objects need higher threshold)
        if normalized_dist < 0.5:
            base_threshold *= 1.2
        elif normalized_dist > 2.0:
            base_threshold *= 0.7
        
        return max(0.01, min(0.5, base_threshold))  # Clamp to reasonable range
    
    def _combine_geometric_scores(self,
                                 iou: float,
                                 norm_dist: float,
                                 size_ratio: float,
                                 partial_score: float) -> float:
        """Combine multiple geometric scores into final score."""
        # Distance score (inverse relationship)
        dist_score = 1.0 / (1.0 + norm_dist * self.distance_decay_factor)
        
        # Size score (similarity to 1.0 is good)
        size_score = self._compute_size_similarity({'volume': size_ratio}, {'volume': 1.0})
        
        # Weighted combination
        if iou > 0.3:
            # High IoU - rely mainly on IoU
            return 0.7 * iou + 0.2 * dist_score + 0.1 * size_score
        elif partial_score > 0.6:
            # Partial visibility case
            return 0.3 * iou + 0.4 * partial_score + 0.2 * dist_score + 0.1 * size_score
        else:
            # Low/no overlap - use distance and size
            return 0.2 * iou + 0.5 * dist_score + 0.2 * size_score + 0.1 * partial_score
    
    def _compute_temporal_consistency(self,
                                     detection: Dict,
                                     track: 'TrackedObject3D',
                                     frame_id: int) -> float:
        """Compute temporal consistency boost based on motion prediction."""
        if len(track.world_observations) < 2:
            return 1.0
        
        # Simple velocity-based prediction
        if len(track.world_observations) >= 2:
            # Last two positions
            pos1 = track.world_observations[-2]
            pos2 = track.world_observations[-1]
            
            # Velocity
            velocity = pos2 - pos1
            
            # Predicted position
            predicted_pos = pos2 + velocity
            
            # Compare with detection
            pred_error = torch.norm(detection['bbox_world']['center'] - predicted_pos).item()
            
            # Convert to boost factor
            boost = 1.0 + 0.5 * np.exp(-pred_error)  # Up to 1.5x boost
            
            return min(boost, 1.5)
        
        return 1.0