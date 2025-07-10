"""Semantic Loop Closure Verification for MAST3R-SLAM"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
from mast3r_slam.config import config
from mast3r_slam.semantic_integration import SemanticSLAMBackend
import torch.nn.functional as F


class SemanticLoopClosureFilter:
    """
    Add semantic verification to MAST3R's loop closure pipeline.
    
    This filter runs AFTER geometric verification to provide an additional
    layer of robustness by rejecting loop closures that are geometrically
    plausible but semantically inconsistent.
    """
    
    def __init__(self, 
                 semantic_backend: SemanticSLAMBackend,
                 min_semantic_overlap: float = 0.3,
                 min_instance_overlap: float = 0.2,
                 histogram_similarity_threshold: float = 0.5):
        """
        Args:
            semantic_backend: The semantic SLAM backend with keyframe semantics
            min_semantic_overlap: Minimum fraction of matched points with consistent labels
            min_instance_overlap: Minimum overlap for instance matching
            histogram_similarity_threshold: Minimum cosine similarity for label histograms
        """
        self.semantic_backend = semantic_backend
        self.min_semantic_overlap = min_semantic_overlap
        self.min_instance_overlap = min_instance_overlap
        self.histogram_similarity_threshold = histogram_similarity_threshold
        
        # Statistics tracking
        self.stats = {
            'total_checks': 0,
            'accepted': 0,
            'rejected_semantic': 0,
            'rejected_histogram': 0,
            'no_semantic_data': 0
        }
        
    def verify_semantic_consistency(self,
                                   kf_idx1: int,
                                   kf_idx2: int,
                                   matches: Dict) -> Tuple[bool, float, Dict]:
        """
        Verify semantic consistency between two keyframes using MAST3R matches.
        
        Args:
            kf_idx1, kf_idx2: Keyframe indices
            matches: Dictionary containing:
                - 'conf': [N, H, W] confidence maps
                - 'i_j': indices from frame i to j
                - 'j_i': indices from frame j to i
                - 'valid': validity mask
                
        Returns:
            (is_valid, semantic_score, details)
        """
        self.stats['total_checks'] += 1
        
        # Get semantic data for both keyframes
        sem1 = self.semantic_backend.keyframe_semantics.get(kf_idx1)
        sem2 = self.semantic_backend.keyframe_semantics.get(kf_idx2)
        
        if sem1 is None or sem2 is None:
            # No semantic data available - accept based on geometry alone
            self.stats['no_semantic_data'] += 1
            return True, 1.0, {'reason': 'no_semantic_data'}
            
        labels1, conf1 = sem1
        labels2, conf2 = sem2
        
        # Extract matched point indices from MAST3R format
        # MAST3R provides pixel correspondences, we need to sample semantics
        matched_indices_1, matched_indices_2 = self._extract_match_indices(matches)
        
        if len(matched_indices_1) == 0:
            return True, 1.0, {'reason': 'no_matches'}
            
        # Sample semantic labels at matched points
        matched_labels_1 = labels1[matched_indices_1]
        matched_labels_2 = labels2[matched_indices_2]
        
        # Filter out background (label 0) and low confidence matches
        min_conf = 0.5
        valid_mask = (matched_labels_1 > 0) & (matched_labels_2 > 0)
        if 'confidence' in matches:
            match_conf = matches['confidence']
            if len(match_conf) == len(valid_mask):
                valid_mask &= match_conf > min_conf
        
        if valid_mask.sum() == 0:
            return True, 1.0, {'reason': 'no_valid_semantic_matches'}
            
        # Check 1: Point-wise semantic consistency
        consistent = matched_labels_1[valid_mask] == matched_labels_2[valid_mask]
        semantic_overlap = consistent.float().mean().item()
        
        if semantic_overlap < self.min_semantic_overlap:
            self.stats['rejected_semantic'] += 1
            return False, semantic_overlap, {
                'reason': 'low_semantic_overlap',
                'overlap': semantic_overlap,
                'threshold': self.min_semantic_overlap
            }
            
        # Check 2: Global label distribution similarity
        hist_similarity = self._compute_histogram_similarity(labels1, labels2)
        
        if hist_similarity < self.histogram_similarity_threshold:
            self.stats['rejected_histogram'] += 1
            return False, semantic_overlap, {
                'reason': 'different_label_distribution',
                'histogram_similarity': hist_similarity,
                'threshold': self.histogram_similarity_threshold
            }
            
        # Check 3: Instance-level consistency (optional, more strict)
        instance_score = self._check_instance_consistency(
            matched_labels_1[valid_mask],
            matched_labels_2[valid_mask]
        )
        
        # Combine scores
        final_score = 0.6 * semantic_overlap + 0.3 * hist_similarity + 0.1 * instance_score
        
        self.stats['accepted'] += 1
        return True, final_score, {
            'semantic_overlap': semantic_overlap,
            'histogram_similarity': hist_similarity,
            'instance_score': instance_score,
            'final_score': final_score
        }
        
    def _extract_match_indices(self, matches: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract matched point indices from MAST3R match format."""
        if 'indices' in matches:
            # Direct index format
            indices = matches['indices']
            return indices[:, 0], indices[:, 1]
            
        elif 'i_j' in matches and 'valid' in matches:
            # Confidence map format - extract valid matches
            i_j = matches['i_j']
            j_i = matches['j_i']
            valid = matches['valid']
            
            # Get valid match locations
            valid_coords = torch.where(valid)
            if len(valid_coords[0]) > 0:
                # Sample up to 1000 matches for efficiency
                n_samples = min(1000, len(valid_coords[0]))
                sample_idx = torch.randperm(len(valid_coords[0]))[:n_samples]
                
                idx1 = valid_coords[0][sample_idx]
                idx2_coords = i_j[valid_coords][sample_idx]
                
                # Convert 2D coordinates to 1D indices
                # This is simplified - in practice need proper coordinate handling
                return idx1, idx2_coords
                
        return torch.tensor([]), torch.tensor([])
        
    def _compute_histogram_similarity(self, 
                                    labels1: torch.Tensor,
                                    labels2: torch.Tensor) -> float:
        """Compute cosine similarity between label histograms."""
        # Get unique labels (excluding background)
        unique_labels = torch.unique(torch.cat([labels1, labels2]))
        unique_labels = unique_labels[unique_labels > 0]
        
        if len(unique_labels) == 0:
            return 1.0
            
        # Compute normalized histograms
        hist1 = torch.zeros(len(unique_labels))
        hist2 = torch.zeros(len(unique_labels))
        
        for i, label in enumerate(unique_labels):
            hist1[i] = (labels1 == label).sum().float()
            hist2[i] = (labels2 == label).sum().float()
            
        # Normalize
        hist1 = hist1 / (hist1.sum() + 1e-6)
        hist2 = hist2 / (hist2.sum() + 1e-6)
        
        # Cosine similarity
        similarity = F.cosine_similarity(hist1.unsqueeze(0), hist2.unsqueeze(0))
        return similarity.item()
        
    def _check_instance_consistency(self,
                                  labels1: torch.Tensor,
                                  labels2: torch.Tensor) -> float:
        """Check if instances are consistently matched."""
        # Build instance correspondence graph
        correspondence = {}
        for l1, l2 in zip(labels1.tolist(), labels2.tolist()):
            if l1 not in correspondence:
                correspondence[l1] = {}
            correspondence[l1][l2] = correspondence[l1].get(l2, 0) + 1
            
        # Check consistency: each instance in frame1 should map mostly to one instance in frame2
        consistency_scores = []
        for l1, l2_counts in correspondence.items():
            total = sum(l2_counts.values())
            max_count = max(l2_counts.values())
            consistency = max_count / total
            consistency_scores.append(consistency)
            
        return np.mean(consistency_scores) if consistency_scores else 0.0
        
    def get_statistics(self) -> Dict:
        """Get verification statistics."""
        stats = self.stats.copy()
        if stats['total_checks'] > 0:
            stats['acceptance_rate'] = stats['accepted'] / stats['total_checks']
            stats['semantic_rejection_rate'] = stats['rejected_semantic'] / stats['total_checks']
            stats['histogram_rejection_rate'] = stats['rejected_histogram'] / stats['total_checks']
        return stats
        
    def reset_statistics(self):
        """Reset statistics counters."""
        self.stats = {
            'total_checks': 0,
            'accepted': 0,
            'rejected_semantic': 0,
            'rejected_histogram': 0,
            'no_semantic_data': 0
        }


class SemanticFactorWeight:
    """
    Optional: Compute semantic-aware weights for factors in optimization.
    """
    
    def __init__(self, base_weight: float = 1.0, semantic_boost: float = 0.2):
        self.base_weight = base_weight
        self.semantic_boost = semantic_boost
        
    def compute_weight(self, 
                      geometric_confidence: float,
                      semantic_score: float) -> float:
        """
        Combine geometric and semantic confidence into edge weight.
        
        Args:
            geometric_confidence: MAST3R matching confidence
            semantic_score: Semantic consistency score [0, 1]
            
        Returns:
            Combined weight for optimization
        """
        # Simple linear combination
        # Could be more sophisticated (e.g., learned)
        semantic_multiplier = 1.0 + self.semantic_boost * semantic_score
        return geometric_confidence * semantic_multiplier