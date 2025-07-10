"""Enhanced FactorGraph with semantic loop closure verification"""

import torch
from typing import Optional, Dict, List, Tuple
from mast3r_slam.global_opt import FactorGraph
from mast3r_slam.semantic_loop_closure import SemanticLoopClosureFilter
from mast3r_slam.config import config


class SemanticFactorGraph(FactorGraph):
    """
    Extended FactorGraph that adds semantic verification to loop closures.
    
    Inherits from the original FactorGraph and adds semantic filtering
    as an additional verification step after geometric matching.
    """
    
    def __init__(self, 
                 model, 
                 frames, 
                 K=None, 
                 device="cuda",
                 semantic_filter: Optional[SemanticLoopClosureFilter] = None):
        super().__init__(model, frames, K, device)
        
        self.semantic_filter = semantic_filter
        self.semantic_scores = {}  # Store semantic scores for each edge
        
        # Additional configuration for semantic loop closure
        self.semantic_cfg = config.get("semantic_loop_closure", {})
        self.enable_semantic_verification = self.semantic_cfg.get("enabled", True)
        self.semantic_weight_factor = self.semantic_cfg.get("weight_factor", 0.0)  # 0 = no weighting
        
    def add_factors(self, ii, jj, min_match_frac, is_reloc=False):
        """
        Extended add_factors with semantic verification.
        
        The pipeline:
        1. Perform MAST3R matching (parent class)
        2. Apply geometric verification (parent class)
        3. Apply semantic verification (new)
        4. Add accepted edges to factor graph
        """
        # Get features and perform matching
        kf_ii = [self.frames[idx] for idx in ii]
        kf_jj = [self.frames[idx] for idx in jj]
        feat_i = torch.cat([kf_i.feat for kf_i in kf_ii])
        feat_j = torch.cat([kf_j.feat for kf_j in kf_jj])
        pos_i = torch.cat([kf_i.pos for kf_i in kf_ii])
        pos_j = torch.cat([kf_j.pos for kf_j in kf_jj])
        shape_i = [kf_i.img_true_shape for kf_i in kf_ii]
        shape_j = [kf_j.img_true_shape for kf_j in kf_jj]

        # Perform MAST3R matching
        from mast3r_slam.mast3r_utils import mast3r_match_symmetric
        (
            idx_i2j,
            idx_j2i,
            valid_match_j,
            valid_match_i,
            Qii,
            Qjj,
            Qji,
            Qij,
        ) = mast3r_match_symmetric(
            self.model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j
        )

        batch_inds = torch.arange(idx_i2j.shape[0], device=idx_i2j.device)[
            :, None
        ].repeat(1, idx_i2j.shape[1])
        Qj = torch.sqrt(Qii[batch_inds, idx_i2j] * Qji)
        Qi = torch.sqrt(Qjj[batch_inds, idx_j2i] * Qij)

        # Apply confidence threshold
        valid_Qj = Qj > self.cfg["Q_conf"]
        valid_Qi = Qi > self.cfg["Q_conf"]
        valid_j = valid_match_j & valid_Qj
        valid_i = valid_match_i & valid_Qi
        
        # Compute match fractions
        nj = valid_j.shape[1] * valid_j.shape[2]
        ni = valid_i.shape[1] * valid_i.shape[2]
        match_frac_j = valid_j.sum(dim=(1, 2)) / nj
        match_frac_i = valid_i.sum(dim=(1, 2)) / ni

        ii_tensor = torch.as_tensor(ii, device=self.device)
        jj_tensor = torch.as_tensor(jj, device=self.device)

        # Geometric verification
        invalid_edges = torch.minimum(match_frac_j, match_frac_i) < min_match_frac
        consecutive_edges = ii_tensor == (jj_tensor - 1)
        invalid_edges = (~consecutive_edges) & invalid_edges

        if invalid_edges.any() and is_reloc:
            return False

        # Apply semantic verification for non-consecutive edges
        if self.semantic_filter is not None and self.enable_semantic_verification:
            semantic_valid = torch.ones_like(invalid_edges, dtype=torch.bool)
            
            for idx, (i, j) in enumerate(zip(ii, jj)):
                # Skip consecutive frames and already invalid edges
                if consecutive_edges[idx] or invalid_edges[idx]:
                    continue
                    
                # Prepare matches for semantic verification
                matches = {
                    'indices': torch.stack([
                        torch.where(valid_j[idx].flatten())[0],
                        idx_i2j[idx][valid_j[idx]].flatten()
                    ], dim=1),
                    'confidence': Qj[idx][valid_j[idx]].flatten(),
                    'valid': valid_j[idx]
                }
                
                # Perform semantic verification
                is_valid, sem_score, details = self.semantic_filter.verify_semantic_consistency(
                    i, j, matches
                )
                
                if not is_valid:
                    semantic_valid[idx] = False
                    if config.get("verbose", False):
                        print(f"Semantic rejection: KF {i}-{j}, score: {sem_score:.3f}, reason: {details['reason']}")
                else:
                    # Store semantic score for potential use in optimization
                    self.semantic_scores[(i, j)] = sem_score
                    if config.get("verbose", False) and sem_score < 0.5:
                        print(f"Semantic warning: KF {i}-{j}, low score: {sem_score:.3f}")
                        
            # Update invalid edges based on semantic verification
            invalid_edges = invalid_edges | (~semantic_valid)

        # Filter valid edges
        valid_edges = ~invalid_edges
        ii_tensor = ii_tensor[valid_edges]
        jj_tensor = jj_tensor[valid_edges]
        idx_i2j = idx_i2j[valid_edges]
        idx_j2i = idx_j2i[valid_edges]
        valid_match_j = valid_match_j[valid_edges]
        valid_match_i = valid_match_i[valid_edges]
        Qj = Qj[valid_edges]
        Qi = Qi[valid_edges]

        # Apply semantic weighting if enabled
        if self.semantic_weight_factor > 0 and self.semantic_filter is not None:
            for idx, (i, j) in enumerate(zip(ii_tensor.tolist(), jj_tensor.tolist())):
                if (i, j) in self.semantic_scores:
                    sem_score = self.semantic_scores[(i, j)]
                    # Boost confidence based on semantic score
                    weight_boost = 1.0 + self.semantic_weight_factor * (sem_score - 0.5)
                    Qj[idx] *= weight_boost
                    Qi[idx] *= weight_boost

        # Add edges to factor graph
        self.ii = torch.cat([self.ii, ii_tensor])
        self.jj = torch.cat([self.jj, jj_tensor])
        self.idx_ii2jj = torch.cat([self.idx_ii2jj, idx_i2j])
        self.idx_jj2ii = torch.cat([self.idx_jj2ii, idx_j2i])
        self.valid_match_j = torch.cat([self.valid_match_j, valid_match_j])
        self.valid_match_i = torch.cat([self.valid_match_i, valid_match_i])
        self.Q_ii2jj = torch.cat([self.Q_ii2jj, Qj])
        self.Q_jj2ii = torch.cat([self.Q_jj2ii, Qi])

        added_new_edges = valid_edges.sum() > 0
        return added_new_edges
        
    def get_semantic_statistics(self) -> Dict:
        """Get statistics about semantic verification."""
        if self.semantic_filter is None:
            return {}
            
        stats = self.semantic_filter.get_statistics()
        stats['average_semantic_score'] = (
            sum(self.semantic_scores.values()) / len(self.semantic_scores)
            if self.semantic_scores else 0.0
        )
        stats['num_edges_with_semantic_score'] = len(self.semantic_scores)
        return stats


def create_semantic_factor_graph(model, frames, K, device, semantic_backend=None):
    """
    Factory function to create a SemanticFactorGraph with proper configuration.
    
    Args:
        model: MAST3R model
        frames: SharedKeyframes
        K: Camera intrinsics
        device: Compute device
        semantic_backend: Optional SemanticSLAMBackend for semantic verification
        
    Returns:
        SemanticFactorGraph or regular FactorGraph based on configuration
    """
    semantic_lc_config = config.get("semantic_loop_closure", {})
    
    if semantic_backend is not None and semantic_lc_config.get("enabled", False):
        # Create semantic filter
        semantic_filter = SemanticLoopClosureFilter(
            semantic_backend,
            min_semantic_overlap=semantic_lc_config.get("min_semantic_overlap", 0.3),
            min_instance_overlap=semantic_lc_config.get("min_instance_overlap", 0.2),
            histogram_similarity_threshold=semantic_lc_config.get("histogram_similarity", 0.5)
        )
        
        print("Using SemanticFactorGraph with loop closure verification")
        return SemanticFactorGraph(model, frames, K, device, semantic_filter)
    else:
        # Fall back to regular factor graph
        print("Using standard FactorGraph (semantic verification disabled)")
        return FactorGraph(model, frames, K, device)