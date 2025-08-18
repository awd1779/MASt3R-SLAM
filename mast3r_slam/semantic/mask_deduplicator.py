"""Mask deduplication utilities for Grounded SAM2 processor."""

import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict
import logging

logger = logging.getLogger('mast3r_slam.mask_deduplicator')


class MaskDeduplicator:
    """Handles deduplication of overlapping masks with different strategies."""
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
    
    def deduplicate_masks(self, masks: List[np.ndarray], labels: List[str], 
                         scores: List[float], frame_idx: int) -> Tuple[List[np.ndarray], List[str], List[float]]:
        """Complete deduplication pipeline: merge same labels, then resolve cross-label overlaps."""
        if len(masks) <= 1:
            return masks, labels, scores
        
        # Step 1: Merge duplicate labels (keep best score per label)
        masks, labels, scores = self._merge_duplicate_labels(masks, labels, scores, frame_idx)
        
        # Step 2: Resolve cross-label overlaps
        masks, labels = self._resolve_overlaps(masks, labels, scores, frame_idx)
        
        return masks, labels, scores[:len(masks)]  # Trim scores to match final mask count
    
    def _merge_duplicate_labels(self, masks: List[np.ndarray], labels: List[str], 
                               scores: List[float], frame_idx: int) -> Tuple[List[np.ndarray], List[str], List[float]]:
        """Merge masks with identical labels, keeping the highest scoring one."""
        label_groups = defaultdict(list)
        for i, label in enumerate(labels):
            label_groups[label].append(i)
        
        final_masks, final_labels, final_scores = [], [], []
        
        for label, indices in label_groups.items():
            if len(indices) == 1:
                idx = indices[0]
                final_masks.append(masks[idx])
                final_labels.append(label)
                final_scores.append(scores[idx])
            else:
                # Keep highest scoring mask
                best_idx = indices[np.argmax([scores[i] for i in indices])]
                final_masks.append(masks[best_idx])
                final_labels.append(label)
                final_scores.append(scores[best_idx])
                
                if self.debug_mode:
                    logger.info(f"Frame {frame_idx}: Merged {len(indices)} '{label}' masks, "
                              f"kept best score {scores[best_idx]:.3f}")
        
        return final_masks, final_labels, final_scores
    
    def _resolve_overlaps(self, masks: List[np.ndarray], labels: List[str], 
                         scores: List[float], frame_idx: int) -> Tuple[List[np.ndarray], List[str]]:
        """Remove overlapping masks between different labels based on confidence."""
        if len(masks) <= 1:
            return masks, labels
        
        keep_mask = [True] * len(masks)
        removals = 0
        
        # Compare all pairs
        for i in range(len(masks)):
            if not keep_mask[i]:
                continue
                
            for j in range(i + 1, len(masks)):
                if not keep_mask[j] or labels[i] == labels[j]:
                    continue
                
                # Calculate overlap
                mask_i, mask_j = masks[i] > 0.5, masks[j] > 0.5
                intersection = np.sum(mask_i & mask_j)
                
                if intersection == 0:
                    continue
                
                area_i, area_j = np.sum(mask_i), np.sum(mask_j)
                iou = intersection / np.sum(mask_i | mask_j)
                containment = max(intersection / area_i, intersection / area_j)
                
                # Remove if significant overlap
                if iou > 0.8 or containment > 0.9:
                    # Keep higher confidence mask
                    if scores[i] >= scores[j]:
                        keep_mask[j] = False
                        removals += 1
                    else:
                        keep_mask[i] = False
                        removals += 1
                        break
        
        final_masks = [mask for mask, keep in zip(masks, keep_mask) if keep]
        final_labels = [label for label, keep in zip(labels, keep_mask) if keep]
        
        if self.debug_mode and removals > 0:
            logger.info(f"Frame {frame_idx}: Removed {removals} overlapping masks")
        
        return final_masks, final_labels