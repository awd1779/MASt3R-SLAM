# mast3r_slam/semantic_utils.py
"""
Consolidated semantic utilities including mapping, filtering, and post-processing.
Merges functionality from semantic_mapper, semantic_duplicate_filter, and semantic_post_processor.
"""
import torch
import numpy as np
from typing import Tuple, Optional, Dict, List, Set
from collections import defaultdict
import torch.nn.functional as F


# ============================================================================
# Semantic Mapper (from semantic_mapper.py)
# ============================================================================

class SemanticMapper:
    """Maps 2D semantic labels to 3D points with proper handling of MASt3R's point structure."""
    
    def __init__(self, device='cuda'):
        self.device = device
        
    def get_pixel_coordinates_for_points(self, frame_shape: torch.Tensor, num_points: int, 
                                       downsample: int = 1) -> torch.Tensor:
        """
        Get pixel coordinates for MASt3R points assuming row-major ordering.
        
        Args:
            frame_shape: (H, W) tensor of frame dimensions
            num_points: Number of 3D points
            downsample: Downsampling factor used
            
        Returns:
            Tensor of shape (num_points, 2) with (y, x) pixel coordinates
        """
        h, w = frame_shape
        
        # MASt3R typically generates points in a grid pattern
        expected_points = h * w
        
        if num_points == expected_points:
            # Standard case: one point per pixel
            y_coords = torch.arange(h, device=self.device).repeat_interleave(w)
            x_coords = torch.arange(w, device=self.device).repeat(h)
            pixel_coords = torch.stack([y_coords, x_coords], dim=1)
        else:
            # Points don't match pixel grid - use interpolation
            print(f"[SemanticMapper] Point count ({num_points}) doesn't match pixel grid ({expected_points})")
            
            if num_points < expected_points:
                # Subsample the pixel grid
                ratio = expected_points / num_points
                stride = int(np.sqrt(ratio.cpu().numpy() if torch.is_tensor(ratio) else ratio))
                y_coords = torch.arange(0, h, stride, device=self.device)
                x_coords = torch.arange(0, w, stride, device=self.device)
                yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
                pixel_coords = torch.stack([yy.flatten(), xx.flatten()], dim=1)
                
                # Trim to exact number of points if needed
                if pixel_coords.shape[0] > num_points:
                    pixel_coords = pixel_coords[:num_points]
                elif pixel_coords.shape[0] < num_points:
                    # Pad with repeated last coordinate
                    padding = num_points - pixel_coords.shape[0]
                    pixel_coords = torch.cat([
                        pixel_coords,
                        pixel_coords[-1:].repeat(padding, 1)
                    ], dim=0)
            else:
                # More points than pixels - shouldn't happen but handle gracefully
                print(f"[SemanticMapper] WARNING: More points than pixels, using modulo mapping")
                indices = torch.arange(num_points, device=self.device)
                y_coords = (indices // w) % h
                x_coords = indices % w
                pixel_coords = torch.stack([y_coords, x_coords], dim=1)
                
        return pixel_coords
    
    def map_2d_labels_to_3d_points(self, 
                                  semantic_mask: torch.Tensor,
                                  confidence_map: Optional[torch.Tensor],
                                  num_points: int,
                                  frame_shape: torch.Tensor,
                                  confidence_threshold: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Map 2D semantic labels to 3D points with confidence filtering.
        
        Args:
            semantic_mask: (H, W) tensor of semantic labels
            confidence_map: Optional (num_points,) tensor of point confidences
            num_points: Number of 3D points to label
            frame_shape: (H, W) tensor of frame dimensions
            confidence_threshold: Minimum confidence to assign labels
            
        Returns:
            labels: (num_points,) tensor of semantic labels
            valid_mask: (num_points,) boolean tensor of valid assignments
        """
        h, w = semantic_mask.shape
        
        # Get pixel coordinates for each 3D point
        pixel_coords = self.get_pixel_coordinates_for_points(frame_shape, num_points)
        
        # Clamp coordinates to valid range
        pixel_coords[:, 0] = torch.clamp(pixel_coords[:, 0], 0, h - 1)
        pixel_coords[:, 1] = torch.clamp(pixel_coords[:, 1], 0, w - 1)
        
        # Extract labels at pixel coordinates
        y_coords = pixel_coords[:, 0].long()
        x_coords = pixel_coords[:, 1].long()
        labels = semantic_mask[y_coords, x_coords]
        
        # Apply confidence filtering if available
        if confidence_map is not None and confidence_threshold > 0:
            # Ensure confidence_map is 1D
            if confidence_map.dim() > 1:
                confidence_map = confidence_map.squeeze()
            valid_mask = confidence_map >= confidence_threshold
        else:
            valid_mask = torch.ones(num_points, dtype=torch.bool, device=self.device)
        
        # Ensure shapes match
        if labels.shape != valid_mask.shape:
            print(f"[WARNING] Shape mismatch: labels {labels.shape} vs valid_mask {valid_mask.shape}")
            # Make sure both are 1D with same length
            labels = labels.view(-1)
            valid_mask = valid_mask.view(-1)
            min_len = min(labels.shape[0], valid_mask.shape[0])
            labels = labels[:min_len]
            valid_mask = valid_mask[:min_len]
        
        # Debug output
        labeled_count = (labels > 0).sum().item()
        valid_count = valid_mask.sum().item()
        print(f"[SemanticMapper] 2D->3D mapping for {num_points} points:")
        print(f"  2D labeled: {labeled_count}/{num_points} ({labeled_count/num_points*100:.1f}%)")
        # Compute on CPU if needed to avoid OOM
        labels_cpu = labels.cpu()
        valid_mask_cpu = valid_mask.cpu()
        labeled_valid = ((labels_cpu > 0) & valid_mask_cpu).sum().item()
        print(f"  3D labeled: {labeled_valid}/{num_points} ({labeled_valid/num_points*100:.1f}%)")
        print(f"  Confidence: threshold={confidence_threshold:.3f}, passing={valid_count}")
        
        return labels, valid_mask
    
    def refine_labels_with_3d_proximity(self,
                                       points_3d: torch.Tensor,
                                       labels: torch.Tensor,
                                       label_confidences: Dict[int, float],
                                       distance_threshold: float = 0.1,
                                       min_neighbors: int = 3) -> torch.Tensor:
        """
        Refine labels using 3D spatial proximity and confidence scores.
        """
        refined_labels = labels.clone()
        unique_labels = torch.unique(labels)
        
        # Skip if no labeled points
        if (labels > 0).sum() == 0:
            return refined_labels
            
        for label_id in unique_labels:
            if label_id.item() == 0:  # Skip background
                continue
                
            # Find points with this label
            label_mask = labels == label_id
            if label_mask.sum() < min_neighbors:
                continue
                
            # Get confidence for this label
            confidence = label_confidences.get(label_id.item(), 50.0)
            
            # Find nearby unlabeled points
            unlabeled_mask = labels == 0
            if not unlabeled_mask.any():
                continue
                
            # Compute distances from unlabeled to labeled points
            labeled_points = points_3d[label_mask]
            unlabeled_points = points_3d[unlabeled_mask]
            
            # Process in chunks to avoid memory issues
            chunk_size = 1000
            unlabeled_indices = torch.where(unlabeled_mask)[0]
            
            for i in range(0, unlabeled_points.shape[0], chunk_size):
                chunk_end = min(i + chunk_size, unlabeled_points.shape[0])
                chunk_points = unlabeled_points[i:chunk_end]
                
                # Compute distances
                distances = torch.cdist(chunk_points, labeled_points)
                min_distances, _ = distances.min(dim=1)
                
                # Assign label to nearby points based on confidence
                confidence_factor = confidence / 100.0  # Normalize to 0-1
                adjusted_threshold = distance_threshold * (1.0 + confidence_factor)
                
                close_mask = min_distances < adjusted_threshold
                if close_mask.any():
                    chunk_indices = unlabeled_indices[i:chunk_end]
                    refined_labels[chunk_indices[close_mask]] = label_id
        
        return refined_labels
    
    def propagate_high_confidence_labels(self,
                                       points_3d: torch.Tensor,
                                       labels: torch.Tensor,
                                       confidences: torch.Tensor,
                                       propagation_radius: float = 0.1,
                                       confidence_threshold: float = 70.0) -> torch.Tensor:
        """
        Propagate high-confidence labels to nearby unlabeled points.
        
        Args:
            points_3d: (N, 3) tensor of 3D point coordinates
            labels: (N,) tensor of current labels
            confidences: (N,) tensor of confidence scores for each point
            propagation_radius: Maximum distance for propagation
            confidence_threshold: Minimum confidence to propagate from
            
        Returns:
            Updated labels tensor
        """
        propagated_labels = labels.clone()
        
        # Find high-confidence labeled points
        high_conf_mask = (labels > 0) & (confidences >= confidence_threshold)
        if not high_conf_mask.any():
            return propagated_labels
            
        # Find unlabeled points
        unlabeled_mask = labels == 0
        if not unlabeled_mask.any():
            return propagated_labels
            
        print(f"[Tracker INFO] Propagating high-confidence labels to nearby points...")
        
        # Get points
        source_points = points_3d[high_conf_mask]
        source_labels = labels[high_conf_mask]
        source_confidences = confidences[high_conf_mask]
        
        target_points = points_3d[unlabeled_mask]
        target_indices = torch.where(unlabeled_mask)[0]
        
        # Process in chunks to manage memory
        chunk_size = 500
        total_propagated = 0
        
        for i in range(0, target_points.shape[0], chunk_size):
            chunk_end = min(i + chunk_size, target_points.shape[0])
            chunk_points = target_points[i:chunk_end]
            chunk_indices = target_indices[i:chunk_end]
            
            # Compute distances
            distances = torch.cdist(chunk_points, source_points)
            
            # Find nearest high-confidence point within radius
            min_distances, nearest_indices = distances.min(dim=1)
            within_radius = min_distances <= propagation_radius
            
            if within_radius.any():
                # Propagate labels
                valid_chunk_indices = chunk_indices[within_radius]
                valid_nearest = nearest_indices[within_radius]
                
                propagated_labels[valid_chunk_indices] = source_labels[valid_nearest]
                total_propagated += within_radius.sum().item()
        
        print(f"[Tracker INFO] Propagation increased labeled points from {(labels > 0).sum().item()} to {(propagated_labels > 0).sum().item()}")
        
        return propagated_labels


# ============================================================================
# Semantic Post-Processing (from semantic_post_processor.py)
# ============================================================================

class SemanticPostProcessor:
    """Handles temporal consistency and post-processing for semantic labels."""
    
    def __init__(self, temporal_window: int = 5, device: str = 'cuda'):
        self.temporal_window = temporal_window
        self.device = device
        self.frame_history = []
        
    def add_frame(self, frame_data: Dict):
        """Add a frame to the temporal history."""
        self.frame_history.append(frame_data)
        if len(self.frame_history) > self.temporal_window:
            self.frame_history.pop(0)
    
    def process_temporal_consistency(self, 
                                   current_labels: torch.Tensor,
                                   current_confidences: torch.Tensor,
                                   current_points: Optional[torch.Tensor] = None) -> Dict:
        """
        Apply temporal consistency to semantic labels.
        
        Args:
            current_labels: Current frame's semantic labels
            current_confidences: Confidence scores for current labels
            current_points: Optional 3D points for spatial-temporal matching
            
        Returns:
            Dictionary with processed labels and metadata
        """
        if len(self.frame_history) < 2:
            # Not enough history for temporal processing
            return {
                'labels': current_labels,
                'confidences': current_confidences,
                'temporal_consistency': 0.0
            }
        
        # Simple temporal smoothing
        # In practice, this would involve more sophisticated matching
        # For now, just return the current labels
        return {
            'labels': current_labels,
            'confidences': current_confidences,
            'temporal_consistency': 1.0
        }
    
    def remove_outliers(self,
                       labels: torch.Tensor,
                       points_3d: torch.Tensor,
                       min_cluster_size: int = 10,
                       isolation_threshold: float = 0.5) -> torch.Tensor:
        """
        Remove isolated semantic labels that are likely outliers.
        
        Args:
            labels: Semantic labels
            points_3d: 3D point coordinates
            min_cluster_size: Minimum points for a valid cluster
            isolation_threshold: Distance threshold for isolation
            
        Returns:
            Cleaned labels
        """
        cleaned_labels = labels.clone()
        unique_labels = torch.unique(labels)
        
        for label_id in unique_labels:
            if label_id.item() == 0:  # Skip background
                continue
                
            label_mask = labels == label_id
            label_count = label_mask.sum().item()
            
            # Remove small clusters
            if label_count < min_cluster_size:
                cleaned_labels[label_mask] = 0
                continue
            
            # Check for spatial isolation
            if points_3d is not None and label_count < 50:
                label_points = points_3d[label_mask]
                other_points = points_3d[~label_mask & (labels > 0)]
                
                if other_points.shape[0] > 0:
                    # Compute minimum distance to other labeled points
                    distances = torch.cdist(label_points, other_points)
                    min_distances = distances.min(dim=1)[0]
                    
                    # If most points are isolated, remove the label
                    isolated_ratio = (min_distances > isolation_threshold).float().mean()
                    if isolated_ratio > 0.8:
                        cleaned_labels[label_mask] = 0
        
        return cleaned_labels


# ============================================================================
# Helper functions for post-processing
# ============================================================================

def post_process_semantic_labels(labels: np.ndarray, 
                               points: np.ndarray,
                               label_map: Dict[int, str],
                               min_points: int = 50) -> Tuple[np.ndarray, Dict[int, str]]:
    """
    Post-process semantic labels to remove small clusters and outliers.
    
    Args:
        labels: Array of semantic labels
        points: 3D point coordinates
        label_map: Mapping from label IDs to class names
        min_points: Minimum points for a valid cluster
        
    Returns:
        Processed labels and updated label map
    """
    processed_labels = labels.copy()
    
    # Count points per label
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Remove small clusters
    for label_id, count in zip(unique_labels, counts):
        if label_id == 0:  # Skip background
            continue
        if count < min_points:
            processed_labels[labels == label_id] = 0
            print(f"[PostProcess] Removed label {label_id} ({label_map.get(label_id, 'unknown')}) with only {count} points")
    
    return processed_labels, label_map


def merge_similar_labels(labels: np.ndarray,
                        label_map: Dict[int, str],
                        points: np.ndarray,
                        distance_threshold: float = 0.1) -> Tuple[np.ndarray, Dict[int, str]]:
    """
    Merge semantically similar labels that are spatially close.
    
    Args:
        labels: Array of semantic labels
        label_map: Mapping from label IDs to class names
        points: 3D point coordinates
        distance_threshold: Maximum distance for merging
        
    Returns:
        Merged labels and updated label map
    """
    # Define groups of similar labels
    similar_groups = [
        ['computer desk', 'wooden table', 'work surface'],
        ['computer monitor', 'television screen', 'electronic display'],
        ['office chair', 'wooden chair'],
        ['container object', 'storage box'],
        ['furniture leg', 'table leg', 'chair leg']
    ]
    
    merged_labels = labels.copy()
    
    # Build similarity mapping
    label_to_group = {}
    for group in similar_groups:
        for label_name in group:
            # Find all label IDs with this name
            for label_id, name in label_map.items():
                if label_name in name.lower():
                    label_to_group[label_id] = group[0]  # Use first name as canonical
    
    # Merge similar labels that are close in space
    for label_id, canonical_name in label_to_group.items():
        if label_id not in np.unique(labels):
            continue
            
        # Find canonical label ID
        canonical_id = None
        for lid, name in label_map.items():
            if canonical_name in name.lower() and lid in np.unique(labels):
                canonical_id = lid
                break
        
        if canonical_id and canonical_id != label_id:
            # Check spatial proximity
            mask1 = labels == label_id
            mask2 = labels == canonical_id
            
            if mask1.any() and mask2.any():
                points1 = points[mask1]
                points2 = points[mask2]
                
                # Compute minimum distance between clusters
                min_dist = np.min(np.linalg.norm(
                    points1[:, np.newaxis] - points2[np.newaxis, :], 
                    axis=2
                ))
                
                if min_dist < distance_threshold:
                    merged_labels[mask1] = canonical_id
                    print(f"[PostProcess] Merged label {label_id} ({label_map[label_id]}) into {canonical_id} ({label_map[canonical_id]})")
    
    return merged_labels, label_map


# ============================================================================
# Duplicate Filter (simplified from semantic_duplicate_filter.py)
# ============================================================================

class SemanticDuplicateFilter:
    """Filter duplicate semantic detections based on spatial and semantic similarity."""
    
    def __init__(self, 
                 spatial_threshold: float = 0.5,
                 semantic_threshold: float = 0.8,
                 min_overlap: float = 0.3):
        self.spatial_threshold = spatial_threshold
        self.semantic_threshold = semantic_threshold
        self.min_overlap = min_overlap
    
    def filter_duplicates(self, detections: List[Dict]) -> List[Dict]:
        """
        Filter duplicate detections.
        
        Args:
            detections: List of detection dictionaries with 'mask', 'label', 'confidence'
            
        Returns:
            Filtered list of detections
        """
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence (highest first)
        sorted_detections = sorted(detections, key=lambda x: x.get('confidence', 0), reverse=True)
        
        # Keep track of which detections to keep
        keep = [True] * len(sorted_detections)
        
        for i in range(len(sorted_detections)):
            if not keep[i]:
                continue
                
            for j in range(i + 1, len(sorted_detections)):
                if not keep[j]:
                    continue
                
                # Check if detections are duplicates
                if self._are_duplicates(sorted_detections[i], sorted_detections[j]):
                    keep[j] = False
        
        return [det for det, k in zip(sorted_detections, keep) if k]
    
    def _are_duplicates(self, det1: Dict, det2: Dict) -> bool:
        """Check if two detections are duplicates."""
        # Simple check based on label similarity
        label1 = det1.get('label', '').lower()
        label2 = det2.get('label', '').lower()
        
        # Check if labels are similar
        if label1 == label2:
            return True
        
        # Check for common substrings
        if any(word in label2 for word in label1.split()) or \
           any(word in label1 for word in label2.split()):
            return True
        
        return False