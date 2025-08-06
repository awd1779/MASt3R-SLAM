"""
Utilities for object tracking - feature extraction, similarity computation, etc.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import cv2
import logging
from mast3r_slam.semantic_frame import decode_rle


def create_appearance_signature(mask_rle: Dict,
                              mast3r_features: torch.Tensor,
                              shape: Tuple[int, int],
                              feature_conf: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Extract appearance signature from masked region using MASt3R features.
    
    Args:
        mask_rle: RLE-encoded mask dictionary
        mast3r_features: Dense feature map from MASt3R (H, W, D)
        shape: (H, W) shape of the image
        feature_conf: Optional confidence map for features
        
    Returns:
        Normalized feature vector representing object appearance
    """
    # Decode RLE mask
    h, w = shape
    mask = decode_rle(mask_rle, h, w)
    
    # Ensure mask is boolean tensor on same device as features
    mask_tensor = torch.from_numpy(mask).to(
        device=mast3r_features.device, 
        dtype=torch.bool
    )
    
    # Extract features from masked region
    masked_features = mast3r_features[mask_tensor]
    
    if masked_features.shape[0] == 0:
        # No valid features in mask
        return None
        
    # Compute weighted mean if confidence provided
    if feature_conf is not None:
        masked_conf = feature_conf[mask_tensor]
        # Weighted average
        weighted_features = masked_features * masked_conf.unsqueeze(-1)
        signature = weighted_features.sum(dim=0) / (masked_conf.sum() + 1e-8)
    else:
        # Simple mean
        signature = masked_features.mean(dim=0)
    
    # L2 normalize for cosine similarity
    signature = F.normalize(signature, p=2, dim=0)
    
    return signature


def compute_3d_iou(bbox1: Tuple[torch.Tensor, torch.Tensor],
                   bbox2: Tuple[torch.Tensor, torch.Tensor]) -> float:
    """
    Compute 3D bounding box IoU.
    
    Args:
        bbox1: (min_coords, max_coords) for first box
        bbox2: (min_coords, max_coords) for second box
        
    Returns:
        IoU value between 0 and 1
    """
    min1, max1 = bbox1
    min2, max2 = bbox2
    
    # Compute intersection
    inter_min = torch.maximum(min1, min2)
    inter_max = torch.minimum(max1, max2)
    
    # Check if boxes intersect
    inter_dims = inter_max - inter_min
    if (inter_dims < 0).any():
        return 0.0
        
    # Compute volumes
    inter_volume = inter_dims.prod().item()
    vol1 = (max1 - min1).prod().item()
    vol2 = (max2 - min2).prod().item()
    
    # Compute IoU
    union_volume = vol1 + vol2 - inter_volume
    iou = inter_volume / (union_volume + 1e-8)
    
    return iou


def compute_cosine_similarity(sig1: torch.Tensor, sig2: torch.Tensor) -> float:
    """
    Compute cosine similarity between two normalized signatures.
    
    Args:
        sig1: First normalized signature
        sig2: Second normalized signature
        
    Returns:
        Cosine similarity between -1 and 1
    """
    # Assuming signatures are already normalized
    similarity = torch.dot(sig1.flatten(), sig2.flatten()).item()
    return similarity


def compute_centroid_distance(points1: torch.Tensor, points2: torch.Tensor) -> float:
    """
    Compute distance between centroids of two point clouds.
    
    Args:
        points1: Nx3 tensor of 3D points
        points2: Mx3 tensor of 3D points
        
    Returns:
        Euclidean distance between centroids
    """
    if points1.shape[0] == 0 or points2.shape[0] == 0:
        return float('inf')
        
    centroid1 = points1.mean(dim=0)
    centroid2 = points2.mean(dim=0)
    
    distance = torch.norm(centroid2 - centroid1).item()
    return distance


def extract_3d_points_for_mask(mask_rle: Dict,
                              pointmap: torch.Tensor,
                              confidence: torch.Tensor,
                              shape: Tuple[int, int],
                              conf_threshold: float = 0.1) -> torch.Tensor:
    """
    Extract 3D points corresponding to a segmentation mask.
    
    Args:
        mask_rle: RLE-encoded mask
        pointmap: HxWx3 tensor of 3D points
        confidence: HxW tensor of point confidences
        shape: (H, W) image shape
        conf_threshold: Minimum confidence threshold
        
    Returns:
        Nx3 tensor of 3D points
    """
    # Decode mask
    h, w = shape
    mask = decode_rle(mask_rle, h, w)
    mask_tensor = torch.from_numpy(mask).to(device=pointmap.device, dtype=torch.bool)
    
    # Apply confidence threshold
    valid_mask = mask_tensor & (confidence > conf_threshold)
    
    # Extract points
    points = pointmap[valid_mask]
    
    return points


def project_3d_to_2d(points_3d: torch.Tensor,
                    T_CW: torch.Tensor,
                    K: torch.Tensor) -> torch.Tensor:
    """
    Project 3D points to 2D image coordinates.
    
    Args:
        points_3d: Nx3 tensor of 3D points in world coordinates
        T_CW: 4x4 camera-to-world transformation matrix
        K: 3x3 camera intrinsic matrix
        
    Returns:
        Nx2 tensor of 2D pixel coordinates
    """
    # Convert to homogeneous coordinates
    points_homo = torch.cat([points_3d, torch.ones_like(points_3d[:, :1])], dim=1)
    
    # Transform to camera coordinates
    T_WC = torch.inverse(T_CW)
    points_cam = (T_WC @ points_homo.T).T[:, :3]
    
    # Project to image
    points_img_homo = (K @ points_cam.T).T
    points_2d = points_img_homo[:, :2] / points_img_homo[:, 2:3]
    
    return points_2d


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute IoU between two binary masks.
    
    Args:
        mask1: First binary mask
        mask2: Second binary mask
        
    Returns:
        IoU value between 0 and 1
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
        
    iou = intersection / union
    return iou


def visualize_track_associations(image: np.ndarray,
                               current_masks: Dict[int, np.ndarray],
                               track_ids: Dict[int, int],
                               labels: Dict[int, str]) -> np.ndarray:
    """
    Create visualization of tracking results.
    
    Args:
        image: RGB image
        current_masks: Dictionary of instance_id -> mask
        track_ids: Dictionary of instance_id -> track_id  
        labels: Dictionary of instance_id -> label
        
    Returns:
        Visualization image with colored masks and track IDs
    """
    vis_image = image.copy()
    h, w = image.shape[:2]
    
    # Create color map for tracks
    np.random.seed(42)  # Fixed seed for consistent colors
    max_tracks = max(track_ids.values()) if track_ids else 1
    colors = np.random.randint(0, 255, size=(max_tracks + 1, 3))
    
    # Apply colored overlays
    for instance_id, mask in current_masks.items():
        if instance_id not in track_ids:
            continue
            
        track_id = track_ids[instance_id]
        color = colors[track_id]
        
        # Create colored overlay
        overlay = vis_image.copy()
        overlay[mask] = color
        
        # Blend with original
        alpha = 0.5
        vis_image[mask] = cv2.addWeighted(
            vis_image[mask], 1 - alpha,
            overlay[mask], alpha, 0
        )
        
        # Add text label
        y_coords, x_coords = np.where(mask)
        if len(y_coords) > 0:
            cy, cx = int(np.mean(y_coords)), int(np.mean(x_coords))
            label = labels.get(instance_id, 'unknown')
            text = f"{label} #{track_id}"
            
            # Add background for text
            (text_w, text_h), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(vis_image, 
                         (cx - 5, cy - text_h - 5),
                         (cx + text_w + 5, cy + 5),
                         (0, 0, 0), -1)
            
            cv2.putText(vis_image, text, (cx, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (255, 255, 255), 2)
    
    return vis_image


# === New utilities for geometric tracker optimization ===

class TrackingLogger:
    """Centralized logging utilities for object tracking."""
    
    def __init__(self, debug_labels: list = None, debug_all: bool = False):
        self.debug_labels = set(debug_labels) if debug_labels else set()
        self.debug_all = debug_all
        self.logger = logging.getLogger('mast3r_slam.tracking_utils')
    
    def format_position(self, tensor, label: str = "pos") -> str:
        """Safe position formatting for logging."""
        if tensor is None:
            return f"{label}=None"
        
        pos = tensor.cpu().numpy()
        if pos.ndim == 0:
            pos = pos.reshape(1)
        
        vals = [pos.flat[i] if i < pos.size else 0.0 for i in range(3)]
        return f"{label}=[{vals[0]:.2f}, {vals[1]:.2f}, {vals[2]:.2f}]"
    
    def log_detection_extraction(self, detection: dict, instance_id: int):
        """Centralized detection logging."""
        if not self.logger.isEnabledFor(logging.DEBUG):
            return
            
        cam_pos = self.format_position(detection['bbox_cam']['center'], "cam")
        world_pos = self.format_position(detection['bbox_world']['center'], "world")
        volume = detection['bbox_cam']['volume']
        points = detection['bbox_cam']['extraction_stats']['valid_3d_points']
        
        self.logger.debug(f"Extracted {detection['label']} (instance {instance_id}): "
                         f"{cam_pos}, {world_pos}, vol={volume:.3f}m³, pts={points}")
    
    def log_matching_details(self, detection: dict, track_id: int, scores: dict, 
                           frame_id: int, decision: str = "matched"):
        """Centralized matching decision logging."""
        label = detection['label']
        
        # Only log detailed info for debug labels or if debug_all is enabled
        if not (self.debug_all or label in self.debug_labels):
            return
            
        det_pos = self.format_position(detection['bbox_world']['center'], "det")
        
        self.logger.info(f"{label.upper()} {decision.upper()} - Frame {frame_id}, Track {track_id}:")
        self.logger.info(f"  {det_pos}")
        
        if 'track_pos' in scores:
            track_pos = scores['track_pos']
            self.logger.info(f"  track=[{track_pos[0]:.2f}, {track_pos[1]:.2f}, {track_pos[2]:.2f}]")
        
        if 'distance' in scores:
            self.logger.info(f"  Distance: {scores['distance']:.3f}m")
        if 'matching_score' in scores:
            self.logger.info(f"  Score: {scores['matching_score']:.3f}")
        if 'iou_3d' in scores:
            self.logger.info(f"  IoU: {scores['iou_3d']:.3f}")


class ConfigHelper:
    """Helper for configuration access patterns."""
    
    # Static mapping for object categories
    OBJECT_CATEGORIES = {
        frozenset(['wall', 'floor', 'ceiling']): ('large', 1.0),
        frozenset(['chair', 'table', 'sofa', 'bed', 'desk']): ('furniture', 0.5),
        frozenset(['bottle', 'cup', 'mouse', 'keyboard']): ('small', 0.2)
    }
    
    @classmethod
    def get_distance_threshold(cls, label: str, world_thresholds: dict = None) -> float:
        """Get world distance threshold based on object type."""
        for label_set, (category, default_val) in cls.OBJECT_CATEGORIES.items():
            if label in label_set:
                if world_thresholds:
                    return world_thresholds.get(category, default_val)
                return default_val
        
        # Default category
        if world_thresholds:
            return world_thresholds.get('default', 0.3)
        return 0.3


class TrackProcessor:
    """Generic track processing utilities."""
    
    @staticmethod
    def process_tracks(tracks: dict, condition_fn, action_fn, collect_results: bool = False):
        """Generic track processing with condition and action functions."""
        results = [] if collect_results else None
        
        for track_id, track in tracks.items():
            if condition_fn(track_id, track):
                result = action_fn(track_id, track)
                if collect_results and result is not None:
                    results.append(result)
        
        return results
    
    @staticmethod
    def filter_tracks(tracks: dict, condition_fn) -> dict:
        """Filter tracks based on condition function."""
        return {tid: track for tid, track in tracks.items() if condition_fn(tid, track)}


class StatisticsCollector:
    """Unified statistics collection and reporting."""
    
    @staticmethod
    def generate_track_summary(tracks: dict, timing_stats: dict = None) -> dict:
        """Generate comprehensive track summary data."""
        summary = {
            'total_tracks': len(tracks),
            'active_tracks': 0,
            'lost_tracks': 0,
            'label_counts': {},
            'active_labels': {},
            'positions': {},
            'avg_observations': 0,
            'avg_confidence': 0,
            'timing': timing_stats or {}
        }
        
        total_obs = []
        total_conf = []
        
        for track_id, track in tracks.items():
            # Count by status
            if track.lost_frames == 0:
                summary['active_tracks'] += 1
                # Active track positions
                if track.centroid_world is not None:
                    pos = track.centroid_world.cpu().numpy()
                    summary['positions'][track_id] = {
                        'label': track.label,
                        'position': pos.tolist(),
                        'variance': track.get_world_position_variance(),
                        'observations': track.total_observations
                    }
                # Active label counts
                summary['active_labels'][track.label] = summary['active_labels'].get(track.label, 0) + 1
            else:
                summary['lost_tracks'] += 1
            
            # Total label counts
            summary['label_counts'][track.label] = summary['label_counts'].get(track.label, 0) + 1
            
            # Accumulate stats
            total_obs.append(track.total_observations)
            total_conf.append(track.confidence)
        
        summary['avg_observations'] = np.mean(total_obs) if total_obs else 0
        summary['avg_confidence'] = np.mean(total_conf) if total_conf else 0
        
        return summary
    
    @staticmethod
    def log_tracking_stats(summary: dict, frame_id: int):
        """Log tracking statistics from summary."""
        logger = logging.getLogger('mast3r_slam.tracking_utils')
        
        active_labels = summary['active_labels']
        timing = summary['timing']
        
        logger.info(f"Tracking Stats - Frame {frame_id}:")
        logger.info(f"  Active tracks: {summary['active_tracks']} "
                   f"({', '.join(f'{k}:{v}' for k, v in active_labels.items())})")
        logger.info(f"  Total tracks: {summary['total_tracks']}")
        
        if timing:
            avg_times = {k: np.mean(v) for k, v in timing.items() if v}
            logger.info(f"  Timing (ms): " + 
                       ", ".join(f"{k}={v:.1f}" for k, v in avg_times.items()))
    
    @staticmethod  
    def log_world_positions(summary: dict):
        """Log world positions from summary."""
        logger = logging.getLogger('mast3r_slam.tracking_utils')
        
        logger.info("=== World Position Report ===")
        
        positions = summary['positions']
        for track_id in sorted(positions.keys()):
            data = positions[track_id]
            pos = data['position']
            logger.info(f"Track {track_id:3d} ({data['label']:12s}): "
                       f"pos=[{pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:6.2f}], "
                       f"var={data['variance']:.3f}m, obs={data['observations']:3d}")
        
        logger.info("=" * 50)