"""
Utilities for object tracking - feature extraction, similarity computation, etc.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import cv2
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