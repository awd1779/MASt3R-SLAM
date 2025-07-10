"""Semantic Fusion Module for projecting 2D masks to 3D pointmaps."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
import lietorch
from mast3r_slam.semantic_frame import decode_rle


class SemanticPointmapFusion:
    """Handles projection of 2D semantic masks to 3D pointmaps with high performance."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        
    def fuse_semantics_vectorized(self,
                                 X_world: torch.Tensor,  # [N, 3] 3D points in world
                                 semantic_mask: torch.Tensor,  # [H, W] instance IDs
                                 instance_confidences: Dict[int, float],  # instance confidence scores
                                 K: torch.Tensor,  # [3, 3] intrinsics
                                 T_WC: lietorch.Sim3,  # world to camera transform
                                 img_shape: Tuple[int, int],
                                 existing_labels: Optional[torch.Tensor] = None,  # [N] existing labels
                                 existing_conf: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized semantic fusion from 2D mask to 3D pointmap.
        
        Returns:
            semantic_labels: [N] tensor of instance IDs for each 3D point
            semantic_conf: [N] tensor of confidence scores
        """
        N = X_world.shape[0]
        H, W = img_shape
        
        # Initialize outputs
        if existing_labels is None:
            semantic_labels = torch.zeros(N, dtype=torch.int32, device=self.device)
            semantic_conf = torch.zeros(N, dtype=torch.float32, device=self.device)
        else:
            semantic_labels = existing_labels.clone()
            semantic_conf = existing_conf.clone()
        
        # Convert world points to homogeneous coordinates
        X_world_homo = torch.cat([X_world, torch.ones(N, 1, device=self.device)], dim=1)  # [N, 4]
        
        # Transform to camera coordinates
        T_CW = T_WC.inv()
        T_CW_matrix = T_CW.matrix()[0] if T_CW.matrix().dim() == 3 else T_CW.matrix()  # Handle batch dimension
        X_cam_homo = T_CW_matrix @ X_world_homo.T  # [4, N]
        X_cam = X_cam_homo[:3, :].transpose(0, 1)  # [N, 3]
        
        # Project to image plane
        X_img = K @ X_cam.T  # [3, N]
        
        # Normalize by depth
        depth = X_img[2]  # [N]
        valid_depth = depth > 0.1  # Avoid division by zero and points behind camera
        
        # Compute pixel coordinates
        pixels = torch.zeros(2, N, device=self.device)
        pixels[:, valid_depth] = X_img[:2, valid_depth] / depth[valid_depth]
        
        # Round to integer pixel coordinates
        px = torch.round(pixels[0]).long()
        py = torch.round(pixels[1]).long()
        
        # Check bounds
        valid_x = (px >= 0) & (px < W)
        valid_y = (py >= 0) & (py < H)
        valid_mask = valid_depth & valid_x & valid_y
        
        # Sample semantic mask at valid projections
        valid_indices = torch.where(valid_mask)[0]
        if len(valid_indices) > 0:
            # Vectorized sampling
            sampled_labels = semantic_mask[py[valid_indices], px[valid_indices]]
            
            # Get confidence scores for sampled labels
            sampled_conf = torch.zeros_like(sampled_labels, dtype=torch.float32)
            for i, label in enumerate(sampled_labels):
                label_int = label.item()
                if label_int > 0 and label_int in instance_confidences:
                    sampled_conf[i] = instance_confidences[label_int]
            
            # Update based on confidence (only update if new confidence is higher)
            for idx, (point_idx, new_label, new_conf) in enumerate(zip(valid_indices, sampled_labels, sampled_conf)):
                if new_label > 0:  # Skip background (0)
                    if new_conf > semantic_conf[point_idx]:
                        semantic_labels[point_idx] = new_label
                        semantic_conf[point_idx] = new_conf
        
        return semantic_labels, semantic_conf
    
    def fuse_keyframe_semantics(self,
                               keyframe,
                               semantic_data: Dict,
                               K: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse semantic data into a keyframe's 3D pointmap.
        
        Args:
            keyframe: Frame object with X_canon (3D points)
            semantic_data: Dict with masks_rle, instance_ids, confidences
            K: Camera intrinsics
            
        Returns:
            semantic_labels: [N] tensor of instance IDs
            semantic_conf: [N] tensor of confidence scores
        """
        if keyframe.X_canon is None:
            return None, None
            
        # Decode all masks and create a single semantic mask
        # img_shape has shape [1, 2], so extract H, W from the first element
        H, W = int(keyframe.img_shape[0, 0]), int(keyframe.img_shape[0, 1])
        semantic_mask = torch.zeros((H, W), dtype=torch.int32, device=self.device)
        
        # Decode each instance mask
        masks_rle = semantic_data.get('masks_rle', {})
        n_masks = len(masks_rle)
        expected_pixels = H * W
        
        if n_masks == 0:
            return None, None
            
        for instance_id in semantic_data['instance_ids']:
            if instance_id in masks_rle:
                try:
                    rle = masks_rle[instance_id]
                    
                    if 'size' in rle:
                        size = rle['size']
                        if len(size) == 3:
                            # SAM2 includes batch dimension
                            _, orig_h, orig_w = size
                        else:
                            orig_h, orig_w = size
                        # Just decode at the original size 
                        mask = decode_rle(rle, orig_h, orig_w)
                    else:
                        mask = decode_rle(rle, H, W)
                    
                    n_pixels = mask.sum().item()
                    if n_pixels > 0:
                        semantic_mask[mask] = instance_id
                except Exception as e:
                    print(f"    Error decoding mask for instance {instance_id}: {e}")
        
        # Get confidence scores
        confidences = semantic_data.get('confidences', {})
        
        
        # Fuse semantics
        return self.fuse_semantics_vectorized(
            X_world=keyframe.X_canon,
            semantic_mask=semantic_mask,
            instance_confidences=confidences,
            K=K,
            T_WC=keyframe.T_WC,
            img_shape=(H, W)
        )
    
    def batch_fuse_semantics(self,
                            keyframes: List,
                            semantic_keyframes,
                            K: torch.Tensor) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Batch process multiple keyframes for efficiency.
        
        Returns:
            Dict mapping keyframe index to (semantic_labels, semantic_conf)
        """
        results = {}
        
        for idx, kf in enumerate(keyframes):
            if kf is None or kf.X_canon is None:
                continue
                
            # Get semantic data for this keyframe
            semantic_data = semantic_keyframes.get_semantics(idx)
            if semantic_data is None:
                continue
                
            # Fuse semantics
            labels, conf = self.fuse_keyframe_semantics(kf, semantic_data, K)
            if labels is not None:
                results[idx] = (labels, conf)
                
        return results


class SemanticPointmapFusionCUDA:
    """CUDA-accelerated version of semantic fusion (optional implementation)."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._load_cuda_kernel()
        
    def _load_cuda_kernel(self):
        """Load custom CUDA kernel for semantic fusion."""
        try:
            from mast3r_slam.backend import semantic_fusion_cuda
            self.cuda_kernel = semantic_fusion_cuda.semantic_fusion_kernel
            self.use_cuda = True
        except ImportError:
            print("CUDA kernel not available, falling back to PyTorch")
            self.use_cuda = False
            
    def fuse_semantics_cuda(self,
                           X_world: torch.Tensor,
                           semantic_mask: torch.Tensor,
                           instance_confidences: torch.Tensor,
                           K: torch.Tensor,
                           T_CW: torch.Tensor,
                           img_shape: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        CUDA kernel implementation for maximum performance.
        """
        if not self.use_cuda:
            # Fall back to vectorized implementation
            fusion = SemanticPointmapFusion(self.device)
            # Convert T_CW matrix to Sim3 for compatibility
            T_WC = lietorch.Sim3.from_matrix(torch.inverse(T_CW))
            return fusion.fuse_semantics_vectorized(
                X_world, semantic_mask, instance_confidences, K, T_WC, img_shape
            )
        
        # Use CUDA kernel
        N = X_world.shape[0]
        H, W = img_shape
        
        # Prepare outputs
        semantic_labels = torch.zeros(N, dtype=torch.int32, device=self.device)
        semantic_conf = torch.zeros(N, dtype=torch.float32, device=self.device)
        
        # Launch kernel
        threads_per_block = 256
        blocks = (N + threads_per_block - 1) // threads_per_block
        
        self.cuda_kernel(
            blocks, threads_per_block,
            X_world, semantic_mask, instance_confidences,
            K, T_CW, semantic_labels, semantic_conf,
            N, H, W
        )
        
        return semantic_labels, semantic_conf


def project_points_to_image(X_world: torch.Tensor,
                           K: torch.Tensor,
                           T_CW: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Helper function to project 3D points to 2D image coordinates.
    
    Args:
        X_world: [N, 3] world coordinates
        K: [3, 3] intrinsics
        T_CW: [4, 4] camera-from-world transform
        
    Returns:
        pixels: [2, N] pixel coordinates
        valid: [N] boolean mask of valid projections
    """
    N = X_world.shape[0]
    
    # To homogeneous
    X_homo = torch.cat([X_world, torch.ones(N, 1, device=X_world.device)], dim=1)
    
    # Transform to camera
    X_cam = (T_CW @ X_homo.T)[:3]  # [3, N]
    
    # Project
    x_img = K @ X_cam  # [3, N]
    
    # Normalize
    depth = x_img[2]
    valid = depth > 0.1
    
    pixels = x_img[:2] / depth.unsqueeze(0)
    
    return pixels, valid


def compute_semantic_overlap(labels1: torch.Tensor,
                           labels2: torch.Tensor) -> float:
    """
    Compute semantic overlap between two sets of labels.
    Used for loop closure verification.
    """
    # Count label occurrences
    unique1, counts1 = torch.unique(labels1[labels1 > 0], return_counts=True)
    unique2, counts2 = torch.unique(labels2[labels2 > 0], return_counts=True)
    
    if len(unique1) == 0 or len(unique2) == 0:
        return 0.0
        
    # Compute intersection
    intersection = 0
    for label in unique1:
        if label in unique2:
            idx1 = (unique1 == label).nonzero()[0]
            idx2 = (unique2 == label).nonzero()[0]
            intersection += min(counts1[idx1], counts2[idx2])
            
    # Compute union
    union = counts1.sum() + counts2.sum() - intersection
    
    return float(intersection) / float(union) if union > 0 else 0.0