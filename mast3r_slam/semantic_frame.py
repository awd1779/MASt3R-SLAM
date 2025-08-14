import dataclasses
from typing import Optional, Dict, List
import torch
import numpy as np
from mast3r_slam.frame import Frame
import pycocotools.mask as mask_utils


@dataclasses.dataclass
class SemanticFrame(Frame):
    """Extended Frame class with semantic segmentation data."""
    
    # Semantic masks in RLE format for efficiency
    semantic_masks_rle: Optional[Dict[int, Dict]] = None  # {instance_id: rle_mask}
    
    # Instance tracking information
    instance_ids: Optional[List[int]] = None  # List of instance IDs in this frame
    track_ids: Optional[Dict[int, int]] = None  # {instance_id: track_id} mapping
    
    # Semantic labels and confidence
    semantic_labels: Optional[Dict[int, str]] = None  # {instance_id: label}
    semantic_confidences: Optional[Dict[int, float]] = None  # {instance_id: confidence}
    
    # Timing information
    semantic_timestamp: Optional[float] = None  # When semantic data was computed
    
    def has_semantics(self) -> bool:
        """Check if semantic data is available."""
        return self.semantic_masks_rle is not None
    
    def get_instance_mask(self, instance_id: int) -> Optional[torch.Tensor]:
        """Decode RLE mask for a specific instance."""
        if self.semantic_masks_rle is None or instance_id not in self.semantic_masks_rle:
            return None
        
        rle = self.semantic_masks_rle[instance_id]
        return decode_rle(rle, self.img_shape[0].item(), self.img_shape[1].item())
    
    def get_semantic_mask(self) -> Optional[torch.Tensor]:
        """Get full semantic segmentation mask with all instances."""
        if not self.has_semantics():
            return None
        
        h, w = self.img_shape[0].item(), self.img_shape[1].item()
        semantic_mask = torch.zeros((h, w), dtype=torch.int32, device=self.img.device)
        
        for instance_id in self.instance_ids:
            mask = self.get_instance_mask(instance_id)
            if mask is not None:
                semantic_mask[mask > 0] = instance_id
        
        return semantic_mask


class SharedSemanticKeyframes:
    """Shared memory structure for semantic keyframes."""
    
    def __init__(self, manager, max_keyframes=1000, h=480, w=640, device="cuda"):
        self.max_keyframes = max_keyframes
        self.h, self.w = h, w
        self.device = device
        self.lock = manager.RLock()
        
        # Shared lists for semantic data
        self.semantic_masks_rle = manager.list([None] * max_keyframes)
        self.instance_ids = manager.list([None] * max_keyframes)
        self.track_ids = manager.list([None] * max_keyframes)
        self.semantic_labels = manager.list([None] * max_keyframes)
        self.semantic_confidences = manager.list([None] * max_keyframes)
        self.semantic_timestamps = manager.list([None] * max_keyframes)
        
        # Track which keyframes have semantic data
        self.has_semantics = torch.zeros(max_keyframes, dtype=torch.bool, device="cpu").share_memory_()
        
    def update_semantics(self, kf_idx: int, semantic_data: dict):
        """Update semantic data for a keyframe."""
        with self.lock:
            if kf_idx >= self.max_keyframes:
                return
            
            self.semantic_masks_rle[kf_idx] = semantic_data.get('masks_rle')
            self.instance_ids[kf_idx] = semantic_data.get('instance_ids')
            self.track_ids[kf_idx] = semantic_data.get('track_ids')
            self.semantic_labels[kf_idx] = semantic_data.get('labels')
            self.semantic_confidences[kf_idx] = semantic_data.get('confidences')
            self.semantic_timestamps[kf_idx] = semantic_data.get('timestamp')
            self.has_semantics[kf_idx] = 1  # Use 1 instead of True for tensor assignment
    
    def get_semantics(self, kf_idx: int) -> Optional[dict]:
        """Retrieve semantic data for a keyframe."""
        with self.lock:
            if kf_idx >= self.max_keyframes or not self.has_semantics[kf_idx]:
                return None
            
            return {
                'masks_rle': self.semantic_masks_rle[kf_idx],
                'instance_ids': self.instance_ids[kf_idx],
                'track_ids': self.track_ids[kf_idx],
                'labels': self.semantic_labels[kf_idx],
                'confidences': self.semantic_confidences[kf_idx],
                'timestamp': self.semantic_timestamps[kf_idx]
            }
    
    def has_semantic_data(self, kf_idx: int) -> bool:
        """Check if a keyframe has semantic data."""
        with self.lock:
            if kf_idx >= self.max_keyframes:
                return False
            return bool(self.has_semantics[kf_idx].item())


# RLE encoding/decoding utilities using pycocotools
def encode_rle(mask: torch.Tensor) -> dict:
    """Encode binary mask to RLE format using pycocotools."""
    mask_np = mask.cpu().numpy().astype(np.uint8, order='F')  # Fortran order for pycocotools
    rle = mask_utils.encode(mask_np)
    # Convert bytes to list for JSON serialization
    if isinstance(rle['counts'], bytes):
        rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def decode_rle(rle: dict, h: int, w: int) -> torch.Tensor:
    """Decode RLE format to binary mask using pycocotools."""
    # Handle both string and list formats
    if isinstance(rle['counts'], str):
        rle_copy = {'size': rle['size'], 'counts': rle['counts'].encode('utf-8')}
    else:
        rle_copy = rle
    
    mask = mask_utils.decode(rle_copy)
    return torch.from_numpy(mask).to(torch.bool)


def create_semantic_frame(frame: Frame) -> SemanticFrame:
    """Convert a regular Frame to SemanticFrame."""
    return SemanticFrame(
        frame_id=frame.frame_id,
        img=frame.img,
        img_shape=frame.img_shape,
        img_true_shape=frame.img_true_shape,
        uimg=frame.uimg,
        T_WC=frame.T_WC,
        X_canon=frame.X_canon,
        C=frame.C,
        feat=frame.feat,
        pos=frame.pos,
        N=frame.N,
        N_updates=frame.N_updates,
        K=frame.K
    )