import dataclasses
from typing import Optional, Dict, List, Any
import torch
import numpy as np
from mast3r_slam.frame import Frame
import pycocotools.mask as mask_utils


@dataclasses.dataclass
class SemanticData:
    """Consolidated semantic data structure to eliminate duplication."""
    masks_rle: Optional[Dict] = None
    instance_ids: Optional[List[int]] = None
    track_ids: Optional[Dict[int, int]] = None
    labels: Optional[Dict[int, str]] = None
    confidences: Optional[Dict[int, float]] = None
    timestamp: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format for backward compatibility."""
        return {
            'masks_rle': self.masks_rle,
            'instance_ids': self.instance_ids,
            'track_ids': self.track_ids,
            'labels': self.labels,
            'confidences': self.confidences,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SemanticData':
        """Create from dictionary format."""
        return cls(
            masks_rle=data.get('masks_rle'),
            instance_ids=data.get('instance_ids'),
            track_ids=data.get('track_ids'),
            labels=data.get('labels'),
            confidences=data.get('confidences'),
            timestamp=data.get('timestamp')
        )


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
        h, w = self.img_shape[0].item(), self.img_shape[1].item()
        return decode_rle(rle, h, w)
    
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
        
        # Unified semantic data storage
        self.semantic_data = manager.list([None] * max_keyframes)
        
        # Track which keyframes have semantic data
        self.has_semantics = torch.zeros(max_keyframes, dtype=torch.bool, device="cpu").share_memory_()
        
    def update_semantics(self, kf_idx: int, semantic_data: dict):
        """Update semantic data for a keyframe using consolidated structure."""
        with self.lock:
            if kf_idx >= self.max_keyframes:
                return
            
            # Convert dict to SemanticData object for unified storage
            self.semantic_data[kf_idx] = SemanticData.from_dict(semantic_data)
            self.has_semantics[kf_idx] = True
    
    def get_semantics(self, kf_idx: int) -> Optional[dict]:
        """Retrieve semantic data for a keyframe."""
        with self.lock:
            if kf_idx >= self.max_keyframes or not self.has_semantics[kf_idx]:
                return None
            
            semantic_obj = self.semantic_data[kf_idx]
            return semantic_obj.to_dict() if semantic_obj else None
    
    def has_semantic_data(self, kf_idx: int) -> bool:
        """Check if a keyframe has semantic data."""
        if kf_idx >= self.max_keyframes:
            return False
        return bool(self.has_semantics[kf_idx].item())


# RLE encoding/decoding utilities using pycocotools
def encode_rle(mask: torch.Tensor) -> dict:
    """Encode binary mask to RLE format using shared utilities."""
    from .utils import convert_binary_mask_to_rle
    return convert_binary_mask_to_rle(mask)


def decode_rle(rle: dict, h: int, w: int) -> torch.Tensor:
    """Decode RLE format to binary mask using shared utilities."""
    from .utils import convert_rle_to_binary_mask
    return convert_rle_to_binary_mask(rle, h, w)


def create_semantic_frame(frame: Frame) -> SemanticFrame:
    """
    Convert a regular Frame to SemanticFrame using safe dataclass copying.
    
    This replaces manual field copying with dataclass utilities for safer
    maintenance when the base Frame class changes.
    """
    # Use dataclasses.asdict for safe field extraction
    frame_dict = dataclasses.asdict(frame)
    return SemanticFrame(**frame_dict)