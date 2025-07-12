import dataclasses
from typing import Optional, Dict, List
import torch
import numpy as np
from mast3r_slam.frame import Frame
import torch.multiprocessing as mp


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


# RLE encoding/decoding utilities
def encode_rle(mask: torch.Tensor) -> dict:
    """Encode binary mask to RLE format."""
    original_shape = mask.shape
    mask = mask.cpu().numpy().astype(np.uint8).flatten()
    
    # Find run starts and lengths
    runs = []
    current_val = 0
    start_pos = 0
    
    for i in range(len(mask)):
        if mask[i] != current_val:
            runs.append(i - start_pos)
            current_val = mask[i]
            start_pos = i
    
    # Don't forget the last run
    runs.append(len(mask) - start_pos)
    
    # If the mask starts with 1, prepend a 0-length background run
    if len(mask) > 0 and mask[0] == 1:
        runs = [0] + runs
    
    return {'size': list(original_shape), 'counts': runs}


def decode_rle(rle: dict, h: int, w: int) -> torch.Tensor:
    """Decode RLE format to binary mask."""
    counts = rle['counts']
    mask = np.zeros(h * w, dtype=np.uint8)
    
    pos = 0
    val = 0  # Start with background
    for count in counts:
        if count > 0:
            mask[pos:pos+count] = val
        pos += count
        val = 1 - val  # Alternate between 0 and 1
    
    return torch.from_numpy(mask.reshape(h, w)).to(torch.bool)


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