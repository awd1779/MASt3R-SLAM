"""
Shared utilities for semantic processing operations.

This module provides common utilities to eliminate duplication across
semantic processing modules.
"""

import torch
import numpy as np
import colorsys
from pathlib import Path
from typing import Tuple, Optional, Union


def convert_cxcywh_to_xyxy(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """
    Convert bounding boxes from center-width-height format to x1y1x2y2 format.
    
    Args:
        boxes: Tensor of shape (N, 4) in [cx, cy, w, h] format (normalized 0-1)
        width: Image width in pixels
        height: Image height in pixels
    
    Returns:
        Tensor of shape (N, 4) in [x1, y1, x2, y2] format (pixel coordinates)
    """
    if boxes.numel() == 0:
        return boxes
    
    boxes_xyxy = torch.zeros_like(boxes)
    boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * width   # x1
    boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * height  # y1  
    boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * width   # x2
    boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * height  # y2
    
    return boxes_xyxy


def generate_track_color(track_id: int, saturation: float = 0.8, value: float = 0.9) -> np.ndarray:
    """
    Generate a consistent color for each track ID using HSV color space.
    
    This replaces manual HSV-to-RGB conversion with standard library functions.
    
    Args:
        track_id: Unique track identifier
        saturation: HSV saturation component (0.0 to 1.0)
        value: HSV value/brightness component (0.0 to 1.0)
    
    Returns:
        RGB color as numpy array [R, G, B] with values 0-255
    """
    # Use golden ratio for good color distribution
    golden_ratio = 0.618033988749895
    hue = (track_id * golden_ratio) % 1.0
    
    # Convert HSV to RGB using standard library
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    
    return np.array([int(r * 255), int(g * 255), int(b * 255)], dtype=np.uint8)


def ensure_debug_directory(base_dir: Union[str, Path], frame_idx: int) -> Path:
    """
    Create debug directory for a specific frame.
    
    This consolidates the repeated debug directory creation pattern.
    
    Args:
        base_dir: Base debug output directory
        frame_idx: Frame index for directory naming
    
    Returns:
        Path to the created debug directory
    """
    debug_dir = Path(base_dir) / f"frame_{frame_idx:06d}"
    debug_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir


def validate_array_correspondence(*arrays, context: str = "") -> None:
    """
    Validate that all arrays have the same length.
    
    This consolidates repeated validation patterns.
    
    Args:
        *arrays: Variable number of arrays to validate
        context: Context string for error messages
        
    Raises:
        ValueError: If arrays have different lengths
    """
    if not arrays:
        return
    
    lengths = [len(arr) for arr in arrays]
    if len(set(lengths)) > 1:
        raise ValueError(f"Array length mismatch in {context}: {lengths}")


def convert_rle_to_binary_mask(rle: dict, height: int, width: int) -> torch.Tensor:
    """
    Convert RLE format to binary mask tensor.
    
    Args:
        rle: RLE dictionary with 'size' and 'counts' keys
        height: Mask height
        width: Mask width
        
    Returns:
        Binary mask as torch.Tensor
    """
    try:
        import pycocotools.mask as mask_utils
        
        # Handle string/bytes conversion for counts
        rle_copy = {
            'size': rle['size'],
            'counts': rle['counts'].encode('utf-8') if isinstance(rle['counts'], str) else rle['counts']
        }
        
        mask = mask_utils.decode(rle_copy)
        return torch.from_numpy(mask).to(torch.bool)
        
    except ImportError:
        raise ImportError("pycocotools is required for RLE mask decoding")


def convert_binary_mask_to_rle(mask: torch.Tensor) -> dict:
    """
    Convert binary mask tensor to RLE format.
    
    Args:
        mask: Binary mask as torch.Tensor
        
    Returns:
        RLE dictionary with 'size' and 'counts' keys
    """
    try:
        import pycocotools.mask as mask_utils
        
        mask_np = mask.cpu().numpy().astype(np.uint8, order='F')
        rle = mask_utils.encode(mask_np)
        
        # Ensure counts is a string for JSON serialization
        rle['counts'] = rle['counts'].decode('utf-8') if isinstance(rle['counts'], bytes) else rle['counts']
        
        return rle
        
    except ImportError:
        raise ImportError("pycocotools is required for RLE mask encoding")