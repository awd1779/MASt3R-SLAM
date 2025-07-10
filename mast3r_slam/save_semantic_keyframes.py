"""Save semantic keyframes with segmentation masks overlaid."""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mast3r_slam.semantic_frame import decode_rle


def save_semantic_keyframes(savedir: Path, 
                          timestamps,
                          keyframes,
                          semantic_keyframes,
                          semantic_backend,
                          visualize_mode: str = "side_by_side"):
    """
    Save keyframes with semantic segmentation overlays.
    
    Args:
        savedir: Directory to save images
        timestamps: Timestamps for frames
        keyframes: SLAM keyframes
        semantic_keyframes: Semantic data
        semantic_backend: Backend with semantic processing results
        visualize_mode: "overlay", "side_by_side", or "both"
    """
    print(f"Saving semantic keyframes to {savedir}")
    savedir = Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    
    # Only create side-by-side visualizations
    # No subdirectories needed - save directly to savedir
    
    # Get label colors from track manager
    label_colors = {}
    if hasattr(semantic_backend, 'track_manager'):
        tracks = semantic_backend.track_manager.get_all_tracks()
        cmap = cm.get_cmap('tab20')
        
        for track_id, track_info in tracks.items():
            if track_id == 0:  # Background
                label_colors[track_id] = (128, 128, 128)  # Gray
            else:
                color = cmap((track_id - 1) % 20)[:3]
                label_colors[track_id] = tuple(int(c * 255) for c in color)
    
    # Process each keyframe
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        t = timestamps[keyframe.frame_id]
        
        # Get original image
        img = (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Get semantic data
        semantic_data = semantic_keyframes.get_semantics(i)
        
        if semantic_data is None or not semantic_data.get('masks_rle'):
            # Skip frames without semantics
            continue
        
        # Create semantic mask
        h, w = img.shape[:2]
        semantic_mask = np.zeros((h, w), dtype=np.int32)
        instance_masks = {}
        
        # Decode all masks
        for instance_id, rle in semantic_data['masks_rle'].items():
            if 'size' in rle:
                size = rle['size']
                if len(size) == 3:
                    _, mask_h, mask_w = size
                else:
                    mask_h, mask_w = size
                    
                # Decode at original size
                mask = decode_rle(rle, mask_h, mask_w)
                
                # Resize mask to keyframe size if needed
                if (mask_h, mask_w) != (h, w):
                    mask_resized = cv2.resize(
                        mask.astype(np.uint8), 
                        (w, h), 
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                    mask = mask_resized
                
                instance_masks[instance_id] = mask
                
                # Get global track ID
                track_id = semantic_data.get('track_ids', {}).get(instance_id, instance_id)
                semantic_mask[mask] = track_id
        
        # Create side-by-side visualization only
        # Create overlay with semi-transparent segmentation
        overlay = img_bgr.copy()
        
        # Apply colors for each semantic label
        for track_id in np.unique(semantic_mask):
            if track_id == 0:
                continue
            mask = semantic_mask == track_id
            color = label_colors.get(track_id, (255, 255, 255))
            # Semi-transparent overlay
            overlay[mask] = (0.6 * np.array(color[::-1]) + 0.4 * overlay[mask]).astype(np.uint8)
        
        # Combine images side by side
        combined = np.hstack([img_bgr, overlay])
        
        # Add titles
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(combined, "Original", (10, 30), font, 0.8, (255, 255, 255), 2)
        cv2.putText(combined, "Semantic Segmentation", (w + 10, 30), font, 0.8, (255, 255, 255), 2)
        
        # Add legend on the right side
        y_offset = 60
        x_offset = w + 10
        
        # Get unique labels for this frame
        frame_labels = {}
        for instance_id, track_id in semantic_data.get('track_ids', {}).items():
            label = semantic_data.get('labels', {}).get(instance_id, 'unknown')
            if track_id not in frame_labels and track_id != 0:
                frame_labels[track_id] = label
        
        # Draw legend
        for track_id, label in sorted(frame_labels.items(), key=lambda x: x[1]):
            color = label_colors.get(track_id, (255, 255, 255))
            # Draw color box
            cv2.rectangle(combined, (x_offset, y_offset - 10), (x_offset + 20, y_offset + 5), 
                         color[::-1], -1)
            # Add label text
            cv2.putText(combined, label, (x_offset + 25, y_offset), 
                       font, 0.6, (255, 255, 255), 1)
            y_offset += 25
        
        # Save with simple filename
        filename = savedir / f"keyframe_{i:04d}_semantic.png"
        cv2.imwrite(str(filename), combined)
    
    print(f"Saved semantic keyframes to {savedir}")


def create_semantic_overlay(img: np.ndarray, 
                          semantic_mask: np.ndarray,
                          label_colors: Dict[int, tuple],
                          alpha: float = 0.5) -> np.ndarray:
    """Create overlay of semantic mask on image."""
    overlay = img.copy()
    
    # Apply colors for each semantic label
    for label_id in np.unique(semantic_mask):
        if label_id == 0:  # Skip background
            continue
            
        mask = semantic_mask == label_id
        color = label_colors.get(label_id, (255, 255, 255))
        
        # Create colored overlay
        overlay[mask] = (
            alpha * np.array(color[::-1]) +  # BGR
            (1 - alpha) * overlay[mask]
        ).astype(np.uint8)
    
    return overlay


def create_semantic_stats(semantic_keyframes, semantic_backend) -> Dict:
    """Generate statistics about semantic segmentation."""
    stats = {
        'total_keyframes': 0,
        'keyframes_with_semantics': 0,
        'total_instances': 0,
        'unique_labels': set(),
        'label_counts': {},
        'avg_instances_per_frame': 0,
        'coverage_percentage': 0
    }
    
    total_pixels = 0
    labeled_pixels = 0
    
    for i in range(len(semantic_keyframes)):
        semantic_data = semantic_keyframes.get_semantics(i)
        if semantic_data is None:
            continue
            
        stats['keyframes_with_semantics'] += 1
        n_instances = len(semantic_data.get('instance_ids', []))
        stats['total_instances'] += n_instances
        
        # Count labels
        for instance_id, label in semantic_data.get('labels', {}).items():
            stats['unique_labels'].add(label)
            stats['label_counts'][label] = stats['label_counts'].get(label, 0) + 1
        
        # Calculate coverage
        for rle in semantic_data.get('masks_rle', {}).values():
            if 'counts' in rle:
                # Count pixels in RLE
                counts = rle['counts']
                if isinstance(counts, bytes):
                    counts = counts.decode('utf-8')
                
                # RLE alternates between background and foreground
                is_foreground = False
                idx = 0
                for c in counts:
                    if c.isdigit():
                        continue
                    else:
                        # End of number
                        if is_foreground:
                            # This was a foreground run
                            labeled_pixels += int(counts[idx:counts.index(c, idx)])
                        is_foreground = not is_foreground
                        idx = counts.index(c, idx) + 1
                
                # Get total pixels
                if 'size' in rle:
                    size = rle['size']
                    if len(size) == 3:
                        _, h, w = size
                    else:
                        h, w = size
                    total_pixels += h * w
    
    stats['total_keyframes'] = len(semantic_keyframes)
    if stats['keyframes_with_semantics'] > 0:
        stats['avg_instances_per_frame'] = stats['total_instances'] / stats['keyframes_with_semantics']
    
    if total_pixels > 0:
        stats['coverage_percentage'] = (labeled_pixels / total_pixels) * 100
    
    return stats