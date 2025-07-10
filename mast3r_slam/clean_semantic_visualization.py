"""Clean semantic visualization - only keyframe and segmentation side by side."""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Optional
import torch

class CleanSemanticVisualizer:
    """Create clean side-by-side visualizations of keyframes and their segmentation."""
    
    def __init__(self, output_dir: str = "logs/semantic_visualization"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define color map for labels
        self.label_colormap = {
            'person': [255, 0, 0],      # Red
            'chair': [0, 255, 0],       # Green
            'table': [0, 0, 255],       # Blue
            'car': [255, 255, 0],       # Yellow
            'bottle': [255, 0, 255],    # Magenta
            'door': [0, 255, 255],      # Cyan
            'floor': [128, 128, 128],   # Gray
            'wall': [255, 128, 0],      # Orange
            'unknown': [128, 128, 128]  # Gray
        }
    
    def create_side_by_side_visualization(self,
                                        keyframe,
                                        semantic_mask: np.ndarray,
                                        label_names: Dict[int, str],
                                        kf_idx: int,
                                        dataset_name: str = "default"):
        """Create a side-by-side visualization of keyframe and segmentation."""
        
        # Get RGB image
        img_rgb = (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8)
        h, w = img_rgb.shape[:2]
        
        # Create segmentation visualization
        seg_viz = np.zeros_like(img_rgb)
        
        # Apply colors for each semantic label
        unique_labels = np.unique(semantic_mask)
        for label_id in unique_labels:
            if label_id == 0:  # Skip background
                continue
            
            mask = semantic_mask == label_id
            label_name = label_names.get(label_id, 'unknown')
            color = self.label_colormap.get(label_name, self.label_colormap['unknown'])
            seg_viz[mask] = color
        
        # Create overlay (semi-transparent segmentation on original)
        overlay = img_rgb.copy()
        mask_any = semantic_mask > 0
        overlay[mask_any] = (0.6 * seg_viz[mask_any] + 0.4 * img_rgb[mask_any]).astype(np.uint8)
        
        # Create side-by-side image
        side_by_side = np.hstack([img_rgb, overlay])
        
        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # Add titles
        cv2.putText(side_by_side, "Original", (10, 20), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(side_by_side, "Semantic Segmentation", (w + 10, 20), font, font_scale, (255, 255, 255), thickness)
        
        # Add legend on the right side
        y_offset = 40
        x_offset = w + 10
        for label_id in sorted(unique_labels):
            if label_id == 0:
                continue
            label_name = label_names.get(label_id, f'label_{label_id}')
            count = np.sum(semantic_mask == label_id)
            percentage = (count / (h * w)) * 100
            
            # Draw color box
            color = self.label_colormap.get(label_name, self.label_colormap['unknown'])
            cv2.rectangle(side_by_side, (x_offset, y_offset - 10), (x_offset + 20, y_offset), color, -1)
            
            # Add text
            text = f"{label_name}: {percentage:.1f}%"
            cv2.putText(side_by_side, text, (x_offset + 25, y_offset), 
                       font, font_scale, (255, 255, 255), thickness)
            y_offset += 20
        
        # Save the visualization
        output_file = self.output_dir / f"{dataset_name}_keyframe_{kf_idx:04d}_semantic.png"
        cv2.imwrite(str(output_file), cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))
        
        return output_file
    
    def process_all_keyframes(self,
                            keyframes,
                            semantic_keyframes,
                            dataset_name: str = "default",
                            max_frames: Optional[int] = None):
        """Process all keyframes and create visualizations."""
        
        processed = 0
        output_files = []
        
        for kf_idx in range(len(keyframes)):
            if max_frames and processed >= max_frames:
                break
                
            keyframe = keyframes[kf_idx]
            if keyframe is None:
                continue
            
            # Get semantic data
            semantic_data = semantic_keyframes.get(kf_idx)
            if semantic_data is None or not semantic_data.get('masks_rle'):
                continue
            
            # Reconstruct semantic mask
            h, w = keyframe.uimg.shape[:2]
            semantic_mask = np.zeros((h, w), dtype=np.uint8)
            label_names = {}
            next_label_id = 1
            
            for instance_id, rle in semantic_data['masks_rle'].items():
                # Decode mask
                mask = decode_rle(rle, h, w)
                
                # Get label name
                label_name = semantic_data.get('labels', {}).get(instance_id, 'unknown')
                
                # Assign label ID
                if label_name not in label_names.values():
                    label_names[next_label_id] = label_name
                    label_id = next_label_id
                    next_label_id += 1
                else:
                    label_id = [k for k, v in label_names.items() if v == label_name][0]
                
                # Apply mask
                semantic_mask[mask] = label_id
            
            # Create visualization
            output_file = self.create_side_by_side_visualization(
                keyframe, semantic_mask, label_names, kf_idx, dataset_name
            )
            output_files.append(output_file)
            processed += 1
        
        print(f"\nCreated {processed} semantic visualizations in {self.output_dir}")
        return output_files


def decode_rle(rle: Dict, h: int, w: int) -> np.ndarray:
    """Decode RLE mask to binary array."""
    import pycocotools.mask as mask_util
    
    # Handle different RLE formats
    if 'size' in rle:
        size = rle['size']
        if len(size) == 3:
            _, orig_h, orig_w = size
        else:
            orig_h, orig_w = size
    else:
        orig_h, orig_w = h, w
    
    # Decode mask
    mask = mask_util.decode(rle)
    
    # Handle batch dimension if present
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    
    # Resize if needed
    if (orig_h, orig_w) != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    
    return mask.astype(bool)