"""Debug visualization utilities for Grounded SAM2 processor."""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger('mast3r_slam.debug_visualizer')


class DebugVisualizer:
    """Handles all debug visualizations and file outputs for SAM2 processing."""
    
    def __init__(self, debug_dir: Path, save_visualizations: bool = True):
        self.debug_dir = debug_dir
        self.save_visualizations = save_visualizations
        
        if save_visualizations and debug_dir:
            debug_dir.mkdir(parents=True, exist_ok=True)
    
    def visualize_detections(self, image: np.ndarray, boxes: List, labels: List, 
                           scores: List, frame_idx: int, stage: str = "grounding"):
        """Visualize detection boxes and save to file."""
        if not self.save_visualizations or not boxes:
            return
        
        try:
            vis_image = image.copy()
            
            for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
                x1, y1, x2, y2 = [int(coord) for coord in box]
                
                # Draw box
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label with score
                label_text = f"{label}: {score:.3f}"
                cv2.putText(vis_image, label_text, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Save visualization
            output_path = self.debug_dir / f"{stage}_detections.jpg"
            cv2.imwrite(str(output_path), vis_image)
            
        except Exception as e:
            logger.warning(f"Failed to save {stage} detections visualization: {e}")
    
    def visualize_masks(self, image: np.ndarray, masks: List, labels: List, frame_idx: int):
        """Visualize segmentation masks and save to file."""
        if not self.save_visualizations or not masks:
            return
        
        try:
            overlay = image.copy()
            
            for i, (mask, label) in enumerate(zip(masks, labels)):
                if mask is None:
                    continue
                
                # Generate color for this mask
                color = self._get_mask_color(i)
                
                # Apply colored overlay where mask is True
                colored_mask = np.zeros_like(image)
                mask_bool = mask.astype(bool) if mask.dtype != bool else mask
                colored_mask[mask_bool] = color
                
                # Blend with original image
                overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)
            
            # Save visualization
            output_path = self.debug_dir / "segmentation_overlay.jpg"
            cv2.imwrite(str(output_path), overlay)
            
        except Exception as e:
            logger.warning(f"Failed to save mask visualization: {e}")
    
    def save_sam2_raw_output(self, image: np.ndarray, boxes: List, masks: List, 
                           labels: List, scores: List, frame_idx: int):
        """Save raw SAM2 outputs for debugging."""
        if not self.save_visualizations:
            return
        
        try:
            sam2_dir = self.debug_dir / "sam2_raw"
            sam2_dir.mkdir(exist_ok=True)
            
            for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
                box_prefix = f"box_{i:03d}_{label.replace(' ', '_')}"
                
                # Save proposal visualization
                self._save_proposal_image(image, box, label, score, 
                                        sam2_dir / f"{box_prefix}_proposals.jpg")
                
                # Save individual masks
                box_masks = masks[i] if i < len(masks) and masks[i] is not None else []
                for j, mask in enumerate(box_masks[:3]):  # Top 3 masks
                    if mask is not None:
                        mask_score = scores[i] if i < len(scores) else 0.0
                        mask_path = sam2_dir / f"{box_prefix}_mask_{j}_score_{mask_score:.3f}.png"
                        self._save_mask_image(mask, mask_path)
            
        except Exception as e:
            logger.warning(f"Failed to save SAM2 raw output: {e}")
    
    def save_processing_summary(self, detections: Dict, frame_idx: int):
        """Save text summary of processing results."""
        if not self.save_visualizations:
            return
        
        try:
            summary_path = self.debug_dir / "segmentation_summary.txt"
            
            with open(summary_path, 'w') as f:
                f.write(f"Frame {frame_idx} Processing Summary\n")
                f.write("=" * 40 + "\n\n")
                
                if 'instance_ids' in detections:
                    f.write(f"Total instances: {len(detections['instance_ids'])}\n")
                    f.write(f"Labels: {list(detections.get('labels', {}).values())}\n\n")
                    
                    for inst_id in detections['instance_ids']:
                        label = detections.get('labels', {}).get(inst_id, 'unknown')
                        f.write(f"Instance {inst_id}: {label}\n")
                
        except Exception as e:
            logger.warning(f"Failed to save processing summary: {e}")
    
    def _get_mask_color(self, mask_idx: int) -> Tuple[int, int, int]:
        """Generate consistent color for mask visualization."""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (255, 192, 203), (0, 128, 0), (128, 128, 0), (0, 0, 128)
        ]
        return colors[mask_idx % len(colors)]
    
    def _save_proposal_image(self, image: np.ndarray, box: List, label: str, 
                           score: float, output_path: Path):
        """Save box proposal visualization."""
        vis_image = image.copy()
        x1, y1, x2, y2 = [int(coord) for coord in box]
        
        # Draw box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label_text = f"{label}: {score:.3f}"
        cv2.putText(vis_image, label_text, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imwrite(str(output_path), vis_image)
    
    def _save_mask_image(self, mask: np.ndarray, output_path: Path):
        """Save individual mask as image."""
        if mask is not None:
            mask_img = (mask * 255).astype(np.uint8)
            cv2.imwrite(str(output_path), mask_img)