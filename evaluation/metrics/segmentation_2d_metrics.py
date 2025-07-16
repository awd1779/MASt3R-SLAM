"""2D segmentation quality metrics for semantic SLAM."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import cv2
from scipy.ndimage import distance_transform_edt

from .base_metrics import BaseMetrics


class Segmentation2DMetrics(BaseMetrics):
    """Evaluate 2D segmentation quality across frames."""
    
    def __init__(self,
                 num_classes: int,
                 boundary_distances: List[int] = [1, 3, 5],
                 output_dir: Optional[str] = None):
        """Initialize 2D segmentation metrics.
        
        Args:
            num_classes: Number of semantic classes
            boundary_distances: Pixel distances for boundary evaluation
            output_dir: Directory to save outputs
        """
        super().__init__("Segmentation2D", output_dir)
        
        self.num_classes = num_classes
        self.boundary_distances = boundary_distances
        
        self.reset()
    
    def reset(self):
        """Reset internal state."""
        self.results = {}
        self.frame_ious = []
        self.class_ious = defaultdict(list)
        self.boundary_f1_scores = defaultdict(list)
        self.multi_view_consistency = []
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
    
    def compute_iou_per_class(self, 
                             pred_mask: np.ndarray,
                             gt_mask: np.ndarray) -> Dict[int, float]:
        """Compute IoU for each class in the masks.
        
        Args:
            pred_mask: Predicted segmentation mask (H, W)
            gt_mask: Ground truth segmentation mask (H, W)
            
        Returns:
            Dictionary mapping class IDs to IoU values
        """
        iou_per_class = {}
        
        for class_id in range(self.num_classes):
            pred_binary = (pred_mask == class_id)
            gt_binary = (gt_mask == class_id)
            
            intersection = np.logical_and(pred_binary, gt_binary).sum()
            union = np.logical_or(pred_binary, gt_binary).sum()
            
            if union > 0:
                iou = intersection / union
                iou_per_class[class_id] = iou
                self.class_ious[class_id].append(iou)
        
        return iou_per_class
    
    def compute_boundary_f1(self,
                           pred_mask: np.ndarray,
                           gt_mask: np.ndarray,
                           distance_threshold: int = 3) -> float:
        """Compute boundary F1 score.
        
        Args:
            pred_mask: Predicted segmentation mask
            gt_mask: Ground truth segmentation mask
            distance_threshold: Pixel distance threshold for boundary matching
            
        Returns:
            Boundary F1 score
        """
        # Extract boundaries
        pred_boundary = self._extract_boundaries(pred_mask)
        gt_boundary = self._extract_boundaries(gt_mask)
        
        if not np.any(pred_boundary) or not np.any(gt_boundary):
            return 0.0
        
        # Compute distance transforms
        pred_dist = distance_transform_edt(~pred_boundary)
        gt_dist = distance_transform_edt(~gt_boundary)
        
        # Precision: fraction of predicted boundaries within threshold of GT
        pred_matches = pred_dist[pred_boundary] <= distance_threshold
        precision = np.mean(pred_matches) if np.any(pred_boundary) else 0.0
        
        # Recall: fraction of GT boundaries within threshold of prediction
        gt_matches = gt_dist[gt_boundary] <= distance_threshold
        recall = np.mean(gt_matches) if np.any(gt_boundary) else 0.0
        
        # F1 score
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        return f1
    
    def compute_multi_view_consistency(self,
                                     masks_view1: np.ndarray,
                                     masks_view2: np.ndarray,
                                     correspondence: np.ndarray) -> float:
        """Compute consistency between masks from different viewpoints.
        
        Args:
            masks_view1: Segmentation masks from view 1
            masks_view2: Segmentation masks from view 2
            correspondence: Pixel correspondence between views (view1_idx -> view2_idx)
            
        Returns:
            Consistency score [0, 1]
        """
        valid_mask = correspondence >= 0
        view1_labels = masks_view1[valid_mask]
        view2_labels = masks_view2[correspondence[valid_mask]]
        
        consistency = np.mean(view1_labels == view2_labels) if len(view1_labels) > 0 else 0.0
        
        return consistency
    
    def compute_temporal_consistency(self,
                                   masks_t: np.ndarray,
                                   masks_t1: np.ndarray,
                                   optical_flow: np.ndarray) -> float:
        """Compute temporal consistency between consecutive frames.
        
        Args:
            masks_t: Segmentation masks at time t
            masks_t1: Segmentation masks at time t+1
            optical_flow: Optical flow from t to t+1 (H, W, 2)
            
        Returns:
            Temporal consistency score
        """
        h, w = masks_t.shape
        
        # Create mesh grid
        y, x = np.mgrid[0:h, 0:w]
        
        # Warp coordinates using optical flow
        x_warped = x + optical_flow[..., 0]
        y_warped = y + optical_flow[..., 1]
        
        # Clip to image bounds
        x_warped = np.clip(x_warped, 0, w-1).astype(np.int32)
        y_warped = np.clip(y_warped, 0, h-1).astype(np.int32)
        
        # Get warped labels
        warped_labels = masks_t[y.flatten(), x.flatten()]
        next_labels = masks_t1[y_warped.flatten(), x_warped.flatten()]
        
        # Compute consistency
        consistency = np.mean(warped_labels == next_labels)
        
        return consistency
    
    def compute(self,
               prediction: Dict[str, np.ndarray],
               ground_truth: Dict[str, np.ndarray],
               **kwargs) -> Dict[str, float]:
        """Compute 2D segmentation metrics for a single frame.
        
        Args:
            prediction: Dict with 'mask' key containing predicted segmentation
            ground_truth: Dict with 'mask' key containing GT segmentation
            
        Returns:
            Dictionary of metric values
        """
        pred_mask = prediction['mask']
        gt_mask = ground_truth['mask']
        
        # Update confusion matrix
        valid_mask = gt_mask < self.num_classes
        self.confusion_matrix += np.bincount(
            gt_mask[valid_mask] * self.num_classes + pred_mask[valid_mask],
            minlength=self.num_classes * self.num_classes
        ).reshape(self.num_classes, self.num_classes)
        
        # Compute per-class IoU
        class_ious = self.compute_iou_per_class(pred_mask, gt_mask)
        
        # Mean IoU for this frame
        mean_iou = np.mean(list(class_ious.values())) if class_ious else 0.0
        self.frame_ious.append(mean_iou)
        
        # Boundary F1 scores
        boundary_scores = {}
        for dist in self.boundary_distances:
            f1 = self.compute_boundary_f1(pred_mask, gt_mask, dist)
            boundary_scores[f'boundary_f1_{dist}px'] = f1
            self.boundary_f1_scores[dist].append(f1)
        
        # Frame metrics
        frame_metrics = {
            'mean_iou': mean_iou,
            'pixel_accuracy': np.mean(pred_mask[valid_mask] == gt_mask[valid_mask])
        }
        frame_metrics.update(boundary_scores)
        
        # Multi-view consistency if provided
        if 'prev_mask' in prediction and 'correspondence' in kwargs:
            consistency = self.compute_multi_view_consistency(
                prediction['prev_mask'],
                pred_mask,
                kwargs['correspondence']
            )
            frame_metrics['multi_view_consistency'] = consistency
            self.multi_view_consistency.append(consistency)
        
        # Temporal consistency if optical flow provided
        if 'optical_flow' in kwargs and 'prev_mask' in prediction:
            temporal_consistency = self.compute_temporal_consistency(
                prediction['prev_mask'],
                pred_mask,
                kwargs['optical_flow']
            )
            frame_metrics['temporal_consistency'] = temporal_consistency
        
        return frame_metrics
    
    def compute_aggregated_metrics(self) -> Dict[str, float]:
        """Compute aggregated metrics across all frames."""
        metrics = {}
        
        # Overall metrics
        metrics['mean_iou'] = np.mean(self.frame_ious) if self.frame_ious else 0.0
        
        # Per-class mean IoU
        class_mean_ious = {}
        for class_id, ious in self.class_ious.items():
            if ious:
                class_mean_ious[class_id] = np.mean(ious)
        
        metrics['per_class_iou'] = class_mean_ious
        
        # Boundary F1 scores
        for dist in self.boundary_distances:
            scores = self.boundary_f1_scores[dist]
            if scores:
                metrics[f'mean_boundary_f1_{dist}px'] = np.mean(scores)
        
        # Multi-view consistency
        if self.multi_view_consistency:
            metrics['mean_multi_view_consistency'] = np.mean(self.multi_view_consistency)
        
        # Confusion matrix metrics
        if self.confusion_matrix.sum() > 0:
            # Overall accuracy
            metrics['overall_accuracy'] = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
            
            # Per-class accuracy
            class_accuracies = {}
            for i in range(self.num_classes):
                total = self.confusion_matrix[i].sum()
                if total > 0:
                    class_accuracies[i] = self.confusion_matrix[i, i] / total
            metrics['per_class_accuracy'] = class_accuracies
        
        self.results = metrics
        return metrics
    
    def _extract_boundaries(self, mask: np.ndarray) -> np.ndarray:
        """Extract boundaries from segmentation mask."""
        # Use Sobel filters to detect edges
        grad_x = cv2.Sobel(mask.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(mask.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute gradient magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Threshold to get boundaries
        boundaries = gradient_magnitude > 0
        
        return boundaries
    
    def save_confusion_matrix(self, path: str):
        """Save confusion matrix to file."""
        np.save(path, self.confusion_matrix)
    
    def print_results(self):
        """Print evaluation results."""
        if not self.results:
            self.compute_aggregated_metrics()
        
        print("\n=== 2D Segmentation Metrics ===")
        print(f"Mean IoU: {self.results.get('mean_iou', 0):.3f}")
        print(f"Overall Accuracy: {self.results.get('overall_accuracy', 0):.3f}")
        
        # Boundary F1 scores
        print("\nBoundary F1 Scores:")
        for dist in self.boundary_distances:
            key = f'mean_boundary_f1_{dist}px'
            if key in self.results:
                print(f"  {dist}px threshold: {self.results[key]:.3f}")
        
        # Multi-view consistency
        if 'mean_multi_view_consistency' in self.results:
            print(f"\nMulti-view Consistency: {self.results['mean_multi_view_consistency']:.3f}")
        
        # Per-class IoU (top 5)
        if 'per_class_iou' in self.results:
            class_ious = self.results['per_class_iou']
            if class_ious:
                sorted_classes = sorted(class_ious.items(), key=lambda x: x[1], reverse=True)[:5]
                print("\nTop 5 Classes by IoU:")
                for class_id, iou in sorted_classes:
                    print(f"  Class {class_id}: {iou:.3f}")