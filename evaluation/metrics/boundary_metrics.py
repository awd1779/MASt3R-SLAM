"""Boundary accuracy metrics for evaluating segmentation quality."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import ndimage
from sklearn.neighbors import KDTree
import cv2

from .base_metrics import BaseMetrics


class BoundaryAccuracyMetrics(BaseMetrics):
    """Evaluate the accuracy of segmentation boundaries in 3D space."""
    
    def __init__(self,
                 distance_thresholds: Optional[List[float]] = None,
                 dilation_radius: int = 2,
                 output_dir: Optional[str] = None):
        """Initialize boundary accuracy metrics.
        
        Args:
            distance_thresholds: List of distance thresholds for boundary evaluation (meters)
            dilation_radius: Radius for morphological operations
            output_dir: Directory to save outputs
        """
        super().__init__("BoundaryAccuracy", output_dir)
        
        self.distance_thresholds = distance_thresholds or [0.01, 0.02, 0.05, 0.1, 0.2]
        self.dilation_radius = dilation_radius
        
        self.reset()
    
    def reset(self):
        """Reset internal state."""
        self.results = {}
        self.boundary_distances = []
    
    def extract_3d_boundaries(self,
                             points: np.ndarray,
                             labels: np.ndarray,
                             voxel_size: float = 0.01) -> np.ndarray:
        """Extract boundary points from 3D segmentation.
        
        Args:
            points: (N, 3) array of 3D points
            labels: (N,) array of semantic/instance labels
            voxel_size: Size of voxels for boundary detection
            
        Returns:
            (M, 3) array of boundary points
        """
        if len(points) == 0:
            return np.array([])
        
        # Voxelize the point cloud
        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)
        
        # Compute voxel grid dimensions
        grid_size = np.ceil((max_bound - min_bound) / voxel_size).astype(int) + 1
        
        # Map points to voxel indices
        voxel_indices = ((points - min_bound) / voxel_size).astype(int)
        
        # Create voxel grid with labels
        voxel_grid = np.full(grid_size, -1, dtype=np.int32)
        for i, (vx, vy, vz) in enumerate(voxel_indices):
            voxel_grid[vx, vy, vz] = labels[i]
        
        # Find boundary voxels using morphological operations
        boundary_mask = np.zeros_like(voxel_grid, dtype=bool)
        
        for label in np.unique(labels):
            if label < 0:  # Skip background
                continue
            
            # Create binary mask for current label
            label_mask = voxel_grid == label
            
            # Erode to find internal voxels
            eroded = ndimage.binary_erosion(label_mask)
            
            # Boundary is the difference
            label_boundary = label_mask & ~eroded
            boundary_mask |= label_boundary
        
        # Convert boundary voxels back to 3D points
        boundary_voxels = np.argwhere(boundary_mask)
        boundary_points = boundary_voxels * voxel_size + min_bound
        
        return boundary_points
    
    def extract_2d_boundaries(self,
                             image: np.ndarray,
                             labels: np.ndarray) -> np.ndarray:
        """Extract boundary pixels from 2D segmentation mask.
        
        Args:
            image: Input image (H, W, 3)
            labels: Segmentation mask (H, W)
            
        Returns:
            Binary boundary mask (H, W)
        """
        boundaries = np.zeros_like(labels, dtype=np.uint8)
        
        # Find boundaries using gradient
        grad_x = cv2.Sobel(labels.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(labels.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        boundaries[gradient_magnitude > 0] = 1
        
        return boundaries
    
    def compute_boundary_distance(self,
                                 pred_boundaries: np.ndarray,
                                 gt_boundaries: np.ndarray) -> Dict[str, float]:
        """Compute distance metrics between predicted and GT boundaries.
        
        Args:
            pred_boundaries: Predicted boundary points (N, 3)
            gt_boundaries: Ground truth boundary points (M, 3)
            
        Returns:
            Dictionary of distance metrics
        """
        if len(pred_boundaries) == 0 or len(gt_boundaries) == 0:
            return {f'precision_{int(t*100)}cm': 0.0 for t in self.distance_thresholds}
        
        # Build KD-trees for efficient nearest neighbor search
        pred_tree = KDTree(pred_boundaries)
        gt_tree = KDTree(gt_boundaries)
        
        # Compute distances from predicted to GT (for precision)
        pred_to_gt_dist, _ = gt_tree.query(pred_boundaries, k=1)
        pred_to_gt_dist = pred_to_gt_dist.flatten()
        
        # Compute distances from GT to predicted (for recall)
        gt_to_pred_dist, _ = pred_tree.query(gt_boundaries, k=1)
        gt_to_pred_dist = gt_to_pred_dist.flatten()
        
        metrics = {}
        
        # Compute precision/recall at different thresholds
        for threshold in self.distance_thresholds:
            precision = np.mean(pred_to_gt_dist <= threshold)
            recall = np.mean(gt_to_pred_dist <= threshold)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[f'boundary_precision_{int(threshold*100)}cm'] = precision
            metrics[f'boundary_recall_{int(threshold*100)}cm'] = recall
            metrics[f'boundary_f1_{int(threshold*100)}cm'] = f1
        
        # Compute average boundary distance
        metrics['mean_boundary_distance'] = np.mean(pred_to_gt_dist)
        metrics['median_boundary_distance'] = np.median(pred_to_gt_dist)
        metrics['boundary_distance_std'] = np.std(pred_to_gt_dist)
        
        # Store for aggregation
        self.boundary_distances.extend(pred_to_gt_dist.tolist())
        
        return metrics
    
    def compute_trimap_accuracy(self,
                               pred_mask: np.ndarray,
                               gt_mask: np.ndarray,
                               trimap_width: int = 3) -> Dict[str, float]:
        """Compute accuracy within a trimap around boundaries.
        
        Args:
            pred_mask: Predicted segmentation mask
            gt_mask: Ground truth segmentation mask
            trimap_width: Width of the trimap region
            
        Returns:
            Dictionary with trimap accuracy metrics
        """
        # Extract boundaries
        gt_boundaries = self.extract_2d_boundaries(None, gt_mask)
        
        # Create trimap by dilating boundaries
        kernel = np.ones((trimap_width*2+1, trimap_width*2+1), np.uint8)
        trimap = cv2.dilate(gt_boundaries, kernel, iterations=1)
        
        # Compute accuracy within trimap
        trimap_region = trimap > 0
        if np.sum(trimap_region) == 0:
            return {'trimap_accuracy': 1.0, 'trimap_iou': 1.0}
        
        correct = pred_mask[trimap_region] == gt_mask[trimap_region]
        trimap_accuracy = np.mean(correct)
        
        # Compute IoU within trimap
        intersection = np.sum((pred_mask == gt_mask) & trimap_region)
        union = np.sum(trimap_region)
        trimap_iou = intersection / union if union > 0 else 0
        
        return {
            'trimap_accuracy': trimap_accuracy,
            'trimap_iou': trimap_iou,
            'trimap_pixels': int(np.sum(trimap_region))
        }
    
    def compute_boundary_iou(self,
                            pred_boundaries: np.ndarray,
                            gt_boundaries: np.ndarray,
                            threshold: float = 0.05) -> float:
        """Compute IoU for boundary points.
        
        Args:
            pred_boundaries: Predicted boundary points
            gt_boundaries: Ground truth boundary points
            threshold: Distance threshold for matching
            
        Returns:
            Boundary IoU score
        """
        if len(pred_boundaries) == 0 and len(gt_boundaries) == 0:
            return 1.0
        if len(pred_boundaries) == 0 or len(gt_boundaries) == 0:
            return 0.0
        
        # Find matches within threshold
        gt_tree = KDTree(gt_boundaries)
        pred_tree = KDTree(pred_boundaries)
        
        # Predicted boundaries that match GT
        pred_to_gt_dist, _ = gt_tree.query(pred_boundaries, k=1)
        pred_matches = np.sum(pred_to_gt_dist.flatten() <= threshold)
        
        # GT boundaries that match predicted
        gt_to_pred_dist, _ = pred_tree.query(gt_boundaries, k=1)
        gt_matches = np.sum(gt_to_pred_dist.flatten() <= threshold)
        
        # Compute IoU
        intersection = (pred_matches + gt_matches) / 2.0
        union = len(pred_boundaries) + len(gt_boundaries) - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def compute(self,
               prediction: Dict[str, np.ndarray],
               ground_truth: Dict[str, np.ndarray],
               **kwargs) -> Dict[str, float]:
        """Compute boundary accuracy metrics.
        
        Args:
            prediction: Dict with keys:
                - 'points': (N, 3) point cloud
                - 'labels': (N,) semantic/instance labels
                - 'mask_2d': Optional 2D segmentation mask
            ground_truth: Dict with keys:
                - 'points': (M, 3) point cloud
                - 'labels': (M,) semantic/instance labels
                - 'mask_2d': Optional 2D segmentation mask
                
        Returns:
            Dictionary of boundary metrics
        """
        metrics = {}
        
        # Extract 3D boundaries
        if 'points' in prediction and 'points' in ground_truth:
            pred_points = prediction['points']
            pred_labels = prediction['labels']
            gt_points = ground_truth['points']
            gt_labels = ground_truth['labels']
            
            # Extract boundaries
            voxel_size = kwargs.get('voxel_size', 0.02)
            pred_boundaries = self.extract_3d_boundaries(pred_points, pred_labels, voxel_size)
            gt_boundaries = self.extract_3d_boundaries(gt_points, gt_labels, voxel_size)
            
            # Compute 3D boundary metrics
            boundary_metrics = self.compute_boundary_distance(pred_boundaries, gt_boundaries)
            metrics.update(boundary_metrics)
            
            # Compute boundary IoU
            for threshold in self.distance_thresholds:
                biou = self.compute_boundary_iou(pred_boundaries, gt_boundaries, threshold)
                metrics[f'boundary_iou_{int(threshold*100)}cm'] = biou
        
        # Compute 2D metrics if masks provided
        if 'mask_2d' in prediction and 'mask_2d' in ground_truth:
            pred_mask = prediction['mask_2d']
            gt_mask = ground_truth['mask_2d']
            
            # Compute trimap accuracy
            trimap_metrics = self.compute_trimap_accuracy(pred_mask, gt_mask)
            metrics.update(trimap_metrics)
        
        # Per-class boundary metrics if class labels provided
        if 'semantic_labels' in prediction:
            per_class_metrics = self.compute_per_class_boundaries(
                prediction, ground_truth
            )
            metrics.update(per_class_metrics)
        
        self.results.update(metrics)
        return metrics
    
    def compute_per_class_boundaries(self,
                                   prediction: Dict,
                                   ground_truth: Dict) -> Dict[str, float]:
        """Compute boundary metrics per semantic class.
        
        Args:
            prediction: Prediction dictionary
            ground_truth: Ground truth dictionary
            
        Returns:
            Per-class boundary metrics
        """
        pred_points = prediction['points']
        pred_labels = prediction['labels']
        pred_semantic = prediction.get('semantic_labels', pred_labels)
        
        gt_points = ground_truth['points']
        gt_labels = ground_truth['labels']
        gt_semantic = ground_truth.get('semantic_labels', gt_labels)
        
        unique_classes = np.unique(np.concatenate([pred_semantic, gt_semantic]))
        per_class_metrics = {}
        
        for class_id in unique_classes:
            if class_id < 0:  # Skip background
                continue
            
            # Extract points for this class
            pred_class_mask = pred_semantic == class_id
            gt_class_mask = gt_semantic == class_id
            
            if np.sum(pred_class_mask) > 0 and np.sum(gt_class_mask) > 0:
                # Create temporary labels for boundary extraction
                pred_temp_labels = np.zeros_like(pred_labels)
                pred_temp_labels[pred_class_mask] = 1
                
                gt_temp_labels = np.zeros_like(gt_labels)
                gt_temp_labels[gt_class_mask] = 1
                
                # Extract class-specific boundaries
                pred_boundaries = self.extract_3d_boundaries(pred_points, pred_temp_labels)
                gt_boundaries = self.extract_3d_boundaries(gt_points, gt_temp_labels)
                
                # Compute boundary IoU for this class
                biou = self.compute_boundary_iou(pred_boundaries, gt_boundaries, 0.05)
                per_class_metrics[f'boundary_iou_class_{class_id}'] = biou
        
        return per_class_metrics