"""Instance segmentation evaluation metrics for 3D semantic SLAM."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import KDTree

from .base_metrics import BaseMetrics


class InstanceSegmentationMetrics(BaseMetrics):
    """Compute instance segmentation metrics including AP and AR."""
    
    def __init__(self, 
                 iou_thresholds: Optional[List[float]] = None,
                 distance_threshold: float = 0.05,
                 recall_thresholds: Optional[np.ndarray] = None,
                 output_dir: Optional[str] = None):
        """Initialize instance segmentation metrics.
        
        Args:
            iou_thresholds: List of IoU thresholds for AP computation
            distance_threshold: Distance threshold for 3D point matching (meters)
            recall_thresholds: Recall thresholds for AP computation
            output_dir: Directory to save outputs
        """
        super().__init__("InstanceSegmentation", output_dir)
        
        self.iou_thresholds = iou_thresholds or [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.distance_threshold = distance_threshold
        self.recall_thresholds = recall_thresholds or np.linspace(0.0, 1.0, 101)
        
        self.reset()
    
    def reset(self):
        """Reset internal state."""
        self.results = {}
        self.all_detections = []
        self.all_ground_truths = []
        self.per_class_results = defaultdict(list)
    
    def compute_3d_iou(self, 
                      pred_points: np.ndarray, 
                      gt_points: np.ndarray,
                      pred_mask: np.ndarray,
                      gt_mask: np.ndarray) -> float:
        """Compute 3D IoU between predicted and ground truth instances.
        
        Args:
            pred_points: Predicted point cloud (N, 3)
            gt_points: Ground truth point cloud (M, 3)
            pred_mask: Binary mask for predicted instance
            gt_mask: Binary mask for ground truth instance
            
        Returns:
            3D IoU value
        """
        # Extract instance points
        pred_instance = pred_points[pred_mask]
        gt_instance = gt_points[gt_mask]
        
        if len(pred_instance) == 0 or len(gt_instance) == 0:
            return 0.0
        
        # Find overlapping points using KDTree
        tree = KDTree(gt_instance)
        distances, _ = tree.query(pred_instance, k=1)
        
        # Points within threshold are considered overlapping
        overlap_mask = distances.flatten() < self.distance_threshold
        overlap_count = np.sum(overlap_mask)
        
        # Compute IoU
        union_count = len(pred_instance) + len(gt_instance) - overlap_count
        iou = overlap_count / union_count if union_count > 0 else 0.0
        
        return iou
    
    def match_instances(self,
                       pred_instances: Dict[int, np.ndarray],
                       gt_instances: Dict[int, np.ndarray],
                       pred_points: np.ndarray,
                       gt_points: np.ndarray) -> List[Tuple[int, int, float]]:
        """Match predicted instances to ground truth instances.
        
        Args:
            pred_instances: Dict mapping instance IDs to masks
            gt_instances: Dict mapping instance IDs to masks
            pred_points: Predicted point cloud
            gt_points: Ground truth point cloud
            
        Returns:
            List of (pred_id, gt_id, iou) tuples
        """
        pred_ids = list(pred_instances.keys())
        gt_ids = list(gt_instances.keys())
        
        if not pred_ids or not gt_ids:
            return []
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(pred_ids), len(gt_ids)))
        
        for i, pred_id in enumerate(pred_ids):
            for j, gt_id in enumerate(gt_ids):
                iou = self.compute_3d_iou(
                    pred_points, gt_points,
                    pred_instances[pred_id], gt_instances[gt_id]
                )
                iou_matrix[i, j] = iou
        
        # Hungarian matching
        pred_indices, gt_indices = linear_sum_assignment(-iou_matrix)
        
        matches = []
        for pred_idx, gt_idx in zip(pred_indices, gt_indices):
            iou = iou_matrix[pred_idx, gt_idx]
            if iou > 0:  # Only keep matches with positive IoU
                matches.append((pred_ids[pred_idx], gt_ids[gt_idx], iou))
        
        return matches
    
    def compute_ap_ar(self, 
                     detections: List[Dict],
                     ground_truths: List[Dict],
                     iou_threshold: float) -> Tuple[float, float, np.ndarray]:
        """Compute Average Precision and Average Recall at given IoU threshold.
        
        Args:
            detections: List of detection dicts with 'score' and 'matched_gt_id'
            ground_truths: List of ground truth instances
            iou_threshold: IoU threshold for considering a match
            
        Returns:
            Tuple of (AP, AR, precision-recall curve)
        """
        # Sort detections by confidence score
        sorted_detections = sorted(detections, key=lambda x: x['score'], reverse=True)
        
        # Initialize tracking variables
        num_gt = len(ground_truths)
        tp = np.zeros(len(sorted_detections))
        fp = np.zeros(len(sorted_detections))
        gt_matched = set()
        
        # Process each detection
        for i, det in enumerate(sorted_detections):
            if det['iou'] >= iou_threshold and det['matched_gt_id'] not in gt_matched:
                tp[i] = 1
                gt_matched.add(det['matched_gt_id'])
            else:
                fp[i] = 1
        
        # Compute precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / num_gt if num_gt > 0 else np.zeros_like(tp_cumsum)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
        
        # Compute AP using 101-point interpolation
        ap = 0.0
        for t in self.recall_thresholds:
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / len(self.recall_thresholds)
        
        # Compute AR (max recall)
        ar = np.max(recalls) if len(recalls) > 0 else 0.0
        
        return ap, ar, np.column_stack([recalls, precisions])
    
    def compute(self, 
               prediction: Dict[str, np.ndarray],
               ground_truth: Dict[str, np.ndarray],
               **kwargs) -> Dict[str, float]:
        """Compute instance segmentation metrics.
        
        Args:
            prediction: Dict with keys:
                - 'points': (N, 3) point cloud
                - 'instance_labels': (N,) instance IDs
                - 'semantic_labels': (N,) semantic class IDs
                - 'confidence_scores': (N,) or dict of instance scores
            ground_truth: Dict with keys:
                - 'points': (M, 3) point cloud
                - 'instance_labels': (M,) instance IDs
                - 'semantic_labels': (M,) semantic class IDs
                
        Returns:
            Dictionary of metric values
        """
        pred_points = prediction['points']
        pred_instances = prediction['instance_labels']
        pred_semantics = prediction.get('semantic_labels', None)
        pred_scores = prediction.get('confidence_scores', {})
        
        gt_points = ground_truth['points']
        gt_instances = ground_truth['instance_labels']
        gt_semantics = ground_truth.get('semantic_labels', None)
        
        # Group points by instance
        pred_instance_masks = {}
        gt_instance_masks = {}
        
        for inst_id in np.unique(pred_instances):
            if inst_id >= 0:  # Skip background (-1)
                pred_instance_masks[inst_id] = pred_instances == inst_id
        
        for inst_id in np.unique(gt_instances):
            if inst_id >= 0:
                gt_instance_masks[inst_id] = gt_instances == inst_id
        
        # Match instances
        matches = self.match_instances(
            pred_instance_masks, gt_instance_masks,
            pred_points, gt_points
        )
        
        # Prepare detection results
        detections = []
        for pred_id, mask in pred_instance_masks.items():
            score = pred_scores.get(pred_id, 1.0) if isinstance(pred_scores, dict) else float(np.mean(pred_scores[mask]))
            
            # Find best matching GT
            matched_gt_id = None
            best_iou = 0.0
            for match_pred_id, match_gt_id, iou in matches:
                if match_pred_id == pred_id and iou > best_iou:
                    matched_gt_id = match_gt_id
                    best_iou = iou
            
            detections.append({
                'pred_id': pred_id,
                'matched_gt_id': matched_gt_id,
                'iou': best_iou,
                'score': score,
                'num_points': np.sum(mask)
            })
        
        ground_truths = [{'gt_id': gt_id, 'num_points': np.sum(mask)} 
                        for gt_id, mask in gt_instance_masks.items()]
        
        # Store for aggregated computation
        self.all_detections.extend(detections)
        self.all_ground_truths.extend(ground_truths)
        
        # Compute metrics at different IoU thresholds
        metrics = {}
        ap_list = []
        ar_list = []
        
        for iou_thresh in self.iou_thresholds:
            ap, ar, _ = self.compute_ap_ar(detections, ground_truths, iou_thresh)
            metrics[f'AP_{int(iou_thresh*100)}'] = ap
            metrics[f'AR_{int(iou_thresh*100)}'] = ar
            ap_list.append(ap)
            ar_list.append(ar)
        
        # Compute mAP and mAR
        metrics['mAP'] = np.mean(ap_list)
        metrics['mAR'] = np.mean(ar_list)
        
        # Compute AP at different IoU ranges
        metrics['mAP_50'] = np.mean([metrics[f'AP_{i}'] for i in range(50, 100, 5)])
        metrics['mAP_75'] = np.mean([metrics[f'AP_{i}'] for i in range(75, 100, 5)])
        
        # Instance-level statistics
        metrics['num_pred_instances'] = len(pred_instance_masks)
        metrics['num_gt_instances'] = len(gt_instance_masks)
        metrics['num_matched_instances'] = len([m for m in matches if m[2] > 0.5])
        
        self.results.update(metrics)
        return metrics
    
    def compute_per_class_metrics(self,
                                 predictions: List[Dict],
                                 ground_truths: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Compute per-class instance segmentation metrics.
        
        Args:
            predictions: List of predictions with semantic labels
            ground_truths: List of ground truths with semantic labels
            
        Returns:
            Dict mapping class names to metric dictionaries
        """
        # Group by semantic class
        class_detections = defaultdict(list)
        class_ground_truths = defaultdict(list)
        
        for pred in predictions:
            if 'semantic_label' in pred:
                class_detections[pred['semantic_label']].append(pred)
        
        for gt in ground_truths:
            if 'semantic_label' in gt:
                class_ground_truths[gt['semantic_label']].append(gt)
        
        per_class_metrics = {}
        
        for class_label in set(class_detections.keys()) | set(class_ground_truths.keys()):
            class_dets = class_detections.get(class_label, [])
            class_gts = class_ground_truths.get(class_label, [])
            
            if class_gts:  # Only compute if there are ground truths
                ap_50, ar_50, _ = self.compute_ap_ar(class_dets, class_gts, 0.5)
                ap_75, ar_75, _ = self.compute_ap_ar(class_dets, class_gts, 0.75)
                
                per_class_metrics[class_label] = {
                    'AP_50': ap_50,
                    'AP_75': ap_75,
                    'AR_50': ar_50,
                    'AR_75': ar_75,
                    'num_instances': len(class_gts)
                }
        
        return per_class_metrics