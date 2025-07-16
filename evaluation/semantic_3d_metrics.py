"""3D semantic segmentation evaluation metrics."""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.neighbors import KDTree
import open3d as o3d
from collections import defaultdict


class Semantic3DMetrics:
    def __init__(self, num_classes: int, class_names: Optional[Dict[int, str]] = None,
                 ignore_label: int = -1):
        """Initialize 3D semantic metrics calculator.
        
        Args:
            num_classes: Number of semantic classes
            class_names: Optional mapping from class IDs to names
            ignore_label: Label to ignore in evaluation (e.g., unlabeled points)
        """
        self.num_classes = num_classes
        self.class_names = class_names or {i: f"class_{i}" for i in range(num_classes)}
        self.ignore_label = ignore_label
        
        # Initialize confusion matrix
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.total_points = 0
        self.matched_points = 0
    
    def compute_nn_correspondence(self, pred_points: np.ndarray, gt_points: np.ndarray,
                                 threshold: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """Find nearest neighbor correspondences between predicted and GT points.
        
        Args:
            pred_points: Predicted point cloud positions (N, 3)
            gt_points: Ground truth point cloud positions (M, 3)
            threshold: Maximum distance threshold for matching
            
        Returns:
            pred_indices: Indices of matched predicted points
            gt_indices: Indices of corresponding GT points
        """
        # Build KD-tree from GT points
        tree = KDTree(gt_points)
        
        # Query nearest neighbors
        distances, indices = tree.query(pred_points, k=1)
        distances = distances.flatten()
        indices = indices.flatten()
        
        # Filter by threshold
        valid_mask = distances < threshold
        pred_indices = np.where(valid_mask)[0]
        gt_indices = indices[valid_mask]
        
        return pred_indices, gt_indices
    
    def update(self, pred_points: np.ndarray, pred_labels: np.ndarray,
               gt_points: np.ndarray, gt_labels: np.ndarray,
               threshold: float = 0.05):
        """Update metrics with a new prediction-ground truth pair.
        
        Args:
            pred_points: Predicted point positions (N, 3)
            pred_labels: Predicted semantic labels (N,)
            gt_points: Ground truth point positions (M, 3)
            gt_labels: Ground truth semantic labels (M,)
            threshold: Distance threshold for point matching
        """
        # Find point correspondences
        pred_indices, gt_indices = self.compute_nn_correspondence(
            pred_points, gt_points, threshold
        )
        
        # Get matched labels
        matched_pred_labels = pred_labels[pred_indices]
        matched_gt_labels = gt_labels[gt_indices]
        
        # Filter out ignored labels
        valid_mask = matched_gt_labels != self.ignore_label
        matched_pred_labels = matched_pred_labels[valid_mask]
        matched_gt_labels = matched_gt_labels[valid_mask]
        
        # Update confusion matrix
        for pred_label, gt_label in zip(matched_pred_labels, matched_gt_labels):
            if 0 <= pred_label < self.num_classes and 0 <= gt_label < self.num_classes:
                self.confusion_matrix[gt_label, pred_label] += 1
        
        self.total_points += len(pred_points)
        self.matched_points += len(pred_indices)
    
    def compute_iou_per_class(self) -> Dict[int, float]:
        """Compute IoU for each class.
        
        Returns:
            Dictionary mapping class IDs to IoU values
        """
        iou_per_class = {}
        
        for class_id in range(self.num_classes):
            true_positive = self.confusion_matrix[class_id, class_id]
            false_positive = self.confusion_matrix[:, class_id].sum() - true_positive
            false_negative = self.confusion_matrix[class_id, :].sum() - true_positive
            
            denominator = true_positive + false_positive + false_negative
            if denominator > 0:
                iou = true_positive / denominator
            else:
                iou = 0.0
            
            iou_per_class[class_id] = iou
        
        return iou_per_class
    
    def compute_accuracy_per_class(self) -> Dict[int, float]:
        """Compute accuracy for each class.
        
        Returns:
            Dictionary mapping class IDs to accuracy values
        """
        acc_per_class = {}
        
        for class_id in range(self.num_classes):
            true_positive = self.confusion_matrix[class_id, class_id]
            total_gt = self.confusion_matrix[class_id, :].sum()
            
            if total_gt > 0:
                acc = true_positive / total_gt
            else:
                acc = 0.0
            
            acc_per_class[class_id] = acc
        
        return acc_per_class
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute all evaluation metrics.
        
        Returns:
            Dictionary containing all metrics
        """
        metrics = {}
        
        # Overall accuracy
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        metrics['overall_accuracy'] = correct / total if total > 0 else 0.0
        
        # Coverage (percentage of predicted points that matched GT)
        metrics['coverage'] = self.matched_points / self.total_points if self.total_points > 0 else 0.0
        
        # Per-class IoU
        iou_per_class = self.compute_iou_per_class()
        for class_id, iou in iou_per_class.items():
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            metrics[f'iou_{class_name}'] = iou
        
        # Mean IoU
        valid_ious = [iou for iou in iou_per_class.values() if iou > 0]
        metrics['mean_iou'] = np.mean(valid_ious) if valid_ious else 0.0
        
        # Per-class accuracy
        acc_per_class = self.compute_accuracy_per_class()
        for class_id, acc in acc_per_class.items():
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            metrics[f'accuracy_{class_name}'] = acc
        
        # Mean accuracy
        valid_accs = [acc for acc in acc_per_class.values() if acc > 0]
        metrics['mean_accuracy'] = np.mean(valid_accs) if valid_accs else 0.0
        
        return metrics
    
    def print_metrics(self):
        """Print evaluation metrics in a formatted way."""
        metrics = self.compute_metrics()
        
        print("\n=== 3D Semantic Segmentation Metrics ===")
        print(f"Overall Accuracy: {metrics['overall_accuracy']:.3f}")
        print(f"Point Coverage: {metrics['coverage']:.3f}")
        print(f"Mean IoU: {metrics['mean_iou']:.3f}")
        print(f"Mean Accuracy: {metrics['mean_accuracy']:.3f}")
        
        print("\nPer-class IoU:")
        for class_id in range(self.num_classes):
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            key = f'iou_{class_name}'
            if key in metrics and metrics[key] > 0:
                print(f"  {class_name}: {metrics[key]:.3f}")
        
        print("\nPer-class Accuracy:")
        for class_id in range(self.num_classes):
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            key = f'accuracy_{class_name}'
            if key in metrics and metrics[key] > 0:
                print(f"  {class_name}: {metrics[key]:.3f}")
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get the confusion matrix."""
        return self.confusion_matrix.copy()


def evaluate_semantic_point_cloud(pred_ply_path: str, gt_mesh_path: str,
                                 gt_face_labels: np.ndarray,
                                 class_mapping: Dict[str, int],
                                 num_classes: int,
                                 distance_threshold: float = 0.05) -> Dict[str, float]:
    """Evaluate a predicted semantic point cloud against ground truth mesh.
    
    Args:
        pred_ply_path: Path to predicted semantic point cloud PLY file
        gt_mesh_path: Path to ground truth mesh
        gt_face_labels: Ground truth face labels
        class_mapping: Mapping from predicted class names to GT class IDs
        num_classes: Total number of classes
        distance_threshold: Distance threshold for point matching
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Load predicted point cloud
    pred_pcd = o3d.io.read_point_cloud(pred_ply_path)
    pred_points = np.asarray(pred_pcd.points)
    
    # Extract predicted labels from PLY
    # This assumes labels are stored as a custom property
    # You may need to adjust based on your PLY format
    import plyfile
    plydata = plyfile.PlyData.read(pred_ply_path)
    
    if 'label' in plydata['vertex']:
        pred_labels = np.array(plydata['vertex']['label'])
    else:
        raise ValueError("No 'label' property found in predicted PLY file")
    
    # Load ground truth mesh and sample points
    gt_mesh = o3d.io.read_triangle_mesh(gt_mesh_path)
    
    # Sample points from GT mesh
    gt_pcd = gt_mesh.sample_points_uniformly(number_of_points=len(pred_points) * 2)
    gt_points = np.asarray(gt_pcd.points)
    
    # Get GT labels for sampled points
    # (This is simplified - you'd need to properly map sampled points to face labels)
    # For now, we'll use a placeholder
    gt_labels = np.zeros(len(gt_points), dtype=np.int32)
    
    # Initialize metrics
    metrics_calculator = Semantic3DMetrics(num_classes)
    
    # Update metrics
    metrics_calculator.update(pred_points, pred_labels, gt_points, gt_labels, distance_threshold)
    
    # Compute and return metrics
    return metrics_calculator.compute_metrics()


def test_metrics():
    """Test the metrics implementation with synthetic data."""
    # Create synthetic data
    num_points = 1000
    num_classes = 5
    
    # Random point clouds
    pred_points = np.random.randn(num_points, 3)
    gt_points = pred_points + np.random.randn(num_points, 3) * 0.01  # Small noise
    
    # Random labels with some correlation
    gt_labels = np.random.randint(0, num_classes, size=num_points)
    pred_labels = gt_labels.copy()
    # Add some errors
    error_mask = np.random.rand(num_points) < 0.2
    pred_labels[error_mask] = np.random.randint(0, num_classes, size=error_mask.sum())
    
    # Create metrics calculator
    metrics = Semantic3DMetrics(num_classes)
    
    # Update with synthetic data
    metrics.update(pred_points, pred_labels, gt_points, gt_labels, threshold=0.05)
    
    # Print results
    metrics.print_metrics()
    
    print("\nConfusion Matrix:")
    print(metrics.get_confusion_matrix())


if __name__ == "__main__":
    test_metrics()