#!/usr/bin/env python3
"""Main script to evaluate 3D semantic segmentation on Replica dataset."""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
import open3d as o3d
from typing import Dict, List, Tuple
import yaml

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.replica_loader import ReplicaSemanticLoader
from evaluation.semantic_3d_metrics import Semantic3DMetrics


def load_predicted_semantic_cloud(ply_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Load predicted semantic point cloud from PLY file.
    
    Args:
        ply_path: Path to PLY file from semantic SLAM system
        
    Returns:
        points: (N, 3) array of 3D points
        labels: (N,) array of semantic labels
        label_map: Dictionary mapping label names to IDs
    """
    import plyfile
    
    # Read PLY file
    plydata = plyfile.PlyData.read(ply_path)
    vertex = plydata['vertex']
    
    # Extract points
    points = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
    
    # Extract labels
    if 'label' in vertex:
        labels = np.array(vertex['label'])
    else:
        raise ValueError(f"No 'label' property found in {ply_path}")
    
    # Load label mapping from accompanying text file
    txt_path = ply_path.replace('.ply', '.txt')
    label_map = {}
    
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            
        # Parse label mapping from statistics file
        for line in lines:
            if 'Label' in line and ':' in line:
                # Format: "Label N: class_name (X points, Y.Z%)"
                parts = line.strip().split(':')
                if len(parts) >= 2:
                    label_id = int(parts[0].split()[-1])
                    class_name = parts[1].split('(')[0].strip()
                    # Remove instance number if present (e.g., "chair_1" -> "chair")
                    if '_' in class_name and class_name.split('_')[-1].isdigit():
                        class_name = '_'.join(class_name.split('_')[:-1])
                    label_map[class_name] = label_id
    
    return points, labels, label_map


def create_vocabulary_mapping(pred_vocabulary: List[str], gt_vocabulary: List[str]) -> Dict[str, str]:
    """Create mapping between predicted and ground truth vocabularies.
    
    For zero-shot evaluation, this should ideally be identity mapping,
    but we handle cases where vocabularies don't match exactly.
    
    Args:
        pred_vocabulary: List of predicted class names
        gt_vocabulary: List of ground truth class names
        
    Returns:
        Dictionary mapping predicted names to GT names
    """
    mapping = {}
    
    # First, try exact matches
    for pred_name in pred_vocabulary:
        if pred_name in gt_vocabulary:
            mapping[pred_name] = pred_name
        else:
            # Try case-insensitive match
            for gt_name in gt_vocabulary:
                if pred_name.lower() == gt_name.lower():
                    mapping[pred_name] = gt_name
                    break
    
    # Handle unmatched predictions
    for pred_name in pred_vocabulary:
        if pred_name not in mapping:
            print(f"Warning: No match found for predicted class '{pred_name}'")
            mapping[pred_name] = None
    
    return mapping


def evaluate_scene(pred_ply_path: str, replica_scene_path: str, 
                  output_dir: str, distance_threshold: float = 0.05) -> Dict:
    """Evaluate semantic segmentation for one scene.
    
    Args:
        pred_ply_path: Path to predicted semantic point cloud
        replica_scene_path: Path to Replica scene directory
        output_dir: Directory to save evaluation results
        distance_threshold: Distance threshold for point matching (in meters)
        
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"\nEvaluating scene: {replica_scene_path}")
    print(f"Prediction file: {pred_ply_path}")
    
    # Load Replica ground truth
    replica_loader = ReplicaSemanticLoader(replica_scene_path)
    gt_vocabulary = replica_loader.get_vocabulary_for_zero_shot()
    print(f"\nGround truth vocabulary ({len(gt_vocabulary)} classes):")
    for i, name in enumerate(gt_vocabulary):
        print(f"  {i}: {name}")
    
    # Load predicted semantic point cloud
    pred_points, pred_labels, pred_label_map = load_predicted_semantic_cloud(pred_ply_path)
    pred_vocabulary = list(pred_label_map.keys())
    print(f"\nPredicted vocabulary ({len(pred_vocabulary)} classes):")
    for name, label_id in pred_label_map.items():
        print(f"  {label_id}: {name}")
    
    # Create vocabulary mapping
    vocab_mapping = create_vocabulary_mapping(pred_vocabulary, gt_vocabulary)
    
    # Load ground truth mesh and labels
    gt_mesh, gt_face_labels = replica_loader.load_semantic_mesh_with_labels()
    
    # Sample points from GT mesh
    num_gt_points = len(pred_points) * 2  # Sample more GT points for better coverage
    gt_points, gt_class_labels = replica_loader.sample_points_from_mesh(
        gt_mesh, gt_face_labels, num_points=num_gt_points
    )
    
    # Create mapping from predicted labels to GT class IDs
    # First, create GT vocabulary to class ID mapping
    gt_class_map = {name: i for i, name in enumerate(gt_vocabulary)}
    
    # Map predicted labels to GT class IDs
    pred_labels_mapped = np.zeros_like(pred_labels)
    for pred_name, pred_id in pred_label_map.items():
        gt_name = vocab_mapping.get(pred_name)
        if gt_name and gt_name in gt_class_map:
            gt_class_id = gt_class_map[gt_name]
            mask = pred_labels == pred_id
            pred_labels_mapped[mask] = gt_class_id
        else:
            # Unknown class
            mask = pred_labels == pred_id
            pred_labels_mapped[mask] = -1
    
    # Initialize metrics calculator
    num_classes = len(gt_vocabulary)
    metrics_calc = Semantic3DMetrics(num_classes, class_names=gt_class_map)
    
    # Update metrics
    metrics_calc.update(pred_points, pred_labels_mapped, gt_points, gt_class_labels, 
                       threshold=distance_threshold)
    
    # Compute metrics
    metrics = metrics_calc.compute_metrics()
    
    # Print results
    metrics_calc.print_metrics()
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics to JSON
    metrics_file = os.path.join(output_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save confusion matrix
    confusion_matrix = metrics_calc.get_confusion_matrix()
    np.save(os.path.join(output_dir, 'confusion_matrix.npy'), confusion_matrix)
    
    # Save vocabulary mapping
    mapping_file = os.path.join(output_dir, 'vocabulary_mapping.json')
    with open(mapping_file, 'w') as f:
        json.dump(vocab_mapping, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate 3D semantic segmentation on Replica dataset')
    parser.add_argument('--pred_ply', type=str, required=True,
                       help='Path to predicted semantic point cloud PLY file')
    parser.add_argument('--replica_scene', type=str, required=True,
                       help='Path to Replica scene directory (e.g., /path/to/apartment_0)')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--distance_threshold', type=float, default=0.05,
                       help='Distance threshold for point matching (in meters)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to semantic SLAM config file (to update vocabulary)')
    
    args = parser.parse_args()
    
    # If config is provided, suggest updating vocabulary for zero-shot evaluation
    if args.config:
        print("\nFor zero-shot evaluation, update your semantic SLAM config with Replica vocabulary:")
        print(f"Config file: {args.config}")
        
        # Load Replica vocabulary
        replica_loader = ReplicaSemanticLoader(args.replica_scene)
        vocabulary = replica_loader.get_vocabulary_for_zero_shot()
        
        print("\nSuggested vocabulary for config:")
        print("semantic_classes:")
        for class_name in vocabulary:
            print(f'  - "{class_name}"')
        
        print("\nRerun semantic SLAM with this vocabulary for true zero-shot evaluation.")
        
        # Ask user if they want to continue with current predictions
        response = input("\nContinue evaluation with current predictions? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Run evaluation
    metrics = evaluate_scene(
        args.pred_ply,
        args.replica_scene,
        args.output_dir,
        args.distance_threshold
    )
    
    print("\n=== Evaluation Complete ===")
    print(f"Mean IoU: {metrics['mean_iou']:.3f}")
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.3f}")


if __name__ == "__main__":
    main()