#!/usr/bin/env python3
"""Comprehensive evaluation script for semantic SLAM system."""

import argparse
import json
import numpy as np
from pathlib import Path
import sys
import os
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.datasets import DatasetFactory
from evaluation.metrics import (
    InstanceSegmentationMetrics,
    TemporalConsistencyMetrics,
    BoundaryAccuracyMetrics,
    EfficiencyMetrics
)
from evaluation.semantic_3d_metrics import Semantic3DMetrics
from evaluation.realtime_evaluator import RealtimeEvaluator


def evaluate_semantic_3d_reconstruction(
    dataset,
    pred_ply_path: str,
    output_dir: str,
    distance_threshold: float = 0.05
) -> Dict:
    """Evaluate 3D semantic reconstruction quality.
    
    Args:
        dataset: Dataset loader instance
        pred_ply_path: Path to predicted semantic point cloud
        output_dir: Directory to save results
        distance_threshold: Distance threshold for point matching
        
    Returns:
        Dictionary of evaluation metrics
    """
    print("\n=== Evaluating 3D Semantic Reconstruction ===")
    
    # Load predicted point cloud
    from evaluation.evaluate_semantic_3d import load_predicted_semantic_cloud
    pred_points, pred_labels, pred_label_map = load_predicted_semantic_cloud(pred_ply_path)
    
    # Load ground truth
    gt_data = dataset.load_ground_truth_3d()
    gt_points = gt_data['points']
    gt_labels = gt_data['semantic_labels']
    
    # Get semantic classes
    semantic_classes = dataset.get_semantic_classes()
    num_classes = len(semantic_classes)
    
    # Initialize metrics
    metrics = Semantic3DMetrics(
        num_classes,
        class_names={i: name for i, name in enumerate(semantic_classes)}
    )
    
    # Compute metrics
    metrics.update(pred_points, pred_labels, gt_points, gt_labels, distance_threshold)
    
    # Get results
    results = metrics.compute_metrics()
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    metrics.save_results(os.path.join(output_dir, "semantic_3d_metrics.json"))
    
    # Print summary
    metrics.print_metrics()
    
    return results


def evaluate_instance_segmentation(
    dataset,
    predictions_dir: str,
    output_dir: str
) -> Dict:
    """Evaluate instance segmentation performance.
    
    Args:
        dataset: Dataset loader instance
        predictions_dir: Directory containing predicted instance masks
        output_dir: Directory to save results
        
    Returns:
        Dictionary of evaluation metrics
    """
    print("\n=== Evaluating Instance Segmentation ===")
    
    # Initialize metrics
    instance_metrics = InstanceSegmentationMetrics(output_dir=output_dir)
    
    # Process each frame
    for frame_idx in range(len(dataset)):
        frame_data = dataset[frame_idx]
        
        # Load predictions
        pred_path = Path(predictions_dir) / f"instance_{frame_data['frame_id']:06d}.npz"
        if not pred_path.exists():
            continue
        
        pred_data = np.load(pred_path)
        
        # Prepare data
        prediction = {
            'points': pred_data['points'],
            'instance_labels': pred_data['instance_labels'],
            'semantic_labels': pred_data.get('semantic_labels'),
            'confidence_scores': pred_data.get('confidence_scores', {})
        }
        
        ground_truth = {
            'points': frame_data.get('gt_points', np.array([])),
            'instance_labels': frame_data.get('instance', np.array([])),
            'semantic_labels': frame_data.get('semantic', np.array([]))
        }
        
        # Update metrics
        if len(prediction['points']) > 0 and len(ground_truth['points']) > 0:
            instance_metrics.update(prediction, ground_truth)
    
    # Get results
    results = instance_metrics.compute_metrics()
    
    # Save results
    instance_metrics.save_results()
    
    # Print summary
    instance_metrics.print_results()
    
    return results


def evaluate_temporal_consistency(
    dataset,
    predictions_dir: str,
    output_dir: str
) -> Dict:
    """Evaluate temporal consistency of predictions.
    
    Args:
        dataset: Dataset loader instance
        predictions_dir: Directory containing predictions
        output_dir: Directory to save results
        
    Returns:
        Dictionary of temporal metrics
    """
    print("\n=== Evaluating Temporal Consistency ===")
    
    # Initialize metrics
    temporal_metrics = TemporalConsistencyMetrics(output_dir=output_dir)
    
    # Process frames sequentially
    for frame_idx in range(len(dataset)):
        frame_data = dataset[frame_idx]
        
        # Load predictions
        pred_path = Path(predictions_dir) / f"pred_{frame_data['frame_id']:06d}.npz"
        if not pred_path.exists():
            continue
        
        pred_data = np.load(pred_path)
        
        # Extract instances
        pred_instances = []
        if 'instance_labels' in pred_data:
            unique_instances = np.unique(pred_data['instance_labels'])
            for inst_id in unique_instances:
                if inst_id > 0:
                    mask = pred_data['instance_labels'] == inst_id
                    pred_instances.append({
                        'track_id': inst_id,
                        'mask': mask,
                        'label': pred_data['semantic_labels'][mask][0] if 'semantic_labels' in pred_data else 0
                    })
        
        # Ground truth instances
        gt_instances = []
        if 'instance' in frame_data:
            unique_instances = np.unique(frame_data['instance'])
            for inst_id in unique_instances:
                if inst_id > 0:
                    mask = frame_data['instance'] == inst_id
                    gt_instances.append({
                        'track_id': inst_id,
                        'mask': mask,
                        'label': frame_data['semantic'][mask][0] if 'semantic' in frame_data else 0
                    })
        
        # Add frame to temporal metrics
        temporal_metrics.add_frame(
            frame_data['frame_id'],
            pred_instances,
            gt_instances
        )
    
    # Compute metrics
    results = temporal_metrics.compute({}, {})
    
    # Save results
    temporal_metrics.save_results()
    
    # Print summary
    temporal_metrics.print_results()
    
    # Visualize tracks
    temporal_metrics.visualize_tracks(os.path.join(output_dir, "tracks_visualization.png"))
    
    return results


def run_full_evaluation(args):
    """Run comprehensive evaluation."""
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print(f"\nLoading dataset: {args.dataset_type} - {args.scene_name}")
    dataset = DatasetFactory.create_dataset(
        args.dataset_type,
        args.dataset_path,
        args.scene_name,
        load_semantics=True,
        load_instances=True,
        subsample=args.subsample
    )
    
    # Validate dataset
    validation = dataset.validate_dataset()
    if not validation['is_valid']:
        print("Dataset validation failed:")
        for key, value in validation.items():
            print(f"  {key}: {value}")
        return
    
    print(f"Dataset loaded: {len(dataset)} frames")
    print(f"Semantic classes: {dataset.get_semantic_classes()}")
    
    # Export dataset metadata
    dataset.export_metadata(output_dir / "dataset_metadata.json")
    
    results = {}
    
    # 1. Evaluate 3D reconstruction if PLY provided
    if args.pred_ply:
        results['semantic_3d'] = evaluate_semantic_3d_reconstruction(
            dataset,
            args.pred_ply,
            output_dir / "semantic_3d",
            args.distance_threshold
        )
    
    # 2. Evaluate instance segmentation if predictions provided
    if args.predictions_dir and Path(args.predictions_dir).exists():
        results['instance_segmentation'] = evaluate_instance_segmentation(
            dataset,
            args.predictions_dir,
            output_dir / "instance_segmentation"
        )
        
        # 3. Evaluate temporal consistency
        results['temporal_consistency'] = evaluate_temporal_consistency(
            dataset,
            args.predictions_dir,
            output_dir / "temporal_consistency"
        )
    
    # 4. Run real-time evaluation simulation if requested
    if args.simulate_realtime:
        print("\n=== Simulating Real-time Evaluation ===")
        realtime_eval = RealtimeEvaluator(
            output_dir / "realtime",
            dataset.get_semantic_classes(),
            eval_interval=args.eval_interval,
            enable_plotting=args.show_plots
        )
        
        realtime_eval.start()
        
        # Simulate processing frames
        for frame_idx in range(min(len(dataset), args.max_frames)):
            frame_data = dataset[frame_idx]
            
            # Simulate processing time
            processing_time = np.random.uniform(0.02, 0.05)  # 20-50ms
            
            realtime_eval.add_frame(
                frame_id=frame_data['frame_id'],
                rgb=frame_data['rgb'],
                depth=frame_data.get('depth'),
                pred_semantic=frame_data.get('semantic'),  # Using GT as pred for simulation
                pred_instance=frame_data.get('instance'),
                gt_semantic=frame_data.get('semantic'),
                gt_instance=frame_data.get('instance'),
                pose=frame_data['pose'],
                processing_time=processing_time
            )
        
        realtime_eval.stop()
    
    # 5. Generate final report
    generate_evaluation_report(results, output_dir)
    
    print(f"\nEvaluation complete! Results saved to: {output_dir}")


def generate_evaluation_report(results: Dict, output_dir: Path):
    """Generate comprehensive evaluation report."""
    report_path = output_dir / "evaluation_report.json"
    
    # Save full results
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate summary
    summary_path = output_dir / "evaluation_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("=== Semantic SLAM Evaluation Summary ===\n\n")
        
        if 'semantic_3d' in results:
            f.write("3D Semantic Reconstruction:\n")
            f.write(f"  Mean IoU: {results['semantic_3d'].get('mean_iou', 0):.3f}\n")
            f.write(f"  Coverage: {results['semantic_3d'].get('coverage', 0):.3f}\n\n")
        
        if 'instance_segmentation' in results:
            f.write("Instance Segmentation:\n")
            f.write(f"  mAP: {results['instance_segmentation'].get('mAP', 0):.3f}\n")
            f.write(f"  mAP@50: {results['instance_segmentation'].get('mAP_50', 0):.3f}\n\n")
        
        if 'temporal_consistency' in results:
            f.write("Temporal Consistency:\n")
            f.write(f"  MOTA: {results['temporal_consistency'].get('MOTA', 0):.3f}\n")
            f.write(f"  Track Purity: {results['temporal_consistency'].get('track_purity', 0):.3f}\n\n")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive semantic SLAM evaluation')
    
    # Dataset arguments
    parser.add_argument('--dataset_type', type=str, required=True,
                       choices=['replica', 'scannet', 'kitti360', 'tum_rgbd'],
                       help='Type of dataset')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to dataset root')
    parser.add_argument('--scene_name', type=str, required=True,
                       help='Scene name to evaluate')
    parser.add_argument('--subsample', type=int, default=1,
                       help='Frame subsampling factor')
    
    # Evaluation inputs
    parser.add_argument('--pred_ply', type=str,
                       help='Path to predicted semantic point cloud PLY')
    parser.add_argument('--predictions_dir', type=str,
                       help='Directory containing frame-wise predictions')
    
    # Evaluation parameters
    parser.add_argument('--distance_threshold', type=float, default=0.05,
                       help='Distance threshold for 3D point matching (meters)')
    parser.add_argument('--eval_interval', type=int, default=30,
                       help='Frames between evaluations')
    
    # Real-time simulation
    parser.add_argument('--simulate_realtime', action='store_true',
                       help='Simulate real-time evaluation')
    parser.add_argument('--max_frames', type=int, default=1000,
                       help='Maximum frames for real-time simulation')
    parser.add_argument('--show_plots', action='store_true',
                       help='Show live plots during evaluation')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    run_full_evaluation(args)


if __name__ == "__main__":
    main()