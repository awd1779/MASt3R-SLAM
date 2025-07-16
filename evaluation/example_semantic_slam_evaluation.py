#!/usr/bin/env python3
"""Example script showing how to integrate the enhanced evaluation framework with semantic SLAM."""

import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.realtime_evaluator import RealtimeEvaluator
from evaluation.datasets import DatasetFactory
from evaluation.run_comprehensive_evaluation import evaluate_semantic_3d_reconstruction


def example_realtime_evaluation():
    """Example of real-time evaluation during SLAM execution."""
    print("=== Real-time Evaluation Example ===\n")
    
    # Define semantic classes (should match your SLAM configuration)
    semantic_classes = ["wall", "floor", "chair", "table", "door", "window", "bed", "sofa"]
    
    # Initialize real-time evaluator
    evaluator = RealtimeEvaluator(
        output_dir="./example_realtime_eval",
        semantic_classes=semantic_classes,
        eval_interval=30,  # Evaluate every 30 frames
        plot_interval=60,  # Update plots every 60 frames
        enable_plotting=True
    )
    
    # Start evaluation thread
    evaluator.start()
    
    # Simulate SLAM processing
    print("Simulating SLAM processing...")
    for frame_id in range(300):  # Process 300 frames
        # Simulate frame data (in real use, this comes from your SLAM system)
        h, w = 480, 640
        
        # Simulate predictions (random for example)
        pred_semantic = np.random.randint(0, len(semantic_classes), (h, w))
        pred_instance = np.random.randint(0, 10, (h, w))
        
        # Simulate ground truth (similar to predictions with some noise)
        gt_semantic = pred_semantic.copy()
        gt_semantic[np.random.rand(h, w) > 0.9] = np.random.randint(0, len(semantic_classes))
        gt_instance = pred_instance.copy()
        
        # Simulate other data
        rgb = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        depth = np.random.rand(h, w) * 5.0  # 0-5 meters
        pose = np.eye(4)  # Identity for simplicity
        processing_time = np.random.uniform(0.02, 0.04)  # 20-40ms
        
        # Add frame to evaluator
        evaluator.add_frame(
            frame_id=frame_id,
            rgb=rgb,
            depth=depth,
            pred_semantic=pred_semantic,
            pred_instance=pred_instance,
            gt_semantic=gt_semantic,
            gt_instance=gt_instance,
            pose=pose,
            processing_time=processing_time
        )
        
        # Print progress
        if frame_id % 50 == 0:
            print(f"Processed {frame_id} frames...")
    
    # Stop evaluation and save results
    evaluator.stop()
    print("\nReal-time evaluation complete!")
    print(f"Results saved to: ./example_realtime_eval")


def example_offline_evaluation():
    """Example of offline evaluation on saved results."""
    print("\n=== Offline Evaluation Example ===\n")
    
    # Load dataset (using Replica as example)
    dataset = DatasetFactory.create_dataset(
        dataset_type="replica",
        dataset_path="/path/to/replica",  # Update this path
        scene_name="apartment_0",
        trajectory_name="circular"  # If using generated trajectory
    )
    
    print(f"Loaded dataset: {len(dataset)} frames")
    print(f"Semantic classes: {dataset.get_semantic_classes()}")
    
    # Evaluate 3D reconstruction
    if Path("output/semantic_dense.ply").exists():
        results = evaluate_semantic_3d_reconstruction(
            dataset=dataset,
            pred_ply_path="output/semantic_dense.ply",
            output_dir="./example_3d_eval",
            distance_threshold=0.05
        )
        
        print(f"\n3D Evaluation Results:")
        print(f"  Mean IoU: {results['mean_iou']:.3f}")
        print(f"  Coverage: {results['coverage']:.3f}")
    else:
        print("No semantic point cloud found. Run SLAM first to generate output/semantic_dense.ply")


def example_custom_metrics():
    """Example of using individual metric modules."""
    print("\n=== Custom Metrics Example ===\n")
    
    from evaluation.metrics import (
        InstanceSegmentationMetrics,
        TemporalConsistencyMetrics,
        BoundaryAccuracyMetrics,
        EfficiencyMetrics
    )
    
    # Instance segmentation metrics
    instance_metrics = InstanceSegmentationMetrics()
    
    # Example data
    pred_points = np.random.rand(1000, 3)
    pred_instances = np.random.randint(0, 5, 1000)
    gt_points = pred_points + np.random.randn(1000, 3) * 0.01
    gt_instances = pred_instances.copy()
    gt_instances[::10] = np.random.randint(0, 5)  # Add some errors
    
    # Compute metrics
    results = instance_metrics.compute(
        prediction={
            'points': pred_points,
            'instance_labels': pred_instances,
            'confidence_scores': {i: 0.9 for i in range(5)}
        },
        ground_truth={
            'points': gt_points,
            'instance_labels': gt_instances
        }
    )
    
    print("Instance Segmentation Metrics:")
    print(f"  mAP: {results['mAP']:.3f}")
    print(f"  mAP@50: {results['mAP_50']:.3f}")
    
    # Efficiency metrics
    efficiency_metrics = EfficiencyMetrics()
    
    # Simulate processing times
    for _ in range(100):
        efficiency_metrics.record_frame_time(np.random.uniform(0.02, 0.04))
    
    fps_metrics = efficiency_metrics.compute_fps_metrics()
    print(f"\nEfficiency Metrics:")
    print(f"  Mean FPS: {fps_metrics['fps_mean']:.1f}")
    print(f"  FPS Std: {fps_metrics['fps_std']:.1f}")


def integration_with_semantic_slam():
    """Example showing how to integrate with the semantic SLAM system."""
    print("\n=== Integration with Semantic SLAM ===\n")
    
    example_code = '''
# In your main_semantic.py or similar:

from evaluation.realtime_evaluator import RealtimeEvaluator

# After initializing SLAM system
evaluator = RealtimeEvaluator(
    output_dir=f"./evaluation/{args.scene_name}",
    semantic_classes=vocabulary,  # Your SLAM vocabulary
    eval_interval=30,
    enable_plotting=args.show_eval_plots
)
evaluator.start()

# In your main loop where you process frames:
for frame_idx, (image_np, depth_np, K, T_WC, gt_data) in enumerate(dataloader):
    # Your SLAM processing
    tracking_result = slam.track_frame(image_np, depth_np, K)
    
    # Get semantic results from your semantic processor
    if semantic_result_queue:
        semantic_data = semantic_result_queue.get()
        pred_semantic = decode_semantic_masks(semantic_data)
        pred_instance = decode_instance_masks(semantic_data)
    
    # Add to evaluator
    evaluator.add_frame(
        frame_id=frame_idx,
        rgb=image_np,
        depth=depth_np,
        pred_semantic=pred_semantic,
        pred_instance=pred_instance,
        gt_semantic=gt_data.get('semantic'),
        gt_instance=gt_data.get('instance'),
        pose=T_WC,
        processing_time=tracking_result.time_taken
    )

# After SLAM completes
evaluator.stop()

# For offline evaluation of final reconstruction
from evaluation.run_comprehensive_evaluation import run_full_evaluation
run_full_evaluation(args)
'''
    
    print("Integration example:")
    print(example_code)


def main():
    """Run all examples."""
    print("Enhanced Semantic SLAM Evaluation Framework Examples")
    print("=" * 50)
    
    # Run examples
    example_realtime_evaluation()
    # example_offline_evaluation()  # Uncomment when dataset path is set
    example_custom_metrics()
    integration_with_semantic_slam()
    
    print("\n" + "=" * 50)
    print("Examples complete! Check the output directories for results.")
    print("\nFor full documentation, see: evaluation/ENHANCED_EVALUATION_FRAMEWORK.md")


if __name__ == "__main__":
    main()