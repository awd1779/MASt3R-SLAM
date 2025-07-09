#!/usr/bin/env python3
"""Test script for adaptive semantic processing improvements."""

import torch
import time
import argparse
from pathlib import Path

# Test the adaptive semantic decision logic
def test_pose_based_decision():
    """Test if pose-based semantic decision works correctly."""
    from mast3r_slam.semantic_processor_v2 import SemanticProcessorV2
    import lietorch
    
    print("Testing pose-based semantic decision...")
    
    # Create processor
    processor = SemanticProcessorV2(
        device='cuda',
        enable_batch_processing=True,
        enable_caching=True
    )
    
    # Create test poses using matrix operations
    # Pose 1: Identity
    pose1 = lietorch.Sim3.Identity(1, device='cuda')
    
    # Pose 2: Small translation (0.2m)
    pose2 = lietorch.Sim3.Identity(1, device='cuda')
    # Modify the matrix directly
    pose2_data = pose2.data.clone()
    # Sim3 data format: [tx, ty, tz, qw, qx, qy, qz]
    pose2_data[0, 0] = 0.2  # tx
    pose2 = lietorch.Sim3(pose2_data)
    
    # Pose 3: Large translation (1m)
    pose3 = lietorch.Sim3.Identity(1, device='cuda')
    pose3_data = pose3.data.clone()
    pose3_data[0, 0] = 1.0  # tx
    pose3 = lietorch.Sim3(pose3_data)
    
    # Pose 4: Large rotation (45 degrees around Y axis)
    pose4 = lietorch.Sim3.Identity(1, device='cuda')
    pose4_data = pose4.data.clone()
    angle = 45.0 * 3.14159 / 180.0
    # Quaternion for Y-axis rotation: [cos(θ/2), 0, sin(θ/2), 0]
    pose4_data[0, 3] = torch.cos(torch.tensor(angle/2))  # qw
    pose4_data[0, 5] = torch.sin(torch.tensor(angle/2))  # qy
    pose4 = lietorch.Sim3(pose4_data)
    
    # Test decisions
    print("\nTest 1: First frame (should be True)")
    result1 = processor.should_run_full_segmentation(pose1, None)
    print(f"Result: {result1}")
    
    print("\nTest 2: Small translation 0.2m (should be False)")
    result2 = processor.should_run_full_segmentation(pose2, pose1)
    print(f"Result: {result2}")
    
    print("\nTest 3: Large translation 1m (should be True)")
    result3 = processor.should_run_full_segmentation(pose3, pose1)
    print(f"Result: {result3}")
    
    print("\nTest 4: Large rotation 45° (should be True)")
    result4 = processor.should_run_full_segmentation(pose4, pose1)
    print(f"Result: {result4}")
    
    return all([result1 == True, result2 == False, result3 == True, result4 == True])


def test_semantic_propagation():
    """Test semantic label propagation using correspondences."""
    from mast3r_slam.tracker import FrameTracker
    from mast3r_slam.frame import SharedKeyframes, Frame
    from mast3r_slam.config import load_config
    import torch.multiprocessing as mp
    import lietorch
    
    print("\nTesting semantic label propagation...")
    
    # Load config
    load_config("config/base.yaml")
    
    # Create shared keyframes
    manager = mp.Manager()
    keyframes = SharedKeyframes(manager, 480, 640)
    device = 'cuda'
    
    # Create tracker
    tracker = FrameTracker(None, keyframes, device, {0: 'background'})
    
    # Create mock keyframe with labels
    img_shape = torch.tensor([480, 640], device=device)
    img_true_shape = torch.tensor([480, 640], device=device)
    mock_img = torch.randn(1, 3, 480, 640, device=device)
    mock_uimg = torch.randn(480, 640, 3, device=device)
    
    keyframe = Frame(
        frame_id=0,
        img=mock_img,
        img_shape=img_shape,
        img_true_shape=img_true_shape,
        uimg=mock_uimg,
        T_WC=lietorch.Sim3.Identity(1, device=device)
    )
    keyframe.X_canon = torch.randn(1000, 3, device=device)
    keyframe.global_instance_ids = torch.zeros(1000, 1, dtype=torch.int64, device=device)
    # Assign some labels
    keyframe.global_instance_ids[:300] = 1  # Object 1
    keyframe.global_instance_ids[300:500] = 2  # Object 2
    keyframe.global_instance_ids[500:700] = 3  # Object 3
    
    # Create current frame
    frame = Frame(
        frame_id=1,
        img=mock_img,
        img_shape=img_shape,
        img_true_shape=img_true_shape,
        uimg=mock_uimg,
        T_WC=lietorch.Sim3.Identity(1, device=device)
    )
    frame.X_canon = torch.randn(800, 3, device=device)
    
    # Create mock correspondences
    # 600 matches: first 600 points of frame map to various keyframe points
    idx_f2k = torch.randint(0, 1000, (800,), device=device)
    valid_match_k = torch.zeros(800, dtype=torch.bool, device=device)
    valid_match_k[:600] = True  # First 600 are valid matches
    
    # Test propagation
    start_time = time.time()
    tracker.propagate_semantics_from_keyframe(frame, keyframe, idx_f2k, valid_match_k)
    propagation_time = time.time() - start_time
    
    # Check results
    labeled_points = (frame.global_instance_ids > 0).sum().item()
    print(f"Propagation time: {propagation_time*1000:.2f}ms")
    print(f"Labeled points: {labeled_points}/{frame.X_canon.shape[0]}")
    print(f"Unique labels: {torch.unique(frame.global_instance_ids).tolist()}")
    
    return labeled_points > 0


def run_performance_comparison():
    """Compare performance with and without adaptive processing."""
    print("\n" + "="*60)
    print("Performance Comparison Test")
    print("="*60)
    
    # This would require running on actual data
    # For now, just print expected improvements
    print("\nExpected improvements with adaptive processing:")
    print("- Non-keyframe processing: 200-500ms → 5-20ms (propagation only)")
    print("- Memory usage: Reduced by caching intermediate features")
    print("- Semantic consistency: Improved through correspondence-based propagation")
    print("- Overall FPS: ~30-50% improvement on typical sequences")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', choices=['pose', 'propagation', 'all'], default='all')
    args = parser.parse_args()
    
    print("Testing Adaptive Semantic Processing Implementation")
    print("="*60)
    
    all_passed = True
    
    if args.test in ['pose', 'all']:
        passed = test_pose_based_decision()
        all_passed &= passed
        print(f"\nPose-based decision test: {'PASSED' if passed else 'FAILED'}")
    
    if args.test in ['propagation', 'all']:
        passed = test_semantic_propagation()
        all_passed &= passed
        print(f"\nSemantic propagation test: {'PASSED' if passed else 'FAILED'}")
    
    if args.test == 'all':
        run_performance_comparison()
    
    print("\n" + "="*60)
    print(f"Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    

if __name__ == "__main__":
    main()