#!/usr/bin/env python3
"""
Benchmark script to compare performance with and without adaptive semantic processing.
"""

import argparse
import time
import torch
import numpy as np
from pathlib import Path

def run_with_adaptive_processing(dataset_path, max_frames=100):
    """Run SLAM with adaptive semantic processing enabled."""
    import subprocess
    import os
    
    # Set environment variable to enable adaptive processing
    env = os.environ.copy()
    env['MAST3R_ADAPTIVE_SEMANTIC'] = '1'
    
    print("Running with ADAPTIVE semantic processing...")
    start_time = time.time()
    
    cmd = [
        'python', 'main.py',
        '--dataset', dataset_path,
        '--no-viz',
        '--save-as', 'adaptive_test'
    ]
    
    # Run for limited frames if specified
    if max_frames > 0:
        env['MAX_FRAMES'] = str(max_frames)
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    elapsed_time = time.time() - start_time
    
    # Parse output for timing information
    semantic_times = []
    fps_values = []
    
    for line in result.stdout.split('\n'):
        if 'Propagating semantics' in line:
            semantic_times.append('propagation')
        elif 'Running full semantics' in line or 'Processing semantics' in line:
            semantic_times.append('full')
        elif 'FPS:' in line:
            try:
                fps = float(line.split('FPS:')[1].strip())
                fps_values.append(fps)
            except:
                pass
    
    return {
        'total_time': elapsed_time,
        'semantic_breakdown': semantic_times,
        'fps_values': fps_values,
        'avg_fps': np.mean(fps_values) if fps_values else 0,
        'output': result.stdout,
        'errors': result.stderr
    }

def analyze_results(adaptive_results):
    """Analyze and print benchmark results."""
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    
    # Semantic processing breakdown
    total_frames = len(adaptive_results['semantic_breakdown'])
    full_processing = adaptive_results['semantic_breakdown'].count('full')
    propagation_only = adaptive_results['semantic_breakdown'].count('propagation')
    
    print(f"\nSemantic Processing Breakdown:")
    print(f"  Total frames processed: {total_frames}")
    print(f"  Full semantic processing: {full_processing} ({full_processing/max(1,total_frames)*100:.1f}%)")
    print(f"  Propagation only: {propagation_only} ({propagation_only/max(1,total_frames)*100:.1f}%)")
    
    # Performance metrics
    print(f"\nPerformance Metrics:")
    print(f"  Total runtime: {adaptive_results['total_time']:.2f}s")
    print(f"  Average FPS: {adaptive_results['avg_fps']:.2f}")
    
    # Estimated savings
    # Assume full processing takes 300ms average, propagation takes 15ms
    full_time_est = full_processing * 0.3
    prop_time_est = propagation_only * 0.015
    baseline_time_est = total_frames * 0.3
    
    time_saved = baseline_time_est - (full_time_est + prop_time_est)
    improvement = (time_saved / baseline_time_est) * 100 if baseline_time_est > 0 else 0
    
    print(f"\nEstimated Time Savings:")
    print(f"  Baseline (all full): {baseline_time_est:.2f}s")
    print(f"  With adaptive: {full_time_est + prop_time_est:.2f}s")
    print(f"  Time saved: {time_saved:.2f}s ({improvement:.1f}% improvement)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='datasets/tum/rgbd_dataset_freiburg1_desk')
    parser.add_argument('--max-frames', type=int, default=100, help='Maximum frames to process')
    args = parser.parse_args()
    
    print("MASt3R-SLAM Adaptive Semantic Processing Benchmark")
    print("="*60)
    
    # Check if dataset exists
    if not Path(args.dataset).exists():
        print(f"Error: Dataset not found at {args.dataset}")
        return
    
    # Run with adaptive processing
    adaptive_results = run_with_adaptive_processing(args.dataset, args.max_frames)
    
    # Analyze results
    analyze_results(adaptive_results)
    
    # Check for errors
    if adaptive_results['errors']:
        print(f"\nErrors encountered:")
        print(adaptive_results['errors'])

if __name__ == "__main__":
    main()