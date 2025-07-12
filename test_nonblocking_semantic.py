#!/usr/bin/env python3
"""
Test script to verify non-blocking semantic SLAM operation.
This script monitors queue sizes and timing to ensure semantic processing
doesn't block the main SLAM pipeline.
"""

import subprocess
import time
import re
import sys
from datetime import datetime


def parse_log_line(line):
    """Extract relevant metrics from log lines."""
    metrics = {}
    
    # Parse FPS
    fps_match = re.search(r'FPS: ([\d.]+)', line)
    if fps_match:
        metrics['fps'] = float(fps_match.group(1))
    
    # Parse semantic queue sizes
    queue_match = re.search(r'Semantic queues: input=(\d+), output=(\d+)', line)
    if queue_match:
        metrics['input_queue'] = int(queue_match.group(1))
        metrics['output_queue'] = int(queue_match.group(2))
    
    # Parse keyframe sends
    kf_send_match = re.search(r'Sent (?:initial )?keyframe (\d+) \(frame (\d+)\) to semantic processor', line)
    if kf_send_match:
        metrics['keyframe_sent'] = {
            'kf_idx': int(kf_send_match.group(1)),
            'frame_id': int(kf_send_match.group(2))
        }
    
    # Parse semantic processing
    semantic_match = re.search(r'Processing semantic frame (\d+) \(keyframe (\d+|None)\)', line)
    if semantic_match:
        metrics['semantic_processing'] = {
            'frame_id': int(semantic_match.group(1)),
            'kf_idx': semantic_match.group(2) if semantic_match.group(2) != 'None' else None
        }
    
    # Parse result processing
    result_match = re.search(r'Processed (\d+) semantic results', line)
    if result_match:
        metrics['results_processed'] = int(result_match.group(1))
    
    return metrics


def run_test(dataset_path, config_path, duration=60):
    """Run the semantic SLAM and monitor its performance."""
    
    print(f"Starting non-blocking semantic SLAM test...")
    print(f"Dataset: {dataset_path}")
    print(f"Config: {config_path}")
    print(f"Test duration: {duration} seconds")
    print("-" * 60)
    
    # Start the process
    cmd = [
        'python', 'main_semantic.py',
        '--dataset', dataset_path,
        '--config', config_path,
        '--no-viz',
        '-v'  # Verbose mode to get detailed logs
    ]
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Monitoring variables
    start_time = time.time()
    fps_values = []
    queue_sizes = []
    keyframes_sent = 0
    semantics_processed = 0
    max_queue_size = 0
    
    try:
        # Monitor the process output
        while True:
            line = process.stdout.readline()
            if not line:
                break
                
            # Print the line
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {line.strip()}")
            
            # Parse metrics
            metrics = parse_log_line(line)
            
            if 'fps' in metrics:
                fps_values.append(metrics['fps'])
            
            if 'input_queue' in metrics:
                queue_sizes.append({
                    'time': time.time() - start_time,
                    'input': metrics['input_queue'],
                    'output': metrics['output_queue']
                })
                max_queue_size = max(max_queue_size, metrics['input_queue'], metrics['output_queue'])
            
            if 'keyframe_sent' in metrics:
                keyframes_sent += 1
            
            if 'results_processed' in metrics:
                semantics_processed = metrics['results_processed']
            
            # Check if we've run long enough
            if time.time() - start_time > duration:
                print("\nTest duration reached, terminating...")
                process.terminate()
                break
                
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        process.terminate()
    
    # Wait for process to finish
    process.wait()
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if fps_values:
        avg_fps = sum(fps_values) / len(fps_values)
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Min FPS: {min(fps_values):.2f}")
        print(f"Max FPS: {max(fps_values):.2f}")
    
    print(f"\nKeyframes sent to semantic: {keyframes_sent}")
    print(f"Semantic results processed: {semantics_processed}")
    print(f"Max queue size observed: {max_queue_size}")
    
    if queue_sizes:
        avg_input_queue = sum(q['input'] for q in queue_sizes) / len(queue_sizes)
        avg_output_queue = sum(q['output'] for q in queue_sizes) / len(queue_sizes)
        print(f"Average input queue size: {avg_input_queue:.1f}")
        print(f"Average output queue size: {avg_output_queue:.1f}")
    
    # Check for non-blocking behavior
    print("\n" + "-" * 60)
    print("NON-BLOCKING VERIFICATION:")
    
    if fps_values and avg_fps > 10:
        print("✓ SLAM maintained good FPS - Non-blocking confirmed")
    else:
        print("✗ Low FPS detected - Possible blocking")
    
    if max_queue_size < 50:
        print("✓ Queue sizes remained reasonable")
    else:
        print("✗ High queue sizes detected - Possible backlog")
    
    if keyframes_sent > 0 and semantics_processed > 0:
        print("✓ Semantic processing is active")
    else:
        print("✗ No semantic processing detected")


if __name__ == "__main__":
    # Default test parameters
    dataset = "datasets/tum/rgbd_dataset_freiburg1_desk"
    config = "config/base.yaml"
    
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    if len(sys.argv) > 2:
        config = sys.argv[2]
    
    run_test(dataset, config, duration=30)