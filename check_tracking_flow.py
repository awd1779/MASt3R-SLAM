#!/usr/bin/env python3
"""Check the tracking flow to understand why only frame 0 is tracked."""

import json

# Load tracking decisions
try:
    with open('logs/logs/tracked_3d_global_improved_filtered/room_0/tracking_decisions_3d.json', 'r') as f:
        decisions = json.load(f)
    print(f"Total tracking decisions: {len(decisions)}")
    
    # Group by frame
    frames = {}
    for d in decisions:
        frame_id = d['frame_id']
        if frame_id not in frames:
            frames[frame_id] = []
        frames[frame_id].append(d)
    
    print(f"Frames with tracking decisions: {sorted(frames.keys())}")
    
    # Check each frame
    for frame_id in sorted(frames.keys()):
        print(f"\nFrame {frame_id}: {len(frames[frame_id])} decisions")
        labels = [d['label'] for d in frames[frame_id]]
        print(f"  Labels: {labels}")
except Exception as e:
    print(f"Error reading tracking decisions: {e}")

print("\n" + "="*60 + "\n")

# Check what semantic data was processed
from pathlib import Path
semantic_files = list(Path('debug_semantic_pipeline').glob('frame_*/segmentation_summary.txt'))
print(f"Semantic segmentation frames: {len(semantic_files)}")

# Extract which frames have tables
table_frames = []
for f in sorted(semantic_files):
    frame_num = int(f.parent.name.split('_')[1])
    with open(f, 'r') as file:
        content = file.read()
        if 'a table:' in content:
            table_frames.append(frame_num)

print(f"Frames with table segmentation: {table_frames}")

# Check logs for tracking activity
import subprocess
try:
    result = subprocess.run(['grep', '-n', 'Geometric3DTracker processing frame', '/dev/null'], 
                          capture_output=True, text=True)
    if result.stdout:
        print("\n=== Tracking activity from logs ===")
        print(result.stdout)
except:
    pass