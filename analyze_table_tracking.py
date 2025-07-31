#!/usr/bin/env python3
"""Analyze table tracking across frames to debug fragmentation issue."""

import json
import numpy as np

# Load tracking decisions
with open('logs/logs/tracked_3d_global_improved_filtered/room_0/tracking_decisions_3d.json', 'r') as f:
    decisions = json.load(f)

# Load tracking mapping
with open('logs/logs/tracked_3d_global_improved_filtered/room_0/room_0_semantic_dense_tracked_3d.tracking.json', 'r') as f:
    tracking_info = json.load(f)

# Analyze table tracking
table_decisions = []
table_instances_by_frame = {}

for decision in decisions:
    if decision['label'] == 'a table':
        table_decisions.append(decision)
        frame_id = decision['frame_id']
        if frame_id not in table_instances_by_frame:
            table_instances_by_frame[frame_id] = []
        table_instances_by_frame[frame_id].append(decision)

print("=== Table Tracking Analysis ===")
print(f"Total table detections: {len(table_decisions)}")
print(f"Frames with tables: {sorted(table_instances_by_frame.keys())}")

# Check for unique track IDs
track_ids = set(d['track_id'] for d in table_decisions)
print(f"Unique table track IDs: {track_ids}")

# Analyze by frame
print("\n=== Table Tracking by Frame ===")
for frame_id in sorted(table_instances_by_frame.keys()):
    instances = table_instances_by_frame[frame_id]
    print(f"\nFrame {frame_id}:")
    for inst in instances:
        print(f"  Instance {inst['instance_id']} -> Track {inst['track_id']} (method: {inst['method']})")

# Check if track 8 is consistently used
track_8_frames = [d['frame_id'] for d in table_decisions if d['track_id'] == 8]
print(f"\n=== Track 8 (table) appears in frames: {sorted(track_8_frames)} ===")

# Load semantic segmentation data to check sizes
import os
import glob

print("\n=== Table Segmentation Sizes by Frame ===")
for frame_dir in sorted(glob.glob('debug_semantic_pipeline/frame_*')):
    frame_num = int(os.path.basename(frame_dir).split('_')[1])
    summary_file = os.path.join(frame_dir, 'segmentation_summary.txt')
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            for line in f:
                if 'a table:' in line:
                    print(f"Frame {frame_num}: {line.strip()}")