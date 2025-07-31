#!/usr/bin/env python3
"""Debug world coordinate consistency across frames."""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze_world_positions():
    """Analyze world positions from tracking decisions and semantic data."""
    
    # Read tracking decisions
    tracking_file = "/home/ubuntu/restart_from_scratch/logs/logs/tracked_3d_global_improved_filtered/room_0/tracking_decisions_3d.json"
    with open(tracking_file, 'r') as f:
        decisions = json.load(f)
    
    # Read semantic data for each keyframe to get positions
    semantic_files = [
        "/home/ubuntu/restart_from_scratch/logs/logs/tracked_3d_global_improved_filtered/room_0/semantic_frame_000000.json",
        "/home/ubuntu/restart_from_scratch/logs/logs/tracked_3d_global_improved_filtered/room_0/semantic_frame_000002.json", 
        "/home/ubuntu/restart_from_scratch/logs/logs/tracked_3d_global_improved_filtered/room_0/semantic_frame_000004.json"
    ]
    
    frame_data = {}
    for i, filepath in enumerate(semantic_files):
        try:
            with open(filepath, 'r') as f:
                frame_data[i*2] = json.load(f)
                print(f"Loaded frame {i*2} semantic data")
        except:
            print(f"Could not load {filepath}")
    
    # Find all table detections across frames
    table_detections = []
    
    for decision in decisions:
        if decision['label'] == 'a table':
            frame_id = decision['frame_id']
            instance_id = decision['instance_id']
            track_id = decision['track_id']
            
            # Try to find world position in semantic data
            if frame_id in frame_data:
                # Check if world positions are stored
                world_positions = frame_data[frame_id].get('world_positions', {})
                if str(instance_id) in world_positions:
                    pos = world_positions[str(instance_id)]
                    table_detections.append({
                        'frame': frame_id,
                        'instance': instance_id,
                        'track': track_id,
                        'world_pos': pos
                    })
                    print(f"Frame {frame_id}: Table (track {track_id}) at {pos}")
    
    # Analyze camera poses if available
    print("\n=== Camera Pose Analysis ===")
    for frame_id in [0, 2, 4]:
        if frame_id in frame_data:
            camera_pose = frame_data[frame_id].get('camera_pose', None)
            if camera_pose:
                print(f"Frame {frame_id}: Camera pose available")
            else:
                print(f"Frame {frame_id}: No camera pose stored")
    
    # Check for other objects that might have been successfully tracked
    print("\n=== Successfully Tracked Objects ===")
    track_counts = defaultdict(lambda: {'count': 0, 'frames': []})
    
    for decision in decisions:
        track_id = decision['track_id']
        label = decision['label']
        frame_id = decision['frame_id']
        
        track_counts[label]['count'] = max(track_counts[label]['count'], 
                                          len(set([d['track_id'] for d in decisions if d['label'] == label])))
        track_counts[label]['frames'].append(frame_id)
    
    # Find objects that were successfully tracked across multiple frames
    for label, info in track_counts.items():
        unique_frames = len(set(info['frames']))
        if unique_frames > 1 and info['count'] == 1:
            print(f"{label}: Successfully tracked across {unique_frames} frames with 1 track")
    
    print("\n=== Objects with Multiple Tracks ===")
    for label, info in track_counts.items():
        if info['count'] > 1:
            print(f"{label}: {info['count']} tracks across {len(set(info['frames']))} frames")

if __name__ == "__main__":
    analyze_world_positions()