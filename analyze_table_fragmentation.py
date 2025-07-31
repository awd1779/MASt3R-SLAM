#!/usr/bin/env python3
"""Analyze table fragmentation in tracking."""

import json
from collections import defaultdict

def analyze_tracking_decisions():
    """Analyze tracking decisions to understand table fragmentation."""
    
    tracking_file = "/home/ubuntu/restart_from_scratch/logs/logs/tracked_3d_global_improved_filtered/room_0/tracking_decisions_3d.json"
    
    with open(tracking_file, 'r') as f:
        decisions = json.load(f)
    
    # Group by track_id to see which frames each track appears in
    tracks = defaultdict(lambda: {'frames': [], 'instances': [], 'label': ''})
    
    for decision in decisions:
        track_id = decision['track_id']
        frame_id = decision['frame_id']
        instance_id = decision['instance_id']
        label = decision['label']
        
        tracks[track_id]['frames'].append(frame_id)
        tracks[track_id]['instances'].append((frame_id, instance_id))
        tracks[track_id]['label'] = label
    
    # Find table tracks
    table_tracks = {}
    for track_id, info in tracks.items():
        if info['label'] == 'a table':
            table_tracks[track_id] = info
    
    print("=== Table Tracking Analysis ===")
    print(f"Found {len(table_tracks)} table tracks:\n")
    
    for track_id, info in sorted(table_tracks.items()):
        print(f"Track {track_id}:")
        print(f"  Appears in frames: {sorted(set(info['frames']))}")
        print(f"  Frame->Instance mapping:")
        for frame_id, instance_id in sorted(info['instances']):
            print(f"    Frame {frame_id}: Instance {instance_id}")
        print()
    
    # Analyze why they're different tracks
    print("=== Frame-by-Frame Table Instances ===")
    frame_tables = defaultdict(list)
    
    for decision in decisions:
        if decision['label'] == 'a table':
            frame_tables[decision['frame_id']].append({
                'instance_id': decision['instance_id'],
                'track_id': decision['track_id']
            })
    
    for frame_id in sorted(frame_tables.keys()):
        print(f"\nFrame {frame_id}:")
        for table in frame_tables[frame_id]:
            print(f"  Instance {table['instance_id']} -> Track {table['track_id']}")
    
    # Check for any other fragmented objects
    print("\n=== Other Fragmented Objects ===")
    label_tracks = defaultdict(list)
    for track_id, info in tracks.items():
        label_tracks[info['label']].append(track_id)
    
    for label, track_ids in sorted(label_tracks.items()):
        if len(track_ids) > 1:
            print(f"{label}: {len(track_ids)} tracks ({track_ids})")

if __name__ == "__main__":
    analyze_tracking_decisions()