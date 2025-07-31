#!/usr/bin/env python3
"""Verify that table tracking is working correctly."""

import json
from collections import defaultdict

def verify_tracking():
    # Load tracking decisions
    tracking_file = "logs/logs/tracked_3d_global_improved_filtered/room_0/tracking_decisions_3d.json"
    with open(tracking_file, 'r') as f:
        decisions = json.load(f)
    
    # Load final tracking summary
    tracking_summary_file = "logs/logs/tracked_3d_global_improved_filtered/room_0/room_0_semantic_dense_tracked_3d.tracking.json"
    with open(tracking_summary_file, 'r') as f:
        summary = json.load(f)
    
    # Analyze table tracks
    table_decisions = [d for d in decisions if d['label'] == 'a table']
    
    print("=== Table Tracking Verification ===")
    print(f"Total table detections: {len(table_decisions)}")
    
    # Group by frame
    by_frame = defaultdict(list)
    for d in table_decisions:
        by_frame[d['frame_id']].append(d)
    
    print("\nFrame-by-frame analysis:")
    for frame_id in sorted(by_frame.keys()):
        print(f"\nFrame {frame_id}:")
        for d in by_frame[frame_id]:
            print(f"  Instance {d['instance_id']} -> Track {d['track_id']} (method: {d['method']})")
    
    # Check unique table tracks
    unique_tracks = set(d['track_id'] for d in table_decisions)
    print(f"\nUnique table tracks: {unique_tracks}")
    print(f"Number of unique table tracks: {len(unique_tracks)}")
    
    # Check if tracks are sequential (might indicate forced matching)
    if len(unique_tracks) == 1:
        print("\n✅ SUCCESS: Table has only ONE track ID!")
        track_id = list(unique_tracks)[0]
        print(f"   All tables mapped to track {track_id}")
    else:
        print("\n❌ ISSUE: Table has multiple track IDs")
        
    # Check tracking summary
    print("\n=== Tracking Summary ===")
    table_tracks_in_summary = []
    for track_id, label in summary['track_to_label'].items():
        if label == 'a table':
            table_tracks_in_summary.append(int(track_id))
    
    print(f"Table tracks in final summary: {sorted(table_tracks_in_summary)}")
    
    # Verify other unique objects
    print("\n=== Other Unique Objects ===")
    unique_objects = ['a floor', 'a ceiling', 'a door', 'a sofa']
    
    for obj_label in unique_objects:
        obj_decisions = [d for d in decisions if d['label'] == obj_label]
        obj_tracks = set(d['track_id'] for d in obj_decisions)
        status = "✅" if len(obj_tracks) == 1 else "❌"
        print(f"{status} {obj_label}: {len(obj_tracks)} track(s) - {obj_tracks}")

if __name__ == "__main__":
    verify_tracking()