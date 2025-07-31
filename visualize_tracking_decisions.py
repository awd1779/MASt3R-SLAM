#!/usr/bin/env python3
"""Visualize tracking decisions to understand fragmentation."""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os

def analyze_tracking_decisions(tracking_file):
    """Analyze tracking decisions to understand fragmentation."""
    
    with open(tracking_file, 'r') as f:
        decisions = json.load(f)
    
    # Group by label
    by_label = defaultdict(list)
    for track in decisions:
        by_label[track['label']].append(track)
    
    print("=== Tracking Analysis ===")
    print(f"Total tracks: {len(decisions)}")
    print()
    
    # Find fragmented objects (multiple tracks per label)
    fragmented = {}
    for label, tracks in by_label.items():
        if len(tracks) > 1:
            fragmented[label] = tracks
            print(f"{label}: {len(tracks)} tracks")
            for track in tracks:
                frames = track['frames']
                print(f"  Track {track['track_id']}: frames {frames[0]}-{frames[1]}, "
                      f"{track['observations']} observations")
    
    return fragmented

def visualize_track_timeline(tracking_file, output_dir):
    """Create timeline visualization of tracks."""
    
    with open(tracking_file, 'r') as f:
        decisions = json.load(f)
    
    # Group by label
    by_label = defaultdict(list)
    for track in decisions:
        by_label[track['label']].append(track)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = 0
    y_labels = []
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    
    for label, tracks in sorted(by_label.items()):
        for i, track in enumerate(tracks):
            frames = track['frames']
            # Draw timeline bar
            ax.barh(y_pos, frames[1] - frames[0] + 1, 
                   left=frames[0], height=0.8,
                   color=colors[hash(label) % 20],
                   alpha=0.7,
                   edgecolor='black',
                   linewidth=1)
            
            # Add track ID
            ax.text(frames[0] + 0.1, y_pos, f"T{track['track_id']}", 
                   va='center', fontsize=8)
            
            y_labels.append(f"{label} ({i+1}/{len(tracks)})")
            y_pos += 1
    
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Frame ID')
    ax.set_title('Track Timeline - Showing Fragmentation')
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'track_timeline.png'), dpi=150)
    print(f"Saved timeline to {output_dir}/track_timeline.png")

def analyze_keyframe_associations(semantic_dir):
    """Analyze which objects appear in which keyframes."""
    
    keyframe_data = {}
    
    # Read semantic data for each keyframe
    for i in range(3):  # 3 keyframes
        semantic_file = os.path.join(semantic_dir, f'semantic_frame_{i*2:06d}.json')
        if os.path.exists(semantic_file):
            with open(semantic_file, 'r') as f:
                data = json.load(f)
                keyframe_data[i] = data
    
    print("\n=== Keyframe Object Analysis ===")
    for kf_id, data in keyframe_data.items():
        print(f"\nKeyframe {kf_id} (frame {kf_id*2}):")
        labels = data.get('labels', {})
        label_counts = defaultdict(int)
        for label in labels.values():
            label_counts[label] += 1
        
        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count} instances")
    
    return keyframe_data

def main():
    # Paths
    base_dir = "/home/ubuntu/restart_from_scratch"
    tracking_file = os.path.join(base_dir, "logs/logs/tracked_3d_global_improved_filtered/room_0/tracking_decisions_3d.json")
    semantic_dir = os.path.join(base_dir, "logs/logs/tracked_3d_global_improved_filtered/room_0")
    output_dir = os.path.join(base_dir, "tracking_analysis")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze tracking decisions
    print("Analyzing tracking decisions...")
    fragmented = analyze_tracking_decisions(tracking_file)
    
    # Create timeline visualization
    print("\nCreating timeline visualization...")
    visualize_track_timeline(tracking_file, output_dir)
    
    # Analyze keyframe associations
    print("\nAnalyzing keyframe data...")
    keyframe_data = analyze_keyframe_associations(semantic_dir)
    
    # Focus on table tracking
    print("\n=== Table Tracking Deep Dive ===")
    if 'a table' in fragmented:
        table_tracks = fragmented['a table']
        print(f"Found {len(table_tracks)} table tracks:")
        
        # Check which keyframes have tables
        for kf_id, data in keyframe_data.items():
            labels = data.get('labels', {})
            track_ids = data.get('track_ids', {})
            
            table_instances = []
            for inst_id, label in labels.items():
                if label == 'a table':
                    track_id = track_ids.get(inst_id, -1)
                    table_instances.append((inst_id, track_id))
            
            if table_instances:
                print(f"\nKeyframe {kf_id}: {len(table_instances)} table instances")
                for inst_id, track_id in table_instances:
                    print(f"  Instance {inst_id} -> Track {track_id}")

if __name__ == "__main__":
    main()