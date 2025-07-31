#!/usr/bin/env python3
"""Test the label-based tracker logic."""

import sys
sys.path.append('/home/ubuntu/restart_from_scratch')

from mast3r_slam.label_based_tracker import LabelBasedTracker

# Test the label tracker
tracker = LabelBasedTracker({
    'unique_objects': ['a table', 'a sofa', 'a floor', 'a ceiling', 'a door']
})

print("=== Testing LabelBasedTracker ===")
print(f"Unique objects: {tracker.unique_objects}")
print(f"Is 'a table' unique? {tracker.is_unique_object('a table')}")
print(f"Is 'a chair' unique? {tracker.is_unique_object('a chair')}")

# Simulate tracking
print("\n=== Simulating tracking ===")

# Frame 0: First table detection
print("Frame 0: First table detection")
existing_track = tracker.get_track_for_unique_object('a table')
print(f"  Existing track for 'a table': {existing_track}")

if existing_track is None:
    print("  Creating new track 8 for 'a table'")
    tracker.assign_track_to_unique_object('a table', 8)

# Frame 2: Second table detection
print("\nFrame 2: Second table detection")
existing_track = tracker.get_track_for_unique_object('a table')
print(f"  Existing track for 'a table': {existing_track}")
print(f"  Should force match: {existing_track is not None}")

# Check stats
print("\n=== Final stats ===")
stats = tracker.get_unique_object_stats()
print(f"Stats: {stats}")