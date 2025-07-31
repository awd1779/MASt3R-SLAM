#!/usr/bin/env python3
"""Check overlap between switch and wall plug masks."""

import cv2
import numpy as np

# Load masks
switch_mask = cv2.imread('debug_semantic_pipeline/frame_000004/mask_08_a switch.png', cv2.IMREAD_GRAYSCALE)
wall_plug_mask = cv2.imread('debug_semantic_pipeline/frame_000004/mask_11_a wall plug.png', cv2.IMREAD_GRAYSCALE)

# Convert to binary
switch_mask = switch_mask > 127
wall_plug_mask = wall_plug_mask > 127

# Calculate overlap
intersection = np.sum(switch_mask & wall_plug_mask)
union = np.sum(switch_mask | wall_plug_mask)

iou = intersection / union if union > 0 else 0

print(f"Switch mask pixels: {np.sum(switch_mask)}")
print(f"Wall plug mask pixels: {np.sum(wall_plug_mask)}")
print(f"Intersection pixels: {intersection}")
print(f"Union pixels: {union}")
print(f"IoU: {iou:.4f}")

# Check if they're in the same region
switch_coords = np.where(switch_mask)
plug_coords = np.where(wall_plug_mask)

if len(switch_coords[0]) > 0 and len(plug_coords[0]) > 0:
    switch_center = (np.mean(switch_coords[0]), np.mean(switch_coords[1]))
    plug_center = (np.mean(plug_coords[0]), np.mean(plug_coords[1]))
    distance = np.sqrt((switch_center[0] - plug_center[0])**2 + (switch_center[1] - plug_center[1])**2)
    print(f"\nSwitch center: ({switch_center[0]:.1f}, {switch_center[1]:.1f})")
    print(f"Wall plug center: ({plug_center[0]:.1f}, {plug_center[1]:.1f})")
    print(f"Distance between centers: {distance:.1f} pixels")