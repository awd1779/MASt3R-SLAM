"""
Fixed version of save_reconstruction function with proper data type handling
"""
import numpy as np
import pathlib
from plyfile import PlyData, PlyElement
import colorsys

def generate_distinct_colors(n_colors, seed=42):
    """
    Generate n distinct colors using a general algorithm that works for any objects.
    """
    np.random.seed(seed)
    colors = []
    
    if n_colors == 0:
        return colors
    
    # Golden angle distribution in HSV space
    golden_angle = 137.508  # degrees
    
    # Use different strategies based on number of colors needed
    if n_colors <= 12:
        # For small sets, use maximally spaced hues with high saturation
        for i in range(n_colors):
            hue = (i * 360.0 / n_colors) % 360
            saturation = 0.7 + (i % 3) * 0.1
            value = 0.8 + (i % 2) * 0.1
            
            r, g, b = colorsys.hsv_to_rgb(hue/360.0, saturation, value)
            colors.append([int(r * 255), int(g * 255), int(b * 255)])
    else:
        # For larger sets, use golden angle for better distribution
        saturation_values = [0.5, 0.7, 0.9]
        value_values = [0.6, 0.75, 0.9]
        
        for i in range(n_colors):
            hue = (i * golden_angle) % 360
            saturation = saturation_values[i % len(saturation_values)]
            value = value_values[(i // len(saturation_values)) % len(value_values)]
            
            # Add small random perturbation
            hue = (hue + np.random.uniform(-10, 10)) % 360
            saturation = np.clip(saturation + np.random.uniform(-0.1, 0.1), 0.3, 1.0)
            value = np.clip(value + np.random.uniform(-0.1, 0.1), 0.4, 1.0)
            
            r, g, b = colorsys.hsv_to_rgb(hue/360.0, saturation, value)
            colors.append([int(r * 255), int(g * 255), int(b * 255)])
    
    return colors


def save_reconstruction_fixed(
    savedir,
    timestamps,
    frames,
    filename="pointcloud.ply",
    min_conf=1.5,
):
    """
    Fixed version that handles labels properly without uint8 overflow
    """
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    
    # ... (earlier code remains the same until line 183) ...
    
    # CRITICAL FIX: Use int32 or int64 for labels instead of uint8
    if not labels_list or all(arr is None or len(arr) == 0 for arr in labels_list):
        labels_global_np = np.zeros(len(pointclouds_np), dtype=np.int32)  # Changed from uint8
    else:
        valid_label_arrays = [lbl_arr for lbl_arr in labels_list if lbl_arr is not None and len(lbl_arr) > 0]
        if not valid_label_arrays:
            labels_global_np = np.zeros(len(pointclouds_np), dtype=np.int32)  # Changed from uint8
        else:
            try:
                # CRITICAL: Don't cast to uint8 here - keep original data type or use int32
                labels_global_np = np.concatenate(valid_label_arrays, axis=0).astype(np.int32)
            except ValueError as e_concat:
                print(f"[ERROR] Failed to concatenate label arrays: {e_concat}. Using default labels.")
                labels_global_np = np.zeros(len(pointclouds_np), dtype=np.int32)
    
    # ... (rest of the code for color assignment) ...
    
    # When saving to PLY, we may need to handle the label property differently
    # since PLY format has limitations on property types


def create_semantic_colors_direct(labels_array, label_to_class_map):
    """
    Alternative approach: Create colors directly from labels without intermediate steps
    """
    # Get unique labels
    unique_labels = np.unique(labels_array)
    print(f"[DEBUG] Found {len(unique_labels)} unique labels")
    print(f"[DEBUG] Label range: {unique_labels.min()} to {unique_labels.max()}")
    
    # Create color mapping
    label_to_color = {}
    
    # Background is always dark gray
    label_to_color[0] = [50, 50, 50]
    
    # Get non-background labels
    non_bg_labels = [l for l in unique_labels if l != 0]
    
    if len(non_bg_labels) > 0:
        # Generate distinct colors
        colors = generate_distinct_colors(len(non_bg_labels))
        
        # Assign colors to labels
        for i, label in enumerate(non_bg_labels):
            label_to_color[label] = colors[i]
            class_name = label_to_class_map.get(int(label), f"unknown_{label}")
            print(f"[DEBUG] Label {label} ({class_name}): RGB{tuple(colors[i])}")
    
    # Create color array
    colors_array = np.zeros((len(labels_array), 3), dtype=np.uint8)
    
    # Assign colors based on labels
    for label, color in label_to_color.items():
        mask = labels_array == label
        if mask.any():
            colors_array[mask] = color
            print(f"[DEBUG] Assigned color to {mask.sum()} points with label {label}")
    
    return colors_array


def diagnose_label_issues(labels_array, label_map):
    """
    Diagnose potential issues with label assignments
    """
    print("\n=== LABEL DIAGNOSIS ===")
    print(f"Labels array dtype: {labels_array.dtype}")
    print(f"Labels array shape: {labels_array.shape}")
    print(f"Min label: {labels_array.min()}, Max label: {labels_array.max()}")
    
    unique_labels, counts = np.unique(labels_array, return_counts=True)
    print(f"\nUnique labels and counts:")
    for label, count in zip(unique_labels, counts):
        class_name = label_map.get(int(label), "unknown")
        print(f"  Label {label} ({class_name}): {count} points")
    
    # Check for potential overflow
    if labels_array.dtype == np.uint8 and labels_array.max() == 255:
        print("\n[WARNING] Possible uint8 overflow detected!")
        print("Some labels may have wrapped around from values > 255")
    
    # Check label map
    print(f"\nLabel map has {len(label_map)} entries")
    if len(label_map) > 256:
        print("[WARNING] More than 256 unique objects but using uint8 labels!")
        print("This WILL cause color assignment issues!")
    
    return unique_labels, counts