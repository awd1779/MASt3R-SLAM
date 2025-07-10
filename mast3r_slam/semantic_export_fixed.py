"""Fixed semantic export for sparse reconstruction."""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional
from plyfile import PlyData, PlyElement
from mast3r_slam.geometry import constrain_points_to_ray
from mast3r_slam.config import config
from mast3r_slam.semantic_frame import decode_rle
import cv2


def save_sparse_semantic_reconstruction(savedir: str, 
                                      filename: str, 
                                      keyframes,
                                      semantic_keyframes,
                                      c_conf_threshold: float = 0.5,
                                      use_semantic_colors: bool = True):
    """
    Save sparse semantic reconstruction using SLAM feature points only.
    
    This uses the same semantic data as dense reconstruction but only
    projects labels onto existing SLAM points.
    """
    savedir = Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    
    all_points = []
    all_colors = []
    all_labels = []
    
    # Define semantic colors
    label_colormap = {
        'person': [255, 0, 0],      # Red
        'chair': [0, 255, 0],       # Green
        'table': [0, 0, 255],       # Blue
        'car': [255, 255, 0],       # Yellow
        'bottle': [255, 0, 255],    # Magenta
        'door': [0, 255, 255],      # Cyan
        'floor': [128, 128, 128],   # Gray
        'wall': [255, 128, 0],      # Orange
        'unknown': [200, 200, 200]  # Light gray
    }
    
    # Global label mapping
    global_label_mapping = {0: 'background'}
    next_global_id = 1
    
    num_labeled_points = 0
    num_total_points = 0
    
    # Process each keyframe
    for kf_idx in range(len(keyframes)):
        keyframe = keyframes[kf_idx]
        
        # Skip if no pointmap
        if keyframe is None or keyframe.X_canon is None:
            continue
            
        # Get semantic data for this keyframe
        semantic_data = semantic_keyframes.get(kf_idx)
        if semantic_data is None or not semantic_data.get('masks_rle'):
            # No semantics - still include points but with label 0
            X_canon = keyframe.X_canon
            if config["use_calib"]:
                X_canon = constrain_points_to_ray(
                    keyframe.img_shape.flatten()[:2], 
                    X_canon[None], 
                    keyframe.K
                ).squeeze(0)
            
            # Transform to world coordinates
            pW = keyframe.T_WC.act(X_canon).cpu().numpy().reshape(-1, 3)
            
            # Get confidence mask
            conf = keyframe.get_average_conf()
            if conf is not None:
                valid = conf.cpu().numpy().astype(np.float32).reshape(-1) > c_conf_threshold
            else:
                valid = np.ones(len(pW), dtype=bool)
            
            # Add points with no labels
            valid_points = pW[valid]
            all_points.append(valid_points)
            all_colors.append(np.ones((len(valid_points), 3)) * 128)  # Gray
            all_labels.append(np.zeros(len(valid_points), dtype=np.int32))
            num_total_points += len(valid_points)
            continue
        
        # Process keyframe with semantics
        h, w = keyframe.img_shape[0, 0].item(), keyframe.img_shape[0, 1].item()
        
        # Create semantic mask for the keyframe
        semantic_mask = np.zeros((h, w), dtype=np.int32)
        local_label_mapping = {}
        next_local_id = 1
        
        # Decode all masks
        for instance_id, rle in semantic_data['masks_rle'].items():
            # Get label name
            label_name = semantic_data.get('labels', {}).get(instance_id, 'unknown')
            confidence = semantic_data.get('confidences', {}).get(instance_id, 1.0)
            
            # Skip low confidence
            if confidence < 0.3:
                continue
            
            # Decode mask
            if 'size' in rle:
                size = rle['size']
                if len(size) == 3:
                    _, mask_h, mask_w = size
                else:
                    mask_h, mask_w = size
            else:
                mask_h, mask_w = h, w
            
            # Decode at original size
            mask = decode_rle(rle, mask_h, mask_w)
            
            # Resize if needed
            if (mask_h, mask_w) != (h, w):
                mask_resized = cv2.resize(
                    mask.astype(np.uint8), 
                    (w, h), 
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                mask = mask_resized
            
            # Assign local label
            local_label_mapping[next_local_id] = label_name
            semantic_mask[mask] = next_local_id
            next_local_id += 1
        
        # Get SLAM points
        X_canon = keyframe.X_canon
        if config["use_calib"]:
            X_canon = constrain_points_to_ray(
                keyframe.img_shape.flatten()[:2], 
                X_canon[None], 
                keyframe.K
            ).squeeze(0)
        
        # Transform to world coordinates
        pW = keyframe.T_WC.act(X_canon).cpu().numpy().reshape(-1, 3)
        
        # Get RGB colors
        img_rgb = (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8)
        
        # Get confidence mask
        conf = keyframe.get_average_conf()
        if conf is not None:
            valid = conf.cpu().numpy().astype(np.float32).reshape(-1) > c_conf_threshold
        else:
            valid = np.ones(len(pW), dtype=bool)
        
        # Map semantic labels to SLAM points
        # SLAM points are stored in a flattened array corresponding to image pixels
        semantic_labels_flat = semantic_mask.reshape(-1)
        point_labels = semantic_labels_flat.copy()
        
        # Remap local labels to global labels
        global_labels = np.zeros_like(point_labels)
        for local_id, label_name in local_label_mapping.items():
            # Find or create global ID
            if label_name not in global_label_mapping.values():
                global_label_mapping[next_global_id] = label_name
                global_id = next_global_id
                next_global_id += 1
            else:
                global_id = [k for k, v in global_label_mapping.items() if v == label_name][0]
            
            # Remap
            mask = point_labels == local_id
            global_labels[mask] = global_id
        
        # Apply validity mask
        valid_points = pW[valid]
        valid_labels = global_labels[valid]
        
        # Get colors - either semantic or RGB
        if use_semantic_colors:
            colors = np.zeros((len(valid_points), 3), dtype=np.uint8)
            for idx, label_id in enumerate(valid_labels):
                if label_id > 0:
                    label_name = global_label_mapping.get(label_id, 'unknown')
                    color = label_colormap.get(label_name, label_colormap['unknown'])
                    colors[idx] = color
                else:
                    # Use RGB color for unlabeled points
                    colors[idx] = img_rgb.reshape(-1, 3)[valid][idx]
        else:
            colors = img_rgb.reshape(-1, 3)[valid]
        
        all_points.append(valid_points)
        all_colors.append(colors)
        all_labels.append(valid_labels)
        
        num_labeled = (valid_labels > 0).sum()
        num_labeled_points += num_labeled
        num_total_points += len(valid_points)
        
        print(f"  Keyframe {kf_idx}: {num_labeled}/{len(valid_points)} labeled SLAM points")
    
    # Concatenate all data
    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    print(f"\nTotal sparse points: {len(all_points)}")
    print(f"Labeled sparse points: {(all_labels > 0).sum()}")
    
    # Save PLY file
    save_semantic_ply(
        savedir / filename,
        all_points,
        all_colors,
        all_labels,
        global_label_mapping
    )
    
    # Save statistics
    stats_file = savedir / f"{filename.split('.')[0]}_stats.txt"
    with open(stats_file, 'w') as f:
        f.write("Sparse Semantic Reconstruction Statistics\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total SLAM points: {num_total_points:,}\n")
        f.write(f"Labeled points: {num_labeled_points:,} ({num_labeled_points/num_total_points*100:.1f}%)\n")
        f.write(f"Keyframes used: {len(keyframes)}\n")
        f.write(f"\nLabel distribution:\n")
        
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        for label_id, count in zip(unique_labels, counts):
            if label_id == 0:
                continue
            label_name = global_label_mapping.get(label_id, 'unknown')
            percentage = count / len(all_labels) * 100
            f.write(f"  {label_name}: {count:,} points ({percentage:.1f}%)\n")
    
    print(f"Saved sparse semantic reconstruction to {savedir / filename}")
    print(f"Saved statistics to {stats_file}")


def save_semantic_ply(filename: str, 
                     points: np.ndarray, 
                     colors: np.ndarray, 
                     labels: np.ndarray,
                     label_names: Optional[Dict[int, str]] = None):
    """Save semantic point cloud in PLY format with labels."""
    colors = colors.astype(np.uint8)
    labels = labels.astype(np.int32)
    
    # Create structured array
    vertex_data = np.zeros(
        len(points),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("label", "i4"),
        ],
    )
    
    vertex_data["x"] = points[:, 0]
    vertex_data["y"] = points[:, 1]
    vertex_data["z"] = points[:, 2]
    vertex_data["red"] = colors[:, 0]
    vertex_data["green"] = colors[:, 1]
    vertex_data["blue"] = colors[:, 2]
    vertex_data["label"] = labels
    
    # Create PLY element
    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    
    # Write PLY file
    ply_data = PlyData([vertex_element], text=False)
    ply_data.write(filename)