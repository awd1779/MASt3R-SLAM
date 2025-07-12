"""Export semantic point clouds in PLY format."""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from plyfile import PlyData, PlyElement
from mast3r_slam.geometry import constrain_points_to_ray
from mast3r_slam.config import config
from mast3r_slam.semantic_integration import SemanticSLAMBackend
import matplotlib.cm as cm
import logging

logger = logging.getLogger('mast3r_slam.semantic_export')


def save_semantic_ply(filename: str, 
                     points: np.ndarray, 
                     colors: np.ndarray, 
                     labels: np.ndarray,
                     label_names: Optional[Dict[int, str]] = None):
    """
    Save semantic point cloud in PLY format with labels.
    
    Args:
        filename: Output PLY file path
        points: [N, 3] array of 3D points
        colors: [N, 3] array of RGB colors (0-255)
        labels: [N] array of semantic labels
        label_names: Optional dict mapping label IDs to names
    """
    colors = colors.astype(np.uint8)
    labels = labels.astype(np.int32)
    
    # Create structured array with semantic labels
    pcd = np.empty(
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
    
    pcd["x"], pcd["y"], pcd["z"] = points.T
    pcd["red"], pcd["green"], pcd["blue"] = colors.T
    pcd["label"] = labels
    
    # Add comments with label names if provided
    comments = []
    if label_names:
        comments.append("Label mapping:")
        for label_id, name in sorted(label_names.items()):
            comments.append(f"  {label_id}: {name}")
    
    # Create PLY element
    vertex_element = PlyElement.describe(pcd, "vertex", comments=comments)
    ply_data = PlyData([vertex_element], text=False, comments=comments)
    ply_data.write(filename)
    

def save_semantic_reconstruction(savedir: str, 
                               filename: str, 
                               keyframes,
                               semantic_backend: SemanticSLAMBackend,
                               c_conf_threshold: float = 0.5,
                               use_semantic_colors: bool = True):
    """
    Save semantic reconstruction as colored PLY file.
    
    Args:
        savedir: Directory to save the PLY file
        filename: Output filename
        keyframes: SharedKeyframes object
        semantic_backend: SemanticSLAMBackend with processed semantics
        c_conf_threshold: Confidence threshold for points
        use_semantic_colors: If True, color by semantic labels; else use RGB
    """
    savedir = Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    
    pointclouds = []
    rgb_colors = []
    semantic_labels = []
    semantic_confidences = []
    
    # Collect all keyframe data
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        
        # Skip if no pointmap
        if keyframe.X_canon is None:
            continue
            
        # Apply calibration constraints if needed
        X_canon = keyframe.X_canon
        if config["use_calib"]:
            X_canon = constrain_points_to_ray(
                keyframe.img_shape.flatten()[:2], 
                X_canon[None], 
                keyframe.K
            ).squeeze(0)
        
        # Transform to world coordinates
        pW = keyframe.T_WC.act(X_canon).cpu().numpy().reshape(-1, 3)
        
        # Get RGB colors from the keyframe image (same as save_reconstruction)
        color = (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3)
        
        # Get confidence mask
        conf = keyframe.get_average_conf()
        if conf is not None:
            valid = conf.cpu().numpy().astype(np.float32).reshape(-1) > c_conf_threshold
        else:
            valid = np.ones(len(pW), dtype=bool)
        
        # Get semantic labels if available
        semantics = semantic_backend.keyframe_semantics.get(i)
        if semantics is not None:
            labels, sem_conf = semantics
            labels_np = labels.cpu().numpy()
            sem_conf_np = sem_conf.cpu().numpy()
        else:
            labels_np = np.zeros(len(pW), dtype=np.int32)
            sem_conf_np = np.zeros(len(pW), dtype=np.float32)
        
        # Apply validity mask
        pointclouds.append(pW[valid])
        rgb_colors.append(color[valid])
        semantic_labels.append(labels_np[valid])
        semantic_confidences.append(sem_conf_np[valid])
    
    # Concatenate all data
    pointclouds = np.concatenate(pointclouds, axis=0)
    rgb_colors = np.concatenate(rgb_colors, axis=0)
    semantic_labels = np.concatenate(semantic_labels, axis=0)
    semantic_confidences = np.concatenate(semantic_confidences, axis=0)
    
    # Generate semantic colors if requested
    if use_semantic_colors:
        colors = generate_semantic_colors(semantic_labels)
    else:
        colors = rgb_colors
    
    # Get label names from track manager with instance numbers
    label_names = {}
    unique_labels = np.unique(semantic_labels)
    
    # Count instances per class for cleaner numbering
    class_counters = {}
    label_to_instance_num = {}
    
    # First pass: assign instance numbers per class
    for label in unique_labels:
        if label > 0:  # Skip background
            track_info = semantic_backend.track_manager.get_track_history(int(label))
            if track_info and 'label' in track_info:
                base_label = track_info['label']
                
                # Get or initialize counter for this class
                if base_label not in class_counters:
                    class_counters[base_label] = 0
                class_counters[base_label] += 1
                
                # Store the instance number for this label
                label_to_instance_num[int(label)] = class_counters[base_label]
                
                # Create label name with instance number
                label_names[int(label)] = f"{base_label}_{class_counters[base_label]}"
    
    # Save PLY file
    save_semantic_ply(
        savedir / filename,
        pointclouds,
        colors,
        semantic_labels,
        label_names
    )
    
    # Count objects by type
    object_type_counts = {}
    total_objects = 0
    
    for label in unique_labels:
        if label > 0:  # Skip background
            name = label_names.get(int(label), f"unknown_{label}")
            if name != 'unknown' and not name.startswith('unknown_'):
                total_objects += 1
                base_label = name.split('_')[0] if '_' in name else name
                if base_label not in object_type_counts:
                    object_type_counts[base_label] = 0
                object_type_counts[base_label] += 1
    
    # Save additional statistics
    stats_file = savedir / f"{filename.split('.')[0]}_stats.txt"
    with open(stats_file, 'w') as f:
        f.write("Semantic Reconstruction Statistics\n")
        f.write("=" * 40 + "\n")
        f.write(f"Total points: {len(pointclouds)}\n")
        f.write(f"Labeled points: {(semantic_labels > 0).sum()}\n")
        f.write(f"Unique instances: {len(unique_labels) - 1}\n")  # Exclude background
        f.write(f"Average semantic confidence: {semantic_confidences[semantic_labels > 0].mean():.3f}\n")
        
        f.write(f"\nOBJECT SUMMARY:\n")
        f.write(f"Total unique objects: {total_objects}\n")
        f.write(f"\nObject counts by type:\n")
        for obj_type, count in sorted(object_type_counts.items()):
            f.write(f"  {obj_type}: {count} instance{'s' if count > 1 else ''}\n")
        
        f.write("\nLabel distribution:\n")
        for label in unique_labels:
            count = (semantic_labels == label).sum()
            percentage = count / len(semantic_labels) * 100
            name = label_names.get(int(label), "background" if label == 0 else f"unknown_{label}")
            f.write(f"  {label} ({name}): {count} points ({percentage:.1f}%)\n")
    
    logger.info(f"Saved semantic reconstruction to {savedir / filename}")
    logger.info(f"Total points: {len(pointclouds)}, Labeled: {(semantic_labels > 0).sum()}")
    logger.info(f"Total unique objects: {total_objects}")
    logger.info("Object counts by type:")
    for obj_type, count in sorted(object_type_counts.items()):
        logger.info(f"  {obj_type}: {count} instance{'s' if count > 1 else ''}")
    

def generate_semantic_colors(labels: np.ndarray, 
                           colormap: str = 'tab20') -> np.ndarray:
    """
    Generate consistent colors for semantic labels.
    
    Args:
        labels: [N] array of semantic labels
        colormap: Matplotlib colormap name
        
    Returns:
        [N, 3] array of RGB colors (0-255)
    """
    cmap = cm.get_cmap(colormap)
    colors = np.zeros((len(labels), 3), dtype=np.uint8)
    
    # Assign colors based on labels
    unique_labels = np.unique(labels)
    for i, label in enumerate(unique_labels):
        if label == 0:  # Background - gray
            color = [128, 128, 128]
        else:
            # Use colormap
            color = (np.array(cmap(i % 20)[:3]) * 255).astype(np.uint8)
        
        mask = labels == label
        colors[mask] = color
    
    return colors


def export_semantic_mesh(savedir: str,
                        filename: str,
                        keyframes,
                        semantic_backend: SemanticSLAMBackend,
                        mesh_resolution: float = 0.05):
    """
    Export semantic mesh using marching cubes (requires additional dependencies).
    This is a placeholder for future mesh generation functionality.
    """
    # TODO: Implement TSDF fusion and marching cubes for mesh generation
    print("Mesh export not yet implemented. Use point cloud export instead.")
    

def create_semantic_video(savedir: str,
                         filename: str,
                         keyframes,
                         semantic_backend: SemanticSLAMBackend,
                         fps: int = 30):
    """
    Create a video showing semantic segmentation results.
    """
    import cv2
    
    savedir = Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    
    # Get frame dimensions
    first_kf = keyframes[0]
    h, w = first_kf.img_shape[0].item(), first_kf.img_shape[1].item()
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(savedir / filename), fourcc, fps, (w*2, h))
    
    for i in range(len(keyframes)):
        kf = keyframes[i]
        
        # Get RGB image
        rgb = (kf.uimg.cpu().numpy() * 255).astype(np.uint8)
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        # Get semantic visualization
        semantics = semantic_backend.keyframe_semantics.get(i)
        if semantics is not None:
            labels, _ = semantics
            
            # Create semantic visualization
            semantic_colors = generate_semantic_colors(labels.cpu().numpy())
            semantic_img = semantic_colors.reshape(h, w, 3)
            semantic_bgr = cv2.cvtColor(semantic_img, cv2.COLOR_RGB2BGR)
            
            # Blend with RGB
            alpha = 0.5
            blended = cv2.addWeighted(rgb_bgr, 1-alpha, semantic_bgr, alpha, 0)
        else:
            blended = rgb_bgr
        
        # Concatenate side by side
        combined = np.hstack([rgb_bgr, blended])
        out.write(combined)
    
    out.release()
    print(f"Saved semantic video to {savedir / filename}")