import pathlib
from typing import Optional
import cv2
import numpy as np
import torch
from mast3r_slam.dataloader import Intrinsics
from mast3r_slam.frame import SharedKeyframes
from mast3r_slam.lietorch_utils import as_SE3
from mast3r_slam.config import config
from plyfile import PlyData, PlyElement
import colorsys

# Define SEMANTIC_COLOR_MAP as empty dict (not used anymore with new color system)
SEMANTIC_COLOR_MAP = {}

# Color visualization configuration
COLOR_CONFIG = {
    "method": "distinct",  # "distinct" for maximally distinct colors, "golden" for golden angle
    "background_color": [50, 50, 50],  # Dark gray for background
}


def generate_distinct_colors(n_colors, seed=42):
    """
    Generate n distinct colors using a general algorithm that works for any objects.
    No hardcoded values or object-specific colors.
    
    Args:
        n_colors: Number of distinct colors needed
        seed: Random seed for reproducibility
        
    Returns:
        List of RGB colors (0-255)
    """
    np.random.seed(seed)
    colors = []
    
    if n_colors == 0:
        return colors
    
    # Method 1: Golden angle distribution in HSV space
    # This ensures maximum spread of hues
    golden_angle = 137.508  # degrees
    
    # Use different strategies based on number of colors needed
    if n_colors <= 12:
        # For small sets, use maximally spaced hues with high saturation
        for i in range(n_colors):
            hue = (i * 360.0 / n_colors) % 360
            # Vary saturation and value slightly to avoid monotony
            saturation = 0.7 + (i % 3) * 0.1
            value = 0.8 + (i % 2) * 0.1
            
            r, g, b = colorsys.hsv_to_rgb(hue/360.0, saturation, value)
            colors.append([int(r * 255), int(g * 255), int(b * 255)])
    
    else:
        # For larger sets, use golden angle for better distribution
        # Also vary saturation and value to create more distinction
        saturation_values = [0.5, 0.7, 0.9]
        value_values = [0.6, 0.75, 0.9]
        
        for i in range(n_colors):
            hue = (i * golden_angle) % 360
            saturation = saturation_values[i % len(saturation_values)]
            value = value_values[(i // len(saturation_values)) % len(value_values)]
            
            # Add small random perturbation to avoid patterns
            hue = (hue + np.random.uniform(-10, 10)) % 360
            saturation = np.clip(saturation + np.random.uniform(-0.1, 0.1), 0.3, 1.0)
            value = np.clip(value + np.random.uniform(-0.1, 0.1), 0.4, 1.0)
            
            r, g, b = colorsys.hsv_to_rgb(hue/360.0, saturation, value)
            colors.append([int(r * 255), int(g * 255), int(b * 255)])
    
    return colors



def prepare_savedir(args, dataset):
    save_dir = pathlib.Path("logs")
    if args.save_as != "default":
        save_dir = save_dir / args.save_as
    save_dir.mkdir(exist_ok=True, parents=True)
    seq_name = dataset.dataset_path.stem
    return save_dir, seq_name


def save_traj(
    logdir,
    logfile,
    timestamps,
    frames: SharedKeyframes,
    intrinsics: Optional[Intrinsics] = None,
):
    logdir = pathlib.Path(logdir)
    logdir.mkdir(exist_ok=True, parents=True)
    logfile = logdir / logfile
    with open(logfile, "w") as f:
        for i in range(len(frames)):
            keyframe = frames[i]
            t = timestamps[keyframe.frame_id]
            T_WC = as_SE3(keyframe.T_WC)
            x, y, z, qx, qy, qz, qw = T_WC.data.squeeze().cpu().numpy()
            f.write(f"{t} {x} {y} {z} {qx} {qy} {qz} {qw}\n")


def save_reconstruction(savedir, filename, keyframes: SharedKeyframes,
                        c_conf_threshold: float,
                        global_id_to_name_map: Optional[dict] = None):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)

    pointclouds_list = []
    colors_list = []
    labels_list = []

    final_label_map_for_ply_comments = global_id_to_name_map if global_id_to_name_map is not None else {0: "background"}
    if 0 not in final_label_map_for_ply_comments:
        final_label_map_for_ply_comments[0] = "background"

    for i in range(len(keyframes)):
        keyframe = keyframes[i]

        kf_global_instance_ids_flat = None
        if keyframe.global_instance_ids is not None and keyframe.global_instance_ids.numel() > 0:
            kf_global_instance_ids_flat = keyframe.global_instance_ids.cpu().numpy().reshape(-1)

        X_canon_for_proj = keyframe.X_canon
        if X_canon_for_proj is None or X_canon_for_proj.numel() == 0:
            print(f"[Warning eval.py] Keyframe {keyframe.frame_id} (idx {i}) has no X_canon points. Skipping.")
            continue

        pW_all_kf_points = keyframe.T_WC.act(X_canon_for_proj).cpu().numpy().reshape(-1, 3)
        uimg_np = keyframe.uimg.cpu().numpy() if isinstance(keyframe.uimg, torch.Tensor) else keyframe.uimg
        color_all_kf_points = (uimg_np * 255).astype(np.uint8).reshape(-1, 3)

        conf_tensor = keyframe.get_average_conf()
        if conf_tensor is None or conf_tensor.numel() == 0:
            print(f"[Warning eval.py] Keyframe {keyframe.frame_id} (idx {i}) has no confidence data. Assuming all points valid for this KF.")
            confidence_all_kf_points = np.ones(X_canon_for_proj.shape[0], dtype=np.float32) * (c_conf_threshold + 1.0)
        else:
            confidence_all_kf_points = conf_tensor.cpu().numpy().astype(np.float32).reshape(-1)

        num_points_in_kf_X_canon = X_canon_for_proj.shape[0]
        if not (len(pW_all_kf_points) == num_points_in_kf_X_canon and \
                len(color_all_kf_points) == num_points_in_kf_X_canon and \
                len(confidence_all_kf_points) == num_points_in_kf_X_canon and \
                (kf_global_instance_ids_flat is None or len(kf_global_instance_ids_flat) == num_points_in_kf_X_canon)):
            print(f"[ERROR eval.py] Mismatch in raw array lengths for keyframe {keyframe.frame_id} (idx {i}). Skipping this keyframe for PLY.")
            print(f"  X_canon points: {num_points_in_kf_X_canon}, pW: {len(pW_all_kf_points)}, color: {len(color_all_kf_points)}, "
                  f"conf: {len(confidence_all_kf_points)}, labels: {len(kf_global_instance_ids_flat) if kf_global_instance_ids_flat is not None else 'None'}")
            continue

        valid_indices = confidence_all_kf_points > c_conf_threshold

        if np.sum(valid_indices) == 0:
            continue

        pointclouds_list.append(pW_all_kf_points[valid_indices])
        colors_list.append(color_all_kf_points[valid_indices])

        if kf_global_instance_ids_flat is not None:
            labels_for_kf = kf_global_instance_ids_flat[valid_indices]
            labels_list.append(labels_for_kf)
            # Debug: Print label statistics for this keyframe
            unique_labels, counts = np.unique(labels_for_kf, return_counts=True)
            non_zero_labels = unique_labels[unique_labels != 0]
            print(f"[DEBUG eval.py] Keyframe {keyframe.frame_id}: {len(labels_for_kf)} valid points, "
                  f"{len(non_zero_labels)} non-background labels: {dict(zip(unique_labels[:10], counts[:10]))}")
        else:
            print(f"[Warning eval.py] Keyframe {keyframe.frame_id} (idx {i}) has no global_instance_ids. Using default label 0 for its {np.sum(valid_indices)} valid points.")
            labels_list.append(np.zeros(np.sum(valid_indices), dtype=np.int32))

    if not pointclouds_list:
        print("[Warning eval.py] No valid points collected from any keyframes. Skipping PLY generation.")
        return

    pointclouds_np = np.concatenate(pointclouds_list, axis=0)
    colors_np = np.concatenate(colors_list, axis=0)

    if not labels_list or all(arr is None or len(arr) == 0 for arr in labels_list):
        labels_global_np = np.zeros(len(pointclouds_np), dtype=np.int32)
    else:
        valid_label_arrays = [lbl_arr for lbl_arr in labels_list if lbl_arr is not None and len(lbl_arr) > 0]
        if not valid_label_arrays:
            labels_global_np = np.zeros(len(pointclouds_np), dtype=np.int32)
        else:
            try:
                labels_global_np = np.concatenate(valid_label_arrays, axis=0).astype(np.int32)
            except ValueError as e_concat:
                print(f"[ERROR eval.py] Failed to concatenate label arrays: {e_concat}. Using default labels.")
                labels_global_np = np.zeros(len(pointclouds_np), dtype=np.int32)

    if len(labels_global_np) != len(pointclouds_np):
        print(f"[CRITICAL ERROR eval.py] Final label count ({len(labels_global_np)}) does not match point count ({len(pointclouds_np)}). Using default labels.")
        labels_global_np = np.zeros(len(pointclouds_np), dtype=np.int32)

    print(f"[DIAGNOSTIC evaluate.py - save_reconstruction FINAL AGGREGATION]")
    print(f"  Total points for PLY: {pointclouds_np.shape[0]}")
    if labels_global_np.size > 0:
        unique_final_labels, counts_final_labels = np.unique(labels_global_np, return_counts=True)
        print(f"  Unique final labels in PLY (ID: count): {list(zip(unique_final_labels.tolist(), counts_final_labels.tolist()))}")
        print(f"  Label data type: {labels_global_np.dtype}")
        print(f"  Max label value: {labels_global_np.max()}")
        print(f"  Min label value: {labels_global_np.min()}")
        if labels_global_np.max() > 255:
            print(f"  [WARNING] Labels exceed uint8 range! Max label ID is {labels_global_np.max()}")
    print(f"  Label map for PLY comments (from tracker): {final_label_map_for_ply_comments}")
    print(f"[DIAGNOSTIC evaluate.py - save_reconstruction END]")

    ply_main_filename = savedir / filename
    save_ply(ply_main_filename, pointclouds_np, colors_np, labels_global_np, final_label_map_for_ply_comments, property_name="quality")

    # Apply post-processing to improve label accuracy
    if labels_global_np.size > 0 and labels_global_np.shape[0] == pointclouds_np.shape[0]:
        try:
            from mast3r_slam.semantic_utils import post_process_semantic_labels, merge_similar_labels
            
            # Post-process to remove outliers
            labels_global_np, final_label_map_for_ply_comments = post_process_semantic_labels(
                labels_global_np, pointclouds_np, final_label_map_for_ply_comments,
                min_cluster_size=50
            )
            
            # Merge similar labels that are close together
            labels_global_np, final_label_map_for_ply_comments = merge_similar_labels(
                labels_global_np, final_label_map_for_ply_comments, pointclouds_np,
                distance_threshold=0.1
            )
            
            print("[INFO] Applied semantic post-processing for improved accuracy")
        except Exception as e:
            print(f"[WARNING] Could not apply post-processing: {e}")
    
    if labels_global_np.size > 0 and labels_global_np.shape[0] == pointclouds_np.shape[0]:
        filepath_obj = pathlib.Path(filename)
        seg_color_ply_name = f"{filepath_obj.stem}_seg_color.ply"
        print(f"[INFO eval.py] Generating PLY with segmentation colors: {seg_color_ply_name}")

        # Use custom semantic color map
        seg_colors_np = np.zeros_like(colors_np)
        print(f"[DEBUG] seg_colors_np shape: {seg_colors_np.shape}, dtype: {seg_colors_np.dtype}")
        print(f"[DEBUG] colors_np shape: {colors_np.shape}, dtype: {colors_np.dtype}")
        print(f"[DEBUG] Initial seg_colors_np min: {seg_colors_np.min()}, max: {seg_colors_np.max()}")
        print(f"[DEBUG] labels_global_np shape: {labels_global_np.shape}, dtype: {labels_global_np.dtype}")
        unique_ids_in_ply = np.unique(labels_global_np)
        print(f"[DEBUG] Unique label IDs in PLY: {unique_ids_in_ply}")
        print(f"[DEBUG] Number of unique labels: {len(unique_ids_in_ply)}")
        print(f"[DEBUG] Label range: {unique_ids_in_ply.min()} to {unique_ids_in_ply.max()}")
        if unique_ids_in_ply.max() > 255:
            print(f"[DEBUG] Labels exceed uint8! This would have caused color issues with the old code.")

        # Simple, general color assignment without hardcoded values
        print(f"[INFO] Generating distinct colors for {len(unique_ids_in_ply)} unique labels")
        
        # Get non-background label IDs
        non_bg_labels = [lid for lid in unique_ids_in_ply if lid != 0]
        
        # Generate distinct colors for all non-background labels
        if len(non_bg_labels) > 0:
            distinct_colors = generate_distinct_colors(len(non_bg_labels), seed=42)
            
            # Create color pool
            random_color_pool = {}
            random_color_pool[0] = COLOR_CONFIG["background_color"]  # Background
            
            # Assign colors to labels
            for i, label_id in enumerate(non_bg_labels):
                random_color_pool[label_id] = distinct_colors[i]
                
            # Debug output
            print(f"[DEBUG] Color assignment summary:")
            print(f"  Background (ID 0): {random_color_pool[0]}")
            print(f"  Generated {len(distinct_colors)} distinct colors for {len(non_bg_labels)} labels")
            
            # Show first few color assignments
            for i, label_id in enumerate(non_bg_labels[:5]):
                class_name = final_label_map_for_ply_comments.get(label_id, "unknown")
                print(f"  ID {label_id} ({class_name}): RGB{tuple(random_color_pool[label_id])}")
            if len(non_bg_labels) > 5:
                print(f"  ... and {len(non_bg_labels) - 5} more labels")
        else:
            # Only background
            random_color_pool = {0: COLOR_CONFIG["background_color"]}

        # Assign colors to points
        for label_id_val in unique_ids_in_ply:
            class_label = final_label_map_for_ply_comments.get(label_id_val, "unknown")
            color_rgb = random_color_pool.get(label_id_val, [30, 30, 30])
            
            # Find all points with this label
            mask = labels_global_np == label_id_val
            num_points = np.sum(mask)
            
            # Assign color
            if num_points > 0:
                seg_colors_np[mask] = np.array(color_rgb, dtype=np.uint8)

        # Debug: Check color distribution
        unique_colors = np.unique(seg_colors_np, axis=0)
        print(f"[DEBUG] Unique colors in seg_colors_np: {len(unique_colors)}")
        for color in unique_colors[:10]:  # Show first 10 unique colors
            count = np.sum(np.all(seg_colors_np == color, axis=1))
            print(f"  Color {color}: {count} points")
        
        # Debug: Check if seg_colors_np is all the same color
        if len(unique_colors) == 1:
            print(f"[WARNING] All points have the same color: {unique_colors[0]}")
        
        # Final debug check before saving
        print(f"[DEBUG] Final seg_colors_np stats:")
        print(f"  Min values: {seg_colors_np.min(axis=0)}")
        print(f"  Max values: {seg_colors_np.max(axis=0)}")
        print(f"  Mean values: {seg_colors_np.mean(axis=0)}")
        print(f"  First 5 colors: {seg_colors_np[:5]}")
        
        seg_color_filename_path = savedir / seg_color_ply_name
        save_ply(seg_color_filename_path, pointclouds_np, seg_colors_np, labels=labels_global_np, label_map=final_label_map_for_ply_comments, property_name="semantic_id")
        print(f"[INFO eval.py] Saved PLY with segmentation colors to {seg_color_filename_path}")


def save_keyframes(savedir, timestamps, keyframes: SharedKeyframes):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)

    mask_savedir = savedir / "masks"
    mask_savedir.mkdir(exist_ok=True, parents=True)

    default_mask_color_bgr = [30, 30, 30]

    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        t = timestamps[keyframe.frame_id]

        img_filename = savedir / f"{t}.png"
        uimg_np = keyframe.uimg.cpu().numpy() if isinstance(keyframe.uimg, torch.Tensor) else keyframe.uimg
        cv2.imwrite(
            str(img_filename),
            cv2.cvtColor((uimg_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        )

        # Force semantic processing if missing (safety check)
        if keyframe.local_instance_mask is None and keyframe.img is not None:
            print(f"[evaluate] WARNING: Keyframe {i} (frame {keyframe.frame_id}) missing local_instance_mask! Force processing...")
            from mast3r_slam.semantic_core import process_frame_for_semantics, TEXT_PROMPTS
            
            try:
                local_mask, local_map = process_frame_for_semantics(
                    image_tensor_chw_0_1_rgb=keyframe.img.squeeze(0),
                    text_prompts_for_clip=TEXT_PROMPTS,
                    enable_debug_viz=False,
                    frame_id=keyframe.frame_id
                )
                keyframe.local_instance_mask = local_mask
                keyframe.local_id_to_class_label_map = local_map
                print(f"[evaluate] Successfully processed semantics for keyframe {i}")
            except Exception as e:
                print(f"[evaluate] Failed to process semantics: {e}")
        
        if keyframe.local_instance_mask is not None and keyframe.local_instance_mask.numel() > 0:
            mask_tensor = keyframe.local_instance_mask.cpu().numpy().astype(np.uint8)
            h, w = mask_tensor.shape
            colored_mask_img = np.zeros((h, w, 3), dtype=np.uint8)

            unique_labels_in_mask = np.unique(mask_tensor)
            # Use custom semantic color map for local masks
            # Need to map local_id to class_label first, then to color
            # This requires the local_id_to_class_label_map from the frame
            # However, the local_instance_mask only has local_ids, not global_ids.
            # The frame.local_id_to_class_label_map is what we need.

            # Generate a pool of random colors for labels not in SEMANTIC_COLOR_MAP
            # Seed for consistent random colors for unmapped labels
            np.random.seed(42)
            random_color_pool_bgr = {label_id: list(np.random.randint(50, 251, size=3)[::-1]) # RGB to BGR
                                     for label_id in unique_labels_in_mask}

            for label_id_val in unique_labels_in_mask:
                class_label = "unknown" # Default if map is None or label not found
                if keyframe.local_id_to_class_label_map is not None:
                    class_label = keyframe.local_id_to_class_label_map.get(label_id_val, "unknown")

                if class_label in SEMANTIC_COLOR_MAP:
                    color_rgb = SEMANTIC_COLOR_MAP[class_label]
                    color_bgr = color_rgb[::-1] # Convert RGB to BGR for OpenCV
                elif label_id_val in random_color_pool_bgr:
                    color_bgr = random_color_pool_bgr[label_id_val]
                else:
                    color_bgr = [30, 30, 30] # Fallback dark gray (BGR)

                colored_mask_img[mask_tensor == label_id_val] = color_bgr

            mask_filename = mask_savedir / f"{t}_local_mask.png"
            cv2.imwrite(str(mask_filename), colored_mask_img)


def save_ply(filename, points, colors, labels, label_map, property_name="quality"):
    colors = colors.astype(np.uint8)
    print(f"[DEBUG save_ply] colors shape: {colors.shape}, points shape: {points.shape}")

    num_points = len(points)
    if len(colors) != num_points:
        raise ValueError(f"Mismatch in array lengths: points ({num_points}), colors ({len(colors)})")

    pcd_dtype_list = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]

    if labels is not None and property_name:
        if len(labels) != num_points:
            raise ValueError(
                f"Mismatch in array lengths for label property '{property_name}': points ({num_points}), labels ({len(labels)})"
            )
        # Check if labels exceed uint8 range
        max_label = labels.max()
        if max_label > 255:
            print(f"[WARNING save_ply] Label values exceed uint8 range (max={max_label}). Using uint16 for PLY property.")
            labels = labels.astype(np.uint16)
            pcd_dtype_list.append((property_name, "u2"))  # u2 for uint16
        else:
            labels = labels.astype(np.uint8)
            pcd_dtype_list.append((property_name, "u1"))
    elif labels is not None and not property_name:
        print(f"[Warning save_ply] Labels were provided but no property_name was specified. Labels will not be saved as a separate property.")

    pcd = np.empty(num_points, dtype=pcd_dtype_list)

    pcd["x"], pcd["y"], pcd["z"] = points.T
    print(f"[DEBUG save_ply] Before assignment - colors.T shape: {colors.T.shape}")
    print(f"[DEBUG save_ply] Sample colors (first 5): {colors[:5]}")
    pcd["red"], pcd["green"], pcd["blue"] = colors.T
    print(f"[DEBUG save_ply] After assignment - sample pcd colors (first 5):")
    for i in range(min(5, len(pcd))):
        print(f"  Point {i}: R={pcd['red'][i]}, G={pcd['green'][i]}, B={pcd['blue'][i]}")

    if labels is not None and property_name:
        pcd[property_name] = labels

    vertex_element = PlyElement.describe(pcd, "vertex")

    comments = []
    if label_map:
        comments.append("label_map_start")
        try:
            sorted_label_map_items = sorted(label_map.items(), key=lambda item: int(item[0]))
        except ValueError:
            sorted_label_map_items = sorted(label_map.items())

        for label_id, name in sorted_label_map_items:
            comments.append(f"label_id {label_id}: {name}")
        comments.append("label_map_end")

    ply_data = PlyData([vertex_element], text=False, comments=comments)
    try:
        ply_data.write(filename)
    except Exception as e_ply_write:
        print(f"[ERROR eval.py] Failed to write PLY file {filename}: {e_ply_write}")
