import pathlib
from typing import Optional
import cv2
import numpy as np
import torch
from mast3r_slam.dataloader import Intrinsics
from mast3r_slam.frame import SharedKeyframes
from mast3r_slam.lietorch_utils import as_SE3
from mast3r_slam.config import config
from mast3r_slam.geometry import constrain_points_to_ray
from plyfile import PlyData, PlyElement


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
    # log
    logdir = pathlib.Path(logdir)
    logdir.mkdir(exist_ok=True, parents=True)
    logfile = logdir / logfile
    with open(logfile, "w") as f:
        # for keyframe_id in frames.keyframe_ids:
        for i in range(len(frames)):
            keyframe = frames[i]
            t = timestamps[keyframe.frame_id]
            if intrinsics is None:
                T_WC = as_SE3(keyframe.T_WC)
            else:
                T_WC = intrinsics.refine_pose_with_calibration(keyframe)
            x, y, z, qx, qy, qz, qw = T_WC.data.numpy().reshape(-1)
            f.write(f"{t} {x} {y} {z} {qx} {qy} {qz} {qw}\n")


def save_reconstruction(savedir, filename, keyframes, c_conf_threshold):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    pointclouds = []
    colors = []
    labels_global = [] # To store label IDs for each point
    global_label_map = {} # To consolidate label ID to name mappings

    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        if config["use_calib"]:
            # This X_canon modification should ideally not happen here if X_canon is truly canonical.
            # Or, if it does, point_labels need to be robust to this.
            # Assuming point_labels corresponds to X_canon *before* this potential modification.
            # For safety, let's get point_labels before X_canon is potentially reassigned.
            kf_point_labels_flat = None
            if keyframe.point_labels is not None:
                # Ensure point_labels corresponds to the original X_canon point count (H*W)
                # X_canon is (H*W, 3), point_labels is (H*W, 1)
                kf_point_labels_flat = keyframe.point_labels.cpu().numpy().reshape(-1)

            X_canon_for_proj = keyframe.X_canon
            if config["use_calib"]: # Re-check as X_canon might be modified by constrain_points_to_ray
                 X_canon_proj = constrain_points_to_ray(
                    keyframe.img_shape.flatten()[:2], keyframe.X_canon[None], keyframe.K
                )
                 X_canon_for_proj = X_canon_proj.squeeze(0) # This was keyframe.X_canon = X_canon.squeeze(0)

        else:
            X_canon_for_proj = keyframe.X_canon
            kf_point_labels_flat = None
            if keyframe.point_labels is not None:
                kf_point_labels_flat = keyframe.point_labels.cpu().numpy().reshape(-1)

        pW = keyframe.T_WC.act(X_canon_for_proj).cpu().numpy().reshape(-1, 3)
        color_flat = (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3)

        # Valid points based on confidence
        confidence_flat = keyframe.get_average_conf().cpu().numpy().astype(np.float32).reshape(-1)
        valid_indices = confidence_flat > c_conf_threshold

        pointclouds.append(pW[valid_indices])
        colors.append(color_flat[valid_indices])

        if kf_point_labels_flat is not None:
            if len(kf_point_labels_flat) == len(valid_indices): # Check if point_labels array matches points count
                labels_for_kf = kf_point_labels_flat[valid_indices]
                labels_global.append(labels_for_kf)
            else:
                # Fallback: if counts don't match (e.g. point_labels was None or wrong size)
                # Create default labels (e.g. 0) for the valid points of this keyframe
                print(f"[Warning] Mismatch or missing point_labels for keyframe {i}. Expected {len(valid_indices)}, got {len(kf_point_labels_flat) if kf_point_labels_flat is not None else 'None'}. Using default label 0.")
                labels_global.append(np.zeros(np.sum(valid_indices), dtype=np.uint8))
        else:
            # If no point_labels for this keyframe, append default labels
            labels_global.append(np.zeros(np.sum(valid_indices), dtype=np.uint8))

        if keyframe.label_map:
            # Naive merge: last seen label for an ID wins. Could be smarter.
            global_label_map.update(keyframe.label_map)

    pointclouds_np = np.concatenate(pointclouds, axis=0)
    colors_np = np.concatenate(colors, axis=0)

    # --- Start Diagnostic Prints for save_reconstruction (before concatenation) ---
    print(f"[DIAGNOSTIC evaluate.py - save_reconstruction PRE-CONCAT]")
    total_labels_collected_count = sum(len(lbl_arr) for lbl_arr in labels_global if lbl_arr is not None)
    print(f"  Number of keyframes processed for PLY: {len(keyframes)}")
    print(f"  Total points collected: {len(pointclouds_np)}")
    print(f"  Total color entries collected: {len(colors_np)}")
    print(f"  Number of label arrays in labels_global: {len(labels_global)}")
    print(f"  Total individual label entries collected (sum of lengths in labels_global): {total_labels_collected_count}")
    # --- End Diagnostic Prints ---

    if not labels_global: # Handle case where no labels were collected at all
        print("[Warning] No labels were collected in labels_global. PLY will have default label 0 for all points.")
        labels_global_np = np.zeros(len(pointclouds_np), dtype=np.uint8)
    elif all(arr is None or len(arr) == 0 for arr in labels_global): # All arrays are None or empty
        print("[Warning] All label arrays in labels_global are None or empty. PLY will have default label 0 for all points.")
        labels_global_np = np.zeros(len(pointclouds_np), dtype=np.uint8)
    else: # Concatenate if there's actual label data
        # Filter out None arrays before concatenation if any slipped through (shouldn't with current logic)
        valid_label_arrays = [lbl_arr for lbl_arr in labels_global if lbl_arr is not None and len(lbl_arr) > 0]
        if not valid_label_arrays:
            print("[Warning] No valid label arrays to concatenate. PLY will have default label 0 for all points.")
            labels_global_np = np.zeros(len(pointclouds_np), dtype=np.uint8)
        else:
            try:
                labels_global_np = np.concatenate(valid_label_arrays, axis=0).astype(np.uint8)
            except ValueError as e_concat:
                print(f"[ERROR] Failed to concatenate label arrays: {e_concat}. Lengths might be inconsistent.")
                print(f"  Lengths of arrays in labels_global: {[len(arr) for arr in valid_label_arrays]}")
                print(f"  This often happens if a keyframe had valid points but its label array was empty or mismatched.")
                print(f"  Saving PLY with default label 0 for all points.")
                labels_global_np = np.zeros(len(pointclouds_np), dtype=np.uint8)


    # --- Start Diagnostic Prints for save_reconstruction (POST-CONCAT) ---
    print(f"[DIAGNOSTIC evaluate.py - save_reconstruction POST-CONCAT]")
    print(f"  Shape of final pointclouds_np: {pointclouds_np.shape}")
    print(f"  Shape of final colors_np: {colors_np.shape}")
    print(f"  Shape of final labels_global_np: {labels_global_np.shape}")

    unique_final_labels_list = []
    if labels_global_np.size > 0:
        unique_final_labels, counts_final_labels = np.unique(labels_global_np, return_counts=True)
        unique_final_labels_list = unique_final_labels.tolist()
        print(f"  Unique final labels (ID: count): {list(zip(unique_final_labels_list, counts_final_labels.tolist()))}")
    else:
        print(f"  labels_global_np is empty.")

    # Reconstruct global_label_map based on actual unique labels found in the data
    # This ensures the PLY header map is accurate even if individual keyframe.label_map was not aggregated.
    # (The previous global_label_map.update(keyframe.label_map) was ineffective due to SharedKeyframes not storing dicts)
    reconstructed_global_label_map = {0: "background"} # Always include background
    if unique_final_labels_list: # If there are any labels at all
        max_id_in_data = 0
        if unique_final_labels_list: # Check if list is not empty
             max_id_in_data = max(unique_final_labels_list) if max(unique_final_labels_list) > 0 else 0

        for i in range(1, int(max_id_in_data) + 1):
            # Only add if the label ID was actually present in the final data, or assume all up to max_id could exist.
            # For SAM's object_X naming, it's safer to just map all up to max_id.
            reconstructed_global_label_map[i] = f"object_{i}"

    print(f"  Reconstructed Global label map for PLY: {reconstructed_global_label_map}")
    print(f"[DIAGNOSTIC evaluate.py - save_reconstruction END]")
    # --- End Diagnostic Prints ---

    save_ply(savedir / filename, pointclouds_np, colors_np, labels_global_np, reconstructed_global_label_map)


def save_keyframes(savedir, timestamps, keyframes: SharedKeyframes):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)

    mask_savedir = savedir / "masks" # Create a subdirectory for masks
    mask_savedir.mkdir(exist_ok=True, parents=True)

    # Define a simple color map for mask visualization (BGR for OpenCV)
    # Matches the GLSL shader for consistency (0:grey, 1:red, 2:green, etc.)
    label_to_color_map_bgr = {
        0: [128, 128, 128],  # Grey for label 0 (background)
        1: [0, 0, 255],      # Red for label 1
        2: [0, 255, 0],      # Green for label 2
        3: [255, 0, 0],      # Blue for label 3
        4: [0, 255, 255],    # Yellow for label 4
        5: [255, 0, 255],    # Magenta for label 5
        6: [255, 255, 0],    # Cyan for label 6
        # Add more if needed, or a default color
    }
    default_mask_color_bgr = [30, 30, 30] # Dark grey for undefined labels

    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        t = timestamps[keyframe.frame_id]

        # Save original keyframe image
        img_filename = savedir / f"{t}.png"
        cv2.imwrite(
            str(img_filename),
            cv2.cvtColor(
                (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
            ),
        )

        # Save segmentation mask if available
        if keyframe.raw_segmentation_mask is not None:
            mask_tensor = keyframe.raw_segmentation_mask.cpu().numpy().astype(np.uint8)
            h, w = mask_tensor.shape
            colored_mask_img = np.zeros((h, w, 3), dtype=np.uint8)

            unique_labels_in_mask = np.unique(mask_tensor)
            for label_id in unique_labels_in_mask:
                color_bgr = label_to_color_map_bgr.get(label_id, default_mask_color_bgr)
                colored_mask_img[mask_tensor == label_id] = color_bgr

            mask_filename = mask_savedir / f"{t}_mask.png"
            cv2.imwrite(str(mask_filename), colored_mask_img)


def save_ply(filename, points, colors, labels, label_map): # Added labels and label_map
    colors = colors.astype(np.uint8)
    labels = labels.astype(np.uint8) # Ensure labels are uchar

    # Combine XYZ, RGB, and Label ID into a structured array
    # Ensure that the number of points, colors, and labels are consistent
    num_points = len(points)
    if len(colors) != num_points or len(labels) != num_points:
        raise ValueError(
            f"Mismatch in array lengths: points ({num_points}), "
            f"colors ({len(colors)}), labels ({len(labels)})"
        )

    pcd_dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("quality", "u1") # Changed "label_id" to "quality" for MeshLab
    ]
    pcd = np.empty(num_points, dtype=pcd_dtype)

    pcd["x"], pcd["y"], pcd["z"] = points.T
    pcd["red"], pcd["green"], pcd["blue"] = colors.T
    pcd["quality"] = labels # Changed "label_id" to "quality"

    vertex_element = PlyElement.describe(pcd, "vertex")

    # Create comments for the label map
    comments = []
    if label_map:
        comments.append("label_map_start")
        for label_id, name in sorted(label_map.items()):
            comments.append(f"label_id {label_id}: {name}")
        comments.append("label_map_end")

    ply_data = PlyData([vertex_element], text=False, comments=comments)
    ply_data.write(filename)
