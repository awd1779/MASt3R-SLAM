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

# Define a custom color map for semantic labels (RGB 0-255)
SEMANTIC_COLOR_MAP = {
    "background": [50, 50, 50],   # Dark Gray
    "monitor":    [0, 0, 255],    # Blue
    "screen":     [0, 0, 200],    # Slightly darker Blue
    "book":       [255, 0, 0],    # Red
    "game controller": [0, 255, 0], # Green
    "chair":      [255, 165, 0],  # Orange
    "desk":       [128, 0, 128],  # Purple
    "table":      [255, 255, 0],  # Yellow
    "cup":        [0, 255, 255],  # Cyan
    "keyboard":   [255, 0, 255],  # Magenta
    "mouse":      [100, 100, 0],  # Dark Yellow/Olive
    "laptop":     [0, 128, 128],  # Teal
}



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
        else:
            print(f"[Warning eval.py] Keyframe {keyframe.frame_id} (idx {i}) has no global_instance_ids. Using default label 0 for its {np.sum(valid_indices)} valid points.")
            labels_list.append(np.zeros(np.sum(valid_indices), dtype=np.uint8))

    if not pointclouds_list:
        print("[Warning eval.py] No valid points collected from any keyframes. Skipping PLY generation.")
        return

    pointclouds_np = np.concatenate(pointclouds_list, axis=0)
    colors_np = np.concatenate(colors_list, axis=0)

    if not labels_list or all(arr is None or len(arr) == 0 for arr in labels_list):
        labels_global_np = np.zeros(len(pointclouds_np), dtype=np.uint8)
    else:
        valid_label_arrays = [lbl_arr for lbl_arr in labels_list if lbl_arr is not None and len(lbl_arr) > 0]
        if not valid_label_arrays:
            labels_global_np = np.zeros(len(pointclouds_np), dtype=np.uint8)
        else:
            try:
                labels_global_np = np.concatenate(valid_label_arrays, axis=0).astype(np.uint8)
            except ValueError as e_concat:
                print(f"[ERROR eval.py] Failed to concatenate label arrays: {e_concat}. Using default labels.")
                labels_global_np = np.zeros(len(pointclouds_np), dtype=np.uint8)

    if len(labels_global_np) != len(pointclouds_np):
        print(f"[CRITICAL ERROR eval.py] Final label count ({len(labels_global_np)}) does not match point count ({len(pointclouds_np)}). Using default labels.")
        labels_global_np = np.zeros(len(pointclouds_np), dtype=np.uint8)

    print(f"[DIAGNOSTIC evaluate.py - save_reconstruction FINAL AGGREGATION]")
    print(f"  Total points for PLY: {pointclouds_np.shape[0]}")
    if labels_global_np.size > 0:
        unique_final_labels, counts_final_labels = np.unique(labels_global_np, return_counts=True)
        print(f"  Unique final labels in PLY (ID: count): {list(zip(unique_final_labels.tolist(), counts_final_labels.tolist()))}")
    print(f"  Label map for PLY comments (from tracker): {final_label_map_for_ply_comments}")
    print(f"[DIAGNOSTIC evaluate.py - save_reconstruction END]")

    ply_main_filename = savedir / filename
    save_ply(ply_main_filename, pointclouds_np, colors_np, labels_global_np, final_label_map_for_ply_comments, property_name="quality")

    if labels_global_np.size > 0 and labels_global_np.shape[0] == pointclouds_np.shape[0]:
        filepath_obj = pathlib.Path(filename)
        seg_color_ply_name = f"{filepath_obj.stem}_seg_color.ply"
        print(f"[INFO eval.py] Generating PLY with segmentation colors: {seg_color_ply_name}")

        # Use custom semantic color map
        seg_colors_np = np.zeros_like(colors_np)
        unique_ids_in_ply = np.unique(labels_global_np)

        # Seed for consistent random colors for unmapped labels
        np.random.seed(42)
        # Generate a pool of random colors for labels not in SEMANTIC_COLOR_MAP
        random_color_pool = {label_id: list(np.random.randint(50, 251, size=3))
                             for label_id in unique_ids_in_ply}

        for label_id_val in unique_ids_in_ply:
            class_label = final_label_map_for_ply_comments.get(label_id_val, "unknown")
            
            # Prioritize SEMANTIC_COLOR_MAP, then random_color_pool, then a default fallback
            if class_label in SEMANTIC_COLOR_MAP:
                color_rgb = SEMANTIC_COLOR_MAP[class_label]
            elif label_id_val in random_color_pool:
                color_rgb = random_color_pool[label_id_val]
            else:
                color_rgb = [30, 30, 30] # Fallback dark gray

            seg_colors_np[labels_global_np == label_id_val] = color_rgb

        seg_color_filename_path = savedir / seg_color_ply_name
        save_ply(seg_color_filename_path, pointclouds_np, seg_colors_np, labels=None, label_map=final_label_map_for_ply_comments, property_name=None)
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
        labels = labels.astype(np.uint8)
        pcd_dtype_list.append((property_name, "u1"))
    elif labels is not None and not property_name:
        print(f"[Warning save_ply] Labels were provided but no property_name was specified. Labels will not be saved as a separate property.")

    pcd = np.empty(num_points, dtype=pcd_dtype_list)

    pcd["x"], pcd["y"], pcd["z"] = points.T
    pcd["red"], pcd["green"], pcd["blue"] = colors.T

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
