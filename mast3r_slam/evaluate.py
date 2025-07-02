import pathlib
from typing import Optional
import cv2
import numpy as np
import torch
from mast3r_slam.dataloader import Intrinsics
from mast3r_slam.frame import SharedKeyframes
from mast3r_slam.lietorch_utils import as_SE3
from mast3r_slam.config import config
# from mast3r_slam.geometry import constrain_points_to_ray # Not strictly needed if X_canon is used as is
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
        for i in range(len(frames)):
            keyframe = frames[i]
            t = timestamps[keyframe.frame_id]
            if intrinsics is None:
                T_WC = as_SE3(keyframe.T_WC)
            else:
                # This refine_pose_with_calibration was specific to some eval setups,
                # for general saving, using keyframe.T_WC is more direct.
                # T_WC = intrinsics.refine_pose_with_calibration(keyframe)
                T_WC = as_SE3(keyframe.T_WC) # Use the pose stored in the keyframe
            x, y, z, qx, qy, qz, qw = T_WC.data.squeeze().cpu().numpy() # Ensure squeeze if T_WC has batch dim 1
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

        # Note: `constrain_points_to_ray` was previously in visualization and here.
        # If `keyframe.X_canon` is already the set of points whose labels are in `keyframe.global_instance_ids`,
        # then further modification by `constrain_points_to_ray` here (if it changes point count/order)
        # would misalign labels. We assume `keyframe.X_canon` is the definitive geometry.
        # If `use_calib` implies X_canon should always be ray-constrained, that should happen before label association.

        pW_all_kf_points = keyframe.T_WC.act(X_canon_for_proj).cpu().numpy().reshape(-1, 3)
        # Ensure uimg is a tensor before .cpu().numpy() if it comes from shared memory as numpy
        uimg_np = keyframe.uimg.cpu().numpy() if isinstance(keyframe.uimg, torch.Tensor) else keyframe.uimg
        color_all_kf_points = (uimg_np * 255).astype(np.uint8).reshape(-1, 3)

        conf_tensor = keyframe.get_average_conf()
        if conf_tensor is None or conf_tensor.numel() == 0:
            print(f"[Warning eval.py] Keyframe {keyframe.frame_id} (idx {i}) has no confidence data. Skipping confidence filter, taking all points.")
            confidence_all_kf_points = np.ones(X_canon_for_proj.shape[0], dtype=np.float32) * (c_conf_threshold + 1.0) # Effectively take all
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
        print("[Warning eval.py] No actual labels were collected for PLY. Using default label 0 for all points.")
        labels_global_np = np.zeros(len(pointclouds_np), dtype=np.uint8)
    else:
        valid_label_arrays = [lbl_arr for lbl_arr in labels_list if lbl_arr is not None and len(lbl_arr) > 0]
        if not valid_label_arrays:
            print("[Warning eval.py] No valid label arrays to concatenate. Using default label 0.")
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
    # ... (other diagnostic prints can be kept or removed as needed) ...
    print(f"  Label map for PLY comments (from tracker): {final_label_map_for_ply_comments}")
    print(f"[DIAGNOSTIC evaluate.py - save_reconstruction END]")

    ply_main_filename = savedir / filename
    save_ply(ply_main_filename, pointclouds_np, colors_np, labels_global_np, final_label_map_for_ply_comments, property_name="quality")

    if labels_global_np.size > 0 and labels_global_np.shape[0] == pointclouds_np.shape[0]:
        filepath_obj = pathlib.Path(filename)
        seg_color_ply_name = f"{filepath_obj.stem}_seg_color.ply"
        print(f"[INFO eval.py] Generating PLY with segmentation colors: {seg_color_ply_name}")

        np.random.seed(42)
        unique_ids_in_ply = np.unique(labels_global_np)
        dynamic_rgb_color_map = {label_id: list(np.random.randint(50, 251, size=3)) # Avoid pure black/white, stay in 0-255 uchar range
                                 for label_id in unique_ids_in_ply if label_id != 0}
        dynamic_rgb_color_map[0] = [128, 128, 128]

        default_seg_color_rgb = [30, 30, 30]

        seg_colors_np = np.zeros_like(colors_np)
        for label_id_val in unique_ids_in_ply:
            color_rgb = dynamic_rgb_color_map.get(label_id_val, default_seg_color_rgb)
            seg_colors_np[labels_global_np == label_id_val] = color_rgb

        seg_color_filename_path = savedir / seg_color_ply_name
        save_ply(seg_color_filename_path, pointclouds_np, seg_colors_np, labels=None, label_map=final_label_map_for_ply_comments, property_name=None)
        print(f"[INFO eval.py] Saved PLY with segmentation colors to {seg_color_filename_path}")


def save_keyframes(savedir, timestamps, keyframes: SharedKeyframes):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)

    mask_savedir = savedir / "masks"
    mask_savedir.mkdir(exist_ok=True, parents=True)

    label_to_color_map_bgr = {
        0: [128, 128, 128], 1: [0, 0, 255], 2: [0, 255, 0], 3: [255, 0, 0],
        4: [0, 255, 255], 5: [255, 0, 255], 6: [255, 255, 0],
    }
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

        # Use local_instance_mask for saving mask images
        if keyframe.local_instance_mask is not None and keyframe.local_instance_mask.numel() > 0:
            mask_tensor = keyframe.local_instance_mask.cpu().numpy().astype(np.uint8)
            h, w = mask_tensor.shape
            colored_mask_img = np.zeros((h, w, 3), dtype=np.uint8)

            # Use a dynamic color map for mask images as well for many SAM segments
            unique_labels_in_mask = np.unique(mask_tensor)
            np.random.seed(42) # Consistent colors with _seg_color.ply if desired
            dynamic_bgr_color_map_mask = {
                label_id: list(np.random.randint(50, 251, size=3).astype(np.uint8))
                for label_id in unique_labels_in_mask if label_id != 0
            }
            dynamic_bgr_color_map_mask[0] = [128,128,128]


            for label_id_val in unique_labels_in_mask:
                color_bgr = dynamic_bgr_color_map_mask.get(label_id_val, default_mask_color_bgr)
                colored_mask_img[mask_tensor == label_id_val] = color_bgr

            mask_filename = mask_savedir / f"{t}_local_mask.png" # Renamed to reflect it's local
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
        # Sort by ID for consistent header, converting keys to int for sorting if they are strings
        try:
            sorted_label_map_items = sorted(label_map.items(), key=lambda item: int(item[0]))
        except ValueError: # Handle non-integer keys if they somehow occur, though less likely now
            sorted_label_map_items = sorted(label_map.items())

        for label_id, name in sorted_label_map_items:
            comments.append(f"label_id {label_id}: {name}")
        comments.append("label_map_end")

    ply_data = PlyData([vertex_element], text=False, comments=comments)
    try:
        ply_data.write(filename)
    except Exception as e_ply_write:
        print(f"[ERROR eval.py] Failed to write PLY file {filename}: {e_ply_write}")
