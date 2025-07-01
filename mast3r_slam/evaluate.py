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

    pointclouds = np.concatenate(pointclouds, axis=0)
    colors = np.concatenate(colors, axis=0)
    labels_global = np.concatenate(labels_global, axis=0).astype(np.uint8) # Ensure uchar for PLY

    save_ply(savedir / filename, pointclouds, colors, labels_global, global_label_map)


def save_keyframes(savedir, timestamps, keyframes: SharedKeyframes):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        t = timestamps[keyframe.frame_id]
        filename = savedir / f"{t}.png"
        cv2.imwrite(
            str(filename),
            cv2.cvtColor(
                (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
            ),
        )


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
        ("label_id", "u1") # Add label_id as unsigned char
    ]
    pcd = np.empty(num_points, dtype=pcd_dtype)

    pcd["x"], pcd["y"], pcd["z"] = points.T
    pcd["red"], pcd["green"], pcd["blue"] = colors.T
    pcd["label_id"] = labels

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
