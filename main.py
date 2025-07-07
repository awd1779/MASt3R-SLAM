import argparse
import datetime
import pathlib
import sys
import time
import cv2
import lietorch
import torch
import tqdm
import yaml
import json # Added for saving semantic map
from mast3r_slam.global_opt import FactorGraph

from mast3r_slam.config import load_config, config, set_global_config
from mast3r_slam.dataloader import Intrinsics, load_dataset
import mast3r_slam.evaluate as eval
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.mast3r_utils import (
    load_mast3r,
    load_retriever,
    mast3r_inference_mono,
)
from mast3r_slam.multiprocess_utils import new_queue, try_get_msg
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.visualization import WindowMsg, run_visualization
import torch.multiprocessing as mp
from mast3r_slam.semantic_processor_v2 import SemanticProcessorV2
from mast3r_slam.semantic_core import TEXT_PROMPTS as SEMANTIC_TEXT_PROMPTS # Use new core module
from mast3r_slam.semantic_utils import SemanticMapper # Use consolidated utils


def relocalization(frame, keyframes, factor_graph, retrieval_database):
    # we are adding and then removing from the keyframe, so we need to be careful.
    # The lock slows viz down but safer this way...
    with keyframes.lock:
        kf_idx = []
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=False,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds
        successful_loop_closure = False
        if kf_idx:
            keyframes.append(frame)
            n_kf = len(keyframes)
            kf_idx = list(kf_idx)  # convert to list
            frame_idx = [n_kf - 1] * len(kf_idx)
            print("RELOCALIZING against kf ", n_kf - 1, " and ", kf_idx)
            if factor_graph.add_factors(
                frame_idx,
                kf_idx,
                config["reloc"]["min_match_frac"],
                is_reloc=config["reloc"]["strict"],
            ):
                retrieval_database.update(
                    frame,
                    add_after_query=True,
                    k=config["retrieval"]["k"],
                    min_thresh=config["retrieval"]["min_thresh"],
                )
                print("Success! Relocalized")
                successful_loop_closure = True
                keyframes.T_WC[n_kf - 1] = keyframes.T_WC[kf_idx[0]].clone()
            else:
                keyframes.pop_last()
                print("Failed to relocalize")

        if successful_loop_closure:
            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()
        return successful_loop_closure


def run_backend(cfg, model, states, keyframes, K):
    set_global_config(cfg)

    device = keyframes.device
    model.to(device)
    factor_graph = FactorGraph(model, keyframes, K, device)
    retrieval_database = load_retriever(model)

    mode = states.get_mode()
    while mode is not Mode.TERMINATED:
        mode = states.get_mode()
        if mode == Mode.INIT or states.is_paused():
            time.sleep(0.01)
            continue
        if mode == Mode.RELOC:
            frame = states.get_frame()
            success = relocalization(frame, keyframes, factor_graph, retrieval_database)
            if success:
                states.set_mode(Mode.TRACKING)
            states.dequeue_reloc()
            continue
        idx = -1
        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks[0]
        if idx == -1:
            time.sleep(0.01)
            continue

        # Graph Construction
        kf_idx = []
        # k to previous consecutive keyframes
        n_consec = 1
        for j in range(min(n_consec, idx)):
            kf_idx.append(idx - 1 - j)
        frame = keyframes[idx]
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=True,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds

        lc_inds = set(retrieval_inds)
        lc_inds.discard(idx - 1)
        if len(lc_inds) > 0:
            print("Database retrieval", idx, ": ", lc_inds)

        kf_idx = set(kf_idx)  # Remove duplicates by using set
        kf_idx.discard(idx)  # Remove current kf idx if included
        kf_idx = list(kf_idx)  # convert to list
        frame_idx = [idx] * len(kf_idx)
        if kf_idx:
            factor_graph.add_factors(
                kf_idx, frame_idx, config["local_opt"]["min_match_frac"]
            )

        with states.lock:
            states.edges_ii[:] = factor_graph.ii.cpu().tolist()
            states.edges_jj[:] = factor_graph.jj.cpu().tolist()

        if config["use_calib"]:
            factor_graph.solve_GN_calib()
        else:
            factor_graph.solve_GN_rays()

        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks.pop(0)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = "cuda:0"
    save_frames = False
    datetime_now = str(datetime.datetime.now()).replace(" ", "_")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/tum/rgbd_dataset_freiburg1_desk")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--save-as", default="default")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--calib", default="")

    args = parser.parse_args()

    load_config(args.config)
    print(args.dataset)
    print(config)

    manager = mp.Manager()
    main2viz = new_queue(manager, args.no_viz)
    viz2main = new_queue(manager, args.no_viz)

    dataset = load_dataset(args.dataset)
    dataset.subsample(config["dataset"]["subsample"])
    h, w = dataset.get_img_shape()[0]

    if args.calib:
        with open(args.calib, "r") as f:
            intrinsics = yaml.load(f, Loader=yaml.SafeLoader)
        config["use_calib"] = True
        dataset.use_calibration = True
        dataset.camera_intrinsics = Intrinsics.from_calib(
            dataset.img_size,
            intrinsics["width"],
            intrinsics["height"],
            intrinsics["calibration"],
        )

    keyframes = SharedKeyframes(manager, h, w)
    states = SharedStates(manager, h, w)

    if not args.no_viz:
        viz = mp.Process(
            target=run_visualization,
            args=(config, states, keyframes, main2viz, viz2main),
        )
        viz.start()

    model = load_mast3r(device=device)
    model.share_memory()

    has_calib = dataset.has_calib()
    use_calib = config["use_calib"]

    if use_calib and not has_calib:
        print("[Warning] No calibration provided for this dataset!")
        sys.exit(0)
    K = None
    if use_calib:
        K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(
            device, dtype=torch.float32
        )
        keyframes.set_intrinsics(K)

    # remove the trajectory from the previous run
    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        traj_file = save_dir / f"{seq_name}.txt"
        # recon_file = save_dir / f"{seq_name}.ply" # Path for the main PLY
        # seg_color_recon_file = save_dir / f"{seq_name}_seg_color.ply" # Path for the seg color PLY

        # It's good practice to remove potentially multiple output files if they exist
        for ply_suffix in [".ply", "_seg_color.ply"]:
            recon_file_to_check = save_dir / f"{seq_name}{ply_suffix}"
            if recon_file_to_check.exists():
                recon_file_to_check.unlink()

        if traj_file.exists():
            traj_file.unlink()


    # Initialize the map for global instance IDs to their class names
    # This map will be populated by FrameTracker
    g_initial_global_id_to_class_label_map = {0: "background"}

    # Instantiate FrameTracker, passing the initial map and semantic processor
    tracker = FrameTracker(model, keyframes, device, g_initial_global_id_to_class_label_map, semantic_processor=None)  # Will initialize after creating processor
    
    # Create semantic mapper for improved point-to-pixel alignment
    semantic_mapper = SemanticMapper(device=device)
    
    # Initialize SemanticProcessorV2 with optimizations
    semantic_processor = SemanticProcessorV2(
        device=device,
        enable_batch_processing=True,
        enable_caching=True,
        enable_post_processing=True,
        enable_ensemble=False,  # Disable for speed, set True for accuracy
        cache_dir=pathlib.Path("semantic_cache") if dataset.save_results else None,
        text_prompts=SEMANTIC_TEXT_PROMPTS
    )
    semantic_processor.initialize_models()
    print("[INFO] SemanticProcessorV2 initialized with batch processing and caching enabled")
    
    # Update tracker with semantic processor
    tracker.semantic_processor = semantic_processor
    
    last_msg = WindowMsg()

    backend = mp.Process(target=run_backend, args=(config, model, states, keyframes, K))
    backend.start()

    i = 0
    fps_timer = time.time()

    frames = []

    while True:
        mode = states.get_mode()
        msg = try_get_msg(viz2main)
        last_msg = msg if msg is not None else last_msg
        if last_msg.is_terminated:
            states.set_mode(Mode.TERMINATED)
            break

        if last_msg.is_paused and not last_msg.next:
            states.pause()
            time.sleep(0.01)
            continue

        if not last_msg.is_paused:
            states.unpause()

        if i == len(dataset):
            states.set_mode(Mode.TERMINATED)
            break

        timestamp, img = dataset[i]
        if save_frames:
            frames.append(img)

        # get frames last camera pose
        T_WC = (
            lietorch.Sim3.Identity(1, device=device)
            if i == 0
            else states.get_frame().T_WC
        )
        frame = create_frame(i, img, T_WC, img_size=dataset.img_size, device=device)

        if mode == Mode.INIT:
            # Initialize via mono inference, and encoded features neeed for database
            X_init, C_init = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X_init, C_init) # Populates X_canon

            # --- New: Process semantics for the first frame ---
            print(f"[INFO main.py] Processing semantics for initial frame {frame.frame_id}...")
            try:
                # Use SemanticProcessorV2 for better performance
                frame_data = {
                    'image_tensor': frame.img.squeeze(0),  # (C,H,W)
                    'frame_id': frame.frame_id,
                    'pose': frame.T_WC.matrix()[0] if hasattr(frame, 'T_WC') and frame.T_WC is not None else None
                }
                result = semantic_processor.process_frame(frame_data, mode='full', use_cache=False)  # No cache for first frame
                local_mask = result['local_instance_mask']
                local_map = result['local_id_to_class_map']
                frame.local_instance_mask = local_mask.to(device if local_mask is not None else None)
                frame.local_id_to_class_label_map = local_map
                # global_instance_ids for this first frame will be set by FrameTracker.track
                # when it handles the "first keyframe" case.
            except Exception as e_semantic_init:
                print(f"[ERROR main.py] Semantic processing failed for initial frame {frame.frame_id}: {e_semantic_init}")
                frame.local_instance_mask = None
                frame.local_id_to_class_label_map = None
            # --- End Semantic Processing for first frame ---

            # Initialize global_instance_ids for the first frame based on its local semantic masks
            if frame.local_instance_mask is not None and frame.X_canon is not None:
                num_points = frame.X_canon.shape[0]
                frame.global_instance_ids = torch.zeros(num_points, 1, dtype=torch.int64, device=device)
                
                # Use improved semantic mapper for point-to-pixel alignment
                # Get confidence map and ensure it's 1D
                conf_map = None
                if hasattr(frame, 'C') and frame.C is not None:
                    conf_map = frame.C.squeeze()  # Remove any extra dimensions
                    if conf_map.dim() > 1:
                        conf_map = conf_map.view(-1)  # Flatten to 1D
                
                labels, valid_mask = semantic_mapper.map_2d_labels_to_3d_points(
                    semantic_mask=frame.local_instance_mask,
                    confidence_map=conf_map,
                    num_points=num_points,
                    frame_shape=(frame.local_instance_mask.shape[0], frame.local_instance_mask.shape[1]),  # Use mask shape
                    confidence_threshold=0.1  # Very low threshold for maximum coverage
                )
                
                # Build confidence map for each label
                label_confidences = {}
                
                # Assign global IDs to labeled points
                unique_local_ids = torch.unique(labels)
                for local_id_tensor in unique_local_ids:
                    local_id = local_id_tensor.item()
                    if local_id == 0:  # Skip background
                        continue
                    
                    # Find all points with this label
                    label_mask = (labels == local_id) & valid_mask
                    
                    if label_mask.any():
                        # Assign a new global ID
                        new_global_id = tracker.g_next_global_id
                        frame.global_instance_ids[label_mask] = new_global_id
                        
                        # Update the global semantic map
                        class_name = frame.local_id_to_class_label_map.get(local_id, f"unknown_local_id_{local_id}")
                        tracker.g_global_id_to_class_label_map[new_global_id] = class_name
                        tracker.g_next_global_id += 1
                        
                        # Track confidence for this label (dummy value for now)
                        label_confidences[new_global_id] = 50.0  # Default confidence
                        
                        labeled_count = label_mask.sum().item()
                        print(f"[DEBUG] Label {local_id} ({class_name}): {labeled_count} points labeled with global ID {new_global_id}")
                
                # Apply 3D refinement if we have 3D points
                if frame.X_canon is not None and frame.global_instance_ids.sum() > 0:
                    print(f"[INFO] Applying 3D spatial refinement to initial labels...")
                    refined_labels = semantic_mapper.refine_labels_with_3d_proximity(
                        points_3d=frame.X_canon,
                        labels=frame.global_instance_ids.view(-1),
                        label_confidences=label_confidences,
                        distance_threshold=0.1,
                        min_neighbors=3
                    )
                    frame.global_instance_ids = refined_labels.view(-1, 1)
                
                print(f"[INFO main.py] Initialized global IDs for first frame. Map: {tracker.g_global_id_to_class_label_map}")

            keyframes.append(frame) # Add after semantic processing
            states.queue_global_optimization(len(keyframes) - 1)
            states.set_mode(Mode.TRACKING) # Important: set mode AFTER potentially calling tracker for first frame processing if needed
            states.set_frame(frame)
            i += 1
            continue

        if mode == Mode.TRACKING:
            add_new_kf, match_info, try_reloc = tracker.track(frame)
            if try_reloc:
                states.set_mode(Mode.RELOC)
            states.set_frame(frame)

        elif mode == Mode.RELOC:
            X, C = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X, C)
            states.set_frame(frame)
            states.queue_reloc()
            # In single threaded mode, make sure relocalization happen for every frame
            while config["single_thread"]:
                with states.lock:
                    if states.reloc_sem.value == 0:
                        break
                time.sleep(0.01)

        else:
            raise Exception("Invalid mode")

        if add_new_kf:
            # Ensure frame has semantic data before adding as keyframe
            if frame.local_instance_mask is None and frame.img is not None:
                print(f"[INFO main.py] Processing semantics for new keyframe {frame.frame_id}")
                try:
                    frame_data = {
                        'image_tensor': frame.img.squeeze(0),
                        'frame_id': frame.frame_id,
                        'pose': frame.T_WC.matrix()[0] if hasattr(frame, 'T_WC') and frame.T_WC is not None else None,
                        'global_ids': frame.global_instance_ids
                    }
                    result = semantic_processor.process_frame(frame_data, mode='full', use_cache=False)
                    frame.local_instance_mask = result['local_instance_mask'].to(device)
                    frame.local_id_to_class_label_map = result['local_id_to_class_map']
                except Exception as e:
                    print(f"[ERROR main.py] Semantic processing failed for keyframe {frame.frame_id}: {e}")
            
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
            # In single threaded mode, wait for the backend to finish
            while config["single_thread"]:
                with states.lock:
                    if len(states.global_optimizer_tasks) == 0:
                        break
                time.sleep(0.01)
        # log time
        if i % 30 == 0:
            FPS = i / (time.time() - fps_timer)
            print(f"FPS: {FPS}")
        i += 1

    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        eval.save_traj(save_dir, f"{seq_name}.txt", dataset.timestamps, keyframes)

        # Retrieve the final populated map from the tracker instance
        final_global_semantic_map = tracker.g_global_id_to_class_label_map

        eval.save_reconstruction(
            save_dir,
            f"{seq_name}.ply", # Base filename for PLY files
            keyframes,
            last_msg.C_conf_threshold, # Or however this is determined
            global_id_to_name_map=final_global_semantic_map # Pass the populated map
        )
        # Save the global_id_to_name_map to a JSON file
        semantic_map_file = save_dir / f"{seq_name}_semantic_map.json"
        # Convert keys to strings for JSON serialization if they are integers
        serializable_map = {str(k): v for k, v in final_global_semantic_map.items()}
        with open(semantic_map_file, 'w') as f:
            json.dump(serializable_map, f, indent=4)
        print(f"Saved semantic map to {semantic_map_file}")

        eval.save_keyframes(
            save_dir / "keyframes" / seq_name, dataset.timestamps, keyframes
        )
    if save_frames:
        savedir = pathlib.Path(f"logs/frames/{datetime_now}")
        savedir.mkdir(exist_ok=True, parents=True)
        for i, frame in tqdm.tqdm(enumerate(frames), total=len(frames)):
            frame = (frame * 255).clip(0, 255)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{savedir}/{i}.png", frame)

    print("done")
    
    # Print semantic processor performance summary
    print("\n[SemanticProcessorV2] Performance Summary:")
    perf_summary = semantic_processor.get_performance_summary()
    for key, stats in perf_summary.items():
        if isinstance(stats, dict) and 'mean_ms' in stats:
            print(f"  {key}: mean={stats['mean_ms']:.1f}ms, total={stats['total_s']:.2f}s, count={stats['count']}")
    
    # Clean up semantic processor resources
    semantic_processor.shutdown()
    
    backend.join()
    if not args.no_viz:
        viz.join()
