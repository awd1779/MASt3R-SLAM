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
from pathlib import Path
from mast3r_slam.global_opt import FactorGraph

from mast3r_slam.config import load_config, config, set_global_config
from mast3r_slam.dataloader import Intrinsics, load_dataset
import mast3r_slam.evaluate as eval
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.mast3r_utils import (
    load_mast3r,
    load_retriever,
    mast3r_inference_mono,
    resize_img,
)
from mast3r_slam.multiprocess_utils import new_queue, try_get_msg
from mast3r_slam.tracker import FrameTracker
from mast3r_slam.visualization import WindowMsg, run_visualization
import torch.multiprocessing as mp

# Import semantic components
from mast3r_slam.semantic_frame import SharedSemanticKeyframes, create_semantic_frame
from mast3r_slam.semantic_processor import start_semantic_processor
from mast3r_slam.grounded_sam2_real import start_real_grounded_sam2_processor
from mast3r_slam.grounded_sam2_config import GroundedSAM2ModelSelector


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


def run_backend(cfg, model, states, keyframes, semantic_keyframes, semantic_result_queue, K):
    # Use semantic backend if semantic segmentation is enabled
    if config.get("semantic_segmentation", {}).get("enabled", False):
        from mast3r_slam.semantic_backend_integration import run_semantic_backend
        return run_semantic_backend(cfg, model, states, keyframes, semantic_keyframes, semantic_result_queue, K)
    
    # Otherwise use standard backend
    set_global_config(cfg)

    device = keyframes.device
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

        with states.lock:
            try:
                states.global_optimizer_tasks.pop(0)
            except:
                continue

        retrieval_inds = retrieval_database.update(
            keyframes[idx],
            add_after_query=True,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        if config["global_opt"]["use_loop_closures"]:
            loop_kf_idx = []
            loop_kf_idx += retrieval_inds
            successful_loop_closure = False
            if loop_kf_idx:
                loop_kf_idx = list(loop_kf_idx)  # convert to list
                frame_idx = [idx] * len(loop_kf_idx)
                if factor_graph.add_factors(
                    frame_idx,
                    loop_kf_idx,
                    config["global_opt"]["loop_confidence_threshold"],
                ):
                    successful_loop_closure = True
            with states.lock:
                states.edges_ii.append(
                    factor_graph.ii.cpu().numpy()
                )
                states.edges_jj.append(
                    factor_graph.jj.cpu().numpy()
                )
        print(
            f"KF {idx} | Matching | #factors: {len(factor_graph.ii)}, #frames {len(keyframes)}"
        )

        if config["use_calib"]:
            factor_graph.solve_GN_calib()
        else:
            factor_graph.solve_GN_rays()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    device = torch.device("cuda:0")
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
    
    # Initialize semantic components if enabled
    semantic_keyframes = None
    semantic_processor = None
    semantic_frame_queue = None
    semantic_result_queue = None
    keyframe_saver = None
    
    if config.get("semantic_segmentation", {}).get("enabled", False):
        print("Initializing semantic segmentation...")
        semantic_keyframes = SharedSemanticKeyframes(manager, max_keyframes=1000, h=h, w=w)
        semantic_frame_queue = manager.Queue()
        semantic_result_queue = manager.Queue()
        
        # Get initial vocabulary from config
        vocabulary = config["semantic_segmentation"].get("initial_vocabulary", 
                                                         ["person", "chair", "table", "car", "bottle"])
        
        # Get device and model settings
        semantic_device = config["semantic_segmentation"]["grounded_sam2"].get("device", "cuda:1")
        use_mock = config["semantic_segmentation"].get("use_mock", False)
        
        if use_mock:
            print("Using mock semantic processor")
            # Start mock semantic processor
            semantic_processor = start_semantic_processor(
                semantic_frame_queue, 
                semantic_result_queue,
                vocabulary,
                semantic_device
            )
        else:
            print("Using real Grounded-SAM2 processor")
            # Configure model selector based on config
            model_selector = GroundedSAM2ModelSelector(
                target_fps=15,
                max_vram_gb=24,  # We have 22GB available
                quality_priority="quality"
            )
            
            # Force the models from config
            sam2_model = config["semantic_segmentation"]["grounded_sam2"]["model_type"]
            grounding_model = config["semantic_segmentation"]["grounded_sam2"]["grounding_model"]
            model_selector.sam2_model = sam2_model
            model_selector.grounding_model = grounding_model
            
            # Start real Grounded-SAM2 processor
            # Pass model directories explicitly
            sam2_dir = str(Path.home() / "models" / "segment-anything-2")
            grounding_dir = str(Path.home() / "models" / "GroundingDINO")
            
            # Get confidence threshold from config
            confidence_threshold = config["semantic_segmentation"]["grounded_sam2"]["confidence_threshold"]
            
            semantic_processor = start_real_grounded_sam2_processor(
                semantic_frame_queue, 
                semantic_result_queue,
                vocabulary,
                semantic_device,
                model_selector,
                confidence_threshold=confidence_threshold,
                sam2_checkpoint_dir=sam2_dir,
                grounding_dino_checkpoint_dir=grounding_dir
            )

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
        
        # Initialize keyframe saver for dense reconstruction
        if semantic_keyframes is not None:
            from mast3r_slam.keyframe_saver import KeyframeSaver
            keyframe_saver = KeyframeSaver(save_dir / "keyframe_data")
        traj_file = save_dir / f"{seq_name}.txt"
        recon_file = save_dir / f"{seq_name}.ply"
        if traj_file.exists():
            traj_file.unlink()
        if recon_file.exists():
            recon_file.unlink()

    tracker = FrameTracker(model, keyframes, device)
    last_msg = WindowMsg()

    backend = mp.Process(
        target=run_backend, 
        args=(config, model, states, keyframes, semantic_keyframes, semantic_result_queue, K)
    )
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
        # Resize image first to match SLAM processing
        img_resized = resize_img(img, dataset.img_size)
        frame = create_frame(i, img, T_WC, img_size=dataset.img_size, device=device)
        
        # Send frame to semantic processor if enabled
        if semantic_frame_queue is not None:
            # Convert resized image to numpy array for semantic processing
            # Use the unnormalized image which is already in [0,1] range
            img_numpy = (img_resized["unnormalized_img"] * 255).clip(0, 255).astype('uint8')
            semantic_frame_queue.put({
                'img': img_numpy,
                'frame_id': i
            })

        if mode == Mode.INIT:
            # Initialize via mono inference, and encoded features neeed for database
            X_init, C_init = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X_init, C_init)
            keyframes.append(frame)
            kf_idx = len(keyframes) - 1
            states.queue_global_optimization(kf_idx)
            
            # Save first keyframe immediately
            if keyframe_saver is not None:
                keyframe_saver.save_keyframe(kf_idx, frame)
                
            states.set_mode(Mode.TRACKING)
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
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
            
            # Save keyframe data for dense reconstruction
            if keyframe_saver is not None:
                kf_idx = len(keyframes) - 1
                keyframe_saver.save_keyframe(kf_idx, frame)
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

    # Terminate semantic processor
    if semantic_frame_queue is not None:
        semantic_frame_queue.put(None)  # Termination signal
        semantic_processor.join()

    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        eval.save_traj(save_dir, f"{seq_name}.txt", dataset.timestamps, keyframes)
        
        # Save regular reconstruction
        eval.save_reconstruction(
            save_dir,
            f"{seq_name}.ply",
            keyframes,
            last_msg.C_conf_threshold,
        )
        
        # Save semantic reconstruction if enabled
        if semantic_keyframes is not None:
            from mast3r_slam.semantic_export import save_semantic_reconstruction
            from mast3r_slam.track_manager import GlobalTrackManager
            from mast3r_slam.semantic_integration import SemanticSLAMBackend
            
            # Create semantic backend for export
            track_manager = GlobalTrackManager()
            semantic_backend = SemanticSLAMBackend(
                keyframes, semantic_keyframes, track_manager, K, device
            )
            
            # Process all semantic results from queue
            processed_count = 0
            queue_size = semantic_result_queue.qsize()
            print(f"Semantic result queue has {queue_size} items")
            
            # Also check if backend already processed them
            n_keyframes = len(keyframes)
            print(f"Total keyframes: {n_keyframes}")
            
            while True:
                try:
                    semantic_data = semantic_result_queue.get_nowait()
                    frame_id = semantic_data['frame_id']
                    n_instances = len(semantic_data.get('instance_ids', []))
                    print(f"Processing semantic frame {frame_id} with {n_instances} instances")
                    
                    # Find corresponding keyframe
                    found = False
                    for kf_idx in range(len(keyframes)):
                        try:
                            kf = keyframes[kf_idx]
                            # frame_id in Frame is accessed via dataset_idx attribute
                            if kf and hasattr(kf, 'frame_id') and int(kf.frame_id) == int(frame_id):
                                print(f"  Updating semantics for kf_idx={kf_idx}")
                                semantic_keyframes.update_semantics(kf_idx, semantic_data)
                                print(f"  Processing semantic keyframe")
                                semantic_backend.process_semantic_keyframe(kf_idx)
                                
                                # Save semantic data with keyframe
                                if keyframe_saver is not None:
                                    keyframe_saver.save_keyframe(kf_idx, kf, semantic_data)
                                
                                processed_count += 1
                                print(f"✓ Processed semantic data for frame {frame_id} -> keyframe {kf_idx}")
                                found = True
                                break
                        except Exception as e:
                            print(f"  Error at kf_idx={kf_idx}: {e}")
                            import traceback
                            traceback.print_exc()
                            raise
                    
                    if not found:
                        print(f"⚠ No keyframe found for semantic frame {frame_id}")
                except Exception as e:
                    if "empty" not in str(e).lower():
                        print(f"Error processing semantic queue: {e}")
                    break
            
            print(f"Processed {processed_count} semantic frames")
            
            # Export semantic point cloud (sparse SLAM points)
            save_semantic_reconstruction(
                save_dir,
                f"{seq_name}_semantic_sparse.ply",
                keyframes,
                semantic_backend,
                last_msg.C_conf_threshold,
                use_semantic_colors=True
            )
            
            print(f"Saved sparse semantic reconstruction to {save_dir}/{seq_name}_semantic_sparse.ply")
            
            # Export dense semantic point cloud
            from mast3r_slam.dense_semantic_reconstruction import create_dense_semantic_reconstruction
            from mast3r_slam.dense_semantic_reconstruction_v2 import create_dense_semantic_reconstruction_v2
            
            # Try the original version first
            dense_result = create_dense_semantic_reconstruction(
                keyframes,
                semantic_keyframes,
                semantic_backend,
                str(save_dir / f"{seq_name}_semantic_dense.ply"),
                confidence_threshold=0.3,
                use_semantic_colors=True
            )
            
            # Also try the fixed V2 version
            print("\n" + "="*60)
            print("Running Dense Semantic Reconstruction V2 (with fixes)")
            print("="*60)
            
            dense_result_v2 = create_dense_semantic_reconstruction_v2(
                keyframes,
                semantic_keyframes,
                str(save_dir / f"{seq_name}_semantic_dense_v2_fixed.ply"),
                confidence_threshold=0.3,
                use_semantic_colors=True,
                debug=True
            )
            
            if dense_result:
                print(f"\nDense semantic reconstruction summary:")
                print(f"  Total points: {dense_result['num_points']:,}")
                print(f"  Points per keyframe: {dense_result['num_points'] // dense_result['num_keyframes']:,}")
                for label, stats in sorted(dense_result['label_stats'].items()):
                    print(f"  {label}: {stats['count']:,} points ({stats['percentage']:.1f}%)")
            
            # Also build dense cloud from saved keyframe data
            if keyframe_saver is not None:
                print("\n" + "="*60)
                print("Building dense cloud from saved keyframe data...")
                print("="*60)
                
                # Save metadata first
                keyframe_saver.save_metadata()
                
                # Build dense cloud
                from mast3r_slam.keyframe_saver import DenseSemanticBuilder
                builder = DenseSemanticBuilder(save_dir / "keyframe_data")
                dense_points = builder.build_dense_cloud(
                    str(save_dir / f"{seq_name}_semantic_dense_v2.ply"),
                    use_semantic_colors=True
                )
            
            # Save semantic keyframes with overlays
            from mast3r_slam.save_semantic_keyframes import save_semantic_keyframes, create_semantic_stats
            
            save_semantic_keyframes(
                save_dir / "semantic_keyframes" / seq_name,
                dataset.timestamps,
                keyframes,
                semantic_keyframes,
                semantic_backend,
                visualize_mode="both"  # Save both overlay and side-by-side
            )
            
            # Generate and save statistics
            stats = create_semantic_stats(semantic_keyframes, semantic_backend)
            stats_file = save_dir / f"{seq_name}_semantic_stats.json"
            import json
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            
            print(f"Semantic statistics:")
            print(f"  Keyframes with semantics: {stats['keyframes_with_semantics']}/{stats['total_keyframes']}")
            print(f"  Average instances per frame: {stats['avg_instances_per_frame']:.1f}")
            print(f"  Unique labels: {stats['unique_labels']}")
            print(f"  Coverage: {stats['coverage_percentage']:.1f}%")
        
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
    backend.join()
    if not args.no_viz:
        viz.join()