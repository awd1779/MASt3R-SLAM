"""Integration of semantic loop closure into MAST3R-SLAM backend"""

import torch
from mast3r_slam.semantic_factor_graph import create_semantic_factor_graph
from mast3r_slam.semantic_integration import SemanticSLAMBackend
from mast3r_slam.track_manager import GlobalTrackManager
from mast3r_slam.config import config, set_global_config


def run_semantic_backend(cfg, model, states, keyframes, semantic_keyframes, semantic_result_queue, K):
    """
    Enhanced backend with semantic loop closure verification.
    
    This replaces the standard run_backend function when semantic SLAM is enabled.
    """
    set_global_config(cfg)
    
    device = keyframes.device
    
    # Initialize track manager
    track_manager = GlobalTrackManager(
        iou_threshold=0.5,
        feature_threshold=0.7
    )
    
    # Initialize semantic backend
    semantic_backend = SemanticSLAMBackend(
        keyframes, 
        semantic_keyframes,
        track_manager,
        K,
        device
    )
    
    # Create semantic-aware factor graph
    factor_graph = create_semantic_factor_graph(
        model, keyframes, K, device, semantic_backend
    )
    
    # Load retrieval database
    from mast3r_slam.mast3r_utils import load_retriever
    retrieval_database = load_retriever(model)
    
    # Import necessary functions
    from mast3r_slam.frame import Mode
    import time
    
    mode = states.get_mode()
    while mode is not Mode.TERMINATED:
        mode = states.get_mode()
        
        # Process semantic results continuously
        try:
            semantic_data = semantic_result_queue.get_nowait()
            frame_id = semantic_data['frame_id']
            # Debug: print available keyframe IDs
            if config.get("verbose", False) and frame_id % 5 == 0:
                kf_ids = [keyframes[i].frame_id if i < len(keyframes) else None for i in range(min(5, len(keyframes)))]
                print(f"Looking for frame {frame_id}, available keyframes: {kf_ids}")
            
            # Find corresponding keyframe index
            found = False
            for kf_idx in range(len(keyframes)):
                kf = keyframes[kf_idx]
                if kf and hasattr(kf, 'frame_id') and kf.frame_id == frame_id:
                    semantic_keyframes.update_semantics(kf_idx, semantic_data)
                    semantic_backend.process_semantic_keyframe(kf_idx)
                    print(f"✓ Updated semantics for keyframe {kf_idx} (frame {frame_id})")
                    found = True
                    break
            
            if not found and config.get("verbose", False):
                print(f"⚠ No keyframe found for semantic frame {frame_id}")
        except Exception as e:
            if str(e) != "":  # Ignore empty queue exceptions
                print(f"Semantic processing error: {e}")
        
        if mode == Mode.INIT or states.is_paused():
            time.sleep(0.01)
            continue
            
        if mode == Mode.RELOC:
            frame = states.get_frame()
            success = relocalization_with_semantics(
                frame, keyframes, factor_graph, retrieval_database, semantic_backend
            )
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

        # Update retrieval database
        retrieval_inds = retrieval_database.update(
            keyframes[idx],
            add_after_query=True,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        
        # Process loop closures with semantic verification
        if config["global_opt"]["use_loop_closures"]:
            loop_kf_idx = []
            loop_kf_idx += retrieval_inds
            successful_loop_closure = False
            
            if loop_kf_idx:
                loop_kf_idx = list(loop_kf_idx)
                frame_idx = [idx] * len(loop_kf_idx)
                
                # Add factors with semantic verification
                if factor_graph.add_factors(
                    frame_idx,
                    loop_kf_idx,
                    config["global_opt"]["loop_confidence_threshold"],
                ):
                    successful_loop_closure = True
                    
                    # Update track manager on successful loop closure
                    if semantic_backend and successful_loop_closure:
                        for loop_idx in loop_kf_idx:
                            semantic_backend.update_tracks_on_loop_closure(idx, loop_idx)
                            
            with states.lock:
                states.edges_ii.append(
                    factor_graph.edges[factor_graph.ii].reshape(-1, 3).cpu().numpy()
                )
                states.edges_jj.append(
                    factor_graph.edges[factor_graph.jj].reshape(-1, 3).cpu().numpy()
                )
                
        # Print statistics
        print(f"KF {idx} | Matching | #factors: {len(factor_graph.ii)}, #frames {len(keyframes)}")
        
        # Print semantic statistics if available
        if hasattr(factor_graph, 'get_semantic_statistics'):
            sem_stats = factor_graph.get_semantic_statistics()
            if sem_stats.get('total_checks', 0) > 0:
                print(f"  Semantic: {sem_stats['accepted']}/{sem_stats['total_checks']} accepted, "
                      f"avg score: {sem_stats.get('average_semantic_score', 0):.3f}")

        # Run optimization
        if config["use_calib"]:
            factor_graph.solve_GN_calib()
        else:
            factor_graph.solve_GN_rays()


def relocalization_with_semantics(frame, keyframes, factor_graph, retrieval_database, semantic_backend):
    """
    Enhanced relocalization with semantic verification.
    """
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
            kf_idx = list(kf_idx)
            frame_idx = [n_kf - 1] * len(kf_idx)
            
            print("RELOCALIZING against kf ", n_kf - 1, " and ", kf_idx)
            
            # Add factors with semantic verification
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
                
                # Update semantic tracks
                if semantic_backend and successful_loop_closure:
                    semantic_backend.update_tracks_on_loop_closure(n_kf - 1, kf_idx[0])
            else:
                keyframes.pop_last()
                print("Failed to relocalize")

        if successful_loop_closure:
            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()
                
        return successful_loop_closure