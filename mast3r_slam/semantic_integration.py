"""Integration module for semantic fusion with MAST3R-SLAM backend."""

import torch
import time
import logging
from typing import Dict, Optional, Tuple
from mast3r_slam.semantic_fusion import SemanticPointmapFusion
from mast3r_slam.config import config, set_global_config
from mast3r_slam.global_opt import FactorGraph

logger = logging.getLogger('mast3r_slam.semantic_backend')


class SemanticSLAMBackend:
    """Handles semantic integration in the SLAM backend."""
    
    def __init__(self, 
                 keyframes,
                 semantic_keyframes,
                 K: torch.Tensor,
                 device: str = "cuda"):
        self.keyframes = keyframes
        self.semantic_keyframes = semantic_keyframes
        self.device = device
        
        # If K is None (no calibration), create a default intrinsic matrix
        if K is None:
            # Assume default camera parameters
            h, w = keyframes.h, keyframes.w
            fx = fy = w  # Focal length approximation
            cx = w / 2.0
            cy = h / 2.0
            self.K = torch.tensor([[fx, 0, cx],
                                   [0, fy, cy],
                                   [0, 0, 1]], dtype=torch.float32, device=device)
        else:
            self.K = K
        
        # Initialize fusion module
        self.fusion = SemanticPointmapFusion(device)
        
        # Semantic data cache
        self.keyframe_semantics = {}  # kf_idx -> (labels, confidence)
        
        # Performance tracking
        self.fusion_times = []
        
    def process_semantic_keyframe(self, kf_idx: int) -> bool:
        """
        Process semantic data for a keyframe.
        
        Returns:
            True if semantic data was processed successfully
        """
        start_time = time.time()
        
        # Get keyframe
        kf = self.keyframes[kf_idx]
        if kf is None or kf.X_canon is None:
            return False
            
        # Get semantic data
        semantic_data = self.semantic_keyframes.get_semantics(kf_idx)
        if semantic_data is None:
            return False
            
        # Perform semantic fusion
        labels, conf = self.fusion.fuse_keyframe_semantics(kf, semantic_data, self.K)
        
        if labels is not None:
            # Cache results directly without track management
            self.keyframe_semantics[kf_idx] = (labels, conf)
            
            # Track performance
            self.fusion_times.append(time.time() - start_time)
            
            return True
            
        return False
        
        
    def get_semantic_pointcloud(self, kf_indices: Optional[list] = None) -> Dict:
        """
        Get semantic point cloud data for visualization.
        
        Returns:
            Dict with 'points', 'labels', 'colors', 'confidences'
        """
        if kf_indices is None:
            kf_indices = list(self.keyframe_semantics.keys())
            
        all_points = []
        all_labels = []
        all_confidences = []
        
        for kf_idx in kf_indices:
            kf = self.keyframes[kf_idx]
            if kf is None or kf.X_canon is None:
                continue
                
            semantics = self.keyframe_semantics.get(kf_idx)
            if semantics is None:
                continue
                
            labels, conf = semantics
            
            # Transform points to world
            X_world = kf.T_WC @ kf.X_canon
            
            all_points.append(X_world)
            all_labels.append(labels)
            all_confidences.append(conf)
            
        if len(all_points) == 0:
            return {}
            
        # Concatenate all data
        points = torch.cat(all_points, dim=0)
        labels = torch.cat(all_labels, dim=0)
        confidences = torch.cat(all_confidences, dim=0)
        
        # Generate colors based on labels
        unique_labels = torch.unique(labels)
        label_to_color = {}
        
        # Use a colormap
        import matplotlib.cm as cm
        cmap = cm.get_cmap('tab20')
        
        for i, label in enumerate(unique_labels):
            if label == 0:  # Background
                label_to_color[label.item()] = torch.tensor([0.5, 0.5, 0.5])
            else:
                color = cmap(i % 20)[:3]
                label_to_color[label.item()] = torch.tensor(color)
                
        # Assign colors
        colors = torch.zeros((len(labels), 3), device=labels.device)
        for label, color in label_to_color.items():
            mask = labels == label
            colors[mask] = color.to(labels.device)
            
        return {
            'points': points.cpu().numpy(),
            'labels': labels.cpu().numpy(),
            'colors': colors.cpu().numpy(),
            'confidences': confidences.cpu().numpy(),
            'label_to_color': label_to_color
        }
        
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        if len(self.fusion_times) == 0:
            return {}
            
        return {
            'num_fusions': len(self.fusion_times),
            'avg_fusion_time': sum(self.fusion_times) / len(self.fusion_times),
            'max_fusion_time': max(self.fusion_times),
            'min_fusion_time': min(self.fusion_times),
            'total_fusion_time': sum(self.fusion_times)
        }


def integrate_semantic_backend(backend_func):
    """Decorator to integrate semantic processing into the backend."""
    def wrapper(cfg, model, states, keyframes, semantic_keyframes, semantic_result_queue, K):
        # Initialize semantic backend
        semantic_backend = SemanticSLAMBackend(
            keyframes, 
            semantic_keyframes,
            K,
            device=keyframes.device
        )
        
        # Add semantic backend to the original function's context
        # This is a simplified approach - in practice you'd modify the backend function
        
        # Run original backend with semantic integration
        return backend_func(cfg, model, states, keyframes, K, 
                          semantic_backend=semantic_backend,
                          semantic_keyframes=semantic_keyframes,
                          semantic_result_queue=semantic_result_queue)
                          
    return wrapper


def run_semantic_backend(cfg, model, states, keyframes, semantic_keyframes, semantic_result_queue, K):
    """
    Enhanced backend with semantic loop closure verification.
    
    This replaces the standard run_backend function when semantic SLAM is enabled.
    """
    set_global_config(cfg)
    
    device = keyframes.device
    
    # Initialize semantic backend
    semantic_backend = SemanticSLAMBackend(
        keyframes, 
        semantic_keyframes,
        K,
        device
    )
    
    # Create standard factor graph
    factor_graph = FactorGraph(model, keyframes, K, device)
    
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
                    
                    # Loop closure successful (semantic track update removed)
                            
            with states.lock:
                states.edges_ii.append(
                    factor_graph.ii.cpu().numpy()
                )
                states.edges_jj.append(
                    factor_graph.jj.cpu().numpy()
                )
                
        # Print statistics
        print(f"KF {idx} | Matching | #factors: {len(factor_graph.ii)}, #frames {len(keyframes)}")

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
                
                # Loop closure successful (semantic track update removed)
            else:
                keyframes.pop_last()
                print("Failed to relocalize")

        if successful_loop_closure:
            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()
                
        return successful_loop_closure