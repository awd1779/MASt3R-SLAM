"""Integration module for semantic fusion with MAST3R-SLAM backend."""

import torch
import time
from typing import Dict, Optional, Tuple
from mast3r_slam.semantic_fusion import SemanticPointmapFusion, compute_semantic_overlap
from mast3r_slam.track_manager import GlobalTrackManager
from mast3r_slam.config import config


class SemanticSLAMBackend:
    """Handles semantic integration in the SLAM backend."""
    
    def __init__(self, 
                 keyframes,
                 semantic_keyframes,
                 track_manager: GlobalTrackManager,
                 K: torch.Tensor,
                 device: str = "cuda"):
        self.keyframes = keyframes
        self.semantic_keyframes = semantic_keyframes
        self.track_manager = track_manager
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
            # Update track manager with local-to-global mapping
            local_tracks = {}
            for instance_id in semantic_data['instance_ids']:
                local_tracks[instance_id] = {
                    'track_id': semantic_data['track_ids'].get(instance_id, instance_id),
                    'label': semantic_data['labels'].get(instance_id, 'unknown'),
                    'confidence': semantic_data['confidences'].get(instance_id, 1.0)
                }
                
            # Get global track mapping
            local_to_global = self.track_manager.add_frame_tracks(
                kf_idx, local_tracks
            )
            
            # Remap labels to global track IDs
            global_labels = torch.zeros_like(labels)
            for local_id, global_id in local_to_global.items():
                mask = labels == local_id
                global_labels[mask] = global_id
                
            # Cache results
            self.keyframe_semantics[kf_idx] = (global_labels, conf)
            
            # Track performance
            self.fusion_times.append(time.time() - start_time)
            
            return True
            
        return False
        
    def verify_loop_closure_semantics(self, 
                                     kf_idx1: int, 
                                     kf_idx2: int) -> Tuple[bool, float]:
        """
        Verify loop closure using semantic consistency.
        
        Returns:
            (is_valid, overlap_score)
        """
        if not config["semantic_segmentation"]["backend"]["enable_semantic_verification"]:
            return True, 1.0
            
        # Get semantic labels for both keyframes
        labels1 = self.keyframe_semantics.get(kf_idx1)
        labels2 = self.keyframe_semantics.get(kf_idx2)
        
        if labels1 is None or labels2 is None:
            # No semantic data available, accept loop closure
            return True, 0.0
            
        # Compute semantic overlap
        overlap = compute_semantic_overlap(labels1[0], labels2[0])
        
        # Threshold for accepting loop closure
        min_overlap = config["semantic_segmentation"]["backend"].get("min_semantic_overlap", 0.3)
        
        return overlap >= min_overlap, overlap
        
    def update_tracks_on_loop_closure(self, kf_idx1: int, kf_idx2: int):
        """Update track manager when loop closure is detected."""
        labels1 = self.keyframe_semantics.get(kf_idx1)
        labels2 = self.keyframe_semantics.get(kf_idx2)
        
        if labels1 is not None and labels2 is not None:
            self.track_manager.merge_tracks_on_loop_closure(
                kf_idx1, kf_idx2, labels1[0], labels2[0]
            )
            
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