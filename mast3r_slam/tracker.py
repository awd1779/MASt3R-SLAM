import torch
from collections import Counter # Added import for Counter
from mast3r_slam.frame import Frame
from mast3r_slam.geometry import (
    act_Sim3,
    point_to_ray_dist,
    get_pixel_coords,
    constrain_points_to_ray,
    project_calib,
)
from mast3r_slam.nonlinear_optimizer import check_convergence, huber
from mast3r_slam.config import config
from mast3r_slam.mast3r_utils import mast3r_match_asymmetric
from mast3r_slam.semantic_core import TEXT_PROMPTS as SEMANTIC_TEXT_PROMPTS # Use new core module
from collections import Counter # Ensure Counter is imported (already added previously)
from mast3r_slam.semantic_utils import SemanticMapper # Use consolidated utils


class FrameTracker:
    def __init__(self, model, frames, device, initial_global_id_to_class_label_map: dict, semantic_processor=None):
        self.cfg = config["tracking"]
        self.model = model
        self.keyframes = frames # This is a SharedKeyframes instance
        self.device = device

        # For managing global instance IDs and their semantic labels
        self.g_next_global_id = 1  # Start global IDs from 1 (0 is background)
        self.g_global_id_to_class_label_map = initial_global_id_to_class_label_map
        # Ensure background is in the map if not already passed (though main.py should handle it)
        if 0 not in self.g_global_id_to_class_label_map:
            self.g_global_id_to_class_label_map[0] = "background"

        self.reset_idx_f2k()
        
        # Create semantic mapper for improved labeling
        self.semantic_mapper = SemanticMapper(device=device)
        
        # Track label confidences for refinement
        self.label_confidences = {0: 100.0}  # Background has high confidence
        
        # Store semantic processor instance
        self.semantic_processor = semantic_processor

    # Initialize with identity indexing of size (1,n)
    def reset_idx_f2k(self):
        self.idx_f2k = None

    def track(self, frame: Frame):
        keyframe = self.keyframes.last_keyframe()

        idx_f2k, valid_match_k, Xff, Cff, Qff, Xkf, Ckf, Qkf = mast3r_match_asymmetric(
            self.model, frame, keyframe, idx_i2j_init=self.idx_f2k
        )
        # Save idx for next
        self.idx_f2k = idx_f2k.clone()

        # Get rid of batch dim
        idx_f2k = idx_f2k[0]
        valid_match_k = valid_match_k[0]

        Qk = torch.sqrt(Qff[idx_f2k] * Qkf)

        # Update keyframe pointmap after registration (need pose)
        # This populates frame.X_canon and frame.C
        frame.update_pointmap(Xff, Cff)

        # Initialize global_instance_ids for the current frame
        # It should have the same number of elements as X_canon points.
        num_points_current_frame = frame.X_canon.shape[0]
        frame.global_instance_ids = torch.zeros(num_points_current_frame, 1, dtype=torch.int64, device=self.device)

        # Process semantics for ALL frames to ensure they have local masks when they become keyframes
        if frame.local_instance_mask is None and frame.img is not None and self.semantic_processor is not None:
            print(f"[Tracker] Processing semantics for frame {frame.frame_id}")
            try:
                frame_data = {
                    'image_tensor': frame.img.squeeze(0),
                    'frame_id': frame.frame_id,
                    'pose': frame.T_WC.matrix()[0] if hasattr(frame, 'T_WC') and frame.T_WC is not None else None,
                    'global_ids': None  # Will be populated after
                }
                # Use fast mode with caching for non-keyframes
                result = self.semantic_processor.process_frame(frame_data, mode='fast', use_cache=True)
                frame.local_instance_mask = result['local_instance_mask'].to(self.device)
                frame.local_id_to_class_label_map = result['local_id_to_class_map']
            except Exception as e:
                print(f"[Tracker] Semantic processing failed for frame {frame.frame_id}: {e}")
                frame.local_instance_mask = None
                frame.local_id_to_class_label_map = None

        # --- Start: 3D Instance Association Logic ---
        # IMPROVED: First assign labels to ALL points based on 2D mask, then merge with keyframe
        if frame.local_instance_mask is not None and frame.local_id_to_class_label_map is not None:
            # Map 2D labels to ALL 3D points in current frame
            current_labels, current_valid_mask = self.semantic_mapper.map_2d_labels_to_3d_points(
                semantic_mask=frame.local_instance_mask,
                confidence_map=frame.C if hasattr(frame, 'C') and frame.C is not None else None,
                num_points=num_points_current_frame,
                frame_shape=(frame.local_instance_mask.shape[0], frame.local_instance_mask.shape[1]),
                confidence_threshold=0.5  # Lowered threshold for better coverage
            )
            
            # First: Assign initial global IDs to all labeled points
            unique_local_ids = torch.unique(current_labels)
            local_to_new_global = {}
            
            for local_id in unique_local_ids:
                local_id_val = local_id.item()
                if local_id_val == 0:  # Skip background
                    continue
                    
                label_mask = (current_labels == local_id) & current_valid_mask
                if label_mask.any():
                    # Temporarily assign new global IDs
                    new_global_id = self.g_next_global_id
                    local_to_new_global[local_id_val] = new_global_id
                    frame.global_instance_ids[label_mask] = new_global_id
                    
                    # Store class info
                    class_name = frame.local_id_to_class_label_map.get(local_id_val, f"unknown_{local_id_val}")
                    self.g_global_id_to_class_label_map[new_global_id] = class_name
                    self.label_confidences[new_global_id] = 40.0  # Initial confidence
                    self.g_next_global_id += 1
        
        # Second: Merge with keyframe labels if available
        if keyframe is not None and keyframe.global_instance_ids is not None and \
           frame.local_instance_mask is not None and frame.local_id_to_class_label_map is not None:

            num_points_keyframe = Xkf.shape[0]
            if keyframe.global_instance_ids.numel() != num_points_keyframe:
                print(f"[Tracker WARN] Keyframe {keyframe.frame_id} global_instance_ids numel ({keyframe.global_instance_ids.numel()}) "
                      f"does not match its point count ({num_points_keyframe}). Skipping global ID propagation.")
            else:
                keyframe_global_ids_flat = keyframe.global_instance_ids.view(-1)

                # Track which global IDs need to be merged
                global_id_merge_map = {}  # Maps current frame's global ID -> keyframe's global ID
                
                # Iterate through current frame points that have a valid match in the keyframe
                for p_curr_idx in range(num_points_current_frame):
                    p_kf_idx = idx_f2k[p_curr_idx].item()
                    
                    if 0 <= p_kf_idx < num_points_keyframe:
                        prev_global_id = keyframe_global_ids_flat[p_kf_idx].item()
                        curr_global_id = frame.global_instance_ids[p_curr_idx].item()
                        
                        # Case 1: Both have labels - check if they should merge
                        if prev_global_id != 0 and curr_global_id != 0 and prev_global_id != curr_global_id:
                            prev_class = self.g_global_id_to_class_label_map.get(prev_global_id, "")
                            curr_class = self.g_global_id_to_class_label_map.get(curr_global_id, "")
                            
                            # If same class, merge them
                            if prev_class == curr_class or curr_class.startswith("unknown_"):
                                global_id_merge_map[curr_global_id] = prev_global_id
                        
                        # Case 2: Current has no label but keyframe does - propagate
                        elif curr_global_id == 0 and prev_global_id != 0:
                            frame.global_instance_ids[p_curr_idx] = prev_global_id

                # Apply the merges
                for curr_global_id, prev_global_id in global_id_merge_map.items():
                    # Replace all instances of curr_global_id with prev_global_id
                    frame.global_instance_ids[frame.global_instance_ids == curr_global_id] = prev_global_id
                    
                    # Update confidence if keyframe has higher confidence
                    if prev_global_id in self.label_confidences and curr_global_id in self.label_confidences:
                        self.label_confidences[prev_global_id] = max(
                            self.label_confidences[prev_global_id],
                            self.label_confidences[curr_global_id]
                        )
                    
                    # Clean up the temporary global ID
                    if curr_global_id in self.g_global_id_to_class_label_map:
                        del self.g_global_id_to_class_label_map[curr_global_id]
                    if curr_global_id in self.label_confidences:
                        del self.label_confidences[curr_global_id]
                    
                    # Decrement next ID if we just removed the latest one
                    if curr_global_id == self.g_next_global_id - 1:
                        self.g_next_global_id -= 1
        
        # Log statistics for debugging
        non_background = (frame.global_instance_ids != 0).sum().item()
        total_points = frame.global_instance_ids.numel()
        print(f"[Tracker] Frame {frame.frame_id}: {non_background}/{total_points} points labeled ({non_background/total_points*100:.1f}%)")
        
        # AGGRESSIVE SEMANTIC PROPAGATION - Only run if we have too many background points
        background_ratio = (frame.global_instance_ids == 0).sum().item() / total_points
        
        if background_ratio > 0.4 and frame.X_canon is not None:  # If more than 40% background
            print(f"[Tracker] Applying AGGRESSIVE semantic propagation (background ratio: {background_ratio:.1%})")
            
            # Step 1: Propagate high-confidence labels to nearby unlabeled points
            if hasattr(self, 'semantic_mapper') and self.semantic_mapper is not None:
                # Get current label confidences
                label_confidences = {}
                unique_ids = torch.unique(frame.global_instance_ids)
                for gid in unique_ids:
                    if gid.item() > 0:  # Skip background
                        # Use stored confidence or default
                        label_confidences[gid.item()] = self.label_confidences.get(gid.item(), 50.0)
                
                # Propagate high-confidence labels spatially
                print(f"[Tracker] Step 1: Propagating high-confidence labels spatially...")
                propagated_labels = self.semantic_mapper.propagate_high_confidence_labels(
                    points_3d=frame.X_canon,
                    labels=frame.global_instance_ids.view(-1),
                    confidences=torch.tensor([
                        label_confidences.get(lid.item(), 0.0) 
                        for lid in frame.global_instance_ids.view(-1)
                    ], device=self.device),
                    propagation_radius=0.1,  # 10cm radius
                    confidence_threshold=40.0  # Lower threshold for more propagation
                )
                frame.global_instance_ids = propagated_labels.view(-1, 1)
                
                # Update stats
                new_background_ratio = (frame.global_instance_ids == 0).sum().item() / total_points
                print(f"[Tracker] After spatial propagation: {new_background_ratio:.1%} background")
            
            # Step 2: If still too many background points, use nearest neighbor filling
            if (frame.global_instance_ids == 0).sum().item() / total_points > 0.35:
                print(f"[Tracker] Step 2: Applying nearest-neighbor filling for remaining unlabeled points...")
                
                # Find labeled and unlabeled points
                labeled_mask = frame.global_instance_ids.view(-1) > 0
                unlabeled_mask = ~labeled_mask
                
                if labeled_mask.any() and unlabeled_mask.any():
                    labeled_indices = torch.where(labeled_mask)[0]
                    unlabeled_indices = torch.where(unlabeled_mask)[0]
                    
                    # Process in chunks to save memory
                    chunk_size = 500
                    for start_idx in range(0, unlabeled_indices.numel(), chunk_size):
                        end_idx = min(start_idx + chunk_size, unlabeled_indices.numel())
                        chunk_indices = unlabeled_indices[start_idx:end_idx]
                        chunk_points = frame.X_canon[chunk_indices]
                        
                        # Find nearest labeled point for each unlabeled point
                        distances = torch.cdist(chunk_points, frame.X_canon[labeled_indices])
                        min_distances, nearest_indices = distances.min(dim=1)
                        
                        # Only fill if within reasonable distance (20cm)
                        fill_mask = min_distances < 0.2
                        
                        for i in range(chunk_points.shape[0]):
                            if fill_mask[i]:
                                source_idx = labeled_indices[nearest_indices[i]]
                                frame.global_instance_ids[chunk_indices[i]] = frame.global_instance_ids[source_idx]
                
                # Final stats
                final_background_ratio = (frame.global_instance_ids == 0).sum().item() / total_points
                final_non_background = (frame.global_instance_ids != 0).sum().item()
                print(f"[Tracker] AGGRESSIVE propagation complete:")
                print(f"  Final: {final_non_background}/{total_points} labeled ({final_non_background/total_points*100:.1f}%)")
                print(f"  Background reduced from {background_ratio:.1%} to {final_background_ratio:.1%}")
        
        # Update final debug output
        non_background = (frame.global_instance_ids != 0).sum().item()
        total_points = frame.global_instance_ids.numel()
        print(f"[Tracker] Frame {frame.frame_id} FINAL: {non_background}/{total_points} points labeled ({non_background/total_points*100:.1f}%)")
        # --- End: 3D Instance Association Logic (This whole block will be conditional) ---

        # The decision to add a new keyframe (new_kf) is made LATER in this function,
        # after pose optimization.
        # So, we first track, then decide if it's a keyframe, THEN do full semantics if it is.
        # For non-keyframes, we'll do simpler label propagation if possible.

        use_calib = config["use_calib"]
        img_size = frame.img.shape[-2:] # This is the MaSt3R processed image size (e.g., from resize_img)

        # Store keyframe before it's potentially updated by Xkk, Ckf if frame becomes a KF
        # This 'keyframe' is self.keyframes.last_keyframe()
        # We need its global_instance_ids for propagation to non-keyframes or new keyframes.
        last_kf_for_propagation = keyframe

        if use_calib:
            # K = keyframe.K # This might be problematic if keyframe is None (first frame)
            K = self.keyframes.get_intrinsics() if self.keyframes.n_size.value > 0 and hasattr(self.keyframes, 'get_intrinsics') else None
            if K is None and self.keyframes.last_keyframe() is not None: # Fallback if get_intrinsics isn't there but K might be on KF
                 K = self.keyframes.last_keyframe().K
        else:
        
            K = None

        # Get poses and point correspondneces and confidences
        Xf, Xk, T_WCf, T_WCk, Cf, Ck, meas_k, valid_meas_k = self.get_points_poses(
            frame, keyframe, idx_f2k, img_size, use_calib, K
        )

        # Get valid
        # Use canonical confidence average
        valid_Cf = Cf > self.cfg["C_conf"]
        valid_Ck = Ck > self.cfg["C_conf"]
        valid_Q = Qk > self.cfg["Q_conf"]

        valid_opt = valid_match_k & valid_Cf & valid_Ck & valid_Q
        valid_kf = valid_match_k & valid_Q

        match_frac = valid_opt.sum() / valid_opt.numel()
        if match_frac < self.cfg["min_match_frac"]:
            print(f"Skipped frame {frame.frame_id}")
            return False, [], True

        try:
            # Track
            if not use_calib:
                T_WCf, T_CkCf = self.opt_pose_ray_dist_sim3(
                    Xf, Xk, T_WCf, T_WCk, Qk, valid_opt
                )
            else:
                T_WCf, T_CkCf = self.opt_pose_calib_sim3(
                    Xf,
                    Xk,
                    T_WCf,
                    T_WCk,
                    Qk,
                    valid_opt,
                    meas_k,
                    valid_meas_k,
                    K,
                    img_size,
                )
        except Exception as e:
            print(f"Cholesky failed {frame.frame_id}")
            return False, [], True

        frame.T_WC = T_WCf

        # Use pose to transform points to update keyframe
        Xkk = T_CkCf.act(Xkf)
        keyframe.update_pointmap(Xkk, Ckf)
        # write back the fitered pointmap
        self.keyframes[len(self.keyframes) - 1] = keyframe

        # Keyframe selection
        n_valid = valid_kf.sum()
        match_frac_k = n_valid / valid_kf.numel()
        unique_frac_f = (
            torch.unique(idx_f2k[valid_match_k[:, 0]]).shape[0] / valid_kf.numel()
        )

        new_kf = min(match_frac_k, unique_frac_f) < self.cfg["match_frac_thresh"]
        
        print(f"[DEBUG] Frame {frame.frame_id}: match_frac_k={match_frac_k:.3f}, unique_frac_f={unique_frac_f:.3f}, threshold={self.cfg['match_frac_thresh']}, new_kf={new_kf}")

        # Rest idx if new keyframe
        if new_kf:
            # This is a NEW KEYFRAME
            print(f"[Tracker INFO] Frame {frame.frame_id} designated as new keyframe.")
            # 1. Perform full semantic processing (SAM + CLIP)
            if frame.img is not None: # Ensure image data is available
                print(f"[Tracker INFO] Running full semantic processing for new keyframe {frame.frame_id}...")
                print(f"[DEBUG] Keyframe {frame.frame_id} frame.img shape: {frame.img.shape}, dtype: {frame.img.dtype}")
                try:
                    if self.semantic_processor is not None:
                        # Use SemanticProcessorV2 if available
                        # Check if img needs to be accessed differently for keyframes
                        if hasattr(frame, 'rgb') and frame.rgb is not None:
                            print(f"[DEBUG] Using frame.rgb instead, shape: {frame.rgb.shape}")
                            image_tensor = frame.rgb.squeeze(0) if frame.rgb.ndim > 3 else frame.rgb
                        else:
                            image_tensor = frame.img.squeeze(0) if frame.img.ndim > 3 else frame.img
                        
                        frame_data = {
                            'image_tensor': image_tensor,  # (C,H,W)
                            'frame_id': frame.frame_id,
                            'pose': frame.T_WC.matrix()[0] if hasattr(frame, 'T_WC') and frame.T_WC is not None else None,
                            'global_ids': frame.global_instance_ids
                        }
                        result = self.semantic_processor.process_frame(frame_data, mode='full', use_cache=True)
                        local_mask = result['local_instance_mask']
                        local_map = result['local_id_to_class_map']
                    else:
                        # Fallback to original processor
                        from mast3r_slam.semantic_core import process_frame_for_semantics
                        local_mask, local_map = process_frame_for_semantics(
                            image_tensor_chw_0_1_rgb=frame.img.squeeze(0),
                            text_prompts_for_clip=SEMANTIC_TEXT_PROMPTS,
                            enable_debug_viz=True,
                            frame_id=frame.frame_id
                        )
                    frame.local_instance_mask = local_mask.to(self.device if local_mask is not None else None)
                    frame.local_id_to_class_label_map = local_map
                except Exception as e_semantic_kf:
                    print(f"[Tracker ERROR] Semantic processing failed for new keyframe {frame.frame_id}: {e_semantic_kf}")
                    frame.local_instance_mask = None
                    frame.local_id_to_class_label_map = None
            else:
                print(f"[Tracker WARN] Frame {frame.frame_id} (new KF) has no frame.rgb for semantic processing.")
                frame.local_instance_mask = None
                frame.local_id_to_class_label_map = None

            # 2. Initialize and Populate global_instance_ids
            # num_points_current_frame was already defined when frame.X_canon was first populated
            if frame.X_canon is not None:
                # CRITICAL FIX: Don't reinitialize to zeros if frame already has global_instance_ids!
                # The frame was already processed as a regular frame and has labels
                if not hasattr(frame, 'global_instance_ids') or frame.global_instance_ids is None:
                    print(f"[DEBUG] Initializing global_instance_ids for keyframe {frame.frame_id}")
                    frame.global_instance_ids = torch.zeros(num_points_current_frame, 1, dtype=torch.int64, device=self.device)
                else:
                    print(f"[DEBUG] Keyframe {frame.frame_id} already has global_instance_ids, keeping existing labels")
                    # Verify the size matches
                    if frame.global_instance_ids.numel() != num_points_current_frame:
                        print(f"[WARN] Size mismatch: existing global_ids {frame.global_instance_ids.numel()} vs current points {num_points_current_frame}")
                        frame.global_instance_ids = torch.zeros(num_points_current_frame, 1, dtype=torch.int64, device=self.device)

                # 3. Perform 3D Instance Association (using last_kf_for_propagation)
                if last_kf_for_propagation is not None and \
                   last_kf_for_propagation.global_instance_ids is not None and \
                   frame.local_instance_mask is not None and \
                   frame.local_id_to_class_label_map is not None:

                    num_points_last_kf = Xkf.shape[0] # Xkf corresponds to last_kf_for_propagation.X_canon
                    if last_kf_for_propagation.global_instance_ids.numel() != num_points_last_kf:
                        print(f"[Tracker WARN] Last KF {last_kf_for_propagation.frame_id} global_ids numel "
                              f"({last_kf_for_propagation.global_instance_ids.numel()}) != point count ({num_points_last_kf}). Skipping propagation.")
                    else:
                        last_kf_global_ids_flat = last_kf_for_propagation.global_instance_ids.view(-1)
                        
                        # Use semantic mapper for proper alignment
                        current_labels, current_valid_mask = self.semantic_mapper.map_2d_labels_to_3d_points(
                            semantic_mask=frame.local_instance_mask,
                            confidence_map=frame.C if hasattr(frame, 'C') and frame.C is not None else None,
                            num_points=num_points_current_frame,
                            frame_shape=(frame.local_instance_mask.shape[0], frame.local_instance_mask.shape[1]),
                            confidence_threshold=1.0
                        )
                        
                        local_id_to_global_id_candidates = {}
                        for p_curr_idx in range(num_points_current_frame):
                            p_kf_idx = idx_f2k[p_curr_idx].item()
                            if 0 <= p_kf_idx < num_points_last_kf and current_valid_mask[p_curr_idx]:
                                prev_global_id = last_kf_global_ids_flat[p_kf_idx].item()
                                current_local_sam_id = current_labels[p_curr_idx].item()
                                if current_local_sam_id != 0 and prev_global_id != 0:
                                    if current_local_sam_id not in local_id_to_global_id_candidates:
                                        local_id_to_global_id_candidates[current_local_sam_id] = []
                                    local_id_to_global_id_candidates[current_local_sam_id].append(prev_global_id)

                        for local_id, candidates in local_id_to_global_id_candidates.items():
                            if candidates:
                                chosen_global_id = Counter(candidates).most_common(1)[0][0]
                                # Use the mapper's labels for consistency
                                label_mask = (current_labels == local_id) & current_valid_mask
                                frame.global_instance_ids[label_mask] = chosen_global_id

                    # Assign New Global IDs for this new keyframe
                    # FIXED: Process ALL local segments, not just those without propagated labels
                    print(f"[DEBUG Tracker] Processing new objects for keyframe {frame.frame_id}")
                    unique_local_ids = torch.unique(current_labels)
                    new_objects_count = 0
                    
                    for local_id_tensor in unique_local_ids:
                        local_id = local_id_tensor.item()
                        if local_id == 0: continue
                        
                        label_mask = (current_labels == local_id) & current_valid_mask
                        if not label_mask.any():
                            continue
                            
                        # Check if this segment needs a new global ID
                        # Either it wasn't matched to previous keyframe OR it has unlabeled points
                        unassigned_mask = label_mask & (frame.global_instance_ids.view(-1) == 0)
                        
                        if unassigned_mask.any():
                            # This segment has unlabeled points - needs a global ID
                            
                            # Check if this is entirely new or partially propagated
                            is_entirely_new = (frame.global_instance_ids[label_mask] == 0).all()
                            
                            if is_entirely_new:
                                # Entirely new object - assign new global ID to all points
                                new_global_id = self.g_next_global_id
                                frame.global_instance_ids[label_mask] = new_global_id
                                class_name = frame.local_id_to_class_label_map.get(local_id, f"unclassified_local_id_{local_id}")
                                self.g_global_id_to_class_label_map[new_global_id] = class_name
                                self.label_confidences[new_global_id] = 45.0
                                self.g_next_global_id += 1
                                new_objects_count += 1
                                
                                labeled_count = label_mask.sum().item()
                                print(f"[DEBUG Tracker] New object - Label {local_id} ({class_name}): {labeled_count} points with global ID {new_global_id}")
                            else:
                                # Partially propagated - extend existing label to unlabeled parts
                                # Find the most common non-zero global ID in this segment
                                segment_global_ids = frame.global_instance_ids[label_mask].view(-1)
                                non_zero_ids = segment_global_ids[segment_global_ids > 0]
                                
                                if non_zero_ids.numel() > 0:
                                    # Use mode to find most common ID
                                    most_common_id = non_zero_ids.mode()[0].item()
                                    frame.global_instance_ids[unassigned_mask] = most_common_id
                                    extended_count = unassigned_mask.sum().item()
                                    print(f"[DEBUG Tracker] Extended existing label {most_common_id} to {extended_count} additional points")
                    
                    print(f"[DEBUG Tracker] Keyframe {frame.frame_id}: Added {new_objects_count} new objects after propagation")
                elif frame.local_instance_mask is not None: # Is a new keyframe, but no prior keyframe to propagate from (e.g. first frame)
                    print(f"[Tracker INFO] New keyframe {frame.frame_id} is the first or has no prior KF with global IDs. Initializing from its own SAM.")
                    
                    # Use improved semantic mapper
                    labels, valid_mask = self.semantic_mapper.map_2d_labels_to_3d_points(
                        semantic_mask=frame.local_instance_mask,
                        confidence_map=frame.C if hasattr(frame, 'C') and frame.C is not None else None,
                        num_points=num_points_current_frame,
                        frame_shape=(frame.local_instance_mask.shape[0], frame.local_instance_mask.shape[1]),
                        confidence_threshold=1.5  # Lower threshold for better coverage
                    )
                    
                    # Assign global IDs
                    unique_local_ids = torch.unique(labels)
                    for local_id_tensor in unique_local_ids:
                        local_id = local_id_tensor.item()
                        if local_id == 0: continue
                        
                        label_mask = (labels == local_id) & valid_mask
                        if label_mask.any():
                            new_global_id = self.g_next_global_id
                            frame.global_instance_ids[label_mask] = new_global_id
                            class_name = frame.local_id_to_class_label_map.get(local_id, f"unclassified_local_id_{local_id}")
                            self.g_global_id_to_class_label_map[new_global_id] = class_name
                            self.label_confidences[new_global_id] = 50.0  # Default confidence
                            self.g_next_global_id += 1
                            
                            labeled_count = label_mask.sum().item()
                            print(f"[DEBUG Tracker] KF {frame.frame_id} - Label {local_id} ({class_name}): {labeled_count} points labeled")
                    
                    # Apply 3D refinement
                    if frame.X_canon is not None and frame.global_instance_ids.sum() > 0:
                        print(f"[DEBUG Tracker] Applying 3D refinement for keyframe {frame.frame_id}")
                        refined_labels = self.semantic_mapper.refine_labels_with_3d_proximity(
                            points_3d=frame.X_canon,
                            labels=frame.global_instance_ids.view(-1),
                            label_confidences=self.label_confidences,
                            distance_threshold=0.1,
                            min_neighbors=3
                        )
                        frame.global_instance_ids = refined_labels.view(-1, 1)
                        
                # Log final statistics for new keyframe
                if frame.global_instance_ids is not None:
                    non_background = (frame.global_instance_ids != 0).sum().item()
                    total_points = frame.global_instance_ids.numel()
                    unique_labels = torch.unique(frame.global_instance_ids)
                    print(f"[DEBUG Tracker] Keyframe {frame.frame_id} final stats:")
                    print(f"  - Total points: {total_points}")
                    print(f"  - Labeled points: {non_background} ({non_background/total_points*100:.1f}%)")
                    print(f"  - Unique labels: {unique_labels.tolist()}")
                    print(f"  - Global semantic map size: {len(self.g_global_id_to_class_label_map)} classes")
                    print(f"  - Global semantic map: {self.g_global_id_to_class_label_map}")
            else: # frame.X_canon is None, should not happen if update_pointmap was called
                 print(f"[Tracker WARN] New keyframe {frame.frame_id} has no X_canon. Cannot assign global_instance_ids.")

            self.reset_idx_f2k() # Reset for next tracking sequence against this new KF

        else: # This is an INTERMEDIATE frame, not a new keyframe
            print(f"[Tracker INFO] Frame {frame.frame_id} is an intermediate frame (not keyframe)")
            # Propagate global_instance_ids from the last keyframe (last_kf_for_propagation)
            # using idx_f2k (which maps current frame points to last_kf_for_propagation points)
            if last_kf_for_propagation is not None and \
               last_kf_for_propagation.global_instance_ids is not None and \
               frame.X_canon is not None: # Ensure current frame has points

                # num_points_current_frame defined earlier
                # num_points_last_kf = Xkf.shape[0] (Xkf was from matching against last_kf_for_propagation)
                num_points_last_kf = Xkf.shape[0]


                if last_kf_for_propagation.global_instance_ids.numel() == num_points_last_kf and \
                   idx_f2k.shape[0] == num_points_current_frame:

                    frame.global_instance_ids = torch.zeros(num_points_current_frame, 1, dtype=torch.int64, device=self.device)
                    last_kf_global_ids_flat = last_kf_for_propagation.global_instance_ids.view(-1)

                    # Count how many points will get labels
                    labeled_count = 0
                    for p_curr_idx in range(num_points_current_frame):
                        p_kf_idx = idx_f2k[p_curr_idx].item()
                        if 0 <= p_kf_idx < num_points_last_kf:
                            frame.global_instance_ids.view(-1)[p_curr_idx] = last_kf_global_ids_flat[p_kf_idx]
                            if last_kf_global_ids_flat[p_kf_idx] > 0:  # Non-background
                                labeled_count += 1
                    
                    # Try to propagate high-confidence labels to nearby unlabeled points
                    if frame.X_canon is not None and labeled_count > 0:
                        print(f"[Tracker INFO] Propagating high-confidence labels to nearby points...")
                        updated_labels = self.semantic_mapper.propagate_high_confidence_labels(
                            points_3d=frame.X_canon,
                            labels=frame.global_instance_ids.view(-1),
                            confidences=torch.full((num_points_current_frame,), 60.0, device=self.device),  # Assume propagated labels have good confidence
                            propagation_radius=0.05,
                            confidence_threshold=50.0
                        )
                        
                        new_labeled_count = (updated_labels > 0).sum().item()
                        if new_labeled_count > labeled_count:
                            frame.global_instance_ids = updated_labels.view(-1, 1)
                            print(f"[Tracker INFO] Propagation increased labeled points from {labeled_count} to {new_labeled_count}")
                            labeled_count = new_labeled_count
                    
                    print(f"[Tracker DBG] Propagated global IDs to intermediate frame {frame.frame_id} from KF {last_kf_for_propagation.frame_id}: {labeled_count}/{num_points_current_frame} points labeled")
                else:
                    # print(f"[Tracker WARN] Cannot propagate global IDs to intermediate frame {frame.frame_id}: Mismatch in keyframe data or idx_f2k.")
                    # frame.global_instance_ids remains zeros or None
                    if frame.X_canon is not None: # Still ensure it exists if X_canon does
                         frame.global_instance_ids = torch.zeros(num_points_current_frame, 1, dtype=torch.int64, device=self.device)


        return (
            new_kf, # This is the boolean indicating if current frame became a keyframe
            [
                keyframe.X_canon,
                keyframe.get_average_conf(),
                frame.X_canon,
                frame.get_average_conf(),
                Qkf,
                Qff,
            ],
            False,
        )

    def get_points_poses(self, frame, keyframe, idx_f2k, img_size, use_calib, K=None):
        Xf = frame.X_canon
        Xk = keyframe.X_canon
        T_WCf = frame.T_WC
        T_WCk = keyframe.T_WC

        # Average confidence
        Cf = frame.get_average_conf()
        Ck = keyframe.get_average_conf()

        meas_k = None
        valid_meas_k = None

        if use_calib:
            Xf = constrain_points_to_ray(img_size, Xf[None], K).squeeze(0)
            Xk = constrain_points_to_ray(img_size, Xk[None], K).squeeze(0)

            # Setup pixel coordinates
            uv_k = get_pixel_coords(1, img_size, device=Xf.device, dtype=Xf.dtype)
            uv_k = uv_k.view(-1, 2)
            meas_k = torch.cat((uv_k, torch.log(Xk[..., 2:3])), dim=-1)
            # Avoid any bad calcs in log
            valid_meas_k = Xk[..., 2:3] > self.cfg["depth_eps"]
            meas_k[~valid_meas_k.repeat(1, 3)] = 0.0

        return Xf[idx_f2k], Xk, T_WCf, T_WCk, Cf[idx_f2k], Ck, meas_k, valid_meas_k

    def solve(self, sqrt_info, r, J):
        whitened_r = sqrt_info * r
        robust_sqrt_info = sqrt_info * torch.sqrt(
            huber(whitened_r, k=self.cfg["huber"])
        )
        mdim = J.shape[-1]
        A = (robust_sqrt_info[..., None] * J).view(-1, mdim)  # dr_dX
        b = (robust_sqrt_info * r).view(-1, 1)  # z-h
        H = A.T @ A
        g = -A.T @ b
        cost = 0.5 * (b.T @ b).item()

        L = torch.linalg.cholesky(H, upper=False)
        tau_j = torch.cholesky_solve(g, L, upper=False).view(1, -1)

        return tau_j, cost

    def opt_pose_ray_dist_sim3(self, Xf, Xk, T_WCf, T_WCk, Qk, valid):
        last_error = 0
        sqrt_info_ray = 1 / self.cfg["sigma_ray"] * valid * torch.sqrt(Qk)
        sqrt_info_dist = 1 / self.cfg["sigma_dist"] * valid * torch.sqrt(Qk)
        sqrt_info = torch.cat((sqrt_info_ray.repeat(1, 3), sqrt_info_dist), dim=1)

        # Solving for relative pose without scale!
        T_CkCf = T_WCk.inv() * T_WCf

        # Precalculate distance and ray for obs k
        rd_k = point_to_ray_dist(Xk, jacobian=False)

        old_cost = float("inf")
        for step in range(self.cfg["max_iters"]):
            Xf_Ck, dXf_Ck_dT_CkCf = act_Sim3(T_CkCf, Xf, jacobian=True)
            rd_f_Ck, drd_f_Ck_dXf_Ck = point_to_ray_dist(Xf_Ck, jacobian=True)
            # r = z-h(x)
            r = rd_k - rd_f_Ck
            # Jacobian
            J = -drd_f_Ck_dXf_Ck @ dXf_Ck_dT_CkCf

            tau_ij_sim3, new_cost = self.solve(sqrt_info, r, J)
            T_CkCf = T_CkCf.retr(tau_ij_sim3)

            if check_convergence(
                step,
                self.cfg["rel_error"],
                self.cfg["delta_norm"],
                old_cost,
                new_cost,
                tau_ij_sim3,
            ):
                break
            old_cost = new_cost

            if step == self.cfg["max_iters"] - 1:
                print(f"max iters reached {last_error}")

        # Assign new pose based on relative pose
        T_WCf = T_WCk * T_CkCf

        return T_WCf, T_CkCf

    def opt_pose_calib_sim3(
        self, Xf, Xk, T_WCf, T_WCk, Qk, valid, meas_k, valid_meas_k, K, img_size
    ):
        last_error = 0
        sqrt_info_pixel = 1 / self.cfg["sigma_pixel"] * valid * torch.sqrt(Qk)
        sqrt_info_depth = 1 / self.cfg["sigma_depth"] * valid * torch.sqrt(Qk)
        sqrt_info = torch.cat((sqrt_info_pixel.repeat(1, 2), sqrt_info_depth), dim=1)

        # Solving for relative pose without scale!
        T_CkCf = T_WCk.inv() * T_WCf

        old_cost = float("inf")
        for step in range(self.cfg["max_iters"]):
            Xf_Ck, dXf_Ck_dT_CkCf = act_Sim3(T_CkCf, Xf, jacobian=True)
            pzf_Ck, dpzf_Ck_dXf_Ck, valid_proj = project_calib(
                Xf_Ck,
                K,
                img_size,
                jacobian=True,
                border=self.cfg["pixel_border"],
                z_eps=self.cfg["depth_eps"],
            )
            valid2 = valid_proj & valid_meas_k
            sqrt_info2 = valid2 * sqrt_info

            # r = z-h(x)
            r = meas_k - pzf_Ck
            # Jacobian
            J = -dpzf_Ck_dXf_Ck @ dXf_Ck_dT_CkCf

            tau_ij_sim3, new_cost = self.solve(sqrt_info2, r, J)
            T_CkCf = T_CkCf.retr(tau_ij_sim3)

            if check_convergence(
                step,
                self.cfg["rel_error"],
                self.cfg["delta_norm"],
                old_cost,
                new_cost,
                tau_ij_sim3,
            ):
                break
            old_cost = new_cost

            if step == self.cfg["max_iters"] - 1:
                print(f"max iters reached {last_error}")

        # Assign new pose based on relative pose
        T_WCf = T_WCk * T_CkCf

        return T_WCf, T_CkCf
