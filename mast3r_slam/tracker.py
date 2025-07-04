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
from mast3r_slam.semantic_processor import process_frame_for_semantics, TEXT_PROMPTS as SEMANTIC_TEXT_PROMPTS # New Import
from collections import Counter # Ensure Counter is imported (already added previously)


class FrameTracker:
    def __init__(self, model, frames, device, initial_global_id_to_class_label_map: dict):
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

        # --- Start: 3D Instance Association Logic ---
        # This logic runs if we have a valid keyframe to track against and we have its global_instance_ids
        if keyframe is not None and keyframe.global_instance_ids is not None and \
           frame.local_instance_mask is not None and frame.local_id_to_class_label_map is not None:

            # Ensure keyframe.global_instance_ids is flat for easier indexing if it's not already
            # and has the correct number of elements corresponding to keyframe's points (Xkf.shape[0])
            # Xkf was used to get idx_f2k.
            num_points_keyframe = Xkf.shape[0] # Xkf has shape (N_kf_pts, 3)

            if keyframe.global_instance_ids.numel() != num_points_keyframe:
                print(f"[Tracker WARN] Keyframe {keyframe.frame_id} global_instance_ids numel ({keyframe.global_instance_ids.numel()}) "
                      f"does not match its point count ({num_points_keyframe}). Skipping global ID propagation.")
            else:
                keyframe_global_ids_flat = keyframe.global_instance_ids.view(-1)

                # Flatten current frame's local SAM mask to align with point indices
                # frame.local_instance_mask is HxW.
                # Assuming Xff (and thus frame.X_canon) points correspond row-major to HxW.
                current_frame_local_sam_ids_flat = frame.local_instance_mask.view(-1)

                if current_frame_local_sam_ids_flat.numel() != num_points_current_frame:
                    print(f"[Tracker WARN] Current frame {frame.frame_id} local_instance_mask numel ({current_frame_local_sam_ids_flat.numel()}) "
                          f"does not match its point count ({num_points_current_frame}). Skipping global ID propagation.")
                else:
                    local_id_to_global_id_candidates = {}

                    # Iterate through current frame points that have a valid match in the keyframe
                    # idx_f2k[i] gives the index of the point in keyframe that matches point i in current_frame
                    # valid_match_k[i] would indicate if point i in keyframe has a valid match (not directly used here, idx_f2k implies match)
                    # We need to iterate based on current_frame's point indices 0 to N-1
                    for p_curr_idx in range(num_points_current_frame):
                        p_kf_idx = idx_f2k[p_curr_idx].item() # Get the flat index into keyframe points

                        # Check if this match is valid (e.g., sometimes idx_f2k might have placeholder for no match,
                        # or p_kf_idx might be out of bounds if not careful. mast3r_match_asymmetric should give valid indices)
                        # A simple validity check for p_kf_idx:
                        if 0 <= p_kf_idx < num_points_keyframe:
                            prev_global_id = keyframe_global_ids_flat[p_kf_idx].item()
                            current_local_sam_id = current_frame_local_sam_ids_flat[p_curr_idx].item()

                            if current_local_sam_id != 0 and prev_global_id != 0: # Not background, and matched to a known global object
                                if current_local_sam_id not in local_id_to_global_id_candidates:
                                    local_id_to_global_id_candidates[current_local_sam_id] = []
                                local_id_to_global_id_candidates[current_local_sam_id].append(prev_global_id)

                    # Resolve and Propagate Global IDs for Matched Segments
                    for local_id, candidate_global_ids_list in local_id_to_global_id_candidates.items():
                        if candidate_global_ids_list:
                            # Resolve: Pick the most frequent global ID (majority vote)
                            chosen_global_id = Counter(candidate_global_ids_list).most_common(1)[0][0]
                            # Assign this chosen_global_id to all points in current_frame with this local_id
                            frame.global_instance_ids.view(-1)[current_frame_local_sam_ids_flat == local_id] = chosen_global_id

            # Assign New Global IDs for Unmatched New Segments in current_frame
            unique_local_ids_in_current_frame = torch.unique(current_frame_local_sam_ids_flat)
            for local_id_val_tensor in unique_local_ids_in_current_frame:
                local_id_val = local_id_val_tensor.item()
                if local_id_val == 0: # Skip background SAM label
                    continue

                # Check if this local_id_val segment in current_frame still has unassigned global_instance_ids (i.e., all are 0)
                # This means it wasn't propagated from a keyframe match.
                current_segment_mask_flat = (current_frame_local_sam_ids_flat == local_id_val)
                if (frame.global_instance_ids.view(-1)[current_segment_mask_flat] == 0).all():
                    new_global_id = self.g_next_global_id
                    frame.global_instance_ids.view(-1)[current_segment_mask_flat] = new_global_id

                    class_name = frame.local_id_to_class_label_map.get(local_id_val, f"unknown_local_id_{local_id_val}")
                    self.g_global_id_to_class_label_map[new_global_id] = class_name
                    self.g_next_global_id += 1

        elif frame.local_instance_mask is not None and frame.local_id_to_class_label_map is not None:
            # This is likely the first keyframe, or tracking was lost and re-initialized.
            # Populate global_instance_ids based purely on this frame's SAM segmentation.
            print(f"[Tracker INFO] Frame {frame.frame_id}: No valid keyframe for propagation or keyframe lacks global IDs. Initializing global IDs from current frame's SAM.")
            current_frame_local_sam_ids_flat = frame.local_instance_mask.view(-1)
            if current_frame_local_sam_ids_flat.numel() != num_points_current_frame:
                 print(f"[Tracker WARN] First Keyframe {frame.frame_id} local_instance_mask numel ({current_frame_local_sam_ids_flat.numel()}) "
                       f"does not match its point count ({num_points_current_frame}). Cannot init global IDs.")
            else:
                unique_local_ids_in_current_frame = torch.unique(current_frame_local_sam_ids_flat)
                for local_id_val_tensor in unique_local_ids_in_current_frame:
                    local_id_val = local_id_val_tensor.item()
                    if local_id_val == 0: # Skip background
                        continue

                    new_global_id = self.g_next_global_id
                    frame.global_instance_ids.view(-1)[current_frame_local_sam_ids_flat == local_id_val] = new_global_id
                    class_name = frame.local_id_to_class_label_map.get(local_id_val, f"unknown_local_id_{local_id_val}")
                    self.g_global_id_to_class_label_map[new_global_id] = class_name
                    self.g_next_global_id += 1
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

        # Rest idx if new keyframe
        if new_kf:
            # This is a NEW KEYFRAME
            print(f"[Tracker INFO] Frame {frame.frame_id} designated as new keyframe.")
            # 1. Perform full semantic processing (SAM + CLIP)
            if frame.img is not None: # Ensure image data is available
                print(f"[Tracker INFO] Running full semantic processing for new keyframe {frame.frame_id}...")
                try:
                    local_mask, local_map = process_frame_for_semantics(
                        image_tensor_chw_0_1_rgb=frame.img.squeeze(0),
                        text_prompts_for_clip=SEMANTIC_TEXT_PROMPTS
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
                        current_frame_local_sam_ids_flat = frame.local_instance_mask.view(-1)

                        if current_frame_local_sam_ids_flat.numel() == num_points_current_frame:
                            local_id_to_global_id_candidates = {}
                            for p_curr_idx in range(num_points_current_frame):
                                p_kf_idx = idx_f2k[p_curr_idx].item()
                                if 0 <= p_kf_idx < num_points_last_kf:
                                    prev_global_id = last_kf_global_ids_flat[p_kf_idx].item()
                                    current_local_sam_id = current_frame_local_sam_ids_flat[p_curr_idx].item()
                                    if current_local_sam_id != 0 and prev_global_id != 0:
                                        if current_local_sam_id not in local_id_to_global_id_candidates:
                                            local_id_to_global_id_candidates[current_local_sam_id] = []
                                        local_id_to_global_id_candidates[current_local_sam_id].append(prev_global_id)

                            for local_id, candidates in local_id_to_global_id_candidates.items():
                                if candidates:
                                    chosen_global_id = Counter(candidates).most_common(1)[0][0]
                                    frame.global_instance_ids.view(-1)[current_frame_local_sam_ids_flat == local_id] = chosen_global_id

                    # Assign New Global IDs for this new keyframe
                    unique_local_ids = torch.unique(current_frame_local_sam_ids_flat)
                    for local_id_tensor in unique_local_ids:
                        local_id = local_id_tensor.item()
                        if local_id == 0: continue
                        current_segment_mask = (current_frame_local_sam_ids_flat == local_id)
                        # Check if any point in this segment still has global_id 0 (unassigned by propagation)
                        if (frame.global_instance_ids.view(-1)[current_segment_mask] == 0).any(): # Check if *any* part of segment is new
                             # More precise: if the majority/all are 0, or if it wasn't in local_id_to_global_id_candidates
                            is_truly_new_segment = local_id not in local_id_to_global_id_candidates or \
                                                   not local_id_to_global_id_candidates[local_id]

                            if is_truly_new_segment or (frame.global_instance_ids.view(-1)[current_segment_mask] == 0).all():
                                new_global_id = self.g_next_global_id
                                frame.global_instance_ids.view(-1)[current_segment_mask & (frame.global_instance_ids.view(-1) == 0)] = new_global_id
                                class_name = frame.local_id_to_class_label_map.get(local_id, f"unclassified_local_id_{local_id}")
                                self.g_global_id_to_class_label_map[new_global_id] = class_name
                                self.g_next_global_id += 1
                elif frame.local_instance_mask is not None: # Is a new keyframe, but no prior keyframe to propagate from (e.g. first frame)
                    print(f"[Tracker INFO] New keyframe {frame.frame_id} is the first or has no prior KF with global IDs. Initializing from its own SAM.")
                    current_frame_local_sam_ids_flat = frame.local_instance_mask.view(-1)
                    if current_frame_local_sam_ids_flat.numel() == num_points_current_frame:
                        unique_local_ids = torch.unique(current_frame_local_sam_ids_flat)
                        for local_id_tensor in unique_local_ids:
                            local_id = local_id_tensor.item()
                            if local_id == 0: continue
                            new_global_id = self.g_next_global_id
                            frame.global_instance_ids.view(-1)[current_frame_local_sam_ids_flat == local_id] = new_global_id
                            class_name = frame.local_id_to_class_label_map.get(local_id, f"unclassified_local_id_{local_id}")
                            self.g_global_id_to_class_label_map[new_global_id] = class_name
                            self.g_next_global_id += 1
                    else:
                        print(f"[Tracker WARN] KF {frame.frame_id} local_mask numel mismatch with points. Cannot init global IDs.")
            else: # frame.X_canon is None, should not happen if update_pointmap was called
                 print(f"[Tracker WARN] New keyframe {frame.frame_id} has no X_canon. Cannot assign global_instance_ids.")

            self.reset_idx_f2k() # Reset for next tracking sequence against this new KF

        else: # This is an INTERMEDIATE frame, not a new keyframe
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

                    for p_curr_idx in range(num_points_current_frame):
                        p_kf_idx = idx_f2k[p_curr_idx].item()
                        if 0 <= p_kf_idx < num_points_last_kf:
                            frame.global_instance_ids.view(-1)[p_curr_idx] = last_kf_global_ids_flat[p_kf_idx]
                    # print(f"[Tracker DBG] Propagated global IDs to intermediate frame {frame.frame_id} from KF {last_kf_for_propagation.frame_id}")
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
