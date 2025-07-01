import dataclasses
from enum import Enum
from typing import Optional
import lietorch
import torch
from mast3r_slam.mast3r_utils import resize_img
from mast3r_slam.config import config
from .seem_utils import load_sam_model, run_sam_inference # Changed to load_sam_model


class Mode(Enum):
    INIT = 0
    TRACKING = 1
    RELOC = 2
    TERMINATED = 3


@dataclasses.dataclass
class Frame:
    frame_id: int
    img: torch.Tensor
    img_shape: torch.Tensor
    img_true_shape: torch.Tensor
    uimg: torch.Tensor
    T_WC: lietorch.Sim3 = lietorch.Sim3.Identity(1)
    X_canon: Optional[torch.Tensor] = None
    C: Optional[torch.Tensor] = None
    feat: Optional[torch.Tensor] = None
    pos: Optional[torch.Tensor] = None
    raw_segmentation_mask: Optional[torch.Tensor] = None  # HxW tensor of instance IDs
    point_labels: Optional[torch.Tensor] = None  # HxW or N_points x 1 tensor of label IDs for X_canon
    label_map: Optional[dict] = None  # Dict mapping label IDs to string labels {1: "cat", 2: "dog"}
    N: int = 0
    N_updates: int = 0
    K: Optional[torch.Tensor] = None

    def get_score(self, C):
        filtering_score = config["tracking"]["filtering_score"]
        if filtering_score == "median":
            score = torch.median(C)  # Is this slower than mean? Is it worth it?
        elif filtering_score == "mean":
            score = torch.mean(C)
        return score

    def update_pointmap(self, X: torch.Tensor, C: torch.Tensor):
        filtering_mode = config["tracking"]["filtering_mode"]

        if self.N == 0:
            self.X_canon = X.clone()
            self.C = C.clone()
            self.N = 1
            self.N_updates = 1
            if filtering_mode == "best_score":
                self.score = self.get_score(C)
            return

        if filtering_mode == "first":
            if self.N_updates == 1:
                self.X_canon = X.clone()
                self.C = C.clone()
                self.N = 1
        elif filtering_mode == "recent":
            self.X_canon = X.clone()
            self.C = C.clone()
            self.N = 1
        elif filtering_mode == "best_score":
            new_score = self.get_score(C)
            if new_score > self.score:
                self.X_canon = X.clone()
                self.C = C.clone()
                self.N = 1
                self.score = new_score
        elif filtering_mode == "indep_conf":
            new_mask = C > self.C
            self.X_canon[new_mask.repeat(1, 3)] = X[new_mask.repeat(1, 3)]
            self.C[new_mask] = C[new_mask]
            self.N = 1
        elif filtering_mode == "weighted_pointmap":
            self.X_canon = ((self.C * self.X_canon) + (C * X)) / (self.C + C)
            self.C = self.C + C
            self.N += 1
        elif filtering_mode == "weighted_spherical":

            def cartesian_to_spherical(P):
                r = torch.linalg.norm(P, dim=-1, keepdim=True)
                x, y, z = torch.tensor_split(P, 3, dim=-1)
                phi = torch.atan2(y, x)
                theta = torch.acos(z / r)
                spherical = torch.cat((r, phi, theta), dim=-1)
                return spherical

            def spherical_to_cartesian(spherical):
                r, phi, theta = torch.tensor_split(spherical, 3, dim=-1)
                x = r * torch.sin(theta) * torch.cos(phi)
                y = r * torch.sin(theta) * torch.sin(phi)
                z = r * torch.cos(theta)
                P = torch.cat((x, y, z), dim=-1)
                return P

            spherical1 = cartesian_to_spherical(self.X_canon)
            spherical2 = cartesian_to_spherical(X)
            spherical = ((self.C * spherical1) + (C * spherical2)) / (self.C + C)

            self.X_canon = spherical_to_cartesian(spherical)
            self.C = self.C + C
            self.N += 1

        self.N_updates += 1

        # Associate labels with 3D points
        if self.raw_segmentation_mask is not None and self.X_canon is not None:
            # X_canon is (N_points, 3), where N_points is typically h*w
            # raw_segmentation_mask is (h_mask, w_mask)
            # self.img_shape can be tensor([h, w]) or tensor([[h, w]])

            current_img_shape_tensor = self.img_shape.cpu()
            if current_img_shape_tensor.ndim == 1 and current_img_shape_tensor.numel() == 2:
                # Shape is (2,), e.g., tensor([h, w])
                h_proc, w_proc = current_img_shape_tensor.tolist()
            elif current_img_shape_tensor.ndim == 2 and current_img_shape_tensor.shape[0] == 1 and current_img_shape_tensor.shape[1] == 2:
                # Shape is (1, 2), e.g., tensor([[h, w]])
                h_proc, w_proc = current_img_shape_tensor[0].tolist()
            else:
                print(f"[Error] Frame {self.frame_id}: self.img_shape has unexpected format. Shape: {current_img_shape_tensor.shape}, Value: {current_img_shape_tensor}. Cannot determine h_proc, w_proc.")
                self.point_labels = None
                return # Exit early if dimensions can't be determined

            h_mask, w_mask = self.raw_segmentation_mask.shape

            if h_proc == h_mask and w_proc == w_mask:
                # If dimensions match, reshape mask to align with X_canon points
                self.point_labels = self.raw_segmentation_mask.reshape(-1, 1).clone()
            else:
                # Dimensions do not match. This might happen if SEEM output resolution
                # is different from MaSt3R's effective resolution for X_canon.
                # A resize of raw_segmentation_mask might be needed here.
                # For now, print a warning and skip if not perfectly matched.
                # User might need to implement robust resizing (e.g., using nearest neighbor for masks).
                print(f"[Warning] Frame {self.frame_id}: Mismatch between X_canon's implied image dimensions ({h_proc}x{w_proc}) and raw_segmentation_mask dimensions ({h_mask}x{w_mask}). Point labels cannot be directly assigned.")
                # Attempt resize as a fallback - using NEAREST interpolation for masks
                if h_proc * w_proc == self.X_canon.shape[0]: # Check if X_canon matches expected number of points
                    try:
                        resized_mask = torch.nn.functional.interpolate(
                            self.raw_segmentation_mask.float().unsqueeze(0).unsqueeze(0), # B, C, H, W
                            size=(h_proc, w_proc),
                            mode='nearest'
                        ).squeeze(0).squeeze(0).long()
                        self.point_labels = resized_mask.reshape(-1, 1).clone()
                        print(f"[INFO] Frame {self.frame_id}: Resized raw_segmentation_mask from {h_mask}x{w_mask} to {h_proc}x{w_proc} to match X_canon.")
                    except Exception as e:
                        print(f"[Error] Frame {self.frame_id}: Failed to resize raw_segmentation_mask: {e}")
                        self.point_labels = None # Explicitly set to None
                else:
                    self.point_labels = None # Explicitly set to None

            # Ensure point_labels has the same number of entries as points in X_canon (first dim)
            # and is on the same device.
            if self.point_labels is not None:
                if self.point_labels.shape[0] != self.X_canon.shape[0]:
                    print(f"[Warning] Frame {self.frame_id}: Mismatch in number of point labels ({self.point_labels.shape[0]}) and points in X_canon ({self.X_canon.shape[0]}). Resetting point_labels.")
                    self.point_labels = None
                else:
                    self.point_labels = self.point_labels.to(self.X_canon.device)
        return

    def get_average_conf(self):
        return self.C / self.N if self.C is not None else None


def create_frame(i, img, T_WC, img_size=512, device="cuda:0"):
    img = resize_img(img, img_size)
    rgb = img["img"].to(device=device)
    img_shape = torch.tensor(img["true_shape"], device=device)
    img_true_shape = img_shape.clone()
    uimg = torch.from_numpy(img["unnormalized_img"]) / 255.0
    downsample = config["dataset"]["img_downsample"]
    if downsample > 1:
        uimg = uimg[::downsample, ::downsample]
        img_shape = img_shape // downsample # This img_shape is used by Frame, MaSt3R

    frame = Frame(i, rgb, img_shape, img_true_shape, uimg, T_WC)

    # SEEM Integration
    # Determine which image to use for SEEM. `uimg` is (H, W, C) numpy array (0-255)
    # `rgb` is (1, 3, H_proc, W_proc) tensor, normalized.
    # Assuming SEEM works best with less processed images, let's use `uimg` before downsampling if possible,
    # or ensure SEEM's input matches what it expects.
    # For this integration, we'll use the potentially downsampled `uimg` and convert to tensor.
    # The user should adjust this based on their SEEM model's requirements.

    # Convert uimg (H, W, C) numpy to (C, H, W) tensor for SEEM placeholder
    seem_input_image_np = frame.uimg.numpy() # uimg is already a tensor, get its numpy array
    seem_input_tensor = torch.from_numpy(seem_input_image_np).permute(2, 0, 1).float().to(device) # C, H, W

    # Ensure SEEM model is loaded (placeholder will try to load if not)
    # In a real scenario, model loading should be handled more explicitly, e.g., in main.py
    load_seem_model() # TODO: User should manage SEEM model loading path and device

    # Run SEEM inference
    # Vocabulary can be passed here if needed, e.g. from config
    raw_mask, label_map_from_seem = run_seem_inference(seem_input_tensor)

    frame.raw_segmentation_mask = raw_mask.to(device) # Ensure mask is on the same device as other frame data
    frame.label_map = label_map_from_seem

    # Note: frame.point_labels will be populated in update_pointmap
    return frame


class SharedStates:
    def __init__(self, manager, h, w, dtype=torch.float32, device="cuda"):
        self.h, self.w = h, w
        self.dtype = dtype
        self.device = device

        self.lock = manager.RLock()
        self.paused = manager.Value("i", 0)
        self.mode = manager.Value("i", Mode.INIT)
        self.reloc_sem = manager.Value("i", 0)
        self.global_optimizer_tasks = manager.list()
        self.edges_ii = manager.list()
        self.edges_jj = manager.list()

        self.feat_dim = 1024
        self.num_patches = h * w // (16 * 16)

        # fmt:off
        # shared state for the current frame (used for reloc/visualization)
        self.dataset_idx = torch.zeros(1, device=device, dtype=torch.int).share_memory_()
        self.img = torch.zeros(3, h, w, device=device, dtype=dtype).share_memory_()
        self.uimg = torch.zeros(h, w, 3, device="cpu", dtype=dtype).share_memory_()
        self.img_shape = torch.zeros(1, 2, device=device, dtype=torch.int).share_memory_()
        self.img_true_shape = torch.zeros(1, 2, device=device, dtype=torch.int).share_memory_()
        self.T_WC = lietorch.Sim3.Identity(1, device=device, dtype=dtype).data.share_memory_()
        self.X = torch.zeros(h * w, 3, device=device, dtype=dtype).share_memory_()
        self.C = torch.zeros(h * w, 1, device=device, dtype=dtype).share_memory_()
        self.feat = torch.zeros(1, self.num_patches, self.feat_dim, device=device, dtype=dtype).share_memory_()
        self.pos = torch.zeros(1, self.num_patches, 2, device=device, dtype=torch.long).share_memory_()
        # New fields for segmentation data
        self.raw_segmentation_mask = torch.zeros(h, w, device=device, dtype=torch.int64).share_memory_()
        self.point_labels = torch.zeros(h * w, 1, device=device, dtype=torch.int64).share_memory_()
        # label_map (dict) is not directly shareable via share_memory_(). It will be part of the Frame object.
        # fmt: on

    def set_frame(self, frame):
        with self.lock:
            self.dataset_idx[:] = frame.frame_id
            self.img[:] = frame.img
            self.uimg[:] = frame.uimg
            self.img_shape[:] = frame.img_shape
            self.img_true_shape[:] = frame.img_true_shape
            self.T_WC[:] = frame.T_WC.data
            if frame.X_canon is not None:
                self.X[:] = frame.X_canon
            if frame.C is not None:
                self.C[:] = frame.C
            if frame.feat is not None:
                self.feat[:] = frame.feat
            if frame.pos is not None:
                self.pos[:] = frame.pos
            if frame.raw_segmentation_mask is not None:
                # Ensure mask matches expected dimensions, pad or crop if necessary, or assert
                # For now, assume it's hxw
                h_mask, w_mask = frame.raw_segmentation_mask.shape
                h_shared, w_shared = self.raw_segmentation_mask.shape
                if h_mask == h_shared and w_mask == w_shared:
                    self.raw_segmentation_mask[:] = frame.raw_segmentation_mask
                else:
                    # Handle mismatch: either log a warning, resize, or error
                    # This could happen if img_downsample changes mask dimensions relative to SharedStates h,w
                    print(f"[Warning] Mismatch in raw_segmentation_mask dimensions for SharedStates. Expected {h_shared}x{w_shared}, got {h_mask}x{w_mask}. Skipping update for this mask.")
            if frame.point_labels is not None:
                self.point_labels[:] = frame.point_labels
            # frame.label_map is handled by get_frame reconstruction

    def get_frame(self):
        with self.lock:
            frame = Frame(
                int(self.dataset_idx[0]),
                self.img.clone(), # Clone to avoid issues if tensor is further processed
                self.img_shape.clone(),
                self.img_true_shape.clone(),
                self.uimg.clone(),
                lietorch.Sim3(self.T_WC.clone()),
            )
            frame.X_canon = self.X.clone()
            frame.C = self.C.clone()
            frame.feat = self.feat.clone()
            frame.pos = self.pos.clone()
            # Retrieve segmentation data
            frame.raw_segmentation_mask = self.raw_segmentation_mask.clone()
            frame.point_labels = self.point_labels.clone()
            # frame.label_map will be set if it was part of the original frame object
            # that was passed to set_frame. If set_frame only got tensors,
            # then label_map needs to be explicitly managed if it's to be retrieved here.
            # For now, we assume label_map is part of the Frame object passed around,
            # and if `set_frame` was called with a Frame that had it, it's implicitly "there".
            # However, to be safe, it should be explicitly passed if needed after get_frame.
            # Let's assume it's not directly stored/retrieved from shared memory here,
            # but rather set on the Frame object instance after it's created by get_frame,
            # or the original Frame's label_map is passed along by the process managing states.
            # For simplicity in this step, we'll rely on it being part of the Frame object.
            return frame

    def queue_global_optimization(self, idx):
        with self.lock:
            self.global_optimizer_tasks.append(idx)

    def queue_reloc(self):
        with self.lock:
            self.reloc_sem.value += 1

    def dequeue_reloc(self):
        with self.lock:
            if self.reloc_sem.value == 0:
                return
            self.reloc_sem.value -= 1

    def get_mode(self):
        with self.lock:
            return self.mode.value

    def set_mode(self, mode):
        with self.lock:
            self.mode.value = mode

    def pause(self):
        with self.lock:
            self.paused.value = 1

    def unpause(self):
        with self.lock:
            self.paused.value = 0

    def is_paused(self):
        with self.lock:
            return self.paused.value == 1


class SharedKeyframes:
    def __init__(self, manager, h, w, buffer=512, dtype=torch.float32, device="cuda"):
        self.lock = manager.RLock()
        self.n_size = manager.Value("i", 0)

        self.h, self.w = h, w
        self.buffer = buffer
        self.dtype = dtype
        self.device = device

        self.feat_dim = 1024
        self.num_patches = h * w // (16 * 16)

        # fmt:off
        self.dataset_idx = torch.zeros(buffer, device=device, dtype=torch.int).share_memory_()
        self.img = torch.zeros(buffer, 3, h, w, device=device, dtype=dtype).share_memory_()
        self.uimg = torch.zeros(buffer, h, w, 3, device="cpu", dtype=dtype).share_memory_()
        self.img_shape = torch.zeros(buffer, 1, 2, device=device, dtype=torch.int).share_memory_()
        self.img_true_shape = torch.zeros(buffer, 1, 2, device=device, dtype=torch.int).share_memory_()
        self.T_WC = torch.zeros(buffer, 1, lietorch.Sim3.embedded_dim, device=device, dtype=dtype).share_memory_()
        self.X = torch.zeros(buffer, h * w, 3, device=device, dtype=dtype).share_memory_()
        self.C = torch.zeros(buffer, h * w, 1, device=device, dtype=dtype).share_memory_()
        self.N = torch.zeros(buffer, device=device, dtype=torch.int).share_memory_()
        self.N_updates = torch.zeros(buffer, device=device, dtype=torch.int).share_memory_()
        self.feat = torch.zeros(buffer, 1, self.num_patches, self.feat_dim, device=device, dtype=dtype).share_memory_()
        self.pos = torch.zeros(buffer, 1, self.num_patches, 2, device=device, dtype=torch.long).share_memory_()
        self.is_dirty = torch.zeros(buffer, 1, device=device, dtype=torch.bool).share_memory_()
        self.K = torch.zeros(3, 3, device=device, dtype=dtype).share_memory_()
        # New fields for segmentation data
        self.raw_segmentation_mask = torch.zeros(buffer, h, w, device=device, dtype=torch.int64).share_memory_()
        self.point_labels = torch.zeros(buffer, h * w, 1, device=device, dtype=torch.int64).share_memory_()
        # label_map (dict) will be stored as a Python list of dicts, managed by the multiprocessing Manager if needed,
        # or simply as attributes of the Frame objects. For SharedKeyframes, we'll store it as a regular Python list
        # of dictionaries, assuming it's not performance critical for direct tensor sharing here.
        # Or, more simply, it's part of the Frame object and not a separate list here.
        # Let's stick to the principle that label_map is part of the Frame object.
        # self.label_maps = [{} for _ in range(buffer)] # If we were to store them separately.
        # fmt: on

    def __getitem__(self, idx) -> Frame:
        with self.lock:
            # put all of the data into a frame
            kf = Frame(
                int(self.dataset_idx[idx]),
                self.img[idx].clone(),
                self.img_shape[idx].clone(),
                self.img_true_shape[idx].clone(),
                self.uimg[idx].clone(),
                lietorch.Sim3(self.T_WC[idx].clone()),
            )
            kf.X_canon = self.X[idx].clone()
            kf.C = self.C[idx].clone()
            kf.feat = self.feat[idx].clone()
            kf.pos = self.pos[idx].clone()
            kf.N = int(self.N[idx])
            kf.N_updates = int(self.N_updates[idx])
            if config["use_calib"]:
                kf.K = self.K.clone()

            # Retrieve segmentation data
            kf.raw_segmentation_mask = self.raw_segmentation_mask[idx].clone()
            kf.point_labels = self.point_labels[idx].clone()
            # kf.label_map will be set if it was part of the original Frame 'value' in __setitem__
            # For now, assume Frame object carries its own label_map.
            # If self.label_maps list was used: kf.label_map = self.label_maps[idx]
            return kf

    def __setitem__(self, idx, value: Frame) -> None:
        with self.lock:
            self.n_size.value = max(idx + 1, self.n_size.value)

            # set the attributes
            self.dataset_idx[idx] = value.frame_id
            self.img[idx] = value.img
            self.uimg[idx] = value.uimg
            self.img_shape[idx] = value.img_shape
            self.img_true_shape[idx] = value.img_true_shape
            self.T_WC[idx] = value.T_WC.data
            if value.X_canon is not None:
                self.X[idx] = value.X_canon
            if value.C is not None:
                self.C[idx] = value.C
            if value.feat is not None:
                self.feat[idx] = value.feat
            if value.pos is not None:
                self.pos[idx] = value.pos
            self.N[idx] = value.N
            self.N_updates[idx] = value.N_updates

            # Store segmentation data
            if value.raw_segmentation_mask is not None:
                h_mask, w_mask = value.raw_segmentation_mask.shape
                _b_sh, h_sh, w_sh = self.raw_segmentation_mask.shape # Buffer, H, W
                if h_mask == h_sh and w_mask == w_sh:
                    self.raw_segmentation_mask[idx] = value.raw_segmentation_mask
                else:
                    print(f"[Warning] Mismatch in raw_segmentation_mask dimensions for SharedKeyframes idx {idx}. Expected {h_sh}x{w_sh}, got {h_mask}x{w_mask}. Skipping update for this mask.")

            if value.point_labels is not None:
                self.point_labels[idx] = value.point_labels

            # value.label_map is part of the Frame object 'value'.
            # If self.label_maps list was used: self.label_maps[idx] = value.label_map

            self.is_dirty[idx] = True
            return idx

    def __len__(self):
        with self.lock:
            return self.n_size.value

    def append(self, value: Frame):
        with self.lock:
            self[self.n_size.value] = value

    def pop_last(self):
        with self.lock:
            self.n_size.value -= 1

    def last_keyframe(self) -> Optional[Frame]:
        with self.lock:
            if self.n_size.value == 0:
                return None
            return self[self.n_size.value - 1]

    def update_T_WCs(self, T_WCs, idx) -> None:
        with self.lock:
            self.T_WC[idx] = T_WCs.data

    def get_dirty_idx(self):
        with self.lock:
            idx = torch.where(self.is_dirty)[0]
            self.is_dirty[:] = False
            return idx

    def set_intrinsics(self, K):
        assert config["use_calib"]
        with self.lock:
            self.K[:] = K

    def get_intrinsics(self):
        assert config["use_calib"]
        with self.lock:
            return self.K
