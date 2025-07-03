# mast3r_slam/semantic_processor.py
"""
Handles per-frame instance segmentation (SAM) and classification (OpenCLIP).
"""
import torch
import numpy as np
import cv2
from PIL import Image
import pathlib

# --- Attempt to import SAM specific modules ---
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    # print("[Warning SemanticProcessor] segment_anything library not found...") # Less verbose

# --- Attempt to import OpenCLIP specific modules ---
try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False
    # print("[Warning SemanticProcessor] open_clip_torch library not found...") # Less verbose

_sam_mask_generator = None
_clip_model = None
_clip_image_preprocess_val = None # To store the actual image preprocessing part of CLIP's preprocess
_clip_text_tokenizer = None # To store the tokenizer
_clip_text_features_tensor = None
_clip_text_prompts_cache = []

INSTANCE_MODEL_CHECKPOINT_PATH = "checkpoints/sam_vit_b_01ec64.pth"
INSTANCE_MODEL_TYPE = "vit_b"

CLIP_MODEL_NAME = 'ViT-bigG-14'
CLIP_PRETRAINED_DATASET = 'laion2b_s39b_b160k'

TEXT_PROMPTS = [
    "background", "chair", "desk", "table", "monitor", "person", "plant",
    "cup", "book", "keyboard", "mouse", "laptop", "screen"
]
SEMANTIC_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_SAM_POINTS_PER_SIDE = 16
DEFAULT_SAM_PRED_IOU_THRESH = 0.88
DEFAULT_SAM_STABILITY_SCORE_THRESH = 0.95
DEFAULT_SAM_MIN_MASK_REGION_AREA = 150

# --- CLIP Specifics ---
# We need to know the input resolution for CLIP to resize crops appropriately
# For ViT-bigG-14, it's typically 224, but OpenCLIP's preprocess handles it.
# We will extract the resize and normalize components from CLIP's preprocess.
_clip_input_resolution = (224, 224) # Default, will be updated from loaded model if possible


def load_instance_segmentation_model(checkpoint_path: str = INSTANCE_MODEL_CHECKPOINT_PATH,
                                     model_type: str = INSTANCE_MODEL_TYPE,
                                     device: str = SEMANTIC_DEVICE,
                                     points_per_side: int = DEFAULT_SAM_POINTS_PER_SIDE,
                                     pred_iou_thresh: float = DEFAULT_SAM_PRED_IOU_THRESH,
                                     stability_score_thresh: float = DEFAULT_SAM_STABILITY_SCORE_THRESH,
                                     min_mask_region_area: int = DEFAULT_SAM_MIN_MASK_REGION_AREA
                                     ):
    global _sam_mask_generator, SAM_AVAILABLE
    if not SAM_AVAILABLE:
        raise ImportError("segment_anything library is required for instance segmentation.")
    generator_params_match = False
    if _sam_mask_generator is not None:
        if all(hasattr(_sam_mask_generator, attr) for attr in ['_custom_pps', '_custom_iou', '_custom_stab', '_custom_mma']):
            if (_sam_mask_generator._custom_pps == points_per_side and
                _sam_mask_generator._custom_iou == pred_iou_thresh and
                _sam_mask_generator._custom_stab == stability_score_thresh and
                _sam_mask_generator._custom_mma == min_mask_region_area):
                generator_params_match = True
    if generator_params_match:
        return _sam_mask_generator
    try:
        # print(f"[INFO SemanticProcessor] Loading/Re-initializing SAM model: type='{model_type}' from checkpoint='{checkpoint_path}' to device='{device}'")
        sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam_model.to(device=device); sam_model.eval()
        _sam_mask_generator = SamAutomaticMaskGenerator(
            model=sam_model, points_per_side=points_per_side, pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh, min_mask_region_area=min_mask_region_area,
        )
        _sam_mask_generator._custom_pps = points_per_side; _sam_mask_generator._custom_iou = pred_iou_thresh
        _sam_mask_generator._custom_stab = stability_score_thresh; _sam_mask_generator._custom_mma = min_mask_region_area
        print(f"[INFO SemanticProcessor] SAM loaded (pps={points_per_side}, min_area={min_mask_region_area}).")
    except FileNotFoundError: _sam_mask_generator = None; print(f"[ERROR SP] SAM Ckpt not found: {checkpoint_path}"); raise
    except Exception as e: _sam_mask_generator = None; print(f"[ERROR SP] SAM load failed: {e}"); raise e
    return _sam_mask_generator

def load_clip_model(model_name: str = CLIP_MODEL_NAME, pretrained_dataset: str = CLIP_PRETRAINED_DATASET,
                    text_prompts: list = None, device: str = SEMANTIC_DEVICE, force_reload_prompts: bool = False):
    global _clip_model, _clip_image_preprocess_val, _clip_text_tokenizer, _clip_text_features_tensor, \
           _clip_text_prompts_cache, OPEN_CLIP_AVAILABLE, TEXT_PROMPTS, _clip_input_resolution
    if not OPEN_CLIP_AVAILABLE:
        raise ImportError("open_clip_torch library is required for CLIP.")

    current_prompts = text_prompts if text_prompts is not None else TEXT_PROMPTS
    prompts_cached_and_match = (_clip_model is not None and _clip_image_preprocess_val is not None and
                               _clip_text_features_tensor is not None and _clip_text_prompts_cache == current_prompts)
    if prompts_cached_and_match and not force_reload_prompts:
        return _clip_model, _clip_image_preprocess_val, _clip_text_features_tensor, _clip_text_prompts_cache

    try:
        if _clip_model is None or _clip_image_preprocess_val is None:
            print(f"[INFO SemanticProcessor] Loading OpenCLIP model: '{model_name}' with '{pretrained_dataset}' to '{device}'.")
            _clip_model, _, _clip_image_preprocess_val = open_clip.create_model_and_transforms( # _ is train_preprocess
                model_name, pretrained=pretrained_dataset, device=device, jit=False
            )
            _clip_model.eval()
            # Try to get input resolution from the model's visual config if available
            if hasattr(_clip_model, 'visual') and hasattr(_clip_model.visual, 'image_size'):
                 res = _clip_model.visual.image_size
                 if isinstance(res, tuple): _clip_input_resolution = res
                 elif isinstance(res, int): _clip_input_resolution = (res, res)

        if force_reload_prompts or _clip_text_features_tensor is None or _clip_text_prompts_cache != current_prompts:
            print(f"[INFO SemanticProcessor] Tokenizing {len(current_prompts)} prompts for OpenCLIP...")
            _clip_text_tokenizer = open_clip.get_tokenizer(model_name) # Get tokenizer associated with model
            text_tokens = _clip_text_tokenizer(current_prompts).to(device)
            with torch.no_grad():
                _clip_text_features_tensor = _clip_model.encode_text(text_tokens)
                _clip_text_features_tensor /= _clip_text_features_tensor.norm(dim=-1, keepdim=True)
            _clip_text_prompts_cache = list(current_prompts)
            print(f"[INFO SemanticProcessor] OpenCLIP text features loaded/updated.")

        print(f"[INFO SemanticProcessor] OpenCLIP model ready. Input resolution: {_clip_input_resolution}")
    except Exception as e:
        print(f"[ERROR SemanticProcessor] Failed to load OpenCLIP: {e}")
        _clip_model, _clip_image_preprocess_val, _clip_text_features_tensor, _clip_text_prompts_cache = None, None, None, []
        raise e
    return _clip_model, _clip_image_preprocess_val, _clip_text_features_tensor, _clip_text_prompts_cache

def get_tensor_crop_from_mask(image_chw_0_1_rgb: torch.Tensor, binary_mask_hw: torch.Tensor,
                               output_size: tuple = None): # output_size (H,W) for direct resize
    if not binary_mask_hw.any(): return None
    rows, cols = torch.any(binary_mask_hw, axis=1), torch.any(binary_mask_hw, axis=0)
    if not rows.any() or not cols.any(): return None

    ymin, ymax = torch.where(rows)[0][[0, -1]]
    xmin, xmax = torch.where(cols)[0][[0, -1]]

    pad = 0 # Keep padding minimal for now, resize will handle final size
    h, w = image_chw_0_1_rgb.shape[1], image_chw_0_1_rgb.shape[2]
    ymin_pad = max(0, ymin - pad); ymax_pad = min(h - 1, ymax + pad)
    xmin_pad = max(0, xmin - pad); xmax_pad = min(w - 1, xmax + pad)

    cropped_tensor = image_chw_0_1_rgb[:, ymin_pad:ymax_pad+1, xmin_pad:xmax_pad+1]

    if cropped_tensor.numel() == 0 or cropped_tensor.shape[1] == 0 or cropped_tensor.shape[2] == 0: return None

    # Apply mask to make background transparent (or black) before resize, helps CLIP focus
    # Create a 3-channel version of the binary mask for element-wise multiplication
    # binary_mask_hw is H_crop x W_crop after slicing if we use it directly on cropped_tensor
    # It's easier to crop the binary mask too:
    cropped_binary_mask = binary_mask_hw[ymin_pad:ymax_pad+1, xmin_pad:xmax_pad+1].unsqueeze(0) # 1, H_c, W_c
    # Mask out non-object pixels - set to 0 (black).
    # cropped_tensor = cropped_tensor * cropped_binary_mask


    if output_size: # Resize if an output_size is specified (e.g., CLIP input size)
        # Interpolate expects BCHW, input is CHW. Add batch dim, then remove.
        cropped_tensor = torch.nn.functional.interpolate(
            cropped_tensor.unsqueeze(0),
            size=output_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
    return cropped_tensor # Still in 0-1 range, RGB

def process_frame_for_semantics(image_tensor_chw_0_1_rgb: torch.Tensor,
                                text_prompts_for_clip: list = None):
    global _clip_input_resolution # Use the resolution determined at model load
    effective_prompts = text_prompts_for_clip if text_prompts_for_clip is not None else TEXT_PROMPTS

    sam_generator = load_instance_segmentation_model(device=SEMANTIC_DEVICE) # Uses defaults or last set params
    clip_model, clip_img_val_preprocess, text_features_tensor, cached_prompts_list = load_clip_model(
        text_prompts=effective_prompts, device=SEMANTIC_DEVICE
    )
    if cached_prompts_list != effective_prompts:
        _, clip_img_val_preprocess, text_features_tensor, cached_prompts_list = load_clip_model(
            text_prompts=effective_prompts, device=SEMANTIC_DEVICE, force_reload_prompts=True
        )

    _c, h, w = image_tensor_chw_0_1_rgb.shape
    image_hwc_uint8_rgb = (image_tensor_chw_0_1_rgb.permute(1, 2, 0) * 255.0).byte().cpu().numpy()
    sam_masks_data_list = sam_generator.generate(image_hwc_uint8_rgb)

    local_instance_mask = torch.zeros((h,w), dtype=torch.int64, device=SEMANTIC_DEVICE)
    background_label = cached_prompts_list[0] if cached_prompts_list else "background"
    local_id_to_class_label_map = {0: background_label}

    if not sam_masks_data_list:
        # print(f"[INFO SemanticProcessor] SAM found no masks.")
        return local_instance_mask, local_id_to_class_label_map

    sam_masks_data_list = sorted(sam_masks_data_list, key=lambda x: x['area'], reverse=True)

    tensor_crops_for_clip = []
    segment_info_for_classification = []

    for mask_data in sam_masks_data_list:
        segment_torch_bool = torch.from_numpy(mask_data['segmentation'].astype(bool)).to(device=SEMANTIC_DEVICE)
        # Get tensor crop, resized to CLIP's expected input size
        tensor_crop = get_tensor_crop_from_mask(image_tensor_chw_0_1_rgb, segment_torch_bool, output_size=_clip_input_resolution)
        if tensor_crop is not None:
            tensor_crops_for_clip.append(tensor_crop)
            segment_info_for_classification.append({'mask_torch': segment_torch_bool})

    if not tensor_crops_for_clip:
        # print(f"[INFO SemanticProcessor] No valid tensor crops from SAM masks for CLIP.")
        return local_instance_mask, local_id_to_class_label_map

    # Batch preprocess (normalization mainly, as resize is done, and ToTensor is implicit)
    # The _clip_image_preprocess_val is a torchvision.transforms.Compose.
    # It typically includes Resize, CenterCrop, ToTensor, Normalize.
    # Since we've resized and have tensors, we mainly need to replicate Normalize.
    # Let's inspect clip_img_val_preprocess.transforms to find Normalize
    normalize_transform = None
    if hasattr(clip_img_val_preprocess, 'transforms'):
        for t in clip_img_val_preprocess.transforms:
            if isinstance(t, torch.nn.modules.module.Module) and "Normalize" in t.__class__.__name__: # More robust check
                normalize_transform = t
                break
            elif isinstance(t, Image.Image) and hasattr(t, 'mean') and hasattr(t, 'std') : # For older torchvision Normalize
                normalize_transform = t # it is the Normalize object itself
                break


    if not normalize_transform:
        print("[Warning SemanticProcessor] Could not find Normalize transform in CLIP preprocess. Using default imagenet stats.")
        # Fallback to default ImageNet normalization if not found
        normalize_transform = torch.nn.Sequential( # Dummy sequential to have a .mean and .std
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ).to(SEMANTIC_DEVICE)


    try:
        # Stack crops into a batch. They are already C,H_clip,W_clip and 0-1 range.
        batched_image_tensors = torch.stack(tensor_crops_for_clip) # B, C, H_clip, W_clip
        # Apply normalization
        normalized_batched_images = normalize_transform(batched_image_tensors)

    except Exception as e_preproc:
        print(f"[ERROR SemanticProcessor] Batch CLIP tensor preprocessing failed: {e_preproc}")
        return local_instance_mask, local_id_to_class_label_map

    with torch.no_grad():
        batched_image_features = clip_model.encode_image(normalized_batched_images)
        batched_image_features /= batched_image_features.norm(dim=-1, keepdim=True)

    similarity_matrix = (100.0 * batched_image_features @ text_features_tensor.T)
    best_scores, best_indices = similarity_matrix.max(dim=1)

    current_local_id_counter = 1
    for i, seg_info in enumerate(segment_info_for_classification):
        class_label = cached_prompts_list[best_indices[i].item()]
        if class_label != background_label: # Only assign if not classified as background
            segment_torch = seg_info['mask_torch']
            valid_pixels_for_current_id = segment_torch & (local_instance_mask == 0)
            if valid_pixels_for_current_id.any():
                local_instance_mask[valid_pixels_for_current_id] = current_local_id_counter
                local_id_to_class_label_map[current_local_id_counter] = class_label
                current_local_id_counter += 1
                if current_local_id_counter > 255: break
    return local_instance_mask, local_id_to_class_label_map

if __name__ == '__main__':
    if not SAM_AVAILABLE or not OPEN_CLIP_AVAILABLE:
        print("Profile Test SKIPPED: segment_anything or open_clip_torch library not installed.")
    else:
        import cProfile, pstats, io, torchvision # torchvision for fallback Normalize
        print("Testing and Profiling Semantic Processor with SAM and OpenCLIP...")
        print(f"Device: {SEMANTIC_DEVICE}, SAM Checkpoint: {INSTANCE_MODEL_CHECKPOINT_PATH}, CLIP: {CLIP_MODEL_NAME}/{CLIP_PRETRAINED_DATASET}")
        print(f"Using SAM parameters for test: pps={DEFAULT_SAM_POINTS_PER_SIDE}, min_area={DEFAULT_SAM_MIN_MASK_REGION_AREA}")
        print(f"Prompts: {TEXT_PROMPTS}")
        try:
            print("Pre-loading models for profiling...")
            load_instance_segmentation_model( # Load with defaults for this test
                points_per_side=DEFAULT_SAM_POINTS_PER_SIDE, min_mask_region_area=DEFAULT_SAM_MIN_MASK_REGION_AREA,
                pred_iou_thresh=DEFAULT_SAM_PRED_IOU_THRESH, stability_score_thresh=DEFAULT_SAM_STABILITY_SCORE_THRESH
            )
            load_clip_model(text_prompts=TEXT_PROMPTS)
            print("Models pre-loaded.")

            img_to_load = "test_image_for_profiling.png"
            img_rgb_np = None
            try:
                current_script_path = pathlib.Path(__file__).resolve()
                project_root = current_script_path.parents[1]
                img_path = project_root / img_to_load
                print(f"Attempting to load test image from: {img_path}")
                img_bgr_np = cv2.imread(str(img_path))
                if img_bgr_np is None: raise FileNotFoundError(f"Could not read: {img_path}")
                img_rgb_np = cv2.cvtColor(img_bgr_np, cv2.COLOR_BGR2RGB)
                test_h, test_w, _ = img_rgb_np.shape
                print(f"Loaded test image '{img_to_load}' of size {test_h}x{test_w}")
            except Exception as e_img_load:
                print(f"Failed to load '{img_to_load}': {e_img_load}. Using BLACK dummy image.")
                test_h, test_w = 240, 320
                img_rgb_np = np.zeros((test_h, test_w, 3), dtype=np.uint8)

            img_rgb_np_float = img_rgb_np.astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_rgb_np_float).permute(2,0,1).to(SEMANTIC_DEVICE)

            print("\nProfiling process_frame_for_semantics (1 run)...")
            profiler = cProfile.Profile()
            profiler.enable()
            local_mask, id_to_label = process_frame_for_semantics(img_tensor, text_prompts_for_clip=TEXT_PROMPTS)
            profiler.disable()

            print("\n--- Test Run Output ---")
            print("Local Instance Mask shape:", local_mask.shape)
            unique_ids = torch.unique(local_mask).cpu().tolist()
            print("Local Instance Mask unique IDs:", unique_ids)
            print(f"Number of unique non-background segments classified: {len([uid for uid in unique_ids if uid != 0])}")
            print("Local ID to Class Label Map:", id_to_label)

            s = io.StringIO(); ps = pstats.Stats(profiler, stream=s).sort_stats('tottime')
            print("\n--- Top 20 functions by self time (tottime) ---"); ps.print_stats(20); print(s.getvalue())
            s = io.StringIO(); ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            print("\n--- Top 20 functions by cumulative time ---"); ps.print_stats(20); print(s.getvalue())

            if not id_to_label or (len(id_to_label) == 1 and 0 in id_to_label and id_to_label[0]==TEXT_PROMPTS[0]):
                 print("WARNING: id_to_label map effectively empty or only contains background!")
        except FileNotFoundError as e: print(f"[CRITICAL ERROR in __main__] Model checkpoint not found: {e}.")
        except ImportError as e: print(f"[CRITICAL ERROR in __main__] Required library not found: {e}")
        except Exception as e: print(f"An unexpected error during test: {e}"); import traceback; traceback.print_exc()
        print("\nSemantic Processor __main__ test complete.")
