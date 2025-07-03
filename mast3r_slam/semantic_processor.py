# mast3r_slam/semantic_processor.py
"""
Handles per-frame instance segmentation (SAM) and classification (OpenCLIP).
"""
import torch
import numpy as np
import cv2
from PIL import Image
import pathlib # Added for robust path handling

# --- Attempt to import SAM specific modules ---
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("[Warning SemanticProcessor] segment_anything library not found. SAM functionalities will not be available.")
    print("[Warning SemanticProcessor] Please install it via: pip install git+https://github.com/facebookresearch/segment-anything.git")

# --- Attempt to import OpenCLIP specific modules ---
try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False
    print("[Warning SemanticProcessor] open_clip_torch library not found. CLIP functionalities will not be available.")
    print("[Warning SemanticProcessor] Please install it via: pip install open_clip_torch")

_sam_mask_generator = None
_clip_model = None
_clip_preprocess = None
_clip_text_features_tensor = None
_clip_text_prompts_cache = []

# --- Configuration ---
# TODO: USER - Update this path to your downloaded SAM checkpoint
INSTANCE_MODEL_CHECKPOINT_PATH = "checkpoints/sam_vit_b_01ec64.pth"
INSTANCE_MODEL_TYPE = "vit_b" # "vit_b", "vit_l", "vit_h"

# CLIP Configuration (using OpenCLIP)
CLIP_MODEL_NAME = 'ViT-bigG-14'
CLIP_PRETRAINED_DATASET = 'laion2b_s39b_b160k'

# TODO: USER - Define your text prompts for CLIP classification
TEXT_PROMPTS = [
    "background", "chair", "desk", "table", "monitor", "person", "plant",
    "cup", "book", "keyboard", "mouse", "laptop", "screen"
]
SEMANTIC_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Default SAM Parameters (can be overridden in load_instance_segmentation_model) ---
# These are the less aggressive parameters we want for testing with more segments
DEFAULT_SAM_POINTS_PER_SIDE = 16
DEFAULT_SAM_PRED_IOU_THRESH = 0.88
DEFAULT_SAM_STABILITY_SCORE_THRESH = 0.95
DEFAULT_SAM_MIN_MASK_REGION_AREA = 150


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

    # Check if re-initialization is needed based on parameters
    # Store current params on the generator object itself if it exists
    generator_params_match = False
    if _sam_mask_generator is not None:
        if all(hasattr(_sam_mask_generator, attr) for attr in ['_custom_pps', '_custom_iou', '_custom_stab', '_custom_mma']):
            if (_sam_mask_generator._custom_pps == points_per_side and
                _sam_mask_generator._custom_iou == pred_iou_thresh and
                _sam_mask_generator._custom_stab == stability_score_thresh and
                _sam_mask_generator._custom_mma == min_mask_region_area):
                generator_params_match = True

    if generator_params_match:
        # print(f"[INFO SemanticProcessor] SAM Automatic Mask Generator already loaded with matching params (pps={points_per_side}, min_area={min_mask_region_area}).")
        return _sam_mask_generator

    try:
        print(f"[INFO SemanticProcessor] Loading/Re-initializing SAM model: type='{model_type}' from checkpoint='{checkpoint_path}' to device='{device}'")
        sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam_model.to(device=device)
        sam_model.eval()

        _sam_mask_generator = SamAutomaticMaskGenerator(
            model=sam_model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_mask_region_area,
        )
        # Store current parameters for future checks
        _sam_mask_generator._custom_pps = points_per_side
        _sam_mask_generator._custom_iou = pred_iou_thresh
        _sam_mask_generator._custom_stab = stability_score_thresh
        _sam_mask_generator._custom_mma = min_mask_region_area

        print(f"[INFO SemanticProcessor] SAM model and SamAutomaticMaskGenerator loaded/re-initialized (pps={points_per_side}, min_area={min_mask_region_area}).")
    except FileNotFoundError:
        print(f"[ERROR SemanticProcessor] SAM Checkpoint file not found at: {checkpoint_path}")
        _sam_mask_generator = None; raise
    except Exception as e:
        print(f"[ERROR SemanticProcessor] Failed to load SAM model or init generator: {e}")
        _sam_mask_generator = None; raise e
    return _sam_mask_generator

def load_clip_model(model_name: str = CLIP_MODEL_NAME,
                    pretrained_dataset: str = CLIP_PRETRAINED_DATASET,
                    text_prompts: list = None,
                    device: str = SEMANTIC_DEVICE,
                    force_reload_prompts: bool = False):
    global _clip_model, _clip_preprocess, _clip_text_features_tensor, _clip_text_prompts_cache, OPEN_CLIP_AVAILABLE, TEXT_PROMPTS
    if not OPEN_CLIP_AVAILABLE:
        raise ImportError("open_clip_torch library is required for CLIP classification.")
    current_prompts = text_prompts if text_prompts is not None else TEXT_PROMPTS
    prompts_cached_and_match = (_clip_model is not None and _clip_preprocess is not None and
                               _clip_text_features_tensor is not None and _clip_text_prompts_cache == current_prompts)
    if prompts_cached_and_match and not force_reload_prompts:
        return _clip_model, _clip_preprocess, _clip_text_features_tensor, _clip_text_prompts_cache
    try:
        if _clip_model is None or force_reload_prompts or _clip_text_prompts_cache != current_prompts :
            if _clip_model is None:
                 print(f"[INFO SemanticProcessor] Loading OpenCLIP model: '{model_name}' with weights '{pretrained_dataset}' to '{device}'.")
                 _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
                     model_name, pretrained=pretrained_dataset, device=device, jit=False
                 )
                 _clip_model.eval()

            print(f"[INFO SemanticProcessor] Tokenizing {len(current_prompts)} text prompts for OpenCLIP...")
            text_tokens = open_clip.tokenize(current_prompts).to(device)
            with torch.no_grad():
                _clip_text_features_tensor = _clip_model.encode_text(text_tokens)
                _clip_text_features_tensor /= _clip_text_features_tensor.norm(dim=-1, keepdim=True)
            _clip_text_prompts_cache = list(current_prompts)
            print(f"[INFO SemanticProcessor] OpenCLIP model and text features loaded/updated successfully.")
    except Exception as e:
        print(f"[ERROR SemanticProcessor] Failed to load OpenCLIP model or process text prompts: {e}")
        _clip_model, _clip_preprocess, _clip_text_features_tensor, _clip_text_prompts_cache = None, None, None, []
        raise e
    return _clip_model, _clip_preprocess, _clip_text_features_tensor, _clip_text_prompts_cache

def get_image_crop_from_mask(image_chw_0_1_rgb: torch.Tensor, binary_mask_hw: torch.Tensor):
    if not binary_mask_hw.any(): return None
    rows = torch.any(binary_mask_hw, axis=1)
    cols = torch.any(binary_mask_hw, axis=0)
    if not rows.any() or not cols.any(): return None # Should be caught by binary_mask_hw.any()

    ymin, ymax = torch.where(rows)[0][[0, -1]]
    xmin, xmax = torch.where(cols)[0][[0, -1]]

    # Add a small padding to bounding box, clamped to image dimensions
    pad = 5
    h, w = image_chw_0_1_rgb.shape[1], image_chw_0_1_rgb.shape[2]
    ymin_pad = max(0, ymin - pad)
    ymax_pad = min(h - 1, ymax + pad)
    xmin_pad = max(0, xmin - pad)
    xmax_pad = min(w - 1, xmax + pad)

    cropped_tensor = image_chw_0_1_rgb[:, ymin_pad:ymax_pad+1, xmin_pad:xmax_pad+1]

    if cropped_tensor.numel() == 0 or cropped_tensor.shape[1] == 0 or cropped_tensor.shape[2] == 0: return None
    return Image.fromarray((cropped_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

def process_frame_for_semantics(image_tensor_chw_0_1_rgb: torch.Tensor,
                                text_prompts_for_clip: list = None):
    effective_prompts = text_prompts_for_clip if text_prompts_for_clip is not None else TEXT_PROMPTS

    # Load SAM with default (less aggressive) parameters defined at the top of the file
    sam_generator = load_instance_segmentation_model(
        points_per_side=DEFAULT_SAM_POINTS_PER_SIDE,
        min_mask_region_area=DEFAULT_SAM_MIN_MASK_REGION_AREA,
        pred_iou_thresh=DEFAULT_SAM_PRED_IOU_THRESH,
        stability_score_thresh=DEFAULT_SAM_STABILITY_SCORE_THRESH,
        device=SEMANTIC_DEVICE
    )
    clip_model, clip_preprocess, text_features_tensor, cached_prompts_list = load_clip_model(
        text_prompts=effective_prompts, device=SEMANTIC_DEVICE
    )
    if cached_prompts_list != effective_prompts:
        _, _, text_features_tensor, cached_prompts_list = load_clip_model(
            text_prompts=effective_prompts, device=SEMANTIC_DEVICE, force_reload_prompts=True
        )

    _c, h, w = image_tensor_chw_0_1_rgb.shape
    image_hwc_uint8_rgb = (image_tensor_chw_0_1_rgb.permute(1, 2, 0) * 255.0).byte().cpu().numpy()
    sam_masks_data_list = sam_generator.generate(image_hwc_uint8_rgb)

    local_instance_mask = torch.zeros((h,w), dtype=torch.int64, device=SEMANTIC_DEVICE)
    background_label = cached_prompts_list[0] if cached_prompts_list else "background"
    local_id_to_class_label_map = {0: background_label}

    if not sam_masks_data_list:
        print(f"[INFO SemanticProcessor] SAM found no masks.")
        return local_instance_mask, local_id_to_class_label_map

    sam_masks_data_list = sorted(sam_masks_data_list, key=lambda x: x['area'], reverse=True)
    pil_crops_for_clip = []
    segment_info_for_classification = []

    for mask_data in sam_masks_data_list:
        segment_torch = torch.from_numpy(mask_data['segmentation'].astype(bool)).to(device=SEMANTIC_DEVICE)
        pil_crop = get_image_crop_from_mask(image_tensor_chw_0_1_rgb, segment_torch)
        if pil_crop:
            pil_crops_for_clip.append(pil_crop)
            segment_info_for_classification.append({'mask_torch': segment_torch})

    if not pil_crops_for_clip:
        print(f"[INFO SemanticProcessor] No valid crops from SAM masks for CLIP.")
        return local_instance_mask, local_id_to_class_label_map

    try:
        processed_image_inputs = torch.stack([clip_preprocess(crop) for crop in pil_crops_for_clip]).to(SEMANTIC_DEVICE)
    except Exception as e_preproc:
        print(f"[ERROR SemanticProcessor] Batch CLIP preprocessing failed: {e_preproc}")
        return local_instance_mask, local_id_to_class_label_map

    with torch.no_grad():
        batched_image_features = clip_model.encode_image(processed_image_inputs)
        batched_image_features /= batched_image_features.norm(dim=-1, keepdim=True)
    similarity_matrix = (100.0 * batched_image_features @ text_features_tensor.T)
    best_scores, best_indices = similarity_matrix.max(dim=1)

    current_local_id_counter = 1
    for i, seg_info in enumerate(segment_info_for_classification):
        class_label = cached_prompts_list[best_indices[i].item()]
        if class_label != background_label:
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
        import cProfile, pstats, io
        print("Testing and Profiling Semantic Processor with SAM and OpenCLIP...")
        print(f"Device: {SEMANTIC_DEVICE}, SAM Checkpoint: {INSTANCE_MODEL_CHECKPOINT_PATH}, CLIP: {CLIP_MODEL_NAME}/{CLIP_PRETRAINED_DATASET}")
        print(f"Using SAM parameters for test: pps={DEFAULT_SAM_POINTS_PER_SIDE}, min_area={DEFAULT_SAM_MIN_MASK_REGION_AREA}")
        print(f"Prompts: {TEXT_PROMPTS}")

        try:
            print("Pre-loading models for profiling...")
            # Ensure SAM is loaded with test parameters by passing them explicitly
            load_instance_segmentation_model(
                points_per_side=DEFAULT_SAM_POINTS_PER_SIDE, # Use defaults defined at top for test
                min_mask_region_area=DEFAULT_SAM_MIN_MASK_REGION_AREA,
                pred_iou_thresh=DEFAULT_SAM_PRED_IOU_THRESH,
                stability_score_thresh=DEFAULT_SAM_STABILITY_SCORE_THRESH
            )
            load_clip_model(text_prompts=TEXT_PROMPTS)
            print("Models pre-loaded.")

            # --- Load a real image for testing ---
            img_to_load = "test_image_for_profiling.png" # Make sure this image exists
            img_rgb_np = None
            try:
                # Assuming script is run from MASt3R-SLAM root
                current_script_path = pathlib.Path(__file__).resolve()
                project_root = current_script_path.parents[1] # MASt3R-SLAM/mast3r_slam/ -> MASt3R-SLAM/
                img_path = project_root / img_to_load

                print(f"Attempting to load test image from: {img_path}")
                img_bgr_np = cv2.imread(str(img_path))
                if img_bgr_np is None:
                    raise FileNotFoundError(f"Could not read test image at {img_path}")
                img_rgb_np = cv2.cvtColor(img_bgr_np, cv2.COLOR_BGR2RGB)
                dummy_h, dummy_w, _ = img_rgb_np.shape
                print(f"Loaded test image '{img_to_load}' of size {dummy_h}x{dummy_w}")
            except Exception as e_img_load:
                print(f"Failed to load '{img_to_load}': {e_img_load}. Using simple BLACK dummy image instead.")
                dummy_h, dummy_w = 240, 320
                img_rgb_np = np.zeros((dummy_h, dummy_w, 3), dtype=np.uint8) # Black image
            # --- End Load Image ---

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
            print(f"Number of unique non-background segments found: {len([uid for uid in unique_ids if uid != 0])}")
            print("Local ID to Class Label Map:", id_to_label)

            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('tottime')
            print("\n--- Top 20 functions by self time (tottime) ---")
            ps.print_stats(20)
            print(s.getvalue())

            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            print("\n--- Top 20 functions by cumulative time ---")
            ps.print_stats(20)
            print(s.getvalue())

            if not id_to_label or (len(id_to_label) == 1 and 0 in id_to_label and id_to_label[0]==TEXT_PROMPTS[0]):
                 print("WARNING: id_to_label map effectively empty or only contains background!")

        except FileNotFoundError as e: # Specifically for model checkpoints
            print(f"[CRITICAL ERROR in __main__] Model checkpoint file not found: {e}. Please check INSTANCE_MODEL_CHECKPOINT_PATH in semantic_processor.py.")
        except ImportError as e: # For missing libraries
             print(f"[CRITICAL ERROR in __main__] Required library not found: {e}")
        except Exception as e: # For any other errors during test
            print(f"An unexpected error occurred during test: {e}")
            import traceback
            traceback.print_exc()
        print("\nSemantic Processor __main__ test complete.")
