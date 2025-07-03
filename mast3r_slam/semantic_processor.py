# mast3r_slam/semantic_processor.py
"""
Handles per-frame instance segmentation (SAM) and classification (OpenCLIP).
"""
import torch
import numpy as np
import cv2
from PIL import Image

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

INSTANCE_MODEL_CHECKPOINT_PATH = "checkpoints/sam_vit_b_01ec64.pth"
INSTANCE_MODEL_TYPE = "vit_b"

CLIP_MODEL_NAME = 'ViT-bigG-14'
CLIP_PRETRAINED_DATASET = 'laion2b_s39b_b160k'

TEXT_PROMPTS = [
    "background", "chair", "desk", "table", "monitor", "person", "plant",
    "cup", "book", "keyboard", "mouse", "laptop", "screen"
]
SEMANTIC_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_instance_segmentation_model(checkpoint_path: str = INSTANCE_MODEL_CHECKPOINT_PATH,
                                     model_type: str = INSTANCE_MODEL_TYPE,
                                     device: str = SEMANTIC_DEVICE):
    global _sam_mask_generator, SAM_AVAILABLE
    if not SAM_AVAILABLE:
        raise ImportError("segment_anything library is required for instance segmentation.")
    if _sam_mask_generator is not None:
        return _sam_mask_generator
    try:
        print(f"[INFO SemanticProcessor] Loading SAM model: type='{model_type}' from checkpoint='{checkpoint_path}' to device='{device}'")
        sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam_model.to(device=device)
        sam_model.eval()
        _sam_mask_generator = SamAutomaticMaskGenerator(
            model=sam_model, points_per_side=8, pred_iou_thresh=0.90,
            stability_score_thresh=0.96, min_mask_region_area=1000,
        )
        print(f"[INFO SemanticProcessor] SAM model and SamAutomaticMaskGenerator loaded (pps=8, min_area=1000).")
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
            if _clip_model is None: # Load model only if not already loaded
                 print(f"[INFO SemanticProcessor] Loading OpenCLIP model: '{model_name}' with weights '{pretrained_dataset}' to '{device}'.")
                 _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
                     model_name, pretrained=pretrained_dataset, device=device, jit=False # jit=False can sometimes help with large models if issues arise
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
    rows, cols = torch.any(binary_mask_hw, axis=1), torch.any(binary_mask_hw, axis=0)
    if not rows.any() or not cols.any(): return None
    ymin, ymax = torch.where(rows)[0][[0, -1]]
    xmin, xmax = torch.where(cols)[0][[0, -1]]
    cropped_tensor = image_chw_0_1_rgb[:, ymin:ymax+1, xmin:xmax+1]
    if cropped_tensor.numel() == 0 or cropped_tensor.shape[1] == 0 or cropped_tensor.shape[2] == 0: return None
    return Image.fromarray((cropped_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

# Removed individual classify_crop_with_clip function, logic will be batched in process_frame_for_semantics

def process_frame_for_semantics(image_tensor_chw_0_1_rgb: torch.Tensor,
                                text_prompts_for_clip: list = None):
    effective_prompts = text_prompts_for_clip if text_prompts_for_clip is not None else TEXT_PROMPTS
    sam_generator = load_instance_segmentation_model(device=SEMANTIC_DEVICE)
    clip_model, clip_preprocess, text_features_tensor, cached_prompts_list = load_clip_model(
        text_prompts=effective_prompts, device=SEMANTIC_DEVICE
    )
    if cached_prompts_list != effective_prompts: # Re-encode if prompts changed
        _, _, text_features_tensor, cached_prompts_list = load_clip_model(
            text_prompts=effective_prompts, device=SEMANTIC_DEVICE, force_reload_prompts=True
        )

    _c, h, w = image_tensor_chw_0_1_rgb.shape
    image_hwc_uint8_rgb = (image_tensor_chw_0_1_rgb.permute(1, 2, 0) * 255.0).byte().cpu().numpy()

    # print(f"[INFO SemanticProcessor] Running SAM on image {h}x{w}...")
    sam_masks_data_list = sam_generator.generate(image_hwc_uint8_rgb)

    local_instance_mask = torch.zeros((h,w), dtype=torch.int64, device=SEMANTIC_DEVICE)
    # Ensure the first prompt is used for background if available
    background_label = cached_prompts_list[0] if cached_prompts_list else "background"
    local_id_to_class_label_map = {0: background_label}

    if not sam_masks_data_list:
        print(f"[INFO SemanticProcessor] SAM found no masks.")
        return local_instance_mask, local_id_to_class_label_map

    sam_masks_data_list = sorted(sam_masks_data_list, key=lambda x: x['area'], reverse=True)

    # --- Batch CLIP Classification ---
    pil_crops_for_clip = []
    segment_info_for_classification = [] # Stores (local_id, original_mask_torch) for later assignment

    current_local_id_counter = 1 # Starts from 1 for actual objects

    for mask_data in sam_masks_data_list:
        segment_bool_np = mask_data['segmentation']
        segment_torch = torch.from_numpy(segment_bool_np.astype(bool)).to(device=SEMANTIC_DEVICE)

        # Check if this segment would overwrite existing non-background parts (it shouldn't due to sorting by area if logic is right)
        # For assigning local_id, we ensure a pixel gets only one ID.
        # The valid_pixels check here is to decide if this mask contributes to a *new* local_id.
        # If we assign IDs first and then crop, it's simpler.

        # We need to assign a temporary ID to all parts of this mask first to define the region for CLIP
        # This temporary ID is just for grouping pixels for one SAM mask before deciding its final local_id

        pil_crop = get_image_crop_from_mask(image_tensor_chw_0_1_rgb, segment_torch)
        if pil_crop:
            pil_crops_for_clip.append(pil_crop)
            # Store the segment_torch mask and the ID it will get if classified
            segment_info_for_classification.append({'mask_torch': segment_torch})
            # We will assign final local_ids *after* classification and filtering by background

    if not pil_crops_for_clip:
        print(f"[INFO SemanticProcessor] No valid crops obtained from SAM masks for CLIP.")
        return local_instance_mask, local_id_to_class_label_map

    # Preprocess all crops in a batch
    try:
        processed_image_inputs = torch.stack([clip_preprocess(crop) for crop in pil_crops_for_clip]).to(SEMANTIC_DEVICE)
    except Exception as e_preproc:
        print(f"[ERROR SemanticProcessor] Error during batch CLIP preprocessing: {e_preproc}")
        return local_instance_mask, local_id_to_class_label_map # Return empty/background mask

    # Get batched image features
    with torch.no_grad():
        batched_image_features = clip_model.encode_image(processed_image_inputs)
        batched_image_features /= batched_image_features.norm(dim=-1, keepdim=True)

    # Calculate similarities in a batch
    # similarity_matrix is (num_crops, num_text_prompts)
    similarity_matrix = (100.0 * batched_image_features @ text_features_tensor.T)
    best_scores, best_indices = similarity_matrix.max(dim=1)

    # Assign final local_instance_mask IDs and build map
    for i, seg_info in enumerate(segment_info_for_classification):
        class_label = cached_prompts_list[best_indices[i].item()]

        # Only assign a new local_id if the classification is not "background" (or first prompt)
        # AND if the area is not yet assigned in local_instance_mask by a larger, earlier segment.
        # The first prompt in TEXT_PROMPTS is assumed to be the "background" or "ignore" class.
        if class_label != background_label:
            segment_torch = seg_info['mask_torch']
            valid_pixels_for_current_id = segment_torch & (local_instance_mask == 0) # Pixels part of current segment AND currently background

            if valid_pixels_for_current_id.any():
                local_instance_mask[valid_pixels_for_current_id] = current_local_id_counter
                local_id_to_class_label_map[current_local_id_counter] = class_label
                current_local_id_counter += 1
                if current_local_id_counter > 255: # Max for uchar
                    print(f"[Warning SemanticProcessor] Reached local_id {current_local_id_counter-1}. Further distinct segments might be grouped if uchar is limit.")
                    break

    # print(f"[INFO SemanticProcessor] Frame processing complete. Final local mask unique IDs: {torch.unique(local_instance_mask)}. Label map size: {len(local_id_to_class_label_map)}")
    return local_instance_mask, local_id_to_class_label_map

if __name__ == '__main__':
    if not SAM_AVAILABLE or not OPEN_CLIP_AVAILABLE:
        print("Profile Test SKIPPED: segment_anything or open_clip_torch library not installed.")
    else:
        import cProfile, pstats, io
        print("Testing and Profiling Semantic Processor with SAM and OpenCLIP...")
        # ... (rest of the __main__ block is largely the same, ensure it uses the updated logic if needed) ...
        print(f"Device: {SEMANTIC_DEVICE}, SAM Checkpoint: {INSTANCE_MODEL_CHECKPOINT_PATH}, CLIP: {CLIP_MODEL_NAME}/{CLIP_PRETRAINED_DATASET}")
        print(f"Prompts: {TEXT_PROMPTS}")
        try:
            print("Pre-loading models for profiling...")
            load_instance_segmentation_model()
            load_clip_model(text_prompts=TEXT_PROMPTS)
            print("Models pre-loaded.")

            dummy_h, dummy_w = 240, 320
            img_np = np.zeros((dummy_h, dummy_w, 3), dtype=np.uint8)
            cv2.rectangle(img_np, (dummy_w//4, dummy_h//4), (dummy_w//2, dummy_h//2), (0,0,200), -1)
            cv2.circle(img_np, (3*dummy_w//4, 3*dummy_h//4), dummy_h//5, (0,200,0), -1)
            img_rgb_np_float = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_rgb_np_float).permute(2,0,1).to(SEMANTIC_DEVICE)

            print("\nProfiling process_frame_for_semantics (1 run)...")
            profiler = cProfile.Profile()
            profiler.enable()
            local_mask, id_to_label = process_frame_for_semantics(img_tensor, text_prompts_for_clip=TEXT_PROMPTS)
            profiler.disable()

            print("\n--- Test Run Output ---")
            print("Local Instance Mask shape:", local_mask.shape)
            print("Local Instance Mask unique IDs:", torch.unique(local_mask).cpu().tolist())
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
        except FileNotFoundError as e:
            print(f"[CRITICAL ERROR in __main__] Model checkpoint file not found: {e}.")
        except ImportError as e:
             print(f"[CRITICAL ERROR in __main__] Required library not found: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during test: {e}")
            import traceback
            traceback.print_exc()
        print("\nSemantic Processor __main__ test complete.")
