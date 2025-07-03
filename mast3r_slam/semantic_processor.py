# mast3r_slam/semantic_processor.py
"""
Handles per-frame instance segmentation (SAM) and classification (OpenCLIP).
"""
import torch
import numpy as np
import cv2
from PIL import Image # For CLIP preprocessing

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


# --- Models and Preprocessors (will be loaded by functions) ---
_sam_mask_generator = None
_clip_model = None
_clip_preprocess = None
_clip_text_features_tensor = None
_clip_text_prompts_cache = [] # To store the prompts for which features were calculated

# --- Configuration ---
# TODO: USER - Update this path to your downloaded SAM checkpoint
INSTANCE_MODEL_CHECKPOINT_PATH = "checkpoints/sam_vit_b_01ec64.pth"
INSTANCE_MODEL_TYPE = "vit_b" # "vit_b", "vit_l", "vit_h"

# CLIP Configuration (using OpenCLIP)
CLIP_MODEL_NAME = 'ViT-bigG-14'
CLIP_PRETRAINED_DATASET = 'laion2b_s39b_b160k'

# TODO: USER - Define your text prompts for CLIP classification
TEXT_PROMPTS = [
    "background",
    "chair", "desk", "table", "monitor", "person", "plant",
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
            model=sam_model,
            points_per_side=8,         # Reduced for speed
            pred_iou_thresh=0.90,      # More stringent
            stability_score_thresh=0.96, # More stringent
            min_mask_region_area=1000, # Significantly increased to reduce small segments
        )
        print(f"[INFO SemanticProcessor] SAM model and SamAutomaticMaskGenerator loaded (pps=8, min_area=1000).")
    except FileNotFoundError:
        print(f"[ERROR SemanticProcessor] SAM Checkpoint file not found at: {checkpoint_path}")
        _sam_mask_generator = None
        raise
    except Exception as e:
        print(f"[ERROR SemanticProcessor] Failed to load SAM model or initialize generator: {e}")
        _sam_mask_generator = None
        raise e
    return _sam_mask_generator


def load_clip_model(model_name: str = CLIP_MODEL_NAME,
                    pretrained_dataset: str = CLIP_PRETRAINED_DATASET,
                    text_prompts: list = None,
                    device: str = SEMANTIC_DEVICE,
                    force_reload_prompts: bool = False): # Added force_reload_prompts
    global _clip_model, _clip_preprocess, _clip_text_features_tensor, _clip_text_prompts_cache, OPEN_CLIP_AVAILABLE, TEXT_PROMPTS
    if not OPEN_CLIP_AVAILABLE:
        raise ImportError("open_clip_torch library is required for CLIP classification.")

    current_prompts = text_prompts if text_prompts is not None else TEXT_PROMPTS
    prompts_are_cached_and_match = (_clip_model is not None and
                                   _clip_preprocess is not None and
                                   _clip_text_features_tensor is not None and
                                   _clip_text_prompts_cache == current_prompts)

    if prompts_are_cached_and_match and not force_reload_prompts:
        return _clip_model, _clip_preprocess, _clip_text_features_tensor, _clip_text_prompts_cache

    try:
        if _clip_model is None or force_reload_prompts: # Load model only if not loaded or prompts changed significantly
            print(f"[INFO SemanticProcessor] Loading OpenCLIP model: '{model_name}' with weights '{pretrained_dataset}' to '{device}'.")
            _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained_dataset, device=device
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

def classify_crop_with_clip(pil_image_crop: Image.Image, clip_model, clip_preprocess,
                            text_features_tensor: torch.Tensor, text_prompts_list: list,
                            device: str = SEMANTIC_DEVICE):
    if pil_image_crop is None: return text_prompts_list[0] if text_prompts_list else "unknown"
    try:
        image_input = clip_preprocess(pil_image_crop).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features_tensor.T)
        best_idx = similarity[0].argmax(dim=0)
        class_label = text_prompts_list[best_idx.item()]
    except Exception as e:
        print(f"[ERROR SemanticProcessor] CLIP classification failed: {e}")
        class_label = text_prompts_list[0] if text_prompts_list else "classification_failed"
    return class_label

def process_frame_for_semantics(image_tensor_chw_0_1_rgb: torch.Tensor,
                                text_prompts_for_clip: list = None):
    effective_prompts = text_prompts_for_clip if text_prompts_for_clip is not None else TEXT_PROMPTS
    sam_generator = load_instance_segmentation_model(device=SEMANTIC_DEVICE)
    clip_model, clip_preprocess, text_features, cached_prompts = load_clip_model(
        text_prompts=effective_prompts, device=SEMANTIC_DEVICE
    )
    if cached_prompts != effective_prompts: # Re-encode if prompts changed since last load
        _, _, text_features, cached_prompts = load_clip_model(
            text_prompts=effective_prompts, device=SEMANTIC_DEVICE, force_reload_prompts=True
        )

    _c, h, w = image_tensor_chw_0_1_rgb.shape
    image_hwc_uint8_rgb = (image_tensor_chw_0_1_rgb.permute(1, 2, 0) * 255.0).byte().cpu().numpy()
    sam_masks_data_list = sam_generator.generate(image_hwc_uint8_rgb)

    local_instance_mask = torch.zeros((h,w), dtype=torch.int64, device=SEMANTIC_DEVICE)
    local_id_to_class_label_map = {0: cached_prompts[0] if cached_prompts else "background"}

    if not sam_masks_data_list:
        print(f"[INFO SemanticProcessor] SAM found no masks.")
        return local_instance_mask, local_id_to_class_label_map

    sam_masks_data_list = sorted(sam_masks_data_list, key=lambda x: x['area'], reverse=True)
    current_local_id = 1
    for mask_data in sam_masks_data_list:
        segment_torch = torch.from_numpy(mask_data['segmentation'].astype(bool)).to(device=SEMANTIC_DEVICE)
        valid_pixels = segment_torch & (local_instance_mask == 0)
        if valid_pixels.any():
            local_instance_mask[valid_pixels] = current_local_id
            pil_crop = get_image_crop_from_mask(image_tensor_chw_0_1_rgb, segment_torch)
            class_label = classify_crop_with_clip(pil_crop, clip_model, clip_preprocess,
                                                  text_features, cached_prompts, device=SEMANTIC_DEVICE)
            local_id_to_class_label_map[current_local_id] = class_label
            current_local_id += 1
            if current_local_id > 255: break
    return local_instance_mask, local_id_to_class_label_map

if __name__ == '__main__':
    if not SAM_AVAILABLE or not OPEN_CLIP_AVAILABLE:
        print("Profile Test SKIPPED: segment_anything or open_clip_torch library not installed.")
    else:
        import cProfile, pstats, io
        print("Testing and Profiling Semantic Processor with SAM and OpenCLIP...")
        print(f"Device: {SEMANTIC_DEVICE}, SAM Checkpoint: {INSTANCE_MODEL_CHECKPOINT_PATH}, CLIP: {CLIP_MODEL_NAME}/{CLIP_PRETRAINED_DATASET}")
        print(f"Prompts: {TEXT_PROMPTS}")

        try:
            # Pre-load models for fairer profiling of process_frame_for_semantics
            print("Pre-loading models for profiling...")
            load_instance_segmentation_model()
            load_clip_model(text_prompts=TEXT_PROMPTS) # Use global
            print("Models pre-loaded.")

            dummy_h, dummy_w = 240, 320
            img_np = np.zeros((dummy_h, dummy_w, 3), dtype=np.uint8)
            cv2.rectangle(img_np, (dummy_w//4, dummy_h//4), (dummy_w//2, dummy_h//2), (0,0,200), -1) # Red BGR
            cv2.circle(img_np, (3*dummy_w//4, 3*dummy_h//4), dummy_h//5, (0,200,0), -1) # Green BGR
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

            if not id_to_label or (len(id_to_label) == 1 and 0 in id_to_label):
                 print("WARNING: id_to_label map is empty or only contains background!")

            # (Optional Matplotlib visualization - can be slow or fail in some envs)
            # import matplotlib.pyplot as plt ... (code from previous version) ...

        except FileNotFoundError as e:
            print(f"[CRITICAL ERROR in __main__] Model checkpoint file not found: {e}.")
        except ImportError as e:
             print(f"[CRITICAL ERROR in __main__] Required library not found: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during test: {e}")
            import traceback
            traceback.print_exc()
        print("\nSemantic Processor __main__ test complete.")
```
