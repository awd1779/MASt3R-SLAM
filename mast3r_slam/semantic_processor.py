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
CLIP_PRETRAINED_DATASET = 'laion2b_s39b_b160k' # Corrected based on error message

# TODO: USER - Define your text prompts for CLIP classification
TEXT_PROMPTS = [
    "background", # Prompt for background/unclassified (often good to have one)
    "chair",
    "desk",
    "table",
    "monitor",
    "person",
    "plant",
    "cup",
    "book",
    "keyboard",
    "mouse",
    "laptop",
    "screen"
    # Add more prompts as needed for your environment. Using "a photo of a " prefix is common but not strictly required for all CLIP uses.
    # OpenCLIP tokenization might handle it well. Test what works best.
]
# Ensure prompts are descriptive enough for good classification.
# For better matching, you can use templates like "a photo of a {}", "an image of a {}", etc.
# and then format them: [template.format(obj) for obj in base_objects]

SEMANTIC_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_instance_segmentation_model(checkpoint_path: str = INSTANCE_MODEL_CHECKPOINT_PATH,
                                     model_type: str = INSTANCE_MODEL_TYPE,
                                     device: str = SEMANTIC_DEVICE):
    """
    Loads the SAM model and prepares the automatic mask generator.
    """
    global _sam_mask_generator, SAM_AVAILABLE

    if not SAM_AVAILABLE:
        raise ImportError("segment_anything library is required for instance segmentation.")

    if _sam_mask_generator is not None:
        # print("[INFO SemanticProcessor] SAM Automatic Mask Generator already loaded.")
        return _sam_mask_generator

    try:
        print(f"[INFO SemanticProcessor] Loading SAM model: type='{model_type}' from checkpoint='{checkpoint_path}' to device='{device}'")
        sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam_model.to(device=device)
        sam_model.eval()

        _sam_mask_generator = SamAutomaticMaskGenerator(
            model=sam_model,
            points_per_side=16, # Fewer points for potentially faster processing, coarser masks
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=150, # Filter out very small regions
        )
        print(f"[INFO SemanticProcessor] SAM model and SamAutomaticMaskGenerator loaded (points_per_side=16, min_area=150).")
    except FileNotFoundError:
        print(f"[ERROR SemanticProcessor] SAM Checkpoint file not found at: {checkpoint_path}")
        print(f"[ERROR SemanticProcessor] Please download a SAM checkpoint and update INSTANCE_MODEL_CHECKPOINT_PATH.")
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
                    device: str = SEMANTIC_DEVICE):
    """
    Loads the OpenCLIP model, preprocessor and pre-calculates text features.
    Downloads model from Hugging Face if not cached.
    """
    global _clip_model, _clip_preprocess, _clip_text_features_tensor, _clip_text_prompts_cache, OPEN_CLIP_AVAILABLE, TEXT_PROMPTS

    if not OPEN_CLIP_AVAILABLE:
        raise ImportError("open_clip_torch library is required for CLIP classification.")

    current_prompts = text_prompts if text_prompts is not None else TEXT_PROMPTS
    prompts_are_cached = (_clip_model is not None and
                          _clip_preprocess is not None and
                          _clip_text_features_tensor is not None and
                          _clip_text_prompts_cache == current_prompts)

    if prompts_are_cached:
        # print("[INFO SemanticProcessor] OpenCLIP model and text features already loaded and cached.")
        return _clip_model, _clip_preprocess, _clip_text_features_tensor, _clip_text_prompts_cache

    try:
        print(f"[INFO SemanticProcessor] Loading OpenCLIP model: '{model_name}' with weights '{pretrained_dataset}' to '{device}'.")
        # OpenCLIP's create_model_and_transforms handles download from Hugging Face
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained_dataset,
            device=device
        )
        _clip_model.eval()

        print(f"[INFO SemanticProcessor] Tokenizing {len(current_prompts)} text prompts for OpenCLIP...")
        # Tokenize text prompts
        text_tokens = open_clip.tokenize(current_prompts).to(device)

        # Pre-calculate text features
        with torch.no_grad():
            _clip_text_features_tensor = _clip_model.encode_text(text_tokens)
            _clip_text_features_tensor /= _clip_text_features_tensor.norm(dim=-1, keepdim=True)

        _clip_text_prompts_cache = list(current_prompts) # Cache the prompts these features correspond to

        print(f"[INFO SemanticProcessor] OpenCLIP model and text features loaded successfully.")
    except Exception as e:
        print(f"[ERROR SemanticProcessor] Failed to load OpenCLIP model or process text prompts: {e}")
        _clip_model = None
        _clip_preprocess = None
        _clip_text_features_tensor = None
        _clip_text_prompts_cache = []
        raise e

    return _clip_model, _clip_preprocess, _clip_text_features_tensor, _clip_text_prompts_cache


def get_image_crop_from_mask(image_chw_0_1_rgb: torch.Tensor, binary_mask_hw: torch.Tensor):
    """
    Extracts a cropped image region based on the bounding box of a binary mask.
    Converts to PIL Image as CLIP preprocess often expects it.

    Args:
        image_chw_0_1_rgb (torch.Tensor): Original image (C, H, W), RGB, 0-1 range.
        binary_mask_hw (torch.Tensor): Binary mask (H, W) for the segment.

    Returns:
        PIL.Image: Cropped image region as a PIL Image, or None if mask is empty/invalid.
    """
    if not binary_mask_hw.any():
        return None

    rows = torch.any(binary_mask_hw, axis=1)
    cols = torch.any(binary_mask_hw, axis=0)
    if not rows.any() or not cols.any():
        return None

    ymin, ymax = torch.where(rows)[0][[0, -1]]
    xmin, xmax = torch.where(cols)[0][[0, -1]]

    # Crop the original image tensor (CHW)
    cropped_tensor = image_chw_0_1_rgb[:, ymin:ymax+1, xmin:xmax+1]

    if cropped_tensor.numel() == 0 or cropped_tensor.shape[1] == 0 or cropped_tensor.shape[2] == 0:
        return None

    # Convert to HWC, then to NumPy uint8, then to PIL Image
    cropped_np_hwc_uint8 = (cropped_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    pil_image = Image.fromarray(cropped_np_hwc_uint8)

    return pil_image


def classify_crop_with_clip(pil_image_crop: Image.Image,
                            clip_model, clip_preprocess,
                            text_features_tensor: torch.Tensor,
                            text_prompts_list: list,
                            device: str = SEMANTIC_DEVICE):
    """
    Classifies a PIL image crop using OpenCLIP against pre-calculated text features.
    """
    if pil_image_crop is None:
        return text_prompts_list[0] if text_prompts_list else "unknown" # Default to first prompt or "unknown"

    try:
        image_input = clip_preprocess(pil_image_crop).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Calculate similarity
            #logit_scale = clip_model.logit_scale.exp() # Some OpenCLIP models might not have this directly accessible this way for all versions
            # For zero-shot, often use raw cosine similarity scaled by 100 as logits
            similarity = (100.0 * image_features @ text_features_tensor.T) # .softmax(dim=-1) # Softmax if you want probabilities

        best_score, best_idx = similarity[0].max(dim=0)
        class_label = text_prompts_list[best_idx.item()]
        # print(f"  CLIP Scores: {similarity[0].cpu().numpy().round(2)}, Best: {class_label} ({best_score.item():.2f})")

    except Exception as e:
        print(f"[ERROR SemanticProcessor] CLIP classification failed for a crop: {e}")
        class_label = text_prompts_list[0] if text_prompts_list else "classification_failed" # Fallback

    return class_label


def process_frame_for_semantics(image_tensor_chw_0_1_rgb: torch.Tensor,
                                text_prompts_for_clip: list = None):
    """
    Main processing function for a single frame.
    1. Runs instance segmentation (SAM) to get local instance masks.
    2. For each instance mask, crops the region and classifies it using CLIP.
    """
    if text_prompts_for_clip is None:
        text_prompts_for_clip = TEXT_PROMPTS # Use global default if none provided

    # 1. Load models (idempotent)
    sam_generator = load_instance_segmentation_model(device=SEMANTIC_DEVICE)
    clip_model, clip_preprocess, text_features, text_prompts_list_cache = load_clip_model(
        text_prompts=text_prompts_for_clip, device=SEMANTIC_DEVICE
    )
    # Ensure the prompts used for text_features match the current text_prompts_for_clip
    # This is important if text_prompts_for_clip can change dynamically per call
    if text_prompts_list_cache != text_prompts_for_clip:
        print("[Warning SemanticProcessor] Text prompts changed; re-encoding text features for CLIP.")
        clip_model, clip_preprocess, text_features, text_prompts_list_cache = load_clip_model(
             text_prompts=text_prompts_for_clip, device=SEMANTIC_DEVICE, force_reload_prompts=True # Add a way to force reload prompts if API changes
        )


    _c, h, w = image_tensor_chw_0_1_rgb.shape

    # 2. Run Instance Segmentation (SAM)
    # SAM expects HWC uint8 numpy image
    image_hwc_uint8_rgb = (image_tensor_chw_0_1_rgb.permute(1, 2, 0) * 255.0).byte().cpu().numpy()

    # print(f"[INFO SemanticProcessor] Running SAM on image {h}x{w}...")
    sam_masks_data_list = sam_generator.generate(image_hwc_uint8_rgb)

    local_instance_mask = torch.zeros((h,w), dtype=torch.int64, device=SEMANTIC_DEVICE)
    local_id_to_class_label_map = {0: text_prompts_for_clip[0] if text_prompts_for_clip else "background"} # Default for background

    if not sam_masks_data_list:
        print(f"[INFO SemanticProcessor] SAM found no masks for frame.")
        return local_instance_mask, local_id_to_class_label_map

    # Sort by area, largest first, to give them priority in ID assignment if they overlap
    sam_masks_data_list = sorted(sam_masks_data_list, key=lambda x: x['area'], reverse=True)

    current_local_id = 1 # Start actual object IDs from 1
    # print(f"[INFO SemanticProcessor] SAM generated {len(sam_masks_data_list)} raw masks.")

    for mask_data in sam_masks_data_list:
        segment_bool_np = mask_data['segmentation'] # This is HxW boolean NumPy array
        segment_torch = torch.from_numpy(segment_bool_np).to(device=SEMANTIC_DEVICE)

        # Assign current_local_id to pixels where segment is true AND local_instance_mask is still 0 (background)
        # This gives priority to larger masks (due to sorting) in overlapping regions.
        valid_pixels_for_current_id = segment_torch & (local_instance_mask == 0)

        if valid_pixels_for_current_id.any():
            local_instance_mask[valid_pixels_for_current_id] = current_local_id

            # 3. Classify this segment with CLIP
            pil_crop = get_image_crop_from_mask(image_tensor_chw_0_1_rgb, segment_torch) # Pass the original image and the binary mask for this ID

            class_label = "unknown" # Default if crop is bad or classification fails
            if pil_crop:
                class_label = classify_crop_with_clip(pil_crop, clip_model, clip_preprocess,
                                                      text_features, text_prompts_list_cache, # Use cached prompts that match features
                                                      device=SEMANTIC_DEVICE)

            local_id_to_class_label_map[current_local_id] = class_label
            # print(f"  Local ID {current_local_id} (Area: {mask_data['area']}) -> CLIP: {class_label}")

            current_local_id += 1
            if current_local_id > 255: # Safety for uchar if that's a constraint downstream
                # print("[Warning SemanticProcessor] Reached local_id 255. Further distinct segments won't get new IDs if uchar is limit.")
                break

    # print(f"[INFO SemanticProcessor] Frame processing complete. Final local mask unique IDs: {torch.unique(local_instance_mask)}. Label map size: {len(local_id_to_class_label_map)}")
    return local_instance_mask, local_id_to_class_label_map


if __name__ == '__main__':
    print("Testing Semantic Processor with SAM and OpenCLIP...")
    if not SAM_AVAILABLE or not OPEN_CLIP_AVAILABLE:
        print("Required libraries (segment_anything or open_clip_torch) are not installed. Test cannot run.")
    else:
        print(f"Attempting to use device: {SEMANTIC_DEVICE}")
        print(f"SAM Checkpoint: {INSTANCE_MODEL_CHECKPOINT_PATH} (Type: {INSTANCE_MODEL_TYPE})")
        print(f"OpenCLIP Model: {CLIP_MODEL_NAME} (Dataset: {CLIP_PRETRAINED_DATASET})")
        print(f"Default Text Prompts: {TEXT_PROMPTS}")

        try:
            # Create a dummy image (C, H, W), RGB, 0-1 range
            dummy_h, dummy_w = 240, 320
            img_np = np.zeros((dummy_h, dummy_w, 3), dtype=np.uint8)
            # Make a BGR image with OpenCV, then convert
            cv2.rectangle(img_np, (dummy_w//4, dummy_h//4), (dummy_w//2, dummy_h//2), (0,0,200), -1) # Red BGR
            cv2.circle(img_np, (3*dummy_w//4, 3*dummy_h//4), dummy_h//5, (0,200,0), -1)     # Green BGR
            img_np[0:dummy_h//3, dummy_w//2:dummy_w, 2] = 200 # Blue BGR strip

            img_rgb_np_float = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_rgb_np_float).permute(2,0,1).to(SEMANTIC_DEVICE)

            print("\nRunning process_frame_for_semantics...")
            # TEXT_PROMPTS will be used by default from the global scope of the module
            # You can override by: text_prompts_for_clip=["a photo of red", "a photo of green", "a photo of blue", "other"]
            local_mask, id_to_label = process_frame_for_semantics(img_tensor)

            print("\n--- Test Results ---")
            print("Local Instance Mask shape:", local_mask.shape)
            print("Local Instance Mask unique IDs:", torch.unique(local_mask).cpu().tolist())
            print("Local ID to Class Label Map:", id_to_label)

            if not id_to_label or (len(id_to_label) == 1 and 0 in id_to_label):
                 print("WARNING: id_to_label map is empty or only contains background! SAM might not have found segments or CLIP failed.")
            if local_mask.shape != (dummy_h, dummy_w):
                print(f"ERROR: Mask shape {local_mask.shape} does not match image shape {(dummy_h, dummy_w)}")

            # Visualize (optional, requires matplotlib)
            try:
                import matplotlib.pyplot as plt
                print("Attempting to visualize test output using Matplotlib...")
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.imshow(img_rgb_np_float)
                plt.title("Original Test Image (RGB)")
                plt.axis('off')

                colored_seg_mask_vis = np.zeros((dummy_h, dummy_w, 3), dtype=np.uint8)
                unique_ids_in_mask = torch.unique(local_mask).cpu().numpy()

                # Create a color for each unique ID present in the actual data
                np.random.seed(0)
                viz_colors = {uid: np.random.randint(80, 220, size=3) for uid in unique_ids_in_mask if uid !=0}
                viz_colors[0] = [128,128,128] # Grey for background

                for uid_val in unique_ids_in_mask:
                    color_to_use = viz_colors.get(uid_val, [20,20,20]) # Default dark if ID somehow not in map
                    colored_seg_mask_vis[local_mask.cpu().numpy() == uid_val] = color_to_use

                plt.subplot(1, 2, 2)
                plt.imshow(colored_seg_mask_vis)
                clip_labels_display = "\n".join([f"{k}: {v}" for k,v in id_to_label.items()])
                plt.title(f"SAM Segments + CLIP Labels\n{clip_labels_display}", fontsize=8)
                plt.axis('off')

                output_filename = "semantic_processor_test_output.png"
                plt.savefig(output_filename)
                print(f"Saved semantic_processor test output visualization to {output_filename}")
            except ImportError:
                print("Matplotlib not installed, skipping visualization of test output.")
            except Exception as e_vis:
                print(f"Error during test visualization: {e_vis}")

        except FileNotFoundError as e:
            print(f"[CRITICAL ERROR in __main__] A model checkpoint file was not found: {e}. Please check INSTANCE_MODEL_CHECKPOINT_PATH in semantic_processor.py.")
        except ImportError as e:
             print(f"[CRITICAL ERROR in __main__] A required library (segment_anything or open_clip_torch) is not installed or there was an import error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during test: {e}")
            import traceback
            traceback.print_exc()
        print("\nSemantic Processor __main__ test complete.")
