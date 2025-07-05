# mast3r_slam/semantic_processor.py
"""
Handles per-frame instance segmentation (SAM) and classification (OpenCLIP).
"""
import torch
import torchvision
import numpy as np
import cv2
from PIL import Image
import pathlib
import os
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime

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

INSTANCE_MODEL_CHECKPOINT_PATH = "checkpoints/sam_vit_h_4b8939.pth"
INSTANCE_MODEL_TYPE = "vit_h"

CLIP_MODEL_NAME = 'ViT-B-32'
CLIP_PRETRAINED_DATASET = 'laion2b_e16'

TEXT_PROMPTS = [
    # Background and structural
    "background", "wall surface", "wooden floor", "white ceiling", "room corner", "empty space",
    
    # Furniture - more descriptive
    "office chair", "wooden chair", "computer desk", "wooden table", "dining table", "work surface",
    "bookshelf", "storage cabinet", "file drawer", "furniture leg", 
    
    # Electronics - specific
    "computer monitor", "laptop computer", "desktop computer", "television screen", "electronic display",
    "computer keyboard", "computer mouse", "desktop computer", "electronic device",
    
    # Objects - descriptive  
    "coffee cup", "drinking mug", "book spine", "stack of books", "paper document",
    "picture frame", "wall art", "decorative object", "storage box", "container object",
    
    # Room elements - specific
    "interior door", "glass window", "ceiling light", "desk lamp", "table lamp", "light fixture",
    "light switch", "wall outlet", "door frame", "window frame"
]
SEMANTIC_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_SAM_POINTS_PER_SIDE = 32
DEFAULT_SAM_PRED_IOU_THRESH = 0.88
DEFAULT_SAM_STABILITY_SCORE_THRESH = 0.95
DEFAULT_SAM_MIN_MASK_REGION_AREA = 50

# --- CLIP Specifics ---
# We need to know the input resolution for CLIP to resize crops appropriately
# For ViT-bigG-14, it's typically 224, but OpenCLIP's preprocess handles it.
# We will extract the resize and normalize components from CLIP's preprocess.
_clip_input_resolution = (224, 224) # Default, will be updated from loaded model if possible

# --- Debug Visualization Configuration ---
DEBUG_VISUALIZATION = True  # Global flag to enable/disable debug output
DEBUG_OUTPUT_DIR = "debug_semantic_output"  # Directory for debug outputs
DEBUG_SAVE_SAM_MASKS = True  # Save SAM mask overlays
DEBUG_SAVE_CLIP_CROPS = True  # Save individual CLIP input crops
DEBUG_SAVE_CLASSIFICATION_RESULTS = True  # Save classification summary
DEBUG_MAX_MASKS_TO_VISUALIZE = 20  # Limit number of masks to visualize (performance)

def enable_debug_visualization(enable: bool = True, output_dir: str = None):
    """Enable or disable debug visualization globally."""
    global DEBUG_VISUALIZATION, DEBUG_OUTPUT_DIR
    DEBUG_VISUALIZATION = enable
    if output_dir is not None:
        DEBUG_OUTPUT_DIR = output_dir
    print(f"[DEBUG] Semantic processor debug visualization: {'ENABLED' if enable else 'DISABLED'}")
    if enable:
        print(f"[DEBUG] Output directory: {DEBUG_OUTPUT_DIR}")

def set_debug_options(save_sam: bool = True, save_clips: bool = True, 
                     save_classifications: bool = True, max_masks: int = 20):
    """Configure debug visualization options."""
    global DEBUG_SAVE_SAM_MASKS, DEBUG_SAVE_CLIP_CROPS, DEBUG_SAVE_CLASSIFICATION_RESULTS, DEBUG_MAX_MASKS_TO_VISUALIZE
    DEBUG_SAVE_SAM_MASKS = save_sam
    DEBUG_SAVE_CLIP_CROPS = save_clips
    DEBUG_SAVE_CLASSIFICATION_RESULTS = save_classifications
    DEBUG_MAX_MASKS_TO_VISUALIZE = max_masks


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

def create_debug_output_dir(frame_id: int = None) -> pathlib.Path:
    """Create organized debug output directory structure."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if frame_id is not None:
        debug_dir = pathlib.Path(DEBUG_OUTPUT_DIR) / f"frame_{frame_id:06d}_{timestamp}"
    else:
        debug_dir = pathlib.Path(DEBUG_OUTPUT_DIR) / f"session_{timestamp}"
    
    debug_dir.mkdir(parents=True, exist_ok=True)
    (debug_dir / "sam_masks").mkdir(exist_ok=True)
    (debug_dir / "clip_crops").mkdir(exist_ok=True)
    (debug_dir / "classifications").mkdir(exist_ok=True)
    return debug_dir

def save_sam_masks_visualization(image_hwc_uint8_rgb: np.ndarray, sam_masks_data_list: list, 
                                output_dir: pathlib.Path, frame_id: int = 0, 
                                tensor_crops_for_clip: list = None, 
                                predicted_classes: list = None, confidence_scores: list = None):
    """Save SAM mask visualizations with overlays."""
    if not DEBUG_SAVE_SAM_MASKS or not sam_masks_data_list:
        return
    
    # Create overlay image with all masks
    overlay_img = image_hwc_uint8_rgb.copy()
    colors = plt.cm.tab20(np.linspace(0, 1, min(len(sam_masks_data_list), 20)))
    
    # Create combined mask overlay
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Original image
    axes[0].imshow(image_hwc_uint8_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # All masks overlay
    for i, mask_data in enumerate(sam_masks_data_list[:DEBUG_MAX_MASKS_TO_VISUALIZE]):
        mask = mask_data['segmentation']
        color = (np.array(colors[i % len(colors)][:3]) * 255).astype(np.uint8)
        overlay_img[mask] = overlay_img[mask] * 0.6 + color * 0.4
    
    axes[1].imshow(overlay_img)
    axes[1].set_title(f"All SAM Masks Overlay ({len(sam_masks_data_list)} masks)")
    axes[1].axis('off')
    
    # Largest masks individual view
    if len(sam_masks_data_list) > 0:
        largest_mask = sam_masks_data_list[0]['segmentation']
        axes[2].imshow(largest_mask, cmap='gray')
        axes[2].set_title(f"Largest Mask (area: {sam_masks_data_list[0]['area']})")
        axes[2].axis('off')
    
    # Mask count histogram
    areas = [mask['area'] for mask in sam_masks_data_list]
    axes[3].hist(areas, bins=20, alpha=0.7)
    axes[3].set_title(f"Mask Area Distribution")
    axes[3].set_xlabel("Area (pixels)")
    axes[3].set_ylabel("Count")
    
    plt.tight_layout()
    plt.savefig(output_dir / "sam_masks" / f"sam_overview_frame_{frame_id:06d}.png", 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save individual masks with CLIP crops
    for i, mask_data in enumerate(sam_masks_data_list[:DEBUG_MAX_MASKS_TO_VISUALIZE]):
        mask = mask_data['segmentation']
        area = mask_data['area']
        
        # Check if we have corresponding CLIP data
        has_clip_data = (tensor_crops_for_clip is not None and i < len(tensor_crops_for_clip) and
                        predicted_classes is not None and i < len(predicted_classes) and
                        confidence_scores is not None and i < len(confidence_scores))
        
        # Create individual mask visualization with or without CLIP crop
        if has_clip_data:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        else:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original region
        masked_img = image_hwc_uint8_rgb.copy()
        masked_img[~mask] = masked_img[~mask] * 0.3  # Dim non-mask areas
        ax1.imshow(masked_img)
        ax1.set_title(f"Mask {i:02d} - Highlighted Region")
        ax1.axis('off')
        
        # Binary mask
        ax2.imshow(mask, cmap='gray')
        ax2.set_title(f"Binary Mask (area: {area})")
        ax2.axis('off')
        
        # Mask boundary overlay
        boundary_img = image_hwc_uint8_rgb.copy()
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(boundary_img, contours, -1, (255, 0, 0), 2)
        ax3.imshow(boundary_img)
        ax3.set_title(f"Mask Boundary")
        ax3.axis('off')
        
        # Add CLIP crop if available
        if has_clip_data:
            crop_tensor = tensor_crops_for_clip[i]
            crop_np = crop_tensor.permute(1, 2, 0).cpu().numpy()
            crop_np = np.clip(crop_np, 0, 1)
            
            predicted_class = predicted_classes[i]
            confidence = confidence_scores[i]
            
            ax4.imshow(crop_np)
            ax4.set_title(f"CLIP Crop\nPredicted: {predicted_class}\nConfidence: {confidence:.2f}")
            ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / "sam_masks" / f"mask_{i:02d}_area_{area}_frame_{frame_id:06d}.png", 
                    dpi=100, bbox_inches='tight')
        plt.close()

def save_clip_crops_visualization(tensor_crops_for_clip: list, original_image_tensor: torch.Tensor,
                                 output_dir: pathlib.Path, frame_id: int = 0):
    """Save CLIP input crops for inspection."""
    if not DEBUG_SAVE_CLIP_CROPS or not tensor_crops_for_clip:
        return
    
    # Save all crops in a grid
    n_crops = len(tensor_crops_for_clip)
    cols = min(5, n_crops)
    rows = (n_crops + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, crop_tensor in enumerate(tensor_crops_for_clip[:DEBUG_MAX_MASKS_TO_VISUALIZE]):
        if i < len(axes):
            # Convert tensor back to displayable format
            crop_np = crop_tensor.permute(1, 2, 0).cpu().numpy()
            crop_np = np.clip(crop_np, 0, 1)  # Ensure 0-1 range
            
            axes[i].imshow(crop_np)
            axes[i].set_title(f"Crop {i:02d}")
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(tensor_crops_for_clip), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "clip_crops" / f"clip_crops_grid_frame_{frame_id:06d}.png", 
                dpi=100, bbox_inches='tight')
    plt.close()
    
    # Save individual crops with metadata
    for i, crop_tensor in enumerate(tensor_crops_for_clip[:DEBUG_MAX_MASKS_TO_VISUALIZE]):
        crop_np = crop_tensor.permute(1, 2, 0).cpu().numpy()
        crop_np = np.clip(crop_np, 0, 1)
        
        plt.figure(figsize=(6, 6))
        plt.imshow(crop_np)
        plt.title(f"CLIP Input Crop {i:02d}\nSize: {crop_tensor.shape}")
        plt.axis('off')
        plt.savefig(output_dir / "clip_crops" / f"crop_{i:02d}_frame_{frame_id:06d}.png", 
                    dpi=100, bbox_inches='tight')
        plt.close()

def save_classification_results(segment_info_for_classification: list, best_scores: torch.Tensor,
                               best_indices: torch.Tensor, cached_prompts_list: list,
                               original_image_hwc: np.ndarray, output_dir: pathlib.Path, 
                               frame_id: int = 0):
    """Save classification results with confidence scores and visual summary."""
    if not DEBUG_SAVE_CLASSIFICATION_RESULTS or not segment_info_for_classification:
        return
    
    # Create classification summary
    classification_data = []
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Original image
    axes[0].imshow(original_image_hwc)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Classification confidence histogram
    confidence_scores = best_scores.cpu().numpy()
    axes[1].hist(confidence_scores, bins=20, alpha=0.7, color='skyblue')
    axes[1].set_title("Classification Confidence Distribution")
    axes[1].set_xlabel("Confidence Score")
    axes[1].set_ylabel("Count")
    
    # Class distribution
    predicted_classes = [cached_prompts_list[idx.item()] for idx in best_indices]
    unique_classes, counts = np.unique(predicted_classes, return_counts=True)
    axes[2].bar(range(len(unique_classes)), counts, color='lightcoral')
    axes[2].set_xticks(range(len(unique_classes)))
    axes[2].set_xticklabels(unique_classes, rotation=45, ha='right')
    axes[2].set_title("Predicted Class Distribution")
    axes[2].set_ylabel("Count")
    
    # Confidence vs class scatter
    class_indices = [list(unique_classes).index(cls) for cls in predicted_classes]
    axes[3].scatter(class_indices, confidence_scores, alpha=0.6, color='green')
    axes[3].set_xticks(range(len(unique_classes)))
    axes[3].set_xticklabels(unique_classes, rotation=45, ha='right')
    axes[3].set_title("Confidence vs Predicted Class")
    axes[3].set_ylabel("Confidence Score")
    
    # Top confident predictions
    top_k = min(10, len(confidence_scores))
    top_indices = np.argsort(confidence_scores)[-top_k:][::-1]
    top_classes = [predicted_classes[i] for i in top_indices]
    top_scores = [confidence_scores[i] for i in top_indices]
    
    axes[4].barh(range(len(top_classes)), top_scores, color='gold')
    axes[4].set_yticks(range(len(top_classes)))
    axes[4].set_yticklabels([f"{cls} (seg {top_indices[i]})" for i, cls in enumerate(top_classes)])
    axes[4].set_title(f"Top {top_k} Most Confident Predictions")
    axes[4].set_xlabel("Confidence Score")
    
    # Low confidence predictions
    low_indices = np.argsort(confidence_scores)[:top_k]
    low_classes = [predicted_classes[i] for i in low_indices]
    low_scores = [confidence_scores[i] for i in low_indices]
    
    axes[5].barh(range(len(low_classes)), low_scores, color='orange')
    axes[5].set_yticks(range(len(low_classes)))
    axes[5].set_yticklabels([f"{cls} (seg {low_indices[i]})" for i, cls in enumerate(low_classes)])
    axes[5].set_title(f"Lowest {top_k} Confidence Predictions")
    axes[5].set_xlabel("Confidence Score")
    
    plt.tight_layout()
    plt.savefig(output_dir / "classifications" / f"classification_summary_frame_{frame_id:06d}.png", 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save detailed classification data
    for i, seg_info in enumerate(segment_info_for_classification):
        classification_data.append({
            "segment_id": i,
            "predicted_class": cached_prompts_list[best_indices[i].item()],
            "confidence_score": float(best_scores[i].item()),
            "mask_area": int(seg_info['mask_torch'].sum().item())
        })
    
    # Save metadata as JSON
    metadata = {
        "frame_id": frame_id,
        "total_segments": len(segment_info_for_classification),
        "mean_confidence": float(np.mean(confidence_scores)),
        "std_confidence": float(np.std(confidence_scores)),
        "available_classes": cached_prompts_list,
        "classifications": classification_data
    }
    
    with open(output_dir / "classifications" / f"classification_data_frame_{frame_id:06d}.json", 'w') as f:
        json.dump(metadata, f, indent=2)

def process_frame_for_semantics(image_tensor_chw_0_1_rgb: torch.Tensor,
                                text_prompts_for_clip: list = None,
                                enable_debug_viz: bool = False,
                                frame_id: int = 0):
    global _clip_input_resolution # Use the resolution determined at model load
    effective_prompts = text_prompts_for_clip if text_prompts_for_clip is not None else TEXT_PROMPTS
    
    # CRITICAL FIX: Check if input tensor is in MASt3R/DUSt3R normalized range [-1, 1]
    # and convert back to [0, 1] range for proper semantic processing
    if image_tensor_chw_0_1_rgb.min() < -0.1:  # Likely in [-1, 1] range from ImgNorm
        print(f"[DEBUG SemanticProcessor] Input tensor in [-1,1] range, converting to [0,1]")
        print(f"   Before: range=[{image_tensor_chw_0_1_rgb.min():.3f}, {image_tensor_chw_0_1_rgb.max():.3f}]")
        # Convert from [-1, 1] to [0, 1]: (x + 1) / 2
        image_tensor_chw_0_1_rgb = (image_tensor_chw_0_1_rgb + 1.0) / 2.0
        print(f"   After: range=[{image_tensor_chw_0_1_rgb.min():.3f}, {image_tensor_chw_0_1_rgb.max():.3f}]")
    else:
        print(f"[DEBUG SemanticProcessor] Input tensor appears to be in [0,1] range: [{image_tensor_chw_0_1_rgb.min():.3f}, {image_tensor_chw_0_1_rgb.max():.3f}]")

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

    # --- Debug Visualization: SAM Masks (Basic) ---
    debug_output_dir = None
    if enable_debug_viz and DEBUG_VISUALIZATION:
        debug_output_dir = create_debug_output_dir(frame_id)
        print(f"[DEBUG] Saving SAM visualization for frame {frame_id} to {debug_output_dir}")
        # Save basic SAM visualization first (without CLIP data)
        save_sam_masks_visualization(image_hwc_uint8_rgb, sam_masks_data_list, debug_output_dir, frame_id)

    tensor_crops_for_clip = []
    segment_info_for_classification = []

    for mask_data in sam_masks_data_list:
        segment_torch_bool = torch.from_numpy(mask_data['segmentation'].astype(bool)).to(device=SEMANTIC_DEVICE)
        # Try highlight method (performed best in tests)
        # Create full image with highlighted region
        masked_highlight = image_tensor_chw_0_1_rgb.clone()
        masked_highlight[:, ~segment_torch_bool] *= 0.3  # Dim non-mask areas
        
        # Resize to CLIP input size
        import torch.nn.functional as F
        masked_highlight_resized = F.interpolate(
            masked_highlight.unsqueeze(0), size=_clip_input_resolution, 
            mode='bilinear', align_corners=False
        ).squeeze(0)
        
        if masked_highlight_resized is not None:
            tensor_crops_for_clip.append(masked_highlight_resized)
            segment_info_for_classification.append({'mask_torch': segment_torch_bool})

    if not tensor_crops_for_clip:
        # print(f"[INFO SemanticProcessor] No valid tensor crops from SAM masks for CLIP.")
        return local_instance_mask, local_id_to_class_label_map

    # --- Debug Visualization: CLIP Crops ---
    if enable_debug_viz and DEBUG_VISUALIZATION and debug_output_dir is not None:
        print(f"[DEBUG] Saving CLIP crops visualization for frame {frame_id}")
        save_clip_crops_visualization(tensor_crops_for_clip, image_tensor_chw_0_1_rgb, debug_output_dir, frame_id)

    # Batch preprocess (normalization mainly, as resize is done, and ToTensor is implicit)
    # The _clip_image_preprocess_val is a torchvision.transforms.Compose.
    # It typically includes Resize, CenterCrop, ToTensor, Normalize.
    # Since we've resized and have tensors, we mainly need to replicate Normalize.
    # Let's inspect clip_img_val_preprocess.transforms to find Normalize
    normalize_transform = None
    if hasattr(clip_img_val_preprocess, 'transforms'):
        for t in clip_img_val_preprocess.transforms:
            # Check for torchvision Normalize transform
            if hasattr(t, 'mean') and hasattr(t, 'std') and hasattr(t, '__call__'):
                # This is likely a Normalize transform
                normalize_transform = t
                print(f"[DEBUG SemanticProcessor] Found Normalize transform: mean={t.mean}, std={t.std}")
                break

    if not normalize_transform:
        print("[Warning SemanticProcessor] Could not find Normalize transform in CLIP preprocess. Using ImageNet stats for better compatibility.")
        # Many CLIP models work better with ImageNet normalization
        # Try ImageNet stats first: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        normalize_transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )


    try:
        # Stack crops into a batch. They are already C,H_clip,W_clip and 0-1 range.
        batched_image_tensors = torch.stack(tensor_crops_for_clip) # B, C, H_clip, W_clip
        print(f"[DEBUG SemanticProcessor] Batched tensors shape: {batched_image_tensors.shape}, range: [{batched_image_tensors.min():.3f}, {batched_image_tensors.max():.3f}]")
        
        # Ensure tensors are on correct device for normalization
        if hasattr(normalize_transform, 'mean'):
            # Move normalize transform to same device as tensors
            if hasattr(normalize_transform.mean, 'device'):
                if normalize_transform.mean.device != batched_image_tensors.device:
                    normalize_transform = normalize_transform.to(batched_image_tensors.device)
        
        # Apply normalization
        normalized_batched_images = normalize_transform(batched_image_tensors)
        print(f"[DEBUG SemanticProcessor] Normalized tensors range: [{normalized_batched_images.min():.3f}, {normalized_batched_images.max():.3f}]")

    except Exception as e_preproc:
        print(f"[ERROR SemanticProcessor] Batch CLIP tensor preprocessing failed: {e_preproc}")
        import traceback
        traceback.print_exc()
        return local_instance_mask, local_id_to_class_label_map

    with torch.no_grad():
        batched_image_features = clip_model.encode_image(normalized_batched_images)
        batched_image_features /= batched_image_features.norm(dim=-1, keepdim=True)
        print(f"[DEBUG SemanticProcessor] Image features shape: {batched_image_features.shape}")

    similarity_matrix = (100.0 * batched_image_features @ text_features_tensor.T)
    best_scores, best_indices = similarity_matrix.max(dim=1)
    
    # Debug CLIP scores
    print(f"[DEBUG SemanticProcessor] CLIP similarity matrix shape: {similarity_matrix.shape}")
    print(f"[DEBUG SemanticProcessor] Best scores range: [{best_scores.min():.2f}, {best_scores.max():.2f}]")
    print(f"[DEBUG SemanticProcessor] Best scores mean: {best_scores.mean():.2f}")
    
    # Show top predictions for first few crops
    for i in range(min(3, len(best_scores))):
        top_k = min(5, similarity_matrix.shape[1])
        top_scores, top_indices = similarity_matrix[i].topk(top_k)
        print(f"[DEBUG SemanticProcessor] Crop {i} top predictions:")
        for j in range(top_k):
            cls = cached_prompts_list[top_indices[j].item()]
            score = top_scores[j].item()
            print(f"   {j+1}. {cls}: {score:.2f}")

    # --- Debug Visualization: Classification Results ---
    if enable_debug_viz and DEBUG_VISUALIZATION and debug_output_dir is not None:
        print(f"[DEBUG] Saving classification results for frame {frame_id}")
        save_classification_results(segment_info_for_classification, best_scores, best_indices, 
                                   cached_prompts_list, image_hwc_uint8_rgb, debug_output_dir, frame_id)
        
        # Save enhanced SAM masks with CLIP crops and predictions
        print(f"[DEBUG] Saving enhanced SAM masks with CLIP crops for frame {frame_id}")
        predicted_classes = [cached_prompts_list[idx.item()] for idx in best_indices]
        confidence_scores = [score.item() for score in best_scores]
        save_sam_masks_visualization(image_hwc_uint8_rgb, sam_masks_data_list, debug_output_dir, frame_id,
                                   tensor_crops_for_clip, predicted_classes, confidence_scores)

    current_local_id_counter = 1
    for i, seg_info in enumerate(segment_info_for_classification):
        class_label = cached_prompts_list[best_indices[i].item()]
        confidence = best_scores[i].item()
        
        print(f"[DEBUG] Segment {i}: {class_label} (confidence: {confidence:.2f})")
        
        # Assign segments with confidence > threshold, even if classified as background
        # This helps debug what's happening
        confidence_threshold = 20.0  # Lower threshold for now to see what's happening
        if confidence > confidence_threshold:
            segment_torch = seg_info['mask_torch']
            valid_pixels_for_current_id = segment_torch & (local_instance_mask == 0)
            if valid_pixels_for_current_id.any():
                local_instance_mask[valid_pixels_for_current_id] = current_local_id_counter
                local_id_to_class_label_map[current_local_id_counter] = class_label
                print(f"[DEBUG] Assigned {valid_pixels_for_current_id.sum().item()} pixels to local_id {current_local_id_counter} ({class_label})")
                current_local_id_counter += 1
                if current_local_id_counter > 255: break
        else:
            print(f"[DEBUG] Skipped segment {i} due to low confidence ({confidence:.2f} < {confidence_threshold})")
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
            local_mask, id_to_label = process_frame_for_semantics(img_tensor, text_prompts_for_clip=TEXT_PROMPTS,
                                                                 enable_debug_viz=True, frame_id=999)
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
