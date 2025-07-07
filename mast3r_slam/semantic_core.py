# mast3r_slam/semantic_core.py
"""
Core semantic processing constants and utilities.
Minimal version containing only what's actually used in the codebase.
"""
import torch

# Device configuration
SEMANTIC_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model configurations
INSTANCE_MODEL_CHECKPOINT_PATH = "checkpoints/sam_vit_h_4b8939.pth"
INSTANCE_MODEL_TYPE = "vit_h"

# CLIP Model Configuration
CLIP_MODEL_NAME = 'ViT-H-14'
CLIP_PRETRAINED_DATASET = 'laion2b_s32b_b79k'

# SAM parameters
DEFAULT_SAM_POINTS_PER_SIDE = 64
DEFAULT_SAM_PRED_IOU_THRESH = 0.86
DEFAULT_SAM_STABILITY_SCORE_THRESH = 0.92
DEFAULT_SAM_MIN_MASK_REGION_AREA = 50

# Original text prompts
ORIGINAL_TEXT_PROMPTS = [
    # Background and structural - PRIORITIZED for large surfaces
    "background", "wall", "wall surface", "plain wall", "white wall", "painted wall",
    "wooden floor", "white ceiling", "empty space",
    
    # Electronics - moved before furniture to avoid desk misclassification
    "computer monitor", "laptop computer", "desktop computer", "television screen", "electronic display",
    "computer keyboard", "computer mouse", "electronic device",
    
    # Furniture - moved after walls and electronics
    "office chair", "wooden chair", "wooden table", "dining table", "work surface",
    "bookshelf", "storage cabinet", "file drawer", "furniture leg",
    "computer desk",  # MOVED TO END of furniture to prevent wall misclassification
    
    # Objects - descriptive  
    "coffee cup", "drinking mug", "book spine", "stack of books", "paper document",
    "picture frame", "wall art", "decorative object", "storage box", "container object",
    
    # Room elements - specific
    "interior door", "glass window", "ceiling light", "desk lamp", "table lamp", "light fixture",
    "light switch", "wall outlet", "door frame", "window frame"
]

def create_augmented_prompts(base_prompts):
    """Create augmented text prompts with descriptive variations."""
    augmented = []
    templates = [
        "{}",
        "a photo of a {}",
        "a picture of a {}",
    ]
    
    for prompt in base_prompts:
        if prompt == "background":
            augmented.append(prompt)
        else:
            for template in templates:
                augmented.append(template.format(prompt))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_augmented = []
    for item in augmented:
        if item not in seen:
            seen.add(item)
            unique_augmented.append(item)
    
    return unique_augmented

# Create augmented prompts
TEXT_PROMPTS = create_augmented_prompts(ORIGINAL_TEXT_PROMPTS)

# Debug configuration
DEBUG_VISUALIZATION = False
DEBUG_OUTPUT_DIR = "debug_semantic_output"

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
    # This is a stub for compatibility - actual implementation in semantic_processor_v2
    pass


# Model loading functions
def load_instance_segmentation_model(checkpoint_path: str = INSTANCE_MODEL_CHECKPOINT_PATH,
                                   model_type: str = INSTANCE_MODEL_TYPE,
                                   device: str = SEMANTIC_DEVICE,
                                   points_per_side: int = DEFAULT_SAM_POINTS_PER_SIDE,
                                   pred_iou_thresh: float = DEFAULT_SAM_PRED_IOU_THRESH,
                                   stability_score_thresh: float = DEFAULT_SAM_STABILITY_SCORE_THRESH,
                                   min_mask_region_area: int = DEFAULT_SAM_MIN_MASK_REGION_AREA):
    """Load SAM model for instance segmentation."""
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except ImportError:
        raise ImportError("segment_anything library is required for instance segmentation.")
    
    print(f"[INFO SemanticProcessor] Loading SAM model: type='{model_type}' from '{checkpoint_path}'")
    sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam_model.to(device=device)
    sam_model.eval()
    
    mask_generator = SamAutomaticMaskGenerator(
        model=sam_model,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=min_mask_region_area
    )
    
    print(f"[INFO SemanticProcessor] SAM loaded (pps={points_per_side}, min_area={min_mask_region_area}).")
    return mask_generator


def load_clip_model(model_name: str = CLIP_MODEL_NAME, 
                   pretrained_dataset: str = CLIP_PRETRAINED_DATASET,
                   device: str = SEMANTIC_DEVICE):
    """Load CLIP model for classification."""
    try:
        import open_clip
    except ImportError:
        raise ImportError("open_clip library is required for classification.")
    
    print(f"[INFO SemanticProcessor] Loading OpenCLIP model: '{model_name}' with '{pretrained_dataset}' to '{device}'.")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained_dataset, device=device
    )
    model.eval()
    
    # Get input resolution
    input_resolution = model.visual.image_size
    if isinstance(input_resolution, (list, tuple)):
        input_resolution = input_resolution[0]
    
    print(f"[INFO SemanticProcessor] OpenCLIP model ready. Input resolution: ({input_resolution}, {input_resolution})")
    
    # Prepare text features
    tokenizer = open_clip.get_tokenizer(model_name)
    print(f"[INFO SemanticProcessor] Tokenizing {len(TEXT_PROMPTS)} prompts for OpenCLIP...")
    text_inputs = tokenizer(TEXT_PROMPTS).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    print("[INFO SemanticProcessor] OpenCLIP text features loaded/updated.")
    
    return model, preprocess, text_features, tokenizer


# Fallback function for compatibility (simplified version)
def process_frame_for_semantics(image_tensor_chw_0_1_rgb: torch.Tensor,
                               text_prompts_for_clip: list = None,
                               enable_debug_viz: bool = False,
                               frame_id: int = 0):
    """
    Simple fallback semantic processing function.
    This is only used as a fallback in tracker.py when semantic_processor_v2 is not available.
    """
    # Return empty masks for compatibility
    h, w = image_tensor_chw_0_1_rgb.shape[1:3]
    empty_mask = torch.zeros((h, w), dtype=torch.int64, device=image_tensor_chw_0_1_rgb.device)
    empty_map = {}
    return empty_mask, empty_map