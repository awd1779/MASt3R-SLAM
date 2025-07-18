"""Real Grounded-SAM2 processor implementation with proper model loading."""

import os
import sys
import time
import torch
import numpy as np
import cv2
from typing import Optional, Dict, List, Tuple
import torch.multiprocessing as mp
from queue import Empty
from pathlib import Path
import logging

from mast3r_slam.semantic_frame import encode_rle
from mast3r_slam.config import config
from mast3r_slam.grounded_sam2_config import SAM2_MODELS, GROUNDING_MODELS, GroundedSAM2ModelSelector

logger = logging.getLogger('mast3r_slam.grounded_sam2')

# Debug visualization will be controlled by config
DEBUG_OUTPUT_DIR = Path("debug_detections")

def log_debug(debug_mode: bool, message: str, level=logging.DEBUG):
    """Conditional logging based on debug mode."""
    if debug_mode:
        logger.log(level, message)


class RealGroundedSAM2Processor:
    """Real Grounded-SAM2 processor with automatic model path resolution."""
    
    def __init__(self, 
                 frame_queue: mp.Queue,
                 result_queue: mp.Queue,
                 vocabulary: List[str],
                 device: str = "cuda:1",
                 model_selector: Optional[GroundedSAM2ModelSelector] = None,
                 sam2_checkpoint_dir: Optional[str] = None,
                 grounding_dino_checkpoint_dir: Optional[str] = None,
                 confidence_threshold: float = 0.35,
                 dtype: str = "bfloat16",
                 debug_mode: bool = False,
                 save_debug_visualizations: bool = False,
                 deduplication_iou_threshold: float = 0.9,
                 mask_refinement_threshold: float = 0.7):
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.vocabulary = vocabulary
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.dtype_str = dtype
        self.debug_mode = debug_mode
        self.save_debug_visualizations = save_debug_visualizations
        self.deduplication_iou_threshold = deduplication_iou_threshold
        self.mask_refinement_threshold = mask_refinement_threshold
        
        # Convert dtype string to torch dtype
        dtype_mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16
        }
        self.dtype = dtype_mapping.get(dtype, torch.bfloat16)
        
        # Model directories
        self.sam2_checkpoint_dir = sam2_checkpoint_dir
        self.grounding_dino_checkpoint_dir = grounding_dino_checkpoint_dir
        
        # Model selection
        if model_selector is None:
            self.model_selector = GroundedSAM2ModelSelector(
                target_fps=15,
                max_vram_gb=10,
                quality_priority="quality"
            )
        else:
            self.model_selector = model_selector
            
        # Get model names from selector (which should already be set in main_semantic.py)
        if hasattr(self.model_selector, 'sam2_model') and self.model_selector.sam2_model:
            self.sam2_model_name = self.model_selector.sam2_model
            self.grounding_model_name = self.model_selector.grounding_model
        else:
            # Fall back to selection if not pre-configured
            self.sam2_model_name, self.grounding_model_name = self.model_selector.select_models()
        
        # Model instances
        self.sam2_predictor = None
        self.grounding_dino = None
        self.transform = None
        
        # Current frame state for image mode
        self.current_image_set = False
        
        # Track management
        self.current_tracks = {}
        self.next_track_id = 1
        
        # Performance tracking
        self.frame_times = []
        
        # Statistics tracking
        self.class_statistics = {vocab: {
            'detections': 0,
            'successful_masks': 0,
            'avg_confidence': 0.0,
            'avg_mask_size': 0.0,
            'confidences': []
        } for vocab in vocabulary}
        
    def find_model_paths(self) -> Tuple[str, str, str, str]:
        """Automatically find model paths."""
        # Start with custom paths if provided
        search_paths = []
        
        if self.sam2_checkpoint_dir:
            search_paths.append(Path(self.sam2_checkpoint_dir))
        if self.grounding_dino_checkpoint_dir:
            search_paths.append(Path(self.grounding_dino_checkpoint_dir))
            
        # Common locations to search
        search_paths.extend([
            Path.home() / "models",
            Path.home() / "libs",
            Path("/workspace"),
            Path("/content"),  # Colab
            Path.cwd() / "models",
            Path.cwd().parent / "models",
        ])
            
        # Debug: log search paths
        logger.info(f"Searching for models in: {[str(p) for p in search_paths]}")
            
        # SAM2 model info
        sam2_info = SAM2_MODELS[self.sam2_model_name]
        sam2_checkpoint = None
        sam2_config = None
        
        # Grounding DINO model info
        grounding_info = GROUNDING_MODELS[self.grounding_model_name]
        grounding_checkpoint = None
        grounding_config = None
        
        # Search for SAM2 files - prefer non-2.1 versions for compatibility
        sam2_filenames = {
            "hiera_tiny": ["sam2_hiera_tiny.pt", "sam2.1_hiera_tiny.pt"],
            "hiera_small": ["sam2_hiera_small.pt", "sam2.1_hiera_small.pt"], 
            "hiera_b+": ["sam2_hiera_base_plus.pt", "sam2.1_hiera_base_plus.pt"],
            "hiera_base+": ["sam2_hiera_base_plus.pt", "sam2.1_hiera_base_plus.pt"],
            "hiera_large": ["sam2_hiera_large.pt", "sam2.1_hiera_large.pt"]
        }
        
        for path in search_paths:
            # Check for SAM2
            sam2_dirs = [
                path / "segment-anything-2",
                path / "sam2",
                path / "SAM2"
            ]
            
            for sam2_dir in sam2_dirs:
                if sam2_dir.exists():
                    logger.debug(f"  Checking SAM2 dir: {sam2_dir}")
                    # Look for checkpoint
                    ckpt_files = sam2_filenames.get(self.sam2_model_name, [])
                    if not isinstance(ckpt_files, list):
                        ckpt_files = [ckpt_files]
                    
                    for ckpt_file in ckpt_files:
                        logger.debug(f"    Looking for: {ckpt_file}")
                        ckpt_paths = [
                            sam2_dir / "checkpoints" / ckpt_file,
                            sam2_dir / ckpt_file,
                            sam2_dir / "weights" / ckpt_file
                        ]
                        for ckpt_path in ckpt_paths:
                            if ckpt_path.exists():
                                sam2_checkpoint = str(ckpt_path)
                                logger.info(f"  ✓ Found SAM2 checkpoint: {sam2_checkpoint}")
                                break
                        if sam2_checkpoint:
                            break
                                
                    # Look for config
                    config_paths = [
                        sam2_dir / "sam2_configs" / f"{self.sam2_model_name}.yaml",
                        sam2_dir / "configs" / f"{self.sam2_model_name}.yaml",
                    ]
                    for cfg_path in config_paths:
                        if cfg_path.exists():
                            sam2_config = str(cfg_path)
                            break
                            
            # Check for Grounding DINO
            grounding_dirs = [
                path / "GroundingDINO",
                path / "groundingdino",
                path / "grounding-dino",
                path  # Also check directly in the path (e.g., ~/models/)
            ]
            
            for grounding_dir in grounding_dirs:
                if grounding_dir.exists():
                    logger.debug(f"  Checking Grounding DINO dir: {grounding_dir}")
                    # Look for checkpoint
                    ckpt_names = {
                        "grounding_dino_swin-t": "groundingdino_swint_ogc.pth",
                        "grounding_dino_swin-b": "groundingdino_swinb_cogcoor.pth"
                    }
                    ckpt_file = ckpt_names.get(self.grounding_model_name)
                    logger.debug(f"    Looking for checkpoint: {self.grounding_model_name} -> {ckpt_file}")
                    if ckpt_file:
                        ckpt_paths = [
                            grounding_dir / "weights" / ckpt_file,
                            grounding_dir / ckpt_file,
                            grounding_dir / "checkpoints" / ckpt_file
                        ]
                        for ckpt_path in ckpt_paths:
                            if ckpt_path.exists():
                                grounding_checkpoint = str(ckpt_path)
                                logger.info(f"  ✓ Found Grounding DINO checkpoint: {grounding_checkpoint}")
                                break
                        if grounding_checkpoint:
                            break
                                
                    # Look for config
                    config_paths = [
                        grounding_dir / "groundingdino" / "config" / "GroundingDINO_SwinB_cfg.py",
                        grounding_dir / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py",
                        grounding_dir / "config" / "GroundingDINO_SwinB_cfg.py"
                    ]
                    for cfg_path in config_paths:
                        if cfg_path.exists():
                            grounding_config = str(cfg_path)
                            break
                            
        return sam2_checkpoint, sam2_config, grounding_checkpoint, grounding_config
        
    def initialize_models(self):
        """Initialize Grounded-SAM2 models with automatic path finding."""
        try:
            # Fix Grounding DINO path
            grounding_paths = [
                Path.home() / "Grounded-SAM-2",
                Path.home() / "grounded-sam-2", 
                Path("/workspace") / "Grounded-SAM-2",
                Path.cwd() / "Grounded-SAM-2"
            ]
            
            for path in grounding_paths:
                if path.exists() and (path / "grounding_dino").exists():
                    sys.path.insert(0, str(path))
                    break
            
            # Import required modules
            from sam2.build_sam import build_sam2, build_sam2_video_predictor
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from sam2.utils.misc import get_sdpa_settings
            from grounding_dino.groundingdino.util.inference import load_model
            from grounding_dino.groundingdino.util.slconfig import SLConfig
            import grounding_dino.groundingdino.datasets.transforms as T
            from torchvision.ops import box_convert
            
            # Find model paths
            sam2_ckpt, sam2_cfg, grounding_ckpt, grounding_cfg = self.find_model_paths()
            
            if not all([sam2_ckpt, grounding_ckpt]):
                raise FileNotFoundError(
                    f"Could not find model files:\n"
                    f"  SAM2 checkpoint: {sam2_ckpt}\n"
                    f"  Grounding DINO checkpoint: {grounding_ckpt}\n"
                    f"Please check GROUNDED_SAM2_INSTALLATION.md"
                )
                
            logger.info(f"Loading SAM2 model: {self.sam2_model_name}")
            logger.info(f"  Checkpoint: {sam2_ckpt}")
            
            # Determine config name for SAM2
            if "b+" in self.sam2_model_name or "base_plus" in self.sam2_model_name:
                config_name = "sam2_hiera_b+.yaml"
            elif "tiny" in self.sam2_model_name:
                config_name = "sam2_hiera_t.yaml"
            elif "small" in self.sam2_model_name:
                config_name = "sam2_hiera_s.yaml"
            elif "large" in self.sam2_model_name:
                config_name = "sam2_hiera_l.yaml"
            else:
                config_name = "sam2_hiera_b+.yaml"
                
            logger.debug(f"  Config: {config_name}")
            
            # Configure SDPA settings before building SAM2
            # This sets the correct attention backend for your PyTorch/CUDA version
            old_sdpa_settings = get_sdpa_settings()
            
            # Log dtype configuration
            logger.info(f"Using dtype: {self.dtype_str} ({self.dtype})")
            
            # Initialize SAM2 in IMAGE MODE for keyframe segmentation
            # We don't need video tracking between keyframes
            sam2_model = build_sam2(
                config_file=config_name,
                ckpt_path=sam2_ckpt,
                device=self.device
            )
            
            # Convert model to the specified dtype
            sam2_model = sam2_model.to(dtype=self.dtype)
            
            # Fix: Ensure all buffers are properly converted (especially for float16/bfloat16)
            if self.dtype != torch.float32:
                logger.info(f"Converting all buffers to {self.dtype_str}")
                for module in sam2_model.modules():
                    for buffer_name, buffer in module.named_buffers(recurse=False):
                        if buffer.dtype == torch.float32:
                            module.register_buffer(buffer_name, buffer.to(self.dtype))
            
            # Log model dtype for debugging
            logger.info(f"SAM2 model dtype after conversion: {next(sam2_model.parameters()).dtype}")
            
            self.sam2_predictor = SAM2ImagePredictor(sam2_model)
            
            # For non-float32 dtypes, we need to handle input conversion
            if self.dtype != torch.float32:
                logger.info(f"Setting up dtype conversion for {self.dtype_str}")
                
                # Store original model.forward_image method
                original_forward_image = self.sam2_predictor.model.forward_image
                
                # Create a wrapper that converts inputs
                def forward_image_with_dtype(img):
                    # Convert input to model's dtype
                    img = img.to(dtype=self.dtype)
                    return original_forward_image(img)
                
                # Replace the method
                self.sam2_predictor.model.forward_image = forward_image_with_dtype
            
            logger.info(f"Loading Grounding DINO model: {self.grounding_model_name}")
            logger.info(f"  Checkpoint: {grounding_ckpt}")
            
            # Find Grounding DINO config if not provided
            if not grounding_cfg:
                # Try to find it in common locations
                config_paths = [
                    Path.home() / "Grounded-SAM-2" / "grounding_dino" / "groundingdino" / "config" / "GroundingDINO_SwinB_cfg.py",
                    Path.home() / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinB_cfg.py",
                    Path(grounding_ckpt).parent.parent / "groundingdino" / "config" / "GroundingDINO_SwinB_cfg.py",
                    Path(grounding_ckpt).parent.parent / "config" / "GroundingDINO_SwinB_cfg.py"
                ]
                
                for cfg_path in config_paths:
                    if cfg_path.exists():
                        grounding_cfg = str(cfg_path)
                        break
                        
            if grounding_cfg and Path(grounding_cfg).exists():
                logger.debug(f"  Config: {grounding_cfg}")
                # Initialize Grounding DINO - load_model expects config path as string
                self.grounding_dino = load_model(grounding_cfg, grounding_ckpt, device=self.device)
                self.grounding_dino.eval()
            else:
                logger.warning("Config file not found, trying to load without config")
                # Try to load without config (may fail)
                try:
                    self.grounding_dino = load_model(None, grounding_ckpt, device=self.device)
                    self.grounding_dino.eval()
                except:
                    raise FileNotFoundError(
                        "Grounding DINO config not found. Please ensure Grounded-SAM-2 is properly installed."
                    )
            
            # Initialize transform for Grounding DINO
            self.transform = T.Compose([
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            
            # Print success message
            model_info = self.model_selector.get_model_info(
                self.sam2_model_name, 
                self.grounding_model_name
            )
            logger.info("✅ Models loaded successfully!")
            logger.info(f"  Total parameters: {model_info['combined']['total_params']}")
            logger.info(f"  Expected FPS: {model_info['combined']['expected_fps']}")
            logger.info(f"  Total VRAM: {model_info['combined']['total_vram']}")
            
        except ImportError as e:
            logger.error(f"Required packages not installed: {e}")
            logger.error("Please follow GROUNDED_SAM2_INSTALLATION.md")
            logger.warning("Setting models to None to use fallback")
            self.grounding_dino = None
            self.sam2_predictor = None
        except FileNotFoundError as e:
            logger.error(f"Error: {e}")
            logger.warning("Setting models to None to use fallback")
            self.grounding_dino = None
            self.sam2_predictor = None
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            if logger.isEnabledFor(logging.DEBUG):
                import traceback
                traceback.print_exc()
            logger.warning("Setting models to None to use fallback")
            self.grounding_dino = None
            self.sam2_predictor = None
    
    def get_mask_tight_bbox(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Get tight bounding box around mask pixels.
        Returns: (x1, y1, x2, y2) in pixel coordinates
        """
        # Find all non-zero points in the mask
        if mask.dtype == bool:
            y_indices, x_indices = np.where(mask)
        else:
            y_indices, x_indices = np.where(mask > 0.5)
        
        if len(y_indices) == 0:
            return None
        
        # Get tight bounds
        x1 = int(x_indices.min())
        x2 = int(x_indices.max())
        y1 = int(y_indices.min())
        y2 = int(y_indices.max())
        
        return x1, y1, x2, y2
    
    def refine_box_with_mask(self, original_box: np.ndarray, mask: np.ndarray, 
                           refinement_threshold: float = 0.7) -> Tuple[np.ndarray, bool]:
        """Refine bounding box using SAM2 mask.
        
        Args:
            original_box: Original box from Grounding DINO [x1, y1, x2, y2]
            mask: Binary mask from SAM2
            refinement_threshold: If mask box is smaller than this ratio of original, refine it
            
        Returns:
            refined_box: Refined bounding box
            was_refined: Whether the box was refined
        """
        # Get tight bbox from mask
        mask_bbox = self.get_mask_tight_bbox(mask)
        if mask_bbox is None:
            return original_box, False
        
        x1_mask, y1_mask, x2_mask, y2_mask = mask_bbox
        x1_orig, y1_orig, x2_orig, y2_orig = original_box.astype(int)
        
        # Calculate areas
        orig_area = (x2_orig - x1_orig) * (y2_orig - y1_orig)
        mask_area = (x2_mask - x1_mask) * (y2_mask - y1_mask)
        
        # If mask bbox is significantly smaller than original, use it
        if mask_area < orig_area * refinement_threshold:
            # Add small padding to mask bbox
            padding = 5
            h, w = mask.shape
            x1_refined = max(0, x1_mask - padding)
            y1_refined = max(0, y1_mask - padding)
            x2_refined = min(w - 1, x2_mask + padding)
            y2_refined = min(h - 1, y2_mask + padding)
            
            refined_box = np.array([x1_refined, y1_refined, x2_refined, y2_refined])
            return refined_box, True
        
        return original_box, False
    
    def visualize_detections(self, image: np.ndarray, boxes: torch.Tensor, labels: List[str], 
                           scores: torch.Tensor, frame_idx: int, save_path: Path):
        """Visualize bounding boxes on image and save to file."""
        # Create a copy of the image to draw on
        vis_image = image.copy()
        h, w = image.shape[:2]
        
        # Define colors for different object types
        colors = {
            'window': (0, 0, 255),      # Red for window detections
            'default': (0, 255, 0),     # Green for normal detections
            'furniture': (255, 165, 0),  # Orange for furniture
            'structure': (128, 0, 128)   # Purple for walls/ceiling/floor
        }
        
        # Furniture and structure categories
        furniture = ['sofa', 'chair', 'table', 'cushion', 'blanket', 'lamp', 'cabinet']
        structure = ['wall', 'ceiling', 'floor', 'door']
        
        # Draw each box
        for box, label, score in zip(boxes, labels, scores):
            # Get color based on label
            if label == 'window':
                color = colors['window']
                thickness = 3  # Thicker line for window
            elif label in furniture:
                color = colors['furniture']
                thickness = 2
            elif label in structure:
                color = colors['structure']
                thickness = 2
            else:
                color = colors['default']
                thickness = 2
            
            # Convert box to int coordinates
            x1, y1, x2, y2 = box.int().tolist()
            
            # Draw rectangle
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)
            
            # Add label with confidence
            label_text = f"{label} ({score:.2f})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            
            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
            
            # Draw background rectangle for text
            cv2.rectangle(vis_image, (x1, y1 - text_height - 5), (x1 + text_width + 5, y1), color, -1)
            
            # Draw text
            cv2.putText(vis_image, label_text, (x1 + 2, y1 - 5), font, font_scale, (255, 255, 255), font_thickness)
            
            # Log large detections
            box_width = x2 - x1
            box_height = y2 - y1
            if box_width > w * 0.8 or box_height > h * 0.8:
                logger.info(f"Frame {frame_idx}: LARGE {label} detection: {box_width/w:.1%}x{box_height/h:.1%} of image")
        
        # Save image
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), vis_image)
        logger.info(f"Saved detection visualization to {save_path}")
        
        # Also save a text file with detection details
        text_path = save_path.with_suffix('.txt')
        with open(text_path, 'w') as f:
            f.write(f"Frame {frame_idx} detections:\n")
            f.write(f"Image size: {w}x{h}\n\n")
            for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
                x1, y1, x2, y2 = box.tolist()
                box_width = x2 - x1
                box_height = y2 - y1
                f.write(f"{i+1}. {label} (conf: {score:.3f})\n")
                f.write(f"   Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]\n")
                f.write(f"   Size: {box_width:.1f}x{box_height:.1f} ({box_width/w:.1%}x{box_height/h:.1%} of image)\n")
                f.write(f"   Center: ({(x1+x2)/2:.1f}, {(y1+y2)/2:.1f})\n\n")
            
    def ground_objects_in_frame(self, image: np.ndarray, frame_idx: int = -1) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        """Use Grounding DINO to detect objects based on text prompts."""
        if self.grounding_dino is None:
            logger.error("Grounding DINO is None - the model failed to load properly!")
            raise RuntimeError("Grounding DINO model not loaded. Check installation.")
            
        # Fix import path
        grounding_paths = [
            Path.home() / "Grounded-SAM-2",
            Path.home() / "grounded-sam-2", 
            Path("/workspace") / "Grounded-SAM-2"
        ]
        
        for path in grounding_paths:
            if path.exists() and (path / "grounding_dino").exists():
                sys.path.insert(0, str(path))
                break
                
        from grounding_dino.groundingdino.util.inference import predict
        from PIL import Image
        
        # Ensure image is uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Convert numpy array to PIL Image
        image_pil = Image.fromarray(image)
        
        # Apply transforms
        image_transformed, _ = self.transform(image_pil, None)
        
        # Single-object detection for accurate vocabulary matching
        if self.debug_mode:
            logger.info(f"Frame {frame_idx}: Processing {len(self.vocabulary)} object types...")
            # Log detection parameters
            text_thresh = config.get("semantic_segmentation", {}).get("grounded_sam2", {}).get("text_threshold", 0.25)
            logger.info(f"Detection parameters: box_threshold={self.confidence_threshold}, text_threshold={text_thresh}")
        
        all_boxes = []
        all_labels = []
        all_scores = []
        
        # Process each vocabulary word separately to ensure exact matches
        for i, vocab_word in enumerate(self.vocabulary):
            # Use simple prompts - we'll refine boxes using SAM2 masks
            caption = f"a {vocab_word}"
            
            # Special logging for problematic objects
            if vocab_word in ['sofa', 'blanket', 'cushion', 'chair', 'table']:
                logger.info(f"Frame {frame_idx}: INVESTIGATING '{vocab_word}' detection with caption: '{caption}'")
                
                # Test alternative prompts for debugging
                if vocab_word == 'blanket' and self.debug_mode:
                    test_prompts = [
                        "blanket.",
                        "a blanket",
                        "a blanket only",
                        "blanket on sofa",
                        "small blanket"
                    ]
                    logger.info(f"  Testing multiple prompts for '{vocab_word}':")
                    for test_caption in test_prompts:
                        with torch.no_grad():
                            test_boxes, test_logits, test_phrases = predict(
                                model=self.grounding_dino,
                                image=image_transformed,
                                caption=test_caption,
                                box_threshold=self.confidence_threshold,
                                text_threshold=config.get("semantic_segmentation", {}).get("grounded_sam2", {}).get("text_threshold", 0.25),
                                device=self.device
                            )
                        if len(test_boxes) > 0:
                            logger.info(f"    Prompt '{test_caption}': {len(test_boxes)} detections")
                            for j, (box, score, phrase) in enumerate(zip(test_boxes, test_logits, test_phrases)):
                                box_norm = box.tolist()
                                logger.info(f"      {j}: phrase='{phrase}', score={score:.3f}, size=({box_norm[2]:.3f}, {box_norm[3]:.3f})")
                        else:
                            logger.info(f"    Prompt '{test_caption}': No detections")
            
            with torch.no_grad():
                boxes, logits, phrases = predict(
                    model=self.grounding_dino,
                    image=image_transformed,
                    caption=caption,
                    box_threshold=self.confidence_threshold,
                    text_threshold=config.get("semantic_segmentation", {}).get("grounded_sam2", {}).get("text_threshold", 0.25),
                    device=self.device
                )
            
            # Log raw detections before filtering
            if len(boxes) > 0:
                if vocab_word in ['sofa', 'blanket', 'cushion', 'chair', 'table', 'window']:
                    logger.info(f"  Raw detections for '{vocab_word}':")
                    for j, (box, score, phrase) in enumerate(zip(boxes, logits, phrases)):
                        logger.info(f"    {j}: phrase='{phrase}', score={score:.3f}, box={box.tolist()}")
                
                # Save debug visualization for problematic objects
                if self.save_debug_visualizations and vocab_word in ['sofa', 'blanket', 'cushion', 'chair', 'table', 'window']:
                    debug_img = image.copy()
                    h, w = image.shape[:2]
                    for j, (box, score, phrase) in enumerate(zip(boxes, logits, phrases)):
                        # Convert from cxcywh to xyxy
                        cx, cy, bw, bh = box.tolist()
                        x1 = int((cx - bw/2) * w)
                        y1 = int((cy - bh/2) * h)
                        x2 = int((cx + bw/2) * w)
                        y2 = int((cy + bh/2) * h)
                        
                        # Draw box
                        color = (0, 255, 0) if phrase.strip().lower() == vocab_word.lower() else (0, 0, 255)
                        cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
                        
                        # Add label
                        label = f"{phrase} ({score:.2f})"
                        cv2.putText(debug_img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Save debug image
                    debug_path = DEBUG_OUTPUT_DIR / f"frame_{frame_idx}_grounding_{vocab_word}_raw.jpg"
                    cv2.imwrite(str(debug_path), debug_img)
                    logger.info(f"  Saved raw Grounding DINO detections to {debug_path}")
            else:
                if vocab_word in ['sofa', 'blanket', 'cushion', 'chair', 'table']:
                    logger.warning(f"  NO detections for '{vocab_word}' at all! (box_threshold={self.confidence_threshold})")
            
            # Keep detections that match the vocabulary word (exact or partial)
            for box, score, phrase in zip(boxes, logits, phrases):
                # Clean and normalize the detected phrase
                detected_word = phrase.strip().lower()
                vocab_word_clean = vocab_word.lower().strip()
                
                # Log all detections first for debugging
                if self.debug_mode:
                    logger.debug(f"  Checking detection: '{phrase}' (score={score:.3f}) against vocab word '{vocab_word}'")
                
                # Accept exact matches
                if detected_word == vocab_word_clean:
                    all_boxes.append(box)
                    all_labels.append(vocab_word)  # Use original vocabulary word
                    all_scores.append(score)
                    if self.debug_mode:
                        logger.info(f"  ACCEPTED (exact match): {vocab_word} with confidence {score:.3f}")
                        # Log box coordinates in normalized form
                        box_norm = box.tolist()
                        logger.debug(f"    Box (normalized): center=({box_norm[0]:.3f}, {box_norm[1]:.3f}), size=({box_norm[2]:.3f}, {box_norm[3]:.3f})")
                    # Update statistics
                    self.class_statistics[vocab_word]['detections'] += 1
                    self.class_statistics[vocab_word]['confidences'].append(float(score))
                # Accept partial matches where vocab word is in the detected phrase
                elif vocab_word_clean in detected_word:
                    # Additional checks to prevent false matches
                    # e.g., don't match "chair" in "wheelchair" or "table" in "vegetable"
                    word_boundaries = [' ', '-', '_', ',', '.', '/', '\\']
                    
                    # Check if vocab word appears as a separate word in the phrase
                    is_valid_partial = False
                    
                    # Check at start of phrase
                    if detected_word.startswith(vocab_word_clean + ' ') or detected_word.startswith(vocab_word_clean + ','):
                        is_valid_partial = True
                    # Check at end of phrase
                    elif detected_word.endswith(' ' + vocab_word_clean) or detected_word.endswith(',' + vocab_word_clean):
                        is_valid_partial = True
                    # Check in middle with word boundaries
                    else:
                        for boundary1 in word_boundaries:
                            for boundary2 in word_boundaries:
                                if boundary1 + vocab_word_clean + boundary2 in detected_word:
                                    is_valid_partial = True
                                    break
                            if is_valid_partial:
                                break
                    
                    # Special case: if detected phrase is just vocab word + common descriptors
                    if detected_word in [f"a {vocab_word_clean}", f"an {vocab_word_clean}", f"{vocab_word_clean} only"]:
                        is_valid_partial = True
                    
                    if is_valid_partial:
                        all_boxes.append(box)
                        all_labels.append(vocab_word)  # Use original vocabulary word
                        all_scores.append(score)
                        if self.debug_mode:
                            logger.info(f"  ACCEPTED (partial match): '{phrase}' contains '{vocab_word}' with confidence {score:.3f}")
                        # Update statistics
                        self.class_statistics[vocab_word]['detections'] += 1
                        self.class_statistics[vocab_word]['confidences'].append(float(score))
                        # Log box coordinates
                        box_norm = box.tolist()
                        logger.debug(f"    Box (normalized): center=({box_norm[0]:.3f}, {box_norm[1]:.3f}), size=({box_norm[2]:.3f}, {box_norm[3]:.3f})")
                    else:
                        logger.info(f"  REJECTED (invalid partial match): '{phrase}' contains '{vocab_word}' but not as separate word")
                else:
                    logger.debug(f"  REJECTED (no match): '{phrase}' doesn't contain '{vocab_word}'")
        
        # Convert to tensors
        if len(all_boxes) > 0:
            boxes = torch.stack(all_boxes)
            logits = torch.tensor(all_scores)
            phrases = all_labels
            
            # No size-based filtering - process all detections
            logger.info(f"Frame {frame_idx}: Processing all {len(boxes)} detections without size filtering")
            
            # CRITICAL: Deduplicate boxes with high IoU
            # Many objects get detected with nearly identical boxes
            logger.info(f"Frame {frame_idx}: Deduplicating {len(boxes)} detections...")
            unique_indices = []
            
            for i in range(len(boxes)):
                is_duplicate = False
                box_i = boxes[i]
                
                # Check against previously selected boxes
                for j in unique_indices:
                    box_j = boxes[j]
                    
                    # Calculate IoU between normalized boxes
                    # Boxes are in cxcywh format
                    x1_i = box_i[0] - box_i[2]/2
                    y1_i = box_i[1] - box_i[3]/2
                    x2_i = box_i[0] + box_i[2]/2
                    y2_i = box_i[1] + box_i[3]/2
                    
                    x1_j = box_j[0] - box_j[2]/2
                    y1_j = box_j[1] - box_j[3]/2
                    x2_j = box_j[0] + box_j[2]/2
                    y2_j = box_j[1] + box_j[3]/2
                    
                    # Intersection
                    x1 = torch.max(x1_i, x1_j)
                    y1 = torch.max(y1_i, y1_j)
                    x2 = torch.min(x2_i, x2_j)
                    y2 = torch.min(y2_i, y2_j)
                    
                    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
                    
                    # Union
                    area_i = box_i[2] * box_i[3]
                    area_j = box_j[2] * box_j[3]
                    union = area_i + area_j - intersection
                    
                    iou = intersection / (union + 1e-6)
                    
                    # If IoU > threshold, consider it a duplicate
                    if iou > self.deduplication_iou_threshold:
                        # Keep the one with higher confidence
                        if logits[i] <= logits[j]:
                            is_duplicate = True
                            logger.info(f"  Removing duplicate: {phrases[i]} (IoU={iou:.2f} with {phrases[j]})")
                            break
                        else:
                            # Remove the previous one and keep this one
                            unique_indices.remove(j)
                            logger.info(f"  Replacing {phrases[j]} with higher confidence {phrases[i]}")
                
                if not is_duplicate:
                    unique_indices.append(i)
            
            # Keep only unique detections
            if len(unique_indices) < len(boxes):
                logger.info(f"Frame {frame_idx}: Reduced from {len(boxes)} to {len(unique_indices)} unique detections")
                boxes = boxes[unique_indices]
                logits = logits[unique_indices]
                phrases = [phrases[i] for i in unique_indices]
        else:
            boxes = torch.zeros((0, 4))
            logits = torch.zeros(0)
            phrases = []
        
        logger.info(f"Frame {frame_idx}: Total detected: {len(boxes)} objects from vocabulary")
        if len(boxes) > 0:
            # Create summary of detections
            from collections import Counter
            label_counts = Counter(phrases)
            summary = ', '.join([f'{label}({count})' for label, count in label_counts.items()])
            logger.info(f"Frame {frame_idx}: Detection summary: {summary}")
        
        # Convert boxes from normalized to pixel coordinates
        h, w = image.shape[:2]
        
        # Check box format - Grounding DINO returns cxcywh format, need to convert to xyxy
        logger.debug(f"Frame {frame_idx}: Raw box format (first box): {boxes[0] if len(boxes) > 0 else 'None'}")
        
        # Convert from cxcywh to xyxy format
        boxes_xyxy = torch.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1 = cx - w/2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1 = cy - h/2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2 = cx + w/2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2 = cy + h/2
        
        # Scale to pixel coordinates
        boxes_xyxy = boxes_xyxy * torch.tensor([w, h, w, h], device=boxes.device)
        
        # Log diagnostic information about box sizes
        if len(boxes_xyxy) > 0:
            for i, (box, label, score) in enumerate(zip(boxes_xyxy, phrases, logits)):
                box_width = box[2] - box[0]
                box_height = box[3] - box[1]
                width_ratio = box_width / w
                height_ratio = box_height / h
                logger.debug(f"Frame {frame_idx}: {label} box {i}: size={box_width:.0f}x{box_height:.0f} ({width_ratio:.1%}x{height_ratio:.1%} of image), conf={score:.3f}")
                if width_ratio > 0.8 or height_ratio > 0.8:
                    logger.warning(f"Frame {frame_idx}: Large detection for '{label}': {width_ratio:.1%}x{height_ratio:.1%} of image")
        
        # Save visualization if debug mode is enabled
        if self.save_debug_visualizations:
            # Save original image
            DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            orig_path = DEBUG_OUTPUT_DIR / f"frame_{frame_idx}_original.jpg"
            cv2.imwrite(str(orig_path), image)
            
            # Save detection visualization if there are detections
            if len(boxes_xyxy) > 0:
                vis_path = DEBUG_OUTPUT_DIR / f"frame_{frame_idx}_detections.jpg"
                self.visualize_detections(image, boxes_xyxy, phrases, logits, frame_idx, vis_path)
            else:
                logger.info(f"Frame {frame_idx}: No detections found")
        
        return boxes_xyxy, phrases, logits
    
    def segment_frame_with_boxes(self, image: np.ndarray, boxes: torch.Tensor, 
                                 labels: List[str], frame_idx: int) -> Dict:
        """Use SAM2 IMAGE MODE to segment objects given bounding boxes."""
        if self.sam2_predictor is None:
            logger.error("SAM2 predictor is None - the model failed to load properly!")
            raise RuntimeError("SAM2 model not loaded. Check installation.")
            
        # Ensure image is uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
                
        # SAM2 expects RGB image
        if image.shape[2] == 3 and image.dtype == np.uint8:
            # Assume BGR, convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        # Set the image in the predictor (required for image mode)
        logger.debug(f"Setting image in SAM2 predictor")
        # Convert image to model's dtype to avoid mismatch
        with torch.no_grad():
            self.sam2_predictor.set_image(image_rgb)
        self.current_image_set = True
        
        # Process all boxes at once
        masks = []
        track_ids = []
        valid_labels = []  # Track which labels actually got valid masks
        
        logger.debug(f"Frame {frame_idx}: Processing {len(boxes)} detected boxes for segmentation")
        
        # Convert all boxes to numpy array
        if len(boxes) > 0:
            # Boxes should already be in xyxy pixel coordinates
            input_boxes = boxes.cpu().numpy()
            logger.debug(f"Frame {frame_idx}: Input boxes shape: {input_boxes.shape}")
            logger.debug(f"Frame {frame_idx}: First box: {input_boxes[0] if len(input_boxes) > 0 else 'None'}")
            
            # Process boxes individually to avoid overlapping masks
            logger.debug(f"Frame {frame_idx}: Running SAM2 predict individually for {len(input_boxes)} boxes")
            predicted_masks = []
            scores = []
            logits_list = []
            
            try:
                # Process each box separately for better segmentation
                for box_idx, box in enumerate(input_boxes):
                    label = labels[box_idx] if box_idx < len(labels) else "unknown"
                    logger.debug(f"Frame {frame_idx}: Processing box {box_idx} for '{label}'")
                    
                    # SAM2 expects box as (1, 4) array for single box prediction
                    single_box = box.reshape(1, 4)
                    
                    # Predict mask for single box
                    mask, score, logit = self.sam2_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=single_box,
                        multimask_output=False  # Single mask per box
                    )
                    
                    predicted_masks.append(mask[0])  # mask is (1, C, H, W), take first
                    scores.append(score[0])  # score is (1,), take first
                    logits_list.append(logit[0])  # logit is (1, C, H, W), take first
                
                # Convert lists to arrays for consistency
                predicted_masks = np.array(predicted_masks)
                scores = np.array(scores)
                logits = np.array(logits_list)
                
                logger.debug(f"Frame {frame_idx}: SAM2 output: masks shape={predicted_masks.shape}, scores shape={scores.shape}")
                
                # CRITICAL: Check if SAM2 returned the same number of masks as input boxes
                num_boxes = len(input_boxes)
                num_masks = len(predicted_masks)
                num_labels = len(labels)
                
                logger.info(f"Frame {frame_idx}: Input boxes: {num_boxes}, Output masks: {num_masks}, Labels: {num_labels}")
                
                if num_masks != num_boxes:
                    logger.error(f"Frame {frame_idx}: MISMATCH! SAM2 returned {num_masks} masks for {num_boxes} boxes!")
                    logger.error(f"Labels provided: {labels}")
                    logger.error("This will cause some objects to be skipped!")
                
                # Process each mask - use enumerate on masks to ensure we process all returned masks
                for box_idx, mask in enumerate(predicted_masks):
                    if box_idx >= len(labels):
                        logger.error(f"Frame {frame_idx}: Mask {box_idx} has no corresponding label! Skipping.")
                        continue
                    
                    label = labels[box_idx]
                    score = scores[box_idx] if box_idx < len(scores) else 0.0
                    # Score might be a scalar or array
                    score_val = float(score[0] if hasattr(score, '__len__') else score)
                    # Processing mask
                    
                    # mask is (C, H, W) with C=1, need to squeeze
                    logger.debug(f"Frame {frame_idx}, {label}: Mask shape before squeeze: {mask.shape}")
                    if mask.ndim == 3 and mask.shape[0] == 1:
                        mask = mask[0]  # Remove channel dimension
                    logger.debug(f"Frame {frame_idx}, {label}: Mask shape after squeeze: {mask.shape}")
                    
                    mask_pixels = np.sum(mask)
                    mask_total = mask.shape[0] * mask.shape[1]
                    mask_coverage = mask_pixels / mask_total
                    logger.debug(f"Frame {frame_idx}, {label}: Mask coverage: {mask_coverage:.1%} ({mask_pixels}/{mask_total} pixels)")
                    
                    # Validate mask
                    if mask_pixels == 0:
                        logger.error(f"Frame {frame_idx}, {label}: EMPTY MASK! SAM2 failed to generate a mask for this object.")
                        continue  # Skip empty masks
                    
                    if mask_coverage < 0.001:  # Less than 0.1% coverage
                        logger.warning(f"Frame {frame_idx}, {label}: Very small mask ({mask_pixels} pixels). May be invalid.")
                    
                    if mask_coverage > 0.8:
                        logger.warning(f"Frame {frame_idx}: HIGH COVERAGE MASK for '{label}': {mask_coverage:.1%} of image!")
                    
                    # MASK-BASED BOX REFINEMENT
                    original_box = input_boxes[box_idx]
                    refined_box, was_refined = self.refine_box_with_mask(original_box, mask)
                    
                    if was_refined:
                        logger.info(f"Frame {frame_idx}, {label}: Refined box from {original_box.astype(int).tolist()} to {refined_box.tolist()}")
                        # Calculate refinement ratio
                        orig_area = (original_box[2] - original_box[0]) * (original_box[3] - original_box[1])
                        refined_area = (refined_box[2] - refined_box[0]) * (refined_box[3] - refined_box[1])
                        logger.info(f"  Box area reduced to {refined_area/orig_area:.1%} of original")
                        
                        # Update the box for visualization
                        input_boxes[box_idx] = refined_box
                    
                    # Save mask visualization for debugging - save ALL masks during debug
                    if self.save_debug_visualizations:
                        # Save raw mask
                        mask_vis_path = DEBUG_OUTPUT_DIR / f"frame_{frame_idx}_mask_{label}_{box_idx}.jpg"
                        mask_uint8 = (mask * 255).astype(np.uint8) if mask.dtype == bool else ((mask > 0.5) * 255).astype(np.uint8)
                        cv2.imwrite(str(mask_vis_path), mask_uint8)
                        
                        # Save mask overlay on image
                        overlay_path = DEBUG_OUTPUT_DIR / f"frame_{frame_idx}_overlay_{label}_{box_idx}.jpg"
                        overlay = image.copy()
                        mask_bool = mask > 0.5 if mask.dtype != bool else mask
                        # Color the mask region in red with transparency
                        overlay[mask_bool] = overlay[mask_bool] * 0.5 + np.array([0, 0, 255]) * 0.5
                        
                        # Draw both original and refined boxes
                        if was_refined:
                            # Draw original box in yellow (dashed)
                            x1_orig, y1_orig, x2_orig, y2_orig = original_box.astype(int)
                            # Draw dashed rectangle by drawing multiple small rectangles
                            for i in range(0, x2_orig - x1_orig, 10):
                                cv2.line(overlay, (x1_orig + i, y1_orig), (x1_orig + min(i + 5, x2_orig - x1_orig), y1_orig), (0, 255, 255), 2)
                                cv2.line(overlay, (x1_orig + i, y2_orig), (x1_orig + min(i + 5, x2_orig - x1_orig), y2_orig), (0, 255, 255), 2)
                            for i in range(0, y2_orig - y1_orig, 10):
                                cv2.line(overlay, (x1_orig, y1_orig + i), (x1_orig, y1_orig + min(i + 5, y2_orig - y1_orig)), (0, 255, 255), 2)
                                cv2.line(overlay, (x2_orig, y1_orig + i), (x2_orig, y1_orig + min(i + 5, y2_orig - y1_orig)), (0, 255, 255), 2)
                            cv2.putText(overlay, "original", (x1_orig, y1_orig-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        
                        # Draw refined/final box in green
                        x1, y1, x2, y2 = input_boxes[box_idx].astype(int)
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        box_label = f"{label} (refined)" if was_refined else f"{label}"
                        cv2.putText(overlay, box_label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.imwrite(str(overlay_path), overlay)
                        
                        logger.info(f"Saved {label} mask and overlay to {mask_vis_path} and {overlay_path}")
                    
                    # Check mask bounds
                    try:
                        # Convert to binary mask if needed
                        if mask.dtype != bool:
                            mask_binary = mask > 0.5
                        else:
                            mask_binary = mask
                        
                        y_indices, x_indices = np.where(mask_binary)
                        if len(y_indices) > 0:
                            min_y, max_y = y_indices.min(), y_indices.max()
                            min_x, max_x = x_indices.min(), x_indices.max()
                            logger.debug(f"Frame {frame_idx}, {label}: Mask bounds: x=[{min_x}, {max_x}], y=[{min_y}, {max_y}], size={max_x-min_x+1}x{max_y-min_y+1}")
                    except Exception as e:
                        logger.debug(f"Frame {frame_idx}: Error checking mask bounds: {e}")
                    
                    masks.append(mask)
                    valid_labels.append(label)  # Add label only for valid masks
                    
                    # Update mask statistics
                    self.class_statistics[label]['successful_masks'] += 1
                    mask_size = mask_pixels if isinstance(mask_pixels, (int, float)) else mask_pixels.item()
                    if self.class_statistics[label]['avg_mask_size'] == 0:
                        self.class_statistics[label]['avg_mask_size'] = mask_size
                    else:
                        # Running average
                        n = self.class_statistics[label]['successful_masks']
                        self.class_statistics[label]['avg_mask_size'] = (
                            (self.class_statistics[label]['avg_mask_size'] * (n-1) + mask_size) / n
                        )
                    
                    # Generate track ID
                    self.current_tracks[self.next_track_id] = {
                        'label': label,
                        'first_frame': frame_idx,
                        'box': input_boxes[box_idx],
                        'score': score_val
                    }
                    track_ids.append(self.next_track_id)
                    self.next_track_id += 1
                    
            except Exception as e:
                logger.error(f"ERROR in SAM2 predict: {e}")
                import traceback
                traceback.print_exc()
                # Fallback to box masks
                for box_idx, (box, label) in enumerate(zip(input_boxes, labels)):
                    h, w = image.shape[:2]
                    mask = np.zeros((h, w), dtype=bool)
                    x1, y1, x2, y2 = box.astype(int)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    mask[y1:y2, x1:x2] = True
                    masks.append(mask)
                    valid_labels.append(label)
                    track_ids.append(self.next_track_id)
                    self.next_track_id += 1
        
        # Log summary of mask processing
        logger.info(f"Frame {frame_idx}: Processed {len(masks)} valid masks out of {len(boxes)} input boxes")
        if len(masks) < len(boxes):
            logger.warning(f"Frame {frame_idx}: Some objects were skipped! Only {len(masks)}/{len(boxes)} masks were generated.")
            logger.warning(f"Frame {frame_idx}: Labels with masks: {valid_labels}")
            missing_labels = [label for label in labels if label not in valid_labels]
            if missing_labels:
                logger.error(f"Frame {frame_idx}: MISSING MASKS for: {missing_labels}")
                
        return {
            'masks': masks,
            'labels': valid_labels,  # Return only labels that have valid masks
            'track_ids': track_ids,
            'refined_boxes': input_boxes if len(boxes) > 0 else []  # Include refined boxes for reference
        }
    
    def process_frame(self, image: np.ndarray, frame_id: int, keyframe_idx: Optional[int] = None) -> Dict:
        """Process a single frame to generate semantic masks."""
        start_time = time.time()
        h, w = image.shape[:2]
        
        
        # Ground objects in the frame
        boxes, labels, confidences = self.ground_objects_in_frame(image, frame_id)
        
        if len(boxes) == 0:
            return {
                'frame_id': frame_id,
                'keyframe_idx': keyframe_idx,
                'masks_rle': {},
                'instance_ids': [],
                'track_ids': {},
                'labels': {},
                'confidences': {},
                'processing_time': time.time() - start_time
            }
        
        # Segment with SAM2
        segmentation_result = self.segment_frame_with_boxes(
            image, boxes, labels, frame_id
        )
        
        # Check for overlapping masks (diagnostic only)
        if len(segmentation_result['masks']) > 1:
            overlap_matrix = np.zeros((len(segmentation_result['masks']), len(segmentation_result['masks'])))
            for i, mask1 in enumerate(segmentation_result['masks']):
                for j, mask2 in enumerate(segmentation_result['masks']):
                    if i < j:
                        # Ensure masks are boolean arrays
                        mask1_bool = mask1.astype(bool) if mask1.dtype != bool else mask1
                        mask2_bool = mask2.astype(bool) if mask2.dtype != bool else mask2
                        overlap = np.sum(mask1_bool & mask2_bool)
                        total = np.sum(mask1_bool | mask2_bool)
                        if total > 0:
                            iou = overlap / total
                            if iou > 0.5:
                                logger.warning(f"Frame {frame_id}: High overlap (IoU={iou:.2f}) between '{segmentation_result['labels'][i]}' and '{segmentation_result['labels'][j]}'")
        
        # Convert to RLE format
        results = {
            'frame_id': frame_id,
            'keyframe_idx': keyframe_idx,  # Direct mapping to keyframe
            'masks_rle': {},
            'instance_ids': [],
            'track_ids': {},
            'labels': {},
            'confidences': {},
            'processing_time': time.time() - start_time
        }
        
        for idx, (mask, label, conf, track_id) in enumerate(zip(
            segmentation_result['masks'],
            segmentation_result['labels'],
            confidences,
            segmentation_result['track_ids']
        )):
            instance_id = idx + 1
            
            # Encode mask to RLE
            mask_tensor = torch.from_numpy(mask).to(dtype=torch.bool)
            rle_mask = encode_rle(mask_tensor)
            
            # Store results
            results['masks_rle'][instance_id] = rle_mask
            results['instance_ids'].append(instance_id)
            results['track_ids'][instance_id] = track_id
            results['labels'][instance_id] = label
            results['confidences'][instance_id] = float(conf)
            
        # Track performance
        self.frame_times.append(results['processing_time'])
        if len(self.frame_times) % 30 == 0:
            avg_time = np.mean(self.frame_times[-30:])
            logger.info(f"Semantic processing: {avg_time*1000:.1f}ms/frame ({1/avg_time:.1f} FPS)")
            
        return results
    
    
    def export_statistics(self):
        """Export per-class statistics for evaluation."""
        stats = {}
        for vocab, data in self.class_statistics.items():
            if data['detections'] > 0:
                # Calculate average confidence
                avg_conf = np.mean(data['confidences']) if data['confidences'] else 0.0
                stats[vocab] = {
                    'detections': data['detections'],
                    'successful_masks': data['successful_masks'],
                    'detection_success_rate': data['successful_masks'] / data['detections'] if data['detections'] > 0 else 0,
                    'avg_confidence': avg_conf,
                    'avg_mask_size_pixels': data['avg_mask_size']
                }
        
        # Save to file if debug mode
        if self.debug_mode:
            import json
            DEBUG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            stats_file = DEBUG_OUTPUT_DIR / "class_statistics.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Saved class statistics to {stats_file}")
        
        return stats
    
    def run(self):
        """Main processing loop."""
        logger.info(f"Starting Real Grounded-SAM2 semantic processor on {self.device}...")
        self.initialize_models()
        
        while True:
            try:
                frame_data = self.frame_queue.get(timeout=0.1)
                
                if frame_data is None:
                    break
                
                image = frame_data['img']
                frame_id = frame_data['frame_id']
                keyframe_idx = frame_data.get('keyframe_idx')  # Direct keyframe mapping
                
                # Process frame
                semantic_data = self.process_frame(image, frame_id, keyframe_idx)
                
                # Put results in output queue
                self.result_queue.put(semantic_data)
                
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in semantic processor: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Export statistics before shutting down
        self.export_statistics()
        
        logger.info("Semantic processor terminated.")
        if self.frame_times:
            logger.info(f"Average processing time: {np.mean(self.frame_times)*1000:.1f}ms")


def start_real_grounded_sam2_processor(frame_queue: mp.Queue, 
                                      result_queue: mp.Queue,
                                      vocabulary: List[str],
                                      device: str = "cuda:1",
                                      model_selector: Optional[GroundedSAM2ModelSelector] = None,
                                      confidence_threshold: float = 0.35,
                                      **kwargs) -> mp.Process:
    """Start the real Grounded-SAM2 processor in a separate process."""
    processor = RealGroundedSAM2Processor(
        frame_queue, result_queue, vocabulary, device, model_selector, 
        confidence_threshold=confidence_threshold, **kwargs
    )
    process = mp.Process(target=processor.run)
    process.start()
    return process