"""Real Grounded-SAM2 processor implementation - Reduced version."""

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

# Debug visualization directory
DEBUG_OUTPUT_DIR = Path("debug_semantic_pipeline")


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
                 mask_refinement_threshold: float = 0.7,
                 filter_empty_labels: bool = True,
                 min_phrase_length: int = 2,
                 empty_label_max_size: float = 0.4,
                 object_tracker = None):
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.vocabulary = vocabulary
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.debug_mode = debug_mode
        self.save_debug_visualizations = save_debug_visualizations
        self.deduplication_iou_threshold = deduplication_iou_threshold
        self.mask_refinement_threshold = mask_refinement_threshold
        self.filter_empty_labels = filter_empty_labels
        self.min_phrase_length = min_phrase_length
        self.empty_label_max_size = empty_label_max_size
        
        # Convert dtype string to torch dtype
        self.dtype = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}.get(dtype, torch.bfloat16)
        
        # Model directories
        self.sam2_checkpoint_dir = sam2_checkpoint_dir
        self.grounding_dino_checkpoint_dir = grounding_dino_checkpoint_dir
        
        # Model selection
        self.model_selector = model_selector or GroundedSAM2ModelSelector(target_fps=15, max_vram_gb=10, quality_priority="quality")
        self.sam2_model_name = getattr(self.model_selector, 'sam2_model', None) or self.model_selector.select_models()[0]
        self.grounding_model_name = getattr(self.model_selector, 'grounding_model', None) or self.model_selector.select_models()[1]
        
        # Model instances
        self.sam2_predictor = None
        self.grounding_dino = None
        self.transform = None
        
        # Frame processing times
        self.frame_times = []
        
        # Debug counters
        self.debug_frame_counter = 0
        
        # Object tracker (optional)
        self.object_tracker = object_tracker
        if self.object_tracker is not None:
            logger.info(f"GroundedSAM2Processor initialized with object tracker: {type(object_tracker).__name__}")
        else:
            logger.info("GroundedSAM2Processor initialized without object tracker")
        
        # Keyframe access callback (will be set by processor)
        self.keyframe_access_callback = None
        
    def visualize_detections(self, image: np.ndarray, boxes: torch.Tensor, labels: List[str], 
                             scores: torch.Tensor, frame_idx: int, stage: str = "grounding"):
        """Visualize bounding boxes on image and save to file."""
        if not self.save_debug_visualizations:
            return
            
        # Create debug directory
        debug_dir = DEBUG_OUTPUT_DIR / f"frame_{frame_idx:06d}"
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a copy of the image
        vis_image = image.copy()
        h, w = image.shape[:2]
        
        # Define colors for different stages
        colors = {
            'grounding': (0, 255, 0),     # Green for Grounding DINO detections
            'deduped': (255, 165, 0),     # Orange for after deduplication
            'final': (0, 0, 255)          # Red for final detections
        }
        color = colors.get(stage, (255, 255, 255))
        
        # Draw each box
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box.int().tolist()
            
            # Draw rectangle
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Add label with confidence
            label_text = f"{label} ({score:.2f})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            
            # Get text size for background
            (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, font_thickness)
            
            # Draw background rectangle for text
            cv2.rectangle(vis_image, (x1, y1 - text_height - 5), (x1 + text_width + 5, y1), color, -1)
            
            # Draw text
            cv2.putText(vis_image, label_text, (x1 + 2, y1 - 5), font, font_scale, (255, 255, 255), font_thickness)
        
        # Add stage info
        cv2.putText(vis_image, f"Stage: {stage} | Total: {len(boxes)} detections", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Save image
        save_path = debug_dir / f"{stage}_detections.jpg"
        cv2.imwrite(str(save_path), vis_image)
        logger.info(f"Saved {stage} visualization to {save_path}")
        
        # Also save detection details to text file
        text_path = debug_dir / f"{stage}_details.txt"
        with open(text_path, 'w') as f:
            f.write(f"Frame {frame_idx} - {stage} detections\n")
            f.write(f"Total detections: {len(boxes)}\n")
            f.write(f"Confidence threshold: {self.confidence_threshold}\n\n")
            for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
                x1, y1, x2, y2 = box.tolist()
                box_width = x2 - x1
                box_height = y2 - y1
                f.write(f"{i+1}. {label} (conf: {score:.3f})\n")
                f.write(f"   Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]\n")
                f.write(f"   Size: {box_width:.1f}x{box_height:.1f} ({box_width/w*100:.1f}%x{box_height/h*100:.1f}% of image)\n\n")
    
    def visualize_masks(self, image: np.ndarray, masks: List[np.ndarray], labels: List[str], 
                       frame_idx: int):
        """Visualize segmentation masks and save to file."""
        if not self.save_debug_visualizations:
            return
            
        debug_dir = DEBUG_OUTPUT_DIR / f"frame_{frame_idx:06d}"
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Create overlay image
        overlay = image.copy()
        h, w = image.shape[:2]
        
        # Create separate mask for each instance
        for idx, (mask, label) in enumerate(zip(masks, labels)):
            # Create colored mask
            mask_color = np.zeros_like(image)
            color = np.array([
                (idx * 67) % 255,
                (idx * 131) % 255,
                (idx * 193) % 255
            ])
            mask_color[mask > 0] = color
            
            # Apply to overlay
            overlay[mask > 0] = overlay[mask > 0] * 0.5 + mask_color[mask > 0] * 0.5
            
            # Save individual mask
            mask_path = debug_dir / f"mask_{idx:02d}_{label}.png"
            cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))
        
        # Add labels
        for idx, (mask, label) in enumerate(zip(masks, labels)):
            # Find mask center
            mask_indices = np.where(mask > 0)
            if len(mask_indices[0]) > 0:
                cy = int(np.mean(mask_indices[0]))
                cx = int(np.mean(mask_indices[1]))
                cv2.putText(overlay, label, (cx-20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save overlay
        overlay_path = debug_dir / "segmentation_overlay.jpg"
        cv2.imwrite(str(overlay_path), overlay)
        
        # Save summary
        summary_path = debug_dir / "segmentation_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Frame {frame_idx} - Segmentation Results\n")
            f.write(f"Total masks: {len(masks)}\n\n")
            for idx, (mask, label) in enumerate(zip(masks, labels)):
                mask_pixels = np.sum(mask > 0)
                mask_percentage = (mask_pixels / (h * w)) * 100
                f.write(f"{idx+1}. {label}: {mask_pixels} pixels ({mask_percentage:.2f}% of image)\n")
    
    def find_model_paths(self) -> Tuple[str, str, str, str]:
        """Automatically find model paths - simplified version."""
        search_paths = [
            Path(self.sam2_checkpoint_dir) if self.sam2_checkpoint_dir else None,
            Path(self.grounding_dino_checkpoint_dir) if self.grounding_dino_checkpoint_dir else None,
            Path.home() / "models",
            Path.home() / "libs",
            Path("/workspace"),
            Path.cwd() / "models",
        ]
        search_paths = [p for p in search_paths if p]
        
        sam2_checkpoint = sam2_config = grounding_checkpoint = grounding_config = None
        
        # SAM2 filename mapping - check older versions first for compatibility
        sam2_files = {
            "hiera_tiny": ["sam2_hiera_tiny.pt", "sam2.1_hiera_tiny.pt"],
            "hiera_small": ["sam2_hiera_small.pt", "sam2.1_hiera_small.pt"],
            "hiera_b+": ["sam2_hiera_base_plus.pt", "sam2.1_hiera_base_plus.pt"],
            "hiera_base+": ["sam2_hiera_base_plus.pt", "sam2.1_hiera_base_plus.pt"],
            "hiera_large": ["sam2_hiera_large.pt", "sam2.1_hiera_large.pt"]
        }.get(self.sam2_model_name, [])
        
        grounding_files = {
            "grounding_dino_swin-t": "groundingdino_swint_ogc.pth",
            "grounding_dino_swin-b": "groundingdino_swinb_cogcoor.pth",
            "grounding_dino_swin-l": "groundingdino_swinl_cogcoor.pth"
        }.get(self.grounding_model_name, "")
        
        for path in search_paths:
            # Check SAM2
            for sam2_dir in [path / "segment-anything-2", path / "sam2", path / "SAM2"]:
                if sam2_dir.exists() and not sam2_checkpoint:
                    for ckpt_file in sam2_files:
                        for ckpt_path in [sam2_dir / "checkpoints" / ckpt_file, sam2_dir / ckpt_file]:
                            if ckpt_path.exists():
                                sam2_checkpoint = str(ckpt_path)
                                logger.info(f"Found SAM2 checkpoint: {sam2_checkpoint}")
                                break
                        if sam2_checkpoint:
                            break
                            
            # Check Grounding DINO
            for grounding_dir in [path / "GroundingDINO", path / "groundingdino", path]:
                if grounding_dir.exists() and not grounding_checkpoint and grounding_files:
                    for ckpt_path in [grounding_dir / "weights" / grounding_files, grounding_dir / grounding_files]:
                        if ckpt_path.exists():
                            grounding_checkpoint = str(ckpt_path)
                            logger.info(f"Found Grounding DINO checkpoint: {grounding_checkpoint}")
                            break
                            
        return sam2_checkpoint, sam2_config, grounding_checkpoint, grounding_config
        
    def initialize_models(self):
        """Initialize Grounded-SAM2 models with automatic path finding."""
        try:
            # Fix Grounding DINO path
            for path in [Path.home() / "Grounded-SAM-2", Path("/workspace") / "Grounded-SAM-2"]:
                if path.exists() and (path / "grounding_dino").exists():
                    sys.path.insert(0, str(path))
                    break
            
            # Import required modules
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from grounding_dino.groundingdino.util.inference import load_model
            import grounding_dino.groundingdino.datasets.transforms as T
            
            # Find model paths
            sam2_ckpt, _, grounding_ckpt, grounding_cfg = self.find_model_paths()
            
            if not all([sam2_ckpt, grounding_ckpt]):
                raise FileNotFoundError(f"Could not find model files")
                
            logger.info(f"Loading SAM2 model: {self.sam2_model_name}")
            
            # Determine config name for SAM2
            config_map = {
                "tiny": "sam2_hiera_t.yaml",
                "small": "sam2_hiera_s.yaml", 
                "b+": "sam2_hiera_b+.yaml",
                "base_plus": "sam2_hiera_b+.yaml",
                "large": "sam2_hiera_l.yaml"
            }
            config_name = next((v for k, v in config_map.items() if k in self.sam2_model_name), "sam2_hiera_b+.yaml")
            
            # Initialize SAM2 with compatibility handling
            try:
                sam2_model = build_sam2(config_file=config_name, ckpt_path=sam2_ckpt, device=self.device).to(dtype=self.dtype)
            except RuntimeError as e:
                if "Unexpected key(s)" in str(e):
                    logger.warning("SAM2 checkpoint version mismatch detected. Attempting compatibility mode...")
                    # Try loading with strict=False to ignore unexpected keys
                    sam2_model = build_sam2(config_file=config_name, ckpt_path=None, device=self.device).to(dtype=self.dtype)
                    checkpoint = torch.load(sam2_ckpt, map_location=self.device)
                    if "model" in checkpoint:
                        checkpoint = checkpoint["model"]
                    # Load with strict=False to ignore version differences
                    sam2_model.load_state_dict(checkpoint, strict=False)
                    logger.info("Successfully loaded SAM2 model in compatibility mode")
                else:
                    raise
            
            self.sam2_predictor = SAM2ImagePredictor(sam2_model)
            
            # Initialize Grounding DINO
            logger.info(f"Loading Grounding DINO model: {self.grounding_model_name}")
            
            # Find config if not provided
            if not grounding_cfg:
                config_paths = [
                    Path.home() / "Grounded-SAM-2" / "grounding_dino" / "groundingdino" / "config" / "GroundingDINO_SwinB_cfg.py",
                    Path(grounding_ckpt).parent.parent / "groundingdino" / "config" / "GroundingDINO_SwinB_cfg.py"
                ]
                grounding_cfg = next((str(p) for p in config_paths if p.exists()), None)
                
            self.grounding_dino = load_model(grounding_cfg, grounding_ckpt, device=self.device)
            self.grounding_dino.eval()
            
            # Initialize transform
            self.transform = T.Compose([
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            
            logger.info("✅ Models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise RuntimeError("Failed to load models. Check installation.")
    
    def ground_objects_in_frame(self, image: np.ndarray, frame_idx: int = -1) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        """Use Grounding DINO to detect objects based on text prompts."""
        if self.grounding_dino is None:
            raise RuntimeError("Grounding DINO model not loaded")
            
        from grounding_dino.groundingdino.util.inference import predict
        from PIL import Image
        
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255 if image.max() <= 1.0 else image).astype(np.uint8)
        
        # Save original image for debugging
        if self.save_debug_visualizations:
            debug_dir = DEBUG_OUTPUT_DIR / f"frame_{frame_idx:06d}"
            debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_dir / "original_image.jpg"), image)
            
            # Log vocabulary being searched
            vocab_path = debug_dir / "vocabulary.txt"
            with open(vocab_path, 'w') as f:
                f.write(f"Searching for {len(self.vocabulary)} objects:\n")
                for vocab_word in sorted(self.vocabulary):
                    f.write(f"- {vocab_word}\n")
        
        # Apply transforms
        image_transformed, _ = self.transform(Image.fromarray(image), None)
        
        all_boxes, all_labels, all_scores = [], [], []
        filtered_detections = []  # Track what was filtered
        
        # Label filtering parameters
        min_phrase_length = getattr(self, 'min_phrase_length', 2)
        filter_empty_labels = getattr(self, 'filter_empty_labels', True)
        empty_label_max_size = getattr(self, 'empty_label_max_size', 0.4)
        
        # Process each vocabulary word
        for vocab_word in self.vocabulary:
            caption = f"a {vocab_word}"
            
            with torch.no_grad():
                boxes, logits, phrases = predict(
                    model=self.grounding_dino,
                    image=image_transformed,
                    caption=caption,
                    box_threshold=self.confidence_threshold,
                    text_threshold=config.get("semantic_segmentation", {}).get("grounded_sam2", {}).get("text_threshold", 0.25),
                    device=self.device
                )
            
            # Log all raw detections for this vocabulary word
            if self.save_debug_visualizations and len(boxes) > 0:
                logger.info(f"Frame {frame_idx}: {vocab_word} - {len(boxes)} raw detections")
                for box, score, phrase in zip(boxes, logits, phrases):
                    logger.info(f"  - '{phrase}' (score: {score:.3f})")
            
            # Filter and keep valid detections
            for box, score, phrase in zip(boxes, logits, phrases):
                phrase_clean = phrase.strip()
                
                # Filter empty or invalid labels
                if filter_empty_labels and len(phrase_clean) < min_phrase_length:
                    # Always filter empty labels
                    box_area = box[2] * box[3]  # width * height in normalized coords
                    filtered_detections.append({
                        'vocab_word': vocab_word,
                        'detected_phrase': phrase,
                        'score': float(score),
                        'reason': f'Empty label (length={len(phrase_clean)}, area={box_area:.2f})',
                        'box': box.tolist()
                    })
                    continue
                
                all_boxes.append(box)
                all_labels.append(phrase_clean)
                all_scores.append(score)
        
        # Save filtered detections log
        if self.save_debug_visualizations and filtered_detections:
            filtered_path = DEBUG_OUTPUT_DIR / f"frame_{frame_idx:06d}" / "filtered_detections.txt"
            with open(filtered_path, 'w') as f:
                f.write(f"Filtered {len(filtered_detections)} detections:\n\n")
                for det in filtered_detections:
                    f.write(f"Vocabulary: {det['vocab_word']}\n")
                    f.write(f"Detected: '{det['detected_phrase']}' (score: {det['score']:.3f})\n")
                    f.write(f"Reason: {det['reason']}\n\n")
        
        if len(all_boxes) > 0:
            boxes = torch.stack(all_boxes)
            logits = torch.tensor(all_scores)
            
            # Convert to pixel coordinates for visualization before deduplication
            h, w = image.shape[:2]
            boxes_xyxy_before = torch.zeros_like(boxes)
            boxes_xyxy_before[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * w
            boxes_xyxy_before[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * h
            boxes_xyxy_before[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * w
            boxes_xyxy_before[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * h
            
            # Visualize before deduplication
            if self.save_debug_visualizations:
                self.visualize_detections(image, boxes_xyxy_before, all_labels, logits, 
                                        frame_idx, "grounding")
                logger.info(f"Frame {frame_idx}: {len(boxes)} detections before deduplication")
            
            # Deduplicate boxes with high IoU
            num_before = len(boxes)
            boxes, logits, all_labels = self._deduplicate_boxes(boxes, logits, all_labels)
            num_after = len(boxes)
            
            if self.save_debug_visualizations and num_before != num_after:
                logger.info(f"Frame {frame_idx}: Deduplication reduced {num_before} -> {num_after} detections")
            
            # Convert boxes from cxcywh to xyxy pixel coordinates
            boxes_xyxy = torch.zeros_like(boxes)
            boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * w
            boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * h
            boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * w
            boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * h
            
            # Visualize after deduplication
            if self.save_debug_visualizations:
                self.visualize_detections(image, boxes_xyxy, all_labels, logits, 
                                        frame_idx, "deduped")
            
            # Store grounding scores for use in segmentation
            self._last_grounding_scores = logits.tolist()
            
            return boxes_xyxy, all_labels, logits
        
        # No detections found
        if self.save_debug_visualizations:
            no_detection_path = DEBUG_OUTPUT_DIR / f"frame_{frame_idx:06d}" / "no_detections.txt"
            with open(no_detection_path, 'w') as f:
                f.write(f"No detections found for frame {frame_idx}\n")
                f.write(f"Vocabulary size: {len(self.vocabulary)}\n")
                f.write(f"Confidence threshold: {self.confidence_threshold}\n")
                if filtered_detections:
                    f.write(f"\nFiltered {len(filtered_detections)} detections that didn't match vocabulary\n")
        
        return torch.zeros((0, 4)), [], torch.zeros(0)
    
    def _deduplicate_boxes(self, boxes: torch.Tensor, scores: torch.Tensor, labels: List[str]) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """Deduplicate overlapping boxes using IoU threshold."""
        if len(boxes) == 0:
            return boxes, scores, labels
            
        # Convert cxcywh to xyxy for IoU calculation
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        
        keep = []
        order = scores.argsort(descending=True)
        
        while order.numel() > 0:
            i = order[0]
            keep.append(i)
            
            if order.numel() == 1:
                break
                
            # Compute IoU with remaining boxes
            xx1 = torch.max(x1[i], x1[order[1:]])
            yy1 = torch.max(y1[i], y1[order[1:]])
            xx2 = torch.min(x2[i], x2[order[1:]])
            yy2 = torch.min(y2[i], y2[order[1:]])
            
            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            inter = w * h
            
            area_i = (x2[i] - x1[i]) * (y2[i] - y1[i])
            area = (x2[order[1:]] - x1[order[1:]]) * (y2[order[1:]] - y1[order[1:]])
            union = area_i + area - inter
            
            iou = inter / union
            idx = (iou <= self.deduplication_iou_threshold).nonzero().squeeze(1)
            order = order[idx + 1]
        
        keep = torch.tensor(keep, dtype=torch.long)
        return boxes[keep], scores[keep], [labels[i] for i in keep]
    
    def _deduplicate_masks(self, masks: List[np.ndarray], labels: List[str], frame_idx: int) -> Tuple[List[np.ndarray], List[str]]:
        """Deduplicate overlapping segmentation masks using simple size and containment logic."""
        if len(masks) <= 1:
            return masks, labels
        
        # Calculate mask areas
        mask_areas = []
        for i, mask in enumerate(masks):
            area = np.sum(mask > 0)
            mask_areas.append(area)
        
        def should_allow_overlap(idx_i, idx_j, masks_i, masks_j, area_i, area_j):
            """Determine if overlap should be allowed based on size and containment only."""
            
            # Calculate overlap metrics
            intersection = np.sum((masks_i > 0) & (masks_j > 0))
            if intersection == 0:
                return True  # No overlap, both can exist
            
            # Containment ratios
            containment_i_in_j = intersection / area_i if area_i > 0 else 0
            containment_j_in_i = intersection / area_j if area_j > 0 else 0
            
            # Size-based hierarchy: smaller objects can be on/in larger objects
            size_ratio = area_i / area_j if area_j > 0 else 1.0
            
            # If one object is significantly smaller and mostly contained in the larger
            if size_ratio < 0.5 and containment_i_in_j > 0.7:
                # Small object i is mostly within large object j
                return True
            elif size_ratio > 2.0 and containment_j_in_i > 0.7:
                # Small object j is mostly within large object i
                return True
            
            # High containment in either direction suggests valid relationship
            if containment_i_in_j > 0.8 or containment_j_in_i > 0.8:
                return True
            
            return False
        
        # Sort by area (process larger objects first)
        order = sorted(range(len(masks)), key=lambda i: mask_areas[i], reverse=True)
        
        keep = []
        removed = []
        
        for i in order:
            if mask_areas[i] == 0:
                continue  # Skip empty masks
                
            should_keep = True
            
            # Check overlap with already kept masks
            for j in keep:
                # Calculate overlap
                intersection = np.sum((masks[i] > 0) & (masks[j] > 0))
                if intersection == 0:
                    continue  # No overlap
                
                union = np.sum((masks[i] > 0) | (masks[j] > 0))
                iou = intersection / union if union > 0 else 0
                
                # For same object type, always deduplicate
                if labels[i] == labels[j]:
                    if iou > 0.5:  # Lower threshold for same objects
                        should_keep = False
                        removed.append({
                            'label': labels[i],
                            'area': mask_areas[i],
                            'iou': iou,
                            'kept_label': labels[j],
                            'kept_area': mask_areas[j],
                            'reason': 'duplicate_same_type'
                        })
                        break
                else:
                    # For different objects, check if overlap is allowed
                    if not should_allow_overlap(i, j, masks[i], masks[j], mask_areas[i], mask_areas[j]):
                        # High overlap without valid size/containment relationship
                        if iou > 0.6:
                            should_keep = False
                            removed.append({
                                'label': labels[i],
                                'area': mask_areas[i],
                                'iou': iou,
                                'kept_label': labels[j],
                                'kept_area': mask_areas[j],
                                'reason': 'invalid_overlap'
                            })
                            break
            
            if should_keep:
                keep.append(i)
        
        # Log what was removed
        if removed and self.save_debug_visualizations:
            logger.info(f"Frame {frame_idx}: Removed {len(removed)} masks:")
            for r in removed:
                logger.info(f"  - Removed '{r['label']}' (area: {r['area']}) - "
                          f"IoU: {r['iou']:.2f} with '{r['kept_label']}' - Reason: {r['reason']}")
        
        return [masks[i] for i in keep], [labels[i] for i in keep]
    
    def _merge_duplicate_labels(self, masks: List[np.ndarray], labels: List[str], scores: List[float], frame_idx: int) -> Tuple[List[np.ndarray], List[str], List[float]]:
        """Merge masks with duplicate labels, keeping the best one per label."""
        if len(masks) <= 1:
            return masks, labels, scores
        
        from collections import defaultdict
        
        # Group masks by label
        label_groups = defaultdict(list)
        for i, label in enumerate(labels):
            label_groups[label].append(i)
        
        # Process each label group
        final_masks = []
        final_labels = []
        final_scores = []
        
        for label, indices in sorted(label_groups.items()):
            if len(indices) == 1:
                # No duplicates for this label
                final_masks.append(masks[indices[0]])
                final_labels.append(label)
                final_scores.append(scores[indices[0]])
            else:
                # Multiple masks for same label - keep the one with highest score
                group_scores = [scores[i] for i in indices]
                best_idx = indices[np.argmax(group_scores)]
                final_masks.append(masks[best_idx])
                final_labels.append(label)
                final_scores.append(scores[best_idx])
                
                if self.save_debug_visualizations:
                    mask_areas = [np.sum(masks[i] > 0) for i in indices]
                    logger.info(f"Frame {frame_idx}: Merged {len(indices)} '{label}' masks, "
                              f"kept mask with score {max(group_scores):.3f} and {mask_areas[indices.index(best_idx)]} pixels")
        
        return final_masks, final_labels, final_scores
    
    def _resolve_cross_label_overlaps(self, masks: List[np.ndarray], labels: List[str], scores: List[float], frame_idx: int) -> Tuple[List[np.ndarray], List[str]]:
        """Resolve overlaps between masks with different labels (e.g., wall plug vs switch on same object)."""
        if len(masks) <= 1:
            return masks, labels
        
        # Use provided confidence scores
        confidence_scores = scores
        
        # Track which masks to keep
        n_masks = len(masks)
        keep_mask = [True] * n_masks
        removal_log = []
        
        # Compare all pairs of masks
        for i in range(n_masks):
            if not keep_mask[i]:
                continue
                
            for j in range(i + 1, n_masks):
                if not keep_mask[j]:
                    continue
                
                # Skip if same label (already handled by _merge_duplicate_labels)
                if labels[i] == labels[j]:
                    continue
                
                # Calculate IoU and containment
                mask_i = masks[i] > 0.5
                mask_j = masks[j] > 0.5
                
                intersection = np.sum(mask_i & mask_j)
                union = np.sum(mask_i | mask_j)
                
                if union == 0:
                    continue
                
                iou = intersection / union
                
                # Calculate containment ratios
                area_i = np.sum(mask_i)
                area_j = np.sum(mask_j)
                containment_i_in_j = intersection / area_i if area_i > 0 else 0
                containment_j_in_i = intersection / area_j if area_j > 0 else 0
                
                # Remove if high overlap OR one mask is mostly contained in the other
                if iou > 0.8 or containment_i_in_j > 0.9 or containment_j_in_i > 0.9:
                    # Keep the one with higher confidence (larger mask area)
                    if confidence_scores[i] > confidence_scores[j]:
                        keep_mask[j] = False
                        removal_log.append({
                            'removed': labels[j],
                            'kept': labels[i],
                            'iou': iou,
                            'score_removed': confidence_scores[j],
                            'score_kept': confidence_scores[i]
                        })
                    else:
                        keep_mask[i] = False
                        removal_log.append({
                            'removed': labels[i],
                            'kept': labels[j],
                            'iou': iou,
                            'score_removed': confidence_scores[i],
                            'score_kept': confidence_scores[j]
                        })
                        break  # i is removed, no need to check more pairs with i
        
        # Filter masks and labels
        final_masks = [mask for mask, keep in zip(masks, keep_mask) if keep]
        final_labels = [label for label, keep in zip(labels, keep_mask) if keep]
        
        # Log removals
        if removal_log and self.save_debug_visualizations:
            logger.info(f"Frame {frame_idx}: Cross-label deduplication removed {len(removal_log)} masks:")
            for removal in removal_log:
                logger.info(f"  - Removed '{removal['removed']}' (score: {removal['score_removed']:.3f}) "
                          f"in favor of '{removal['kept']}' (score: {removal['score_kept']:.3f}) "
                          f"with IoU: {removal['iou']:.3f}")
        
        return final_masks, final_labels
    
    def _resolve_label_conflicts(self, all_proposals: List[Dict], frame_idx: int) -> Tuple[List[np.ndarray], List[str]]:
        """Resolve label conflicts when multiple labels claim the same physical region."""
        if not all_proposals:
            return [], []
        
        # Group proposals by mask similarity (IoU > 0.8)
        groups = []
        used = set()
        
        for i, prop_i in enumerate(all_proposals):
            if i in used:
                continue
                
            # Start new group
            group = [i]
            used.add(i)
            
            # Find all similar masks
            for j, prop_j in enumerate(all_proposals):
                if j <= i or j in used:
                    continue
                    
                # Calculate IoU between masks
                intersection = np.sum((prop_i['mask'] > 0.5) & (prop_j['mask'] > 0.5))
                union = np.sum((prop_i['mask'] > 0.5) | (prop_j['mask'] > 0.5))
                iou = intersection / union if union > 0 else 0
                
                if iou > 0.8:  # High overlap - same physical object
                    group.append(j)
                    used.add(j)
            
            groups.append(group)
        
        # Select best label for each group
        final_masks = []
        final_labels = []
        
        if self.save_debug_visualizations:
            logger.info(f"Frame {frame_idx}: Found {len(groups)} mask groups from {len(all_proposals)} proposals")
        
        for group_idx, group in enumerate(groups):
            # Calculate combined score for each proposal in the group
            best_score = -1
            best_idx = None
            
            group_labels = []
            for idx in group:
                prop = all_proposals[idx]
                
                # Combined score: grounding confidence * SAM2 quality * coverage ratio
                combined_score = (prop['grounding_score'] * 
                                prop['sam2_score'] * 
                                (0.5 + 0.5 * prop['coverage_ratio']))  # coverage weighted less
                
                group_labels.append(f"{prop['label']}({prop['grounding_score']:.2f})")
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_idx = idx
            
            if best_idx is not None:
                best_prop = all_proposals[best_idx]
                final_masks.append(best_prop['mask'])
                final_labels.append(best_prop['label'])
                
                if self.save_debug_visualizations and len(group) > 1:
                    logger.info(f"  Group {group_idx}: Selected '{best_prop['label']}' "
                              f"(score: {best_score:.3f}) from candidates: {', '.join(group_labels)}")
        
        # Apply final mask deduplication to handle any remaining overlaps
        if len(final_masks) > 1:
            final_masks, final_labels = self._deduplicate_masks(final_masks, final_labels, frame_idx)
            if self.save_debug_visualizations:
                logger.info(f"Frame {frame_idx}: After final deduplication: {len(final_masks)} masks")
        
        return final_masks, final_labels
    
    def _save_sam2_raw_output(self, image: np.ndarray, masks_proposals: np.ndarray, 
                             scores: np.ndarray, box: np.ndarray, label: str,
                             frame_idx: int, box_idx: int):
        """Save raw SAM2 segmentation proposals before any filtering."""
        debug_dir = DEBUG_OUTPUT_DIR / f"frame_{frame_idx:06d}" / "sam2_raw"
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Find best mask index (highest score)
        best_idx = np.argmax(scores)
        
        # Create a figure showing all proposals
        h, w = image.shape[:2]
        combined = np.zeros((h, w * 4, 3), dtype=np.uint8)
        
        # Original image with box
        img_with_box = image.copy()
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img_with_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_with_box, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        combined[:, :w] = img_with_box
        
        # Show each mask proposal
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue
        for i in range(min(3, len(masks_proposals))):
            mask = masks_proposals[i].squeeze()
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            
            # Create colored overlay
            overlay = image.copy()
            mask_bool = mask > 0.5
            overlay[mask_bool] = overlay[mask_bool] * 0.5 + np.array(colors[i]) * 0.5
            
            # Add score text and highlight the selected mask
            score_text = f"Score: {scores[i]:.3f}"
            if i == best_idx:
                score_text += " [SELECTED]"
                # Add border to selected mask
                cv2.rectangle(overlay, (0, 0), (w-1, h-1), colors[i], 3)
            cv2.putText(overlay, score_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            combined[:, (i+1)*w:(i+2)*w] = overlay
        
        # Save combined image
        output_path = debug_dir / f"box_{box_idx:03d}_{label}_proposals.jpg"
        cv2.imwrite(str(output_path), combined)
        
        # Also save individual masks
        for i in range(len(masks_proposals)):
            mask = masks_proposals[i].squeeze()
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            mask_path = debug_dir / f"box_{box_idx:03d}_{label}_mask_{i}_score_{scores[i]:.3f}.png"
            cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))
    
    def _save_all_sam2_masks(self, image: np.ndarray, masks: List[np.ndarray], 
                            labels: List[str], frame_idx: int, stage: str):
        """Save visualization of all SAM2 masks combined."""
        debug_dir = DEBUG_OUTPUT_DIR / f"frame_{frame_idx:06d}" / "sam2_combined"
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Create overlay with all masks
        overlay = image.copy()
        h, w = image.shape[:2]
        
        # Create instance segmentation map
        instance_map = np.zeros((h, w), dtype=np.int32)
        
        # Apply each mask with a different color
        for idx, (mask, label) in enumerate(zip(masks, labels)):
            # Random color for each instance
            color = np.array([
                (idx * 67) % 255,
                (idx * 131) % 255,
                (idx * 193) % 255
            ])
            
            # Apply mask
            mask_bool = mask > 0
            overlay[mask_bool] = overlay[mask_bool] * 0.3 + color * 0.7
            instance_map[mask_bool] = idx + 1
            
            # Add label at centroid
            if np.any(mask_bool):
                y_coords, x_coords = np.where(mask_bool)
                cy, cx = int(np.mean(y_coords)), int(np.mean(x_coords))
                cv2.putText(overlay, f"{idx}: {label}", (cx-30, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Save overlay
        output_path = debug_dir / f"all_masks_{stage}.jpg"
        cv2.imwrite(str(output_path), overlay)
        
        # Save instance map
        instance_path = debug_dir / f"instance_map_{stage}.png"
        cv2.imwrite(str(instance_path), instance_map.astype(np.uint8))
        
        # Save summary
        summary_path = debug_dir / f"summary_{stage}.txt"
        with open(summary_path, 'w') as f:
            f.write(f"SAM2 Segmentation Summary - {stage}\n")
            f.write(f"Total masks: {len(masks)}\n\n")
            for idx, (mask, label) in enumerate(zip(masks, labels)):
                mask_pixels = np.sum(mask > 0)
                mask_percent = (mask_pixels / (h * w)) * 100
                f.write(f"{idx}: {label} - {mask_pixels} pixels ({mask_percent:.2f}%)\n")
    
    def segment_frame_with_boxes(self, image: np.ndarray, boxes: torch.Tensor, 
                                labels: List[str], frame_idx: int) -> Dict:
        """Use SAM2 to segment objects given bounding boxes."""
        if self.sam2_predictor is None:
            raise RuntimeError("SAM2 model not loaded")
            
        # Prepare image
        if image.dtype != np.uint8:
            image = (image * 255 if image.max() <= 1.0 else image).astype(np.uint8)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image
        
        # Get image dimensions
        h, w = image.shape[:2]
        
        # Set image in predictor
        with torch.no_grad():
            self.sam2_predictor.set_image(image_rgb)
        
        masks, valid_labels = [], []
        failed_masks = []  # Track failed segmentations
        valid_scores = []  # Track confidence scores for valid masks
        
        # Get grounding scores if available
        grounding_scores = getattr(self, '_last_grounding_scores', [])
        
        if len(boxes) > 0:
            input_boxes = boxes.cpu().numpy()
            
            # Log segmentation attempt
            if self.save_debug_visualizations:
                logger.info(f"Frame {frame_idx}: Attempting to segment {len(input_boxes)} objects")
            
            # Process each box
            for box_idx, (box, label) in enumerate(zip(input_boxes, labels)):
                try:
                    # Predict mask with multiple proposals
                    masks_proposals, scores, _ = self.sam2_predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=box.reshape(1, 4),
                        multimask_output=True  # Get 3 mask proposals
                    )
                    
                    # Save raw SAM2 proposals if debug mode
                    if self.save_debug_visualizations:
                        self._save_sam2_raw_output(
                            image, masks_proposals, scores, box, label, 
                            frame_idx, box_idx
                        )
                    
                    # Simply select the mask with highest confidence score
                    best_mask_idx = np.argmax(scores)
                    mask = masks_proposals[best_mask_idx]
                    score = scores[best_mask_idx]
                    
                    if self.save_debug_visualizations:
                        logger.info(f"Frame {frame_idx}: Selected mask {best_mask_idx} for {label} with score {score:.3f}")
                    
                    # Handle mask dimensions properly
                    if mask.ndim == 4:  # (1, 1, H, W)
                        mask = mask[0, 0]
                    elif mask.ndim == 3:  # (1, H, W)
                        mask = mask[0]
                    elif mask.ndim == 2:  # (H, W)
                        pass  # Already 2D
                    else:
                        logger.error(f"Unexpected mask shape: {mask.shape}")
                        failed_masks.append({
                            'label': label,
                            'reason': f'Invalid mask shape: {mask.shape}'
                        })
                        continue
                    
                    # Validate mask - filter empty masks
                    mask_pixels = np.sum(mask > 0)
                    if mask_pixels == 0:
                        failed_masks.append({
                            'label': label,
                            'reason': 'Empty mask (0 pixels)',
                            'box': box.tolist()
                        })
                        if self.save_debug_visualizations:
                            logger.warning(f"Frame {frame_idx}: Empty mask for {label}")
                        continue
                    
                    # Check mask quality
                    h, w = mask.shape
                    mask_percentage = (mask_pixels / (h * w)) * 100
                    
                    if self.save_debug_visualizations and mask_percentage > 50:
                        logger.warning(f"Frame {frame_idx}: Large mask for {label}: {mask_percentage:.1f}% of image")
                    
                    masks.append(mask)
                    valid_labels.append(label)
                    # Store grounding score if available
                    if box_idx < len(grounding_scores):
                        valid_scores.append(grounding_scores[box_idx])
                    else:
                        valid_scores.append(mask_pixels)  # Fallback to area
                    
                    if self.save_debug_visualizations:
                        logger.info(f"Frame {frame_idx}: Successfully segmented {label} ({mask_pixels} pixels, {mask_percentage:.1f}%)")
                    
                except Exception as e:
                    logger.error(f"Error segmenting {label}: {e}")
                    failed_masks.append({
                        'label': label,
                        'reason': str(e),
                        'box': box.tolist()
                    })
                    continue
        
        # Save failed segmentations log
        if self.save_debug_visualizations and failed_masks:
            failed_path = DEBUG_OUTPUT_DIR / f"frame_{frame_idx:06d}" / "failed_segmentations.txt"
            with open(failed_path, 'w') as f:
                f.write(f"Failed to segment {len(failed_masks)} objects:\n\n")
                for fail in failed_masks:
                    f.write(f"Label: {fail['label']}\n")
                    f.write(f"Reason: {fail['reason']}\n")
                    if 'box' in fail:
                        f.write(f"Box: {fail['box']}\n")
                    f.write("\n")
        
        # Save all SAM2 masks before merging duplicates (if enabled)
        if self.save_debug_visualizations and len(masks) > 0:
            self._save_all_sam2_masks(image, masks, valid_labels, frame_idx, "before_merge")
        
        # Merge duplicate labels - keep best mask per label
        if len(masks) > 0:
            masks, valid_labels, valid_scores = self._merge_duplicate_labels(masks, valid_labels, valid_scores, frame_idx)
            logger.info(f"Frame {frame_idx}: After merging duplicate labels: {len(masks)} masks")
            
            # Resolve cross-label overlaps (e.g., wall plug vs switch)
            masks, valid_labels = self._resolve_cross_label_overlaps(masks, valid_labels, valid_scores, frame_idx)
            logger.info(f"Frame {frame_idx}: After cross-label deduplication: {len(masks)} masks")
        
        # Visualize successful masks
        if self.save_debug_visualizations and len(masks) > 0:
            self.visualize_masks(image, masks, valid_labels, frame_idx)
            self._save_all_sam2_masks(image, masks, valid_labels, frame_idx, "after_merge")
            
            # Log summary
            logger.info(f"Frame {frame_idx}: Segmentation complete - {len(masks)}/{len(boxes)} successful")
        
        return {
            'masks': masks,
            'labels': valid_labels,
            'refined_boxes': boxes.cpu().numpy() if len(boxes) > 0 else []
        }
    
    def process_frame(self, image: np.ndarray, frame_id: int, keyframe_idx: Optional[int] = None) -> Dict:
        """Process a single frame to generate semantic masks."""
        start_time = time.time()
        
        # Ground objects
        boxes, labels, confidences = self.ground_objects_in_frame(image, frame_id)
        
        if len(boxes) == 0:
            return {
                'frame_id': frame_id,
                'keyframe_idx': keyframe_idx,
                'masks_rle': {},
                'instance_ids': [],
                'labels': {},
                'confidences': {},
                'processing_time': time.time() - start_time
            }
        
        # Segment with SAM2
        segmentation_result = self.segment_frame_with_boxes(image, boxes, labels, frame_id)
        
        # Convert to RLE format
        results = {
            'frame_id': frame_id,
            'keyframe_idx': keyframe_idx,
            'masks_rle': {},
            'instance_ids': [],
            'labels': {},
            'confidences': {},
            'processing_time': time.time() - start_time
        }
        
        for idx, (mask, label, conf) in enumerate(zip(
            segmentation_result['masks'],
            segmentation_result['labels'],
            confidences[:len(segmentation_result['masks'])]
        )):
            instance_id = idx + 1
            
            # Encode mask to RLE
            # Ensure mask is 2D before encoding
            if mask.ndim != 2:
                logger.error(f"Mask has wrong dimensions: {mask.shape}")
                continue
            mask_tensor = torch.from_numpy(mask).to(dtype=torch.bool)
            rle_mask = encode_rle(mask_tensor)
            
            # Store results
            results['masks_rle'][instance_id] = rle_mask
            results['instance_ids'].append(instance_id)
            results['labels'][instance_id] = label
            results['confidences'][instance_id] = float(conf)
            
            # Debug: Log what we're storing
            if self.debug_mode:
                logger.info(f"  Storing instance {instance_id}: label='{label}', conf={conf:.3f}")
        
        # Track performance
        self.frame_times.append(results['processing_time'])
        if len(self.frame_times) % 30 == 0:
            avg_time = np.mean(self.frame_times[-30:])
            logger.info(f"Semantic processing: {avg_time*1000:.1f}ms/frame ({1/avg_time:.1f} FPS)")
        
        # Apply object tracking if enabled
        if self.object_tracker is not None:
            logger.info(f"Applying object tracking to frame {frame_id} with {len(results.get('instance_ids', []))} instances")
            results = self.object_tracker.process_frame(
                results, 
                frame_id,
                keyframe_idx
            )
            logger.info(f"Tracking complete, got {len(results.get('track_ids', {}))} track IDs")
            
            # Include tracking decisions in results for saving later
            if hasattr(self.object_tracker, 'tracking_decisions'):
                results['tracking_decisions'] = self.object_tracker.tracking_decisions.copy()
            
        return results
    
    def set_keyframe_access_callback(self, callback):
        """Set callback for accessing keyframe data."""
        self.keyframe_access_callback = callback
        if self.object_tracker is not None:
            self.object_tracker.set_callbacks(get_3d_points=callback)
    
    def run(self):
        """Main processing loop."""
        logger.info(f"Starting Grounded-SAM2 processor on {self.device}...")
        self.initialize_models()
        
        while True:
            try:
                frame_data = self.frame_queue.get(timeout=0.1)
                
                if frame_data is None:
                    break
                
                # Store 3D data if available (for tracking)
                if 'pointmap' in frame_data:
                    pointmap = torch.from_numpy(frame_data['pointmap']).to(self.device)
                    point_conf = torch.from_numpy(frame_data['point_conf']).to(self.device)
                    shape = frame_data['shape']
                    self.set_3d_data_for_tracking(pointmap, point_conf, shape)
                else:
                    self.current_3d_data = None
                
                # Process frame
                logger.info(f"Processing semantic frame {frame_data['frame_id']} (keyframe {frame_data.get('keyframe_idx')})")
                semantic_data = self.process_frame(
                    frame_data['img'], 
                    frame_data['frame_id'], 
                    frame_data.get('keyframe_idx')
                )
                logger.info(f"Semantic processing complete for frame {frame_data['frame_id']}, found {len(semantic_data.get('instance_ids', []))} instances")
                
                # Put results in output queue
                self.result_queue.put(semantic_data)
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in semantic processor: {e}")
                continue
        
        logger.info("Semantic processor terminated.")


def _run_processor_with_tracker(frame_queue, result_queue, vocabulary, device, 
                               model_selector, confidence_threshold, tracking_config, kwargs):
    """Helper function to run processor with optional tracker creation."""
    # Create tracker in the new process if config is provided
    tracker = None
    if tracking_config is not None:
        from mast3r_slam.object_tracker_simple import SimpleObjectTracker
        tracker = SimpleObjectTracker(tracking_config)
        logger.info("Created object tracker in semantic processor process")
    
    processor = RealGroundedSAM2Processor(
        frame_queue, result_queue, vocabulary, device, model_selector, 
        confidence_threshold=confidence_threshold, 
        object_tracker=tracker,
        **kwargs
    )
    processor.run()


def start_real_grounded_sam2_processor(frame_queue: mp.Queue, 
                                      result_queue: mp.Queue,
                                      vocabulary: List[str],
                                      device: str = "cuda:1",
                                      model_selector: Optional[GroundedSAM2ModelSelector] = None,
                                      confidence_threshold: float = 0.35,
                                      object_tracker = None,
                                      tracking_config = None,
                                      **kwargs) -> mp.Process:
    """Start the real Grounded-SAM2 processor in a separate process."""
    
    process = mp.Process(
        target=_run_processor_with_tracker,
        args=(frame_queue, result_queue, vocabulary, device, model_selector, 
              confidence_threshold, tracking_config, kwargs)
    )
    process.start()
    return process