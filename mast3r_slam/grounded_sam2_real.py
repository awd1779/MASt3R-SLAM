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
from mast3r_slam.debug_visualizer import DebugVisualizer
from mast3r_slam.mask_deduplicator import MaskDeduplicator

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
        
        # Initialize debug visualizer and deduplicator
        self.debug_visualizer = None
        if self.save_debug_visualizations:
            self.debug_visualizer = DebugVisualizer(DEBUG_OUTPUT_DIR, save_debug_visualizations)
        self.mask_deduplicator = MaskDeduplicator(debug_mode=self.debug_mode)
        
    def visualize_detections(self, image: np.ndarray, boxes: torch.Tensor, labels: List[str], 
                             scores: torch.Tensor, frame_idx: int, stage: str = "grounding"):
        """Visualize bounding boxes on image and save to file."""
        if self.debug_visualizer:
            # Create debug directory for this frame
            debug_dir = DEBUG_OUTPUT_DIR / f"frame_{frame_idx:06d}"
            debug_dir.mkdir(parents=True, exist_ok=True)
            self.debug_visualizer.debug_dir = debug_dir
            
            # Convert tensors to lists for visualizer
            boxes_list = boxes.tolist() if hasattr(boxes, 'tolist') else boxes
            scores_list = scores.tolist() if hasattr(scores, 'tolist') else scores
            
            self.debug_visualizer.visualize_detections(image, boxes_list, labels, scores_list, frame_idx, stage)
    
    def visualize_masks(self, image: np.ndarray, masks: List[np.ndarray], labels: List[str], 
                       frame_idx: int):
        """Visualize segmentation masks and save to file."""
        if self.debug_visualizer:
            debug_dir = DEBUG_OUTPUT_DIR / f"frame_{frame_idx:06d}"
            debug_dir.mkdir(parents=True, exist_ok=True)
            self.debug_visualizer.debug_dir = debug_dir
            
            self.debug_visualizer.visualize_masks(image, masks, labels, frame_idx)
    
    def _log_processing_step(self, message: str, frame_idx: int = None, level: str = "info"):
        """Simplified logging for processing steps."""
        if self.debug_mode:
            prefix = f"Frame {frame_idx}: " if frame_idx is not None else ""
            getattr(logger, level)(f"{prefix}{message}")
    
    def _save_failed_segmentations(self, failed_masks: List[Dict], frame_idx: int):
        """Save failed segmentation details to debug file."""
        if not self.save_debug_visualizations:
            return
            
        failed_path = DEBUG_OUTPUT_DIR / f"frame_{frame_idx:06d}" / "failed_segmentations.txt"
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(failed_path, 'w') as f:
            f.write(f"Failed to segment {len(failed_masks)} objects:\n\n")
            for fail in failed_masks:
                f.write(f"Label: {fail['label']}\n")
                f.write(f"Reason: {fail['reason']}\n")
                if 'box' in fail:
                    f.write(f"Box: {fail['box']}\n")
                f.write("\n")

    def _save_frame_debug_info(self, frame_idx: int, data: dict):
        """Save debug information for a frame."""
        if not self.save_debug_visualizations:
            return
        
        debug_dir = DEBUG_OUTPUT_DIR / f"frame_{frame_idx:06d}"
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Save vocabulary if provided
        if 'vocabulary' in data:
            with open(debug_dir / "vocabulary.txt", 'w') as f:
                f.write(f"Searching for {len(data['vocabulary'])} objects:\n")
                for word in sorted(data['vocabulary']):
                    f.write(f"- {word}\n")
        
        # Save filtered detections if provided
        if 'filtered_detections' in data and data['filtered_detections']:
            with open(debug_dir / "filtered_detections.txt", 'w') as f:
                f.write(f"Filtered {len(data['filtered_detections'])} detections:\n\n")
                for det in data['filtered_detections']:
                    f.write(f"Vocabulary: {det['vocab_word']}\n")
                    f.write(f"Detected: '{det['detected_phrase']}' (score: {det['score']:.3f})\n")
                    f.write(f"Reason: {det['reason']}\n\n")
    
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
            
            # Initialize transform to match MAST3R's 512px resolution
            # Using 512 for long side to maintain consistency with MAST3R processing
            self.transform = T.Compose([
                T.RandomResize([512], max_size=512),  # Match MAST3R's 512px processing
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
        
        # Save debug info
        if self.save_debug_visualizations:
            debug_dir = DEBUG_OUTPUT_DIR / f"frame_{frame_idx:06d}"
            debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_dir / "original_image.jpg"), image)
            self._save_frame_debug_info(frame_idx, {'vocabulary': self.vocabulary})
        
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
            
            # Log detections for this vocabulary word
            if len(boxes) > 0:
                self._log_processing_step(f"{vocab_word} - {len(boxes)} raw detections", frame_idx)
            
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
        if filtered_detections:
            self._save_frame_debug_info(frame_idx, {'filtered_detections': filtered_detections})
        
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
            
            # Deduplicate boxes with high IoU
            num_before = len(boxes)
            boxes, logits, all_labels = self._deduplicate_boxes(boxes, logits, all_labels)
            num_after = len(boxes)
            
            if num_before != num_after:
                self._log_processing_step(f"Deduplication reduced {num_before} -> {num_after} detections", frame_idx)
            
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
        self._log_processing_step("No detections found", frame_idx)
        
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
    
    
    
    def _save_sam2_raw_output(self, image: np.ndarray, masks_proposals: np.ndarray, 
                             scores: np.ndarray, box: np.ndarray, label: str,
                             frame_idx: int, box_idx: int):
        """Save raw SAM2 segmentation proposals before any filtering."""
        if self.debug_visualizer:
            # Convert arrays to lists and prepare data
            boxes = [box.tolist()]
            labels = [label]
            scores_list = scores.tolist() if hasattr(scores, 'tolist') else scores
            masks = [masks_proposals]  # Pass raw proposals to visualizer
            
            self.debug_visualizer.save_sam2_raw_output(image, boxes, masks, labels, scores_list, frame_idx)
    
    def _save_all_sam2_masks(self, image: np.ndarray, masks: List[np.ndarray], 
                            labels: List[str], frame_idx: int, stage: str):
        """Save visualization of all SAM2 masks combined."""
        if self.debug_visualizer:
            # Use the existing visualize_masks method for combined mask visualization
            self.visualize_masks(image, masks, labels, frame_idx)
    
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
            
            self._log_processing_step(f"Attempting to segment {len(input_boxes)} objects", frame_idx)
            
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
                    
                    self._log_processing_step(f"Selected mask {best_mask_idx} for {label} with score {score:.3f}", frame_idx)
                    
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
                        self._log_processing_step(f"Empty mask for {label}", frame_idx, "warning")
                        continue
                    
                    # Check mask quality
                    h, w = mask.shape
                    mask_percentage = (mask_pixels / (h * w)) * 100
                    
                    if mask_percentage > 50:
                        self._log_processing_step(f"Large mask for {label}: {mask_percentage:.1f}% of image", frame_idx, "warning")
                    
                    masks.append(mask)
                    valid_labels.append(label)
                    # Store grounding score if available
                    if box_idx < len(grounding_scores):
                        valid_scores.append(grounding_scores[box_idx])
                    else:
                        valid_scores.append(mask_pixels)  # Fallback to area
                    
                    self._log_processing_step(f"Successfully segmented {label} ({mask_pixels} pixels, {mask_percentage:.1f}%)", frame_idx)
                    
                except Exception as e:
                    logger.error(f"Error segmenting {label}: {e}")
                    failed_masks.append({
                        'label': label,
                        'reason': str(e),
                        'box': box.tolist()
                    })
                    continue
        
        # Save failed segmentations log
        if failed_masks:
            self._save_failed_segmentations(failed_masks, frame_idx)
        
        # Save all SAM2 masks before merging duplicates (if enabled)
        if self.save_debug_visualizations and len(masks) > 0:
            self._save_all_sam2_masks(image, masks, valid_labels, frame_idx, "before_merge")
        
        # Deduplicate masks using consolidated logic
        if len(masks) > 0:
            masks, valid_labels, valid_scores = self.mask_deduplicator.deduplicate_masks(
                masks, valid_labels, valid_scores, frame_idx
            )
            self._log_processing_step(f"After deduplication: {len(masks)} masks", frame_idx)
        
        # Visualize successful masks
        if self.save_debug_visualizations and len(masks) > 0:
            self.visualize_masks(image, masks, valid_labels, frame_idx)
            self._save_all_sam2_masks(image, masks, valid_labels, frame_idx, "after_merge")
            
            self._log_processing_step(f"Segmentation complete - {len(masks)}/{len(boxes)} successful", frame_idx)
        
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