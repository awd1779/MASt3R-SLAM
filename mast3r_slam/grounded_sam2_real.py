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
                 mask_refinement_threshold: float = 0.7):
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.vocabulary = vocabulary
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.debug_mode = debug_mode
        self.save_debug_visualizations = save_debug_visualizations
        self.deduplication_iou_threshold = deduplication_iou_threshold
        self.mask_refinement_threshold = mask_refinement_threshold
        
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
        
        # Track management
        self.next_track_id = 1
        self.frame_times = []
        
        # Debug counters
        self.debug_frame_counter = 0
        
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
        
        # SAM2 filename mapping
        sam2_files = {
            "hiera_tiny": ["sam2_hiera_tiny.pt", "sam2.1_hiera_tiny.pt"],
            "hiera_small": ["sam2_hiera_small.pt", "sam2.1_hiera_small.pt"],
            "hiera_b+": ["sam2_hiera_base_plus.pt", "sam2.1_hiera_base_plus.pt"],
            "hiera_base+": ["sam2_hiera_base_plus.pt", "sam2.1_hiera_base_plus.pt"],
            "hiera_large": ["sam2_hiera_large.pt", "sam2.1_hiera_large.pt"]
        }.get(self.sam2_model_name, [])
        
        grounding_files = {
            "grounding_dino_swin-t": "groundingdino_swint_ogc.pth",
            "grounding_dino_swin-b": "groundingdino_swinb_cogcoor.pth"
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
            
            # Initialize SAM2
            sam2_model = build_sam2(config_file=config_name, ckpt_path=sam2_ckpt, device=self.device).to(dtype=self.dtype)
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
            
            # Keep ALL detections without vocabulary filtering
            for box, score, phrase in zip(boxes, logits, phrases):
                all_boxes.append(box)
                all_labels.append(phrase.strip())  # Use the actual detected phrase
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
        
        masks, track_ids, valid_labels = [], [], []
        failed_masks = []  # Track failed segmentations
        
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
                    
                    # Select the mask with highest IoU to the bounding box
                    best_mask_idx = 0
                    best_iou = 0
                    
                    # Convert box to mask for IoU calculation
                    box_mask = np.zeros((h, w), dtype=bool)
                    x1, y1, x2, y2 = box.astype(int)
                    box_mask[y1:y2, x1:x2] = True
                    box_area = (x2 - x1) * (y2 - y1)
                    
                    for i in range(len(masks_proposals)):
                        mask_proposal = masks_proposals[i].squeeze()
                        
                        # Ensure mask is boolean numpy array
                        if isinstance(mask_proposal, torch.Tensor):
                            mask_proposal = mask_proposal.cpu().numpy()
                        mask_proposal = mask_proposal.astype(bool)
                        
                        # Calculate IoU with bounding box
                        intersection = np.sum(mask_proposal & box_mask)
                        union = np.sum(mask_proposal | box_mask)
                        iou = intersection / union if union > 0 else 0
                        
                        # Prefer masks that fit well within the box
                        mask_area = np.sum(mask_proposal)
                        containment = intersection / mask_area if mask_area > 0 else 0
                        
                        # Combined score: IoU + containment
                        combined_score = iou + containment
                        
                        if combined_score > best_iou:
                            best_iou = combined_score
                            best_mask_idx = i
                    
                    mask = masks_proposals[best_mask_idx]
                    score = scores[best_mask_idx]
                    
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
                    track_ids.append(self.next_track_id)
                    self.next_track_id += 1
                    
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
        
        # Visualize successful masks
        if self.save_debug_visualizations and len(masks) > 0:
            self.visualize_masks(image, masks, valid_labels, frame_idx)
            
            # Log summary
            logger.info(f"Frame {frame_idx}: Segmentation complete - {len(masks)}/{len(boxes)} successful")
        
        return {
            'masks': masks,
            'labels': valid_labels,
            'track_ids': track_ids,
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
                'track_ids': {},
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
            'track_ids': {},
            'labels': {},
            'confidences': {},
            'processing_time': time.time() - start_time
        }
        
        for idx, (mask, label, conf, track_id) in enumerate(zip(
            segmentation_result['masks'],
            segmentation_result['labels'],
            confidences[:len(segmentation_result['masks'])],
            segmentation_result['track_ids']
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
            results['track_ids'][instance_id] = track_id
            results['labels'][instance_id] = label
            results['confidences'][instance_id] = float(conf)
            
            # Debug: Log what we're storing
            if self.debug_mode:
                logger.info(f"  Storing instance {instance_id}: label='{label}', conf={conf:.3f}, track_id={track_id}")
        
        # Track performance
        self.frame_times.append(results['processing_time'])
        if len(self.frame_times) % 30 == 0:
            avg_time = np.mean(self.frame_times[-30:])
            logger.info(f"Semantic processing: {avg_time*1000:.1f}ms/frame ({1/avg_time:.1f} FPS)")
            
        return results
    
    def run(self):
        """Main processing loop."""
        logger.info(f"Starting Grounded-SAM2 processor on {self.device}...")
        self.initialize_models()
        
        while True:
            try:
                frame_data = self.frame_queue.get(timeout=0.1)
                
                if frame_data is None:
                    break
                
                # Process frame
                semantic_data = self.process_frame(
                    frame_data['img'], 
                    frame_data['frame_id'], 
                    frame_data.get('keyframe_idx')
                )
                
                # Put results in output queue
                self.result_queue.put(semantic_data)
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in semantic processor: {e}")
                continue
        
        logger.info("Semantic processor terminated.")


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