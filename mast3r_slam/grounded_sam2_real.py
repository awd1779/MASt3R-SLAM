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
                 confidence_threshold: float = 0.35):
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.vocabulary = vocabulary
        self.device = device
        self.confidence_threshold = confidence_threshold
        
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
        logger.debug(f"Searching for models in: {[str(p) for p in search_paths[:3]]}")
            
        # SAM2 model info
        sam2_info = SAM2_MODELS[self.sam2_model_name]
        sam2_checkpoint = None
        sam2_config = None
        
        # Grounding DINO model info
        grounding_info = GROUNDING_MODELS[self.grounding_model_name]
        grounding_checkpoint = None
        grounding_config = None
        
        # Search for SAM2 files - try both versions
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
                path / "grounding-dino"
            ]
            
            for grounding_dir in grounding_dirs:
                if grounding_dir.exists():
                    # Look for checkpoint
                    ckpt_names = {
                        "grounding_dino_swin-t": "groundingdino_swint_ogc.pth",
                        "grounding_dino_swin-b": "groundingdino_swinb_cogcoor.pth"
                    }
                    ckpt_file = ckpt_names.get(self.grounding_model_name)
                    if ckpt_file:
                        ckpt_paths = [
                            grounding_dir / "weights" / ckpt_file,
                            grounding_dir / ckpt_file,
                            grounding_dir / "checkpoints" / ckpt_file
                        ]
                        for ckpt_path in ckpt_paths:
                            if ckpt_path.exists():
                                grounding_checkpoint = str(ckpt_path)
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
            use_mock = config.get("semantic_segmentation", {}).get("use_mock", False)
            
            if use_mock:
                logger.info("Using mock semantic processor (set use_mock=false for real models)")
                return
                
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
            
            # Initialize SAM2 in IMAGE MODE for keyframe segmentation
            # We don't need video tracking between keyframes
            sam2_model = build_sam2(
                config_file=config_name,
                ckpt_path=sam2_ckpt,
                device=self.device
            )
            self.sam2_predictor = SAM2ImagePredictor(sam2_model)
            
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
            
    def ground_objects_in_frame(self, image: np.ndarray) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        """Use Grounding DINO to detect objects based on text prompts."""
        if self.grounding_dino is None:
            logger.warning("Grounding DINO is None, using mock segmentation!")
            logger.warning("This means the real model failed to load properly.")
            # Fallback to mock
            return self._mock_ground_objects(image)
            
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
        
        # Prepare text prompt
        caption = ". ".join(self.vocabulary) + "."
        
        # Run grounding
        print(f"    Running Grounding DINO with caption: {caption}")
        print(f"    Confidence threshold: {self.confidence_threshold}")
        
        with torch.no_grad():
            boxes, logits, phrases = predict(
                model=self.grounding_dino,
                image=image_transformed,
                caption=caption,
                box_threshold=self.confidence_threshold,
                text_threshold=0.25,
                device=self.device
            )
        
        print(f"    Detected {len(boxes)} objects")
        for i, (phrase, score) in enumerate(zip(phrases, logits)):
            print(f"      {i}: {phrase} (conf={score:.3f})")
        
        # Convert boxes from normalized to pixel coordinates
        h, w = image.shape[:2]
        
        # Check box format - Grounding DINO returns cxcywh format, need to convert to xyxy
        print(f"    Raw box format (first box): {boxes[0] if len(boxes) > 0 else 'None'}")
        
        # Convert from cxcywh to xyxy format
        boxes_xyxy = torch.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1 = cx - w/2
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1 = cy - h/2
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2 = cx + w/2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2 = cy + h/2
        
        # Scale to pixel coordinates
        boxes_xyxy = boxes_xyxy * torch.tensor([w, h, w, h], device=boxes.device)
        
        return boxes_xyxy, phrases, logits
    
    def segment_frame_with_boxes(self, image: np.ndarray, boxes: torch.Tensor, 
                                 labels: List[str], frame_idx: int) -> Dict:
        """Use SAM2 IMAGE MODE to segment objects given bounding boxes."""
        if self.sam2_predictor is None:
            print("    WARNING: SAM2 predictor is None, using mock segmentation!")
            return self._mock_segment_frame(image, boxes, labels, frame_idx)
            
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
        print(f"    Setting image in SAM2 predictor (shape: {image_rgb.shape})")
        self.sam2_predictor.set_image(image_rgb)
        self.current_image_set = True
        
        # Process all boxes at once
        masks = []
        track_ids = []
        
        print(f"    Processing {len(boxes)} detected boxes for segmentation...")
        
        # Convert all boxes to numpy array
        if len(boxes) > 0:
            # Boxes should already be in xyxy pixel coordinates
            input_boxes = boxes.cpu().numpy()
            print(f"    Input boxes shape: {input_boxes.shape}")
            print(f"    First box: {input_boxes[0] if len(input_boxes) > 0 else 'None'}")
            
            # Predict masks for all boxes at once
            print(f"    Running SAM2 predict with {len(input_boxes)} boxes...")
            try:
                # SAM2 image mode predict
                predicted_masks, scores, logits = self.sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False  # Single mask per box
                )
                
                print(f"    SAM2 output: masks shape={predicted_masks.shape}, scores shape={scores.shape}")
                
                # Process each mask
                for box_idx, (mask, score, label) in enumerate(zip(predicted_masks, scores, labels)):
                    # Score might be a scalar or array
                    score_val = float(score[0] if hasattr(score, '__len__') else score)
                    print(f"      Processing mask {box_idx}: {label} (score={score_val:.3f})")
                    
                    # mask is (C, H, W) with C=1, need to squeeze
                    print(f"        Mask shape before squeeze: {mask.shape}")
                    if mask.ndim == 3 and mask.shape[0] == 1:
                        mask = mask[0]  # Remove channel dimension
                    print(f"        Mask shape after squeeze: {mask.shape}")
                    
                    mask_pixels = np.sum(mask)
                    mask_total = mask.shape[0] * mask.shape[1]
                    print(f"        Mask pixels: {mask_pixels} ({mask_pixels/mask_total*100:.1f}% coverage)")
                    
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
                            print(f"        Mask bounds: x=[{min_x}, {max_x}], y=[{min_y}, {max_y}], size={max_x-min_x+1}x{max_y-min_y+1}")
                    except Exception as e:
                        print(f"        Error checking mask bounds: {e}")
                    
                    masks.append(mask)
                    
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
                print(f"    ERROR in SAM2 predict: {e}")
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
                    track_ids.append(self.next_track_id)
                    self.next_track_id += 1
                
        return {
            'masks': masks,
            'labels': labels,
            'track_ids': track_ids
        }
    
    def process_frame(self, image: np.ndarray, frame_id: int) -> Dict:
        """Process a single frame to generate semantic masks."""
        start_time = time.time()
        h, w = image.shape[:2]
        
        
        # Ground objects in the frame
        boxes, labels, confidences = self.ground_objects_in_frame(image)
        
        if len(boxes) == 0:
            return {
                'frame_id': frame_id,
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
        
        # Convert to RLE format
        results = {
            'frame_id': frame_id,
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
            print(f"Semantic processing: {avg_time*1000:.1f}ms/frame ({1/avg_time:.1f} FPS)")
            
        return results
    
    def _mock_ground_objects(self, image: np.ndarray) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        """Mock object grounding for fallback."""
        h, w = image.shape[:2]
        num_objects = min(2, len(self.vocabulary))
        boxes = []
        labels = []
        confidences = []
        
        for i in range(num_objects):
            x1 = int(w * 0.2 * (i + 1))
            y1 = int(h * 0.2 * (i + 1))
            x2 = min(x1 + int(w * 0.3), w)
            y2 = min(y1 + int(h * 0.3), h)
            
            boxes.append([x1, y1, x2, y2])
            labels.append(self.vocabulary[i % len(self.vocabulary)])
            confidences.append(0.8 - i * 0.1)
            
        return torch.tensor(boxes, device=self.device), labels, torch.tensor(confidences, device=self.device)
    
    def _mock_segment_frame(self, image: np.ndarray, boxes: torch.Tensor, 
                           labels: List[str], frame_idx: int) -> Dict:
        """Mock segmentation for fallback."""
        h, w = image.shape[:2]
        masks = []
        track_ids = []
        
        for box in boxes:
            mask = np.zeros((h, w), dtype=bool)
            x1, y1, x2, y2 = box.int().tolist()
            mask[y1:y2, x1:x2] = True
            masks.append(mask)
            track_ids.append(self.next_track_id)
            self.next_track_id += 1
            
        return {
            'masks': masks,
            'labels': labels,
            'track_ids': track_ids
        }
    
    def run(self):
        """Main processing loop."""
        print(f"Starting Real Grounded-SAM2 semantic processor on {self.device}...")
        self.initialize_models()
        
        while True:
            try:
                frame_data = self.frame_queue.get(timeout=0.1)
                
                if frame_data is None:
                    break
                
                image = frame_data['img']
                frame_id = frame_data['frame_id']
                
                # Process frame
                semantic_data = self.process_frame(image, frame_id)
                
                # Put results in output queue
                self.result_queue.put(semantic_data)
                
                
            except Empty:
                continue
            except Exception as e:
                print(f"Error in semantic processor: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print("Semantic processor terminated.")
        if self.frame_times:
            print(f"Average processing time: {np.mean(self.frame_times)*1000:.1f}ms")


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