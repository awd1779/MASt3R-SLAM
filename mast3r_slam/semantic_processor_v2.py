import time
import torch
import numpy as np
from typing import Optional, Dict, List, Tuple
import torch.multiprocessing as mp
from queue import Empty
from mast3r_slam.semantic_frame import encode_rle
from mast3r_slam.config import config
from mast3r_slam.grounded_sam2_config import SAM2_MODELS, GROUNDING_MODELS, GroundedSAM2ModelSelector


class GroundedSAM2Processor:
    """Advanced semantic processor with proper Grounded-SAM2 integration."""
    
    def __init__(self, 
                 frame_queue: mp.Queue,
                 result_queue: mp.Queue,
                 vocabulary: List[str],
                 device: str = "cuda:1",
                 model_selector: Optional[GroundedSAM2ModelSelector] = None):
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.vocabulary = vocabulary
        self.device = device
        
        # Model selection
        if model_selector is None:
            # Default: balanced performance
            self.model_selector = GroundedSAM2ModelSelector(
                target_fps=15,  # Minimum 15 FPS for real-time SLAM
                max_vram_gb=10,  # Typical GPU memory available
                quality_priority="quality"  # For AAAI paper
            )
        else:
            self.model_selector = model_selector
            
        # Select models
        self.sam2_model_name, self.grounding_model_name = self.model_selector.select_models()
        
        # Model instances
        self.sam2_predictor = None
        self.grounding_dino = None
        
        # Video tracking state
        self.inference_state = None
        self.video_segments = {}  # frame_idx -> segments
        self.propagated_masks = {}  # frame_idx -> masks
        
        # Track management
        self.current_tracks = {}  # track_id -> track_info
        self.next_track_id = 1
        
    def initialize_models(self):
        """Initialize Grounded-SAM2 models with selected configurations."""
        try:
            # Check if we should use real models or mock
            use_mock = config["semantic_segmentation"].get("use_mock", True)
            
            if use_mock:
                print("Using mock semantic processor (set use_mock=false for real models)")
                return
                
            # Import only if using real models
            from sam2.build_sam import build_sam2_video_predictor
            from groundingdino.util.inference import load_model
            from groundingdino.util.slconfig import SLConfig
            
            # Get model configs
            sam2_config = SAM2_MODELS[self.sam2_model_name]
            grounding_config = GROUNDING_MODELS[self.grounding_model_name]
            
            print(f"Loading SAM2 model: {self.sam2_model_name}")
            print(f"  - Parameters: {sam2_config['params']/1e6:.1f}M")
            print(f"  - Expected FPS: {sam2_config['fps_estimate']}")
            print(f"  - VRAM: {sam2_config['vram_gb']}GB")
            
            # Initialize SAM2 video predictor
            self.sam2_predictor = build_sam2_video_predictor(
                config_file=sam2_config['config'],
                ckpt_path=sam2_config['checkpoint'],
                device=self.device
            )
            
            print(f"\nLoading Grounding DINO model: {self.grounding_model_name}")
            print(f"  - Parameters: {grounding_config['params']/1e6:.1f}M")
            print(f"  - Expected FPS: {grounding_config['fps_estimate']}")
            print(f"  - VRAM: {grounding_config['vram_gb']}GB")
            
            # Initialize Grounding DINO
            args = SLConfig.fromfile(grounding_config['config'])
            self.grounding_dino = load_model(
                args, 
                grounding_config['checkpoint'],
                device=self.device
            )
            
            # Print combined stats
            model_info = self.model_selector.get_model_info(
                self.sam2_model_name, 
                self.grounding_model_name
            )
            print(f"\nCombined model statistics:")
            print(f"  - Total parameters: {model_info['combined']['total_params']}")
            print(f"  - Total VRAM: {model_info['combined']['total_vram']}")
            print(f"  - Expected FPS: {model_info['combined']['expected_fps']}")
            print(f"  - Bottleneck: {model_info['combined']['bottleneck']}")
            
        except ImportError as e:
            print(f"Warning: Required packages not installed: {e}")
            print("Using mock semantic processor. Install with:")
            print("  pip install segment-anything-2")
            print("  pip install groundingdino")
            self.sam2_predictor = None
            self.grounding_dino = None
            
    def ground_objects_in_frame(self, image: np.ndarray) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        """Use Grounding DINO to detect objects based on text prompts."""
        if self.grounding_dino is None:
            return self._mock_ground_objects(image)
            
        from groundingdino.util.inference import predict
        
        # Prepare text prompt
        caption = ". ".join(self.vocabulary) + "."
        
        # Run grounding
        boxes, logits, phrases = predict(
            model=self.grounding_dino,
            image=image,
            caption=caption,
            box_threshold=config["semantic_segmentation"]["grounded_sam2"]["confidence_threshold"],
            text_threshold=0.25,
            device=self.device
        )
        
        return boxes, phrases, logits
    
    def segment_frame_with_boxes(self, image: np.ndarray, boxes: torch.Tensor, 
                                 labels: List[str], frame_idx: int) -> Dict:
        """Use SAM2 to segment objects given bounding boxes."""
        if self.sam2_predictor is None:
            return self._mock_segment_frame(image, boxes, labels, frame_idx)
            
        # Initialize video if first frame
        if self.inference_state is None:
            self.inference_state = self.sam2_predictor.init_state(video_path=None)
            
        # Add frame to video predictor
        self.sam2_predictor.add_new_frame(self.inference_state, frame_idx, image)
        
        # Process each box
        masks = []
        for box_idx, (box, label) in enumerate(zip(boxes, labels)):
            # Convert box to SAM2 format
            box_np = box.cpu().numpy()
            
            # Add prompt for this object
            _, out_obj_ids, out_mask_logits = self.sam2_predictor.add_new_prompt(
                self.inference_state,
                frame_idx=frame_idx,
                obj_id=self.next_track_id,
                bbox=box_np
            )
            
            # Store track info
            self.current_tracks[self.next_track_id] = {
                'label': label,
                'first_frame': frame_idx,
                'box': box_np
            }
            self.next_track_id += 1
            
            masks.append(out_mask_logits)
            
        return self._process_sam2_output(masks, labels, frame_idx)
    
    def propagate_in_video(self, start_frame: int, end_frame: int) -> Dict[int, Dict]:
        """Propagate masks across video frames for temporal consistency."""
        if self.sam2_predictor is None or self.inference_state is None:
            return {}
            
        # Propagate masks
        for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_predictor.propagate_in_video(
            self.inference_state,
            start_frame_idx=start_frame,
            max_frame_num_to_track=end_frame - start_frame
        ):
            self.propagated_masks[out_frame_idx] = {
                'obj_ids': out_obj_ids,
                'masks': out_mask_logits
            }
            
        return self.propagated_masks
    
    def process_frame(self, image: np.ndarray, frame_id: int) -> Dict:
        """Process a single frame to generate semantic masks."""
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
                'timestamp': time.time()
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
            'timestamp': time.time()
        }
        
        for idx, (mask, label, conf) in enumerate(zip(
            segmentation_result['masks'],
            segmentation_result['labels'],
            confidences
        )):
            instance_id = idx + 1
            
            # Encode mask to RLE
            mask_tensor = torch.from_numpy(mask).to(dtype=torch.bool)
            rle_mask = encode_rle(mask_tensor)
            
            # Get track ID from SAM2's tracking
            track_id = segmentation_result.get('track_ids', [self.next_track_id + idx])[idx]
            
            # Store results
            results['masks_rle'][instance_id] = rle_mask
            results['instance_ids'].append(instance_id)
            results['track_ids'][instance_id] = track_id
            results['labels'][instance_id] = label
            results['confidences'][instance_id] = float(conf)
            
        return results
    
    def _mock_ground_objects(self, image: np.ndarray) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        """Mock object grounding for testing."""
        h, w = image.shape[:2]
        
        # Generate mock boxes
        num_objects = min(3, len(self.vocabulary))
        boxes = []
        labels = []
        confidences = []
        
        for i in range(num_objects):
            # Create a box
            x1 = int(w * 0.1 * (i + 1))
            y1 = int(h * 0.1 * (i + 1))
            x2 = min(x1 + int(w * 0.3), w)
            y2 = min(y1 + int(h * 0.3), h)
            
            boxes.append([x1, y1, x2, y2])
            labels.append(self.vocabulary[i % len(self.vocabulary)])
            confidences.append(0.9 - i * 0.1)
            
        return torch.tensor(boxes), labels, torch.tensor(confidences)
    
    def _mock_segment_frame(self, image: np.ndarray, boxes: torch.Tensor, 
                           labels: List[str], frame_idx: int) -> Dict:
        """Mock segmentation for testing."""
        h, w = image.shape[:2]
        masks = []
        track_ids = []
        
        for box in boxes:
            # Create mask from box
            mask = np.zeros((h, w), dtype=bool)
            x1, y1, x2, y2 = box.int().tolist()
            mask[y1:y2, x1:x2] = True
            masks.append(mask)
            
            # Assign track ID
            track_ids.append(self.next_track_id)
            self.next_track_id += 1
            
        return {
            'masks': masks,
            'labels': labels,
            'track_ids': track_ids
        }
    
    def _process_sam2_output(self, masks: List[torch.Tensor], 
                            labels: List[str], frame_idx: int) -> Dict:
        """Process SAM2 output masks."""
        processed_masks = []
        track_ids = []
        
        for mask_logit, track_id in zip(masks, range(self.next_track_id - len(masks), self.next_track_id)):
            # Convert logits to binary mask
            mask = (mask_logit > 0).cpu().numpy()
            processed_masks.append(mask)
            track_ids.append(track_id)
            
        return {
            'masks': processed_masks,
            'labels': labels,
            'track_ids': track_ids
        }
    
    def run(self):
        """Main processing loop."""
        print("Starting Grounded-SAM2 semantic processor...")
        self.initialize_models()
        
        while True:
            try:
                # Get frame from queue with timeout
                frame_data = self.frame_queue.get(timeout=0.1)
                
                if frame_data is None:  # Termination signal
                    break
                
                # Extract data
                image = frame_data['img']  # NumPy array [H, W, 3]
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


def start_grounded_sam2_processor(frame_queue: mp.Queue, 
                                 result_queue: mp.Queue,
                                 vocabulary: List[str],
                                 device: str = "cuda:1",
                                 model_selector: Optional[GroundedSAM2ModelSelector] = None) -> mp.Process:
    """Start the Grounded-SAM2 processor in a separate process."""
    processor = GroundedSAM2Processor(frame_queue, result_queue, vocabulary, device, model_selector)
    process = mp.Process(target=processor.run)
    process.start()
    return process