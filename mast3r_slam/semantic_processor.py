import time
import torch
import numpy as np
from typing import Optional, Dict, List, Tuple
import torch.multiprocessing as mp
from queue import Empty
from mast3r_slam.semantic_frame import encode_rle
from mast3r_slam.config import config
import logging

# Create logger for this module
logger = logging.getLogger(__name__)


class SemanticProcessor:
    """Process semantic segmentation in a separate process to avoid blocking SLAM."""
    
    def __init__(self, 
                 frame_queue: mp.Queue,
                 result_queue: mp.Queue,
                 vocabulary: List[str],
                 device: str = "cuda:1"):
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.vocabulary = vocabulary
        self.device = device
        self.predictor = None
        self.grounding_model = None
        self.sam_model = None
        self.current_tracks = {}  # Track ID management
        self.next_track_id = 1
        
    def initialize_models(self):
        """Initialize Grounded-SAM2 models."""
        try:
            # Import Grounded-SAM2 components
            from grounded_sam2.grounding_dino import GroundingDINO
            from grounded_sam2.sam2 import SAM2VideoPredictor
            
            # Load models based on config
            sam_config = config["semantic_segmentation"]["grounded_sam2"]
            
            # Initialize SAM2 video predictor
            self.sam_model = SAM2VideoPredictor(
                model_type=sam_config["model_type"],
                device=self.device
            )
            
            # Initialize Grounding DINO
            self.grounding_model = GroundingDINO(
                model_type=sam_config["grounding_model"],
                device=self.device
            )
            
            logger.info(f"Semantic models initialized on {self.device}")
            
        except ImportError:
            logger.warning("Grounded-SAM2 not installed. Using mock semantic processor.")
            self.predictor = None
    
    def process_frame(self, image: np.ndarray, frame_id: int, keyframe_idx: Optional[int] = None) -> Dict:
        """Process a single frame to generate semantic masks."""
        if self.predictor is None:
            # Mock processing for testing without Grounded-SAM2
            return self._mock_process_frame(image, frame_id, keyframe_idx)
        
        # Real Grounded-SAM2 processing
        h, w = image.shape[:2]
        results = {
            'frame_id': frame_id,
            'keyframe_idx': keyframe_idx,  # Direct mapping to keyframe
            'masks_rle': {},
            'instance_ids': [],
            'track_ids': {},
            'labels': {},
            'confidences': {},
            'timestamp': time.time()
        }
        
        try:
            # Ground objects using text prompts
            detections = self.grounding_model.predict(
                image=image,
                text_prompt=", ".join(self.vocabulary),
                confidence_threshold=config["semantic_segmentation"]["grounded_sam2"]["confidence_threshold"]
            )
            
            if len(detections.boxes) == 0:
                return results
            
            # Generate masks using SAM2
            masks, scores, labels = self.sam_model.predict(
                image=image,
                boxes=detections.boxes,
                multimask_output=False
            )
            
            # Process each detection
            instance_id = 1
            for mask, score, label_idx in zip(masks, scores, detections.labels):
                if score < config["semantic_segmentation"]["grounded_sam2"]["confidence_threshold"]:
                    continue
                
                # Encode mask to RLE
                mask_tensor = torch.from_numpy(mask).to(dtype=torch.bool)
                rle_mask = encode_rle(mask_tensor)
                
                # Assign track ID (simplified tracking)
                track_id = self._assign_track_id(mask_tensor, label_idx)
                
                # Store results
                results['masks_rle'][instance_id] = rle_mask
                results['instance_ids'].append(instance_id)
                results['track_ids'][instance_id] = track_id
                results['labels'][instance_id] = self.vocabulary[label_idx]
                results['confidences'][instance_id] = float(score)
                
                instance_id += 1
                
        except Exception as e:
            logger.error(f"Error in semantic processing: {e}")
        
        return results
    
    def _mock_process_frame(self, image: np.ndarray, frame_id: int, keyframe_idx: Optional[int] = None) -> Dict:
        """Mock semantic processing for testing."""
        h, w = image.shape[:2]
        
        # Create mock masks for testing
        results = {
            'frame_id': frame_id,
            'keyframe_idx': keyframe_idx,
            'masks_rle': {},
            'instance_ids': [],
            'track_ids': {},
            'labels': {},
            'confidences': {},
            'timestamp': time.time()
        }
        
        # Generate a few mock instances
        num_instances = min(3, len(self.vocabulary))
        for i in range(num_instances):
            instance_id = i + 1
            
            # Create a simple rectangular mask
            mask = torch.zeros((h, w), dtype=torch.bool)
            y_start = int(h * 0.2 * (i + 1))
            y_end = min(y_start + int(h * 0.2), h)
            x_start = int(w * 0.2 * (i + 1))
            x_end = min(x_start + int(w * 0.2), w)
            mask[y_start:y_end, x_start:x_end] = True
            
            # Encode to RLE
            rle_mask = encode_rle(mask)
            
            # Assign track ID
            track_id = self._assign_track_id(mask, i % len(self.vocabulary))
            
            # Store results
            results['masks_rle'][instance_id] = rle_mask
            results['instance_ids'].append(instance_id)
            results['track_ids'][instance_id] = track_id
            results['labels'][instance_id] = self.vocabulary[i % len(self.vocabulary)]
            results['confidences'][instance_id] = 0.9 - (i * 0.1)
        
        return results
    
    def _assign_track_id(self, mask: torch.Tensor, label_idx: int) -> int:
        """Simple track ID assignment (to be improved with proper tracking)."""
        # For now, just assign incrementing IDs
        track_id = self.next_track_id
        self.next_track_id += 1
        return track_id
    
    def run(self):
        """Main processing loop."""
        logger.info("Starting semantic processor...")
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
                keyframe_idx = frame_data.get('keyframe_idx')  # May be None for old-style
                
                # Process frame
                semantic_data = self.process_frame(image, frame_id, keyframe_idx)
                
                # Put results in output queue
                self.result_queue.put(semantic_data)
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in semantic processor: {e}")
                continue
        
        logger.info("Semantic processor terminated.")


def start_semantic_processor(frame_queue: mp.Queue, 
                           result_queue: mp.Queue,
                           vocabulary: List[str],
                           device: str = "cuda:1") -> mp.Process:
    """Start the semantic processor in a separate process."""
    processor = SemanticProcessor(frame_queue, result_queue, vocabulary, device)
    process = mp.Process(target=processor.run)
    process.start()
    return process