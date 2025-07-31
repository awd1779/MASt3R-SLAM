"""Grounded SAM2 processor with 3D geometric tracking support.

This version integrates the Geometric3DTracker for improved tracking accuracy.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import traceback
import multiprocessing as mp
from pathlib import Path

from mast3r_slam.grounded_sam2_real import RealGroundedSAM2Processor
from mast3r_slam.geometric_3d_tracker import Geometric3DTracker

logger = logging.getLogger('mast3r_slam.grounded_sam2_3d')


class RealGroundedSAM2Processor3D(RealGroundedSAM2Processor):
    """Extended processor with 3D geometric tracking."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_keyframe = None  # Store keyframe for 3D processing
    
    def run(self):
        """Main processing loop with 3D tracking support."""
        logger.info("Starting Grounded-SAM2 processor with 3D tracking support...")
        
        # Initialize models (from parent class)
        self.initialize_models()
        
        while True:
            msg = self.frame_queue.get()
            if msg is None:
                logger.info("Received termination signal")
                break
                
            img = msg['img']
            frame_id = msg['frame_id']
            keyframe_idx = msg.get('keyframe_idx')
            keyframe_data = msg.get('keyframe_data')  # Get keyframe data for 3D processing
            
            # Store keyframe data for 3D tracking
            if keyframe_data is not None:
                # Create a mock keyframe object with the necessary attributes
                class MockKeyframe:
                    def __init__(self, data):
                        self.img_shape = data['img_shape']
                        self.X_canon = data['X_canon']
                        self.T_WC = data['T_WC']
                
                self.current_keyframe = MockKeyframe(keyframe_data)
                logger.debug(f"Received keyframe data for frame {frame_id}")
            
            logger.info(f"Processing frame {frame_id} (keyframe_idx: {keyframe_idx}, has_keyframe_data: {keyframe_data is not None})")
            
            # Process the frame
            result = self.process_frame(img, frame_id, keyframe_idx)
            
            # Apply 3D tracking if enabled and keyframe available
            if self.tracker is not None and self.current_keyframe is not None:
                if hasattr(self.tracker, 'process_frame'):  # Check if it's the 3D tracker
                    logger.info(f"Applying 3D tracking for frame {frame_id} with {len(result.get('instance_ids', []))} instances")
                    try:
                        # Use 3D geometric tracking
                        track_assignments = self.tracker.process_frame(
                            self.current_keyframe,
                            result,
                            frame_id
                        )
                    except Exception as e:
                        logger.error(f"Error in 3D tracking: {e}")
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        track_assignments = {}
                    
                    # Update result with track IDs
                    result['track_ids'] = track_assignments
                    
                    # Collect tracking decisions
                    if hasattr(self.tracker, 'tracked_objects'):
                        decisions = []
                        for instance_id, track_id in track_assignments.items():
                            track_info = self.tracker.get_track_info(track_id)
                            if track_info:
                                decisions.append({
                                    'frame_id': frame_id,
                                    'instance_id': instance_id,
                                    'track_id': track_id,
                                    'label': track_info['label'],
                                    'confidence': track_info['confidence'],
                                    'method': '3d_geometric'
                                })
                        result['tracking_decisions'] = decisions
            else:
                # No tracking applied
                logger.warning(f"No tracking applied for frame {frame_id}: tracker={self.tracker is not None}, keyframe={self.current_keyframe is not None}")
                if not hasattr(self, 'tracker') or self.tracker is None:
                    logger.warning("  Reason: No tracker initialized")
                elif self.current_keyframe is None:
                    logger.warning("  Reason: No keyframe data available")
                elif not hasattr(self.tracker, 'process_frame'):
                    logger.warning("  Reason: Tracker does not have process_frame method")
            
            # Send result
            self.result_queue.put(result)
            logger.info(f"Sent result for frame {frame_id} with {len(result['instance_ids'])} instances")
        
        logger.info("Grounded-SAM2 processor with 3D tracking terminated")


def _run_processor_with_3d_tracker(frame_queue, result_queue, vocabulary, device, 
                                  model_selector, confidence_threshold, tracking_config, kwargs):
    """Helper function to run processor with 3D geometric tracker."""
    
    # Create tracker based on config
    tracker = None
    if tracking_config is not None:
        use_3d = tracking_config.get('use_3d_tracking', True)
        
        if use_3d:
            # Check if global tracking is enabled
            use_global = tracking_config.get('use_global_tracking', False)
            
            if use_global:
                from mast3r_slam.geometric_3d_tracker_global import Geometric3DTrackerGlobal
                tracker = Geometric3DTrackerGlobal(tracking_config)
                logger.info("Created 3D geometric tracker with GLOBAL matching")
            else:
                from mast3r_slam.geometric_3d_tracker import Geometric3DTracker
                tracker = Geometric3DTracker(tracking_config)
                logger.info("Created 3D geometric tracker in semantic processor process")
        else:
            from mast3r_slam.object_tracker_simple import SimpleObjectTracker
            tracker = SimpleObjectTracker(tracking_config)
            logger.info("Created simple 2D tracker in semantic processor process")
    
    # Create processor with 3D support
    processor = RealGroundedSAM2Processor3D(
        frame_queue, result_queue, vocabulary, device, model_selector, 
        confidence_threshold=confidence_threshold,
        **kwargs
    )
    
    if tracker is not None:
        processor.tracker = tracker
    
    processor.run()


def start_real_grounded_sam2_processor_3d(
    frame_queue,
    result_queue, 
    vocabulary: List[str],
    device: str = "cuda:1",
    model_selector=None,
    confidence_threshold: float = 0.3,
    dtype: str = "bfloat16",
    sam2_checkpoint_dir: str = None,
    grounding_dino_checkpoint_dir: str = None,
    debug_mode: bool = False,
    save_debug_visualizations: bool = False,
    deduplication_iou_threshold: float = 0.9,
    mask_refinement_threshold: float = 0.7,
    filter_empty_labels: bool = True,
    min_phrase_length: int = 2,
    empty_label_max_size: float = 0.4,
    tracking_config: Optional[Dict] = None
) -> mp.Process:
    """Start Grounded-SAM2 processor with 3D tracking support in a separate process."""
    
    kwargs = {
        'dtype': dtype,
        'sam2_checkpoint_dir': sam2_checkpoint_dir,
        'grounding_dino_checkpoint_dir': grounding_dino_checkpoint_dir,
        'debug_mode': debug_mode,
        'save_debug_visualizations': save_debug_visualizations,
        'deduplication_iou_threshold': deduplication_iou_threshold,
        'mask_refinement_threshold': mask_refinement_threshold,
        'filter_empty_labels': filter_empty_labels,
        'min_phrase_length': min_phrase_length,
        'empty_label_max_size': empty_label_max_size
    }
    
    process = mp.Process(
        target=_run_processor_with_3d_tracker,
        args=(frame_queue, result_queue, vocabulary, device, 
              model_selector, confidence_threshold, tracking_config, kwargs)
    )
    process.start()
    
    return process