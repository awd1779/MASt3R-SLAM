"""
Improved Semantic Processor V2 for MASt3R-SLAM
Integrates batch processing, caching, post-processing, and ensemble models.
"""
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

# Import our new modules
from .semantic_batch_processor import SemanticBatchProcessor, BatchFrame
from .semantic_cache import SemanticFeatureCache, ViewpointCache
from .semantic_ensemble import SemanticEnsemble, CLIPModelConfig

# Import from consolidated utils
from .semantic_utils import (
    SemanticMapper,
    SemanticPostProcessor,
    SemanticDuplicateFilter
)

# Import scene graph (keep separate as it's rarely used)
from .semantic_hierarchy import SemanticSceneGraph

# Import from new consolidated modules
from .semantic_core import (
    load_instance_segmentation_model,
    TEXT_PROMPTS,
    SEMANTIC_DEVICE,
    enable_debug_visualization,
    set_debug_options
)


class SemanticProcessorV2:
    """
    Enhanced semantic processor with performance optimizations and improved accuracy.
    """
    
    def __init__(self,
                 device: str = None,
                 enable_batch_processing: bool = True,
                 enable_caching: bool = True,
                 enable_post_processing: bool = True,
                 enable_ensemble: bool = False,  # Disabled by default for speed
                 cache_dir: Optional[Path] = None,
                 text_prompts: List[str] = None):
        """
        Initialize the improved semantic processor.
        
        Args:
            device: Device for computation
            enable_batch_processing: Use batch processing optimization
            enable_caching: Use semantic caching
            enable_post_processing: Use temporal post-processing
            enable_ensemble: Use multi-model ensemble
            cache_dir: Directory for persistent cache
            text_prompts: Custom text prompts for classification
        """
        self.device = device or SEMANTIC_DEVICE
        self.text_prompts = text_prompts or TEXT_PROMPTS
        
        # Track last processed keyframe for adaptive processing
        self.last_keyframe_data = None
        self.last_keyframe_result = None
        
        # Feature flags
        self.enable_batch_processing = enable_batch_processing
        self.enable_caching = enable_caching
        self.enable_post_processing = enable_post_processing
        self.enable_ensemble = enable_ensemble
        
        # Initialize components
        self.sam_generator = None
        self.clip_model = None
        self.clip_preprocess = None
        self.text_features = None
        
        # Initialize optimizations
        if enable_batch_processing:
            self.batch_processor = SemanticBatchProcessor(
                batch_size=32,
                device=self.device
            )
        
        if enable_caching:
            self.feature_cache = SemanticFeatureCache(
                max_cache_size=1000,
                device=self.device,
                persistent_cache_dir=cache_dir / 'feature_cache' if cache_dir else None
            )
            self.viewpoint_cache = ViewpointCache(
                position_threshold=0.5,
                rotation_threshold=15.0
            )
        
        if enable_post_processing:
            self.post_processor = SemanticPostProcessor(
                temporal_window=5,
                device=self.device
            )
        
        if enable_ensemble:
            self.ensemble = SemanticEnsemble(
                model_configs=[
                    CLIPModelConfig.VITB32_LAION,
                    CLIPModelConfig.VITH14_LAION
                ],
                device=self.device
            )
        
        # Additional components
        self.semantic_mapper = SemanticMapper(device=self.device)
        self.duplicate_filter = SemanticDuplicateFilter()
        
        # Performance tracking
        self.timing_stats = {
            'sam_segmentation': [],
            'clip_inference': [],
            'post_processing': [],
            'total_time': []
        }
        
    def initialize_models(self):
        """Load and initialize all models."""
        print("[SemanticProcessorV2] Initializing models...")
        
        # Load SAM
        self.sam_generator = load_instance_segmentation_model(device=self.device)
        
        # Load CLIP models
        if self.enable_ensemble:
            self.ensemble.load_models(self.text_prompts)
        else:
            # Load single CLIP model
            from .semantic_core import load_clip_model
            self.clip_model, self.clip_preprocess, self.text_features, _ = load_clip_model(
                device=self.device
            )
        
        print("[SemanticProcessorV2] Models initialized successfully")
    
    def process_frame(self,
                     frame_data: Dict,
                     mode: str = 'full',
                     use_cache: bool = True) -> Dict:
        """
        Process a frame with all optimizations.
        
        Args:
            frame_data: Dictionary containing:
                - 'image_tensor': Image tensor (C, H, W) in [0, 1] range
                - 'frame_id': Frame identifier
                - 'pose': Camera pose (optional, for viewpoint caching)
                - 'global_ids': Previous global IDs (optional)
            mode: Processing mode ('full', 'fast', 'cache_only')
            use_cache: Whether to use caching
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        
        image_tensor = frame_data['image_tensor']
        frame_id = frame_data.get('frame_id', 0)
        pose = frame_data.get('pose', None)
        
        # Check input normalization
        if image_tensor.min() < -0.1:
            # Convert from [-1, 1] to [0, 1]
            image_tensor = (image_tensor + 1.0) / 2.0
        
        # Try cache first if enabled
        if use_cache and self.enable_caching and pose is not None:
            cached_result = self.viewpoint_cache.get_cached_semantics(pose)
            if cached_result is not None and mode == 'cache_only':
                return cached_result
        
        # Perform segmentation
        sam_time_start = time.time()
        sam_masks = self._perform_segmentation(image_tensor)
        self.timing_stats['sam_segmentation'].append(time.time() - sam_time_start)
        
        if not sam_masks:
            return self._empty_result(frame_id)
        
        # Perform classification
        clip_time_start = time.time()
        if self.enable_batch_processing and len(sam_masks) > 10:
            # Use batch processing for many masks
            classification_results = self._batch_classification(
                image_tensor, sam_masks, frame_id
            )
        else:
            # Use standard processing for few masks
            classification_results = self._standard_classification(
                image_tensor, sam_masks
            )
        self.timing_stats['clip_inference'].append(time.time() - clip_time_start)
        
        # Build semantic map
        local_instance_mask, local_id_to_class_map = self._build_semantic_map(
            image_tensor.shape[1:], sam_masks, classification_results
        )
        
        # Apply post-processing if enabled
        if self.enable_post_processing and frame_data.get('global_ids') is not None:
            post_time_start = time.time()
            post_results = self._apply_post_processing(
                local_instance_mask,
                classification_results['confidences'],
                frame_data.get('points_3d'),
                frame_data.get('global_ids'),
                frame_id
            )
            local_instance_mask = post_results['labels']
            self.timing_stats['post_processing'].append(time.time() - post_time_start)
        
        # Cache result if enabled
        if use_cache and self.enable_caching and pose is not None:
            self.viewpoint_cache.add_semantics(pose, {
                'local_instance_mask': local_instance_mask,
                'local_id_to_class_map': local_id_to_class_map,
                'classification_results': classification_results
            })
        
        total_time = time.time() - start_time
        self.timing_stats['total_time'].append(total_time)
        
        return {
            'local_instance_mask': local_instance_mask,
            'local_id_to_class_map': local_id_to_class_map,
            'classification_results': classification_results,
            'sam_masks': sam_masks,
            'processing_time': total_time,
            'frame_id': frame_id
        }
    
    def _perform_segmentation(self, image_tensor: torch.Tensor) -> List[Dict]:
        """Perform SAM segmentation."""
        # Convert to numpy uint8
        image_hwc = (image_tensor.permute(1, 2, 0) * 255.0).byte().cpu().numpy()
        
        # Generate masks
        masks = self.sam_generator.generate(image_hwc)
        
        # Sort by area (largest first)
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        return masks
    
    def _batch_classification(self, 
                            image_tensor: torch.Tensor,
                            sam_masks: List[Dict],
                            frame_id: int) -> Dict:
        """Perform batched CLIP classification."""
        # Create batch frame
        batch_frame = BatchFrame(
            frame_id=frame_id,
            image_tensor=image_tensor,
            sam_masks=sam_masks
        )
        
        # Process with batch processor
        if self.enable_ensemble:
            # Use ensemble for batch processing
            results = self._batch_ensemble_classification([batch_frame])
        else:
            # Create normalize transform for batch processing
            import torchvision
            normalize_transform = torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
            
            # Use single model batch processing
            results = self.batch_processor.process_frames(
                [batch_frame],
                self.clip_model,
                self.text_features,
                self.text_prompts,
                normalize_transform
            )
        
        # Extract results for this frame
        frame_results = results.get(frame_id, {})
        
        return {
            'predictions': frame_results.get('predictions', []),
            'confidences': frame_results.get('confidences', []),
            'metadata': frame_results.get('metadata', [])
        }
    
    def _standard_classification(self,
                               image_tensor: torch.Tensor,
                               sam_masks: List[Dict]) -> Dict:
        """Perform standard CLIP classification."""
        predictions = []
        confidences = []
        
        for mask in sam_masks:
            # Extract crop
            crop = self._extract_crop(image_tensor, mask['segmentation'])
            
            if crop is None:
                predictions.append('background')
                confidences.append(0.0)
                continue
            
            # Check cache if enabled
            if self.enable_caching:
                cached_features, missing = self.feature_cache.get_cached_features([crop])
                if 0 not in missing:
                    # Use cached result
                    # (Would need to store predictions with features)
                    pass
            
            # Perform inference
            if self.enable_ensemble:
                result = self.ensemble.predict_ensemble(crop.unsqueeze(0))
                pred_idx = result['predictions'][0].item()
                confidence = result['scores'][0].item()
            else:
                # Single model inference
                with torch.no_grad():
                    # Normalize the crop tensor directly (skip CLIP preprocess which expects PIL)
                    # Use the normalize transform from the original semantic_processor
                    import torchvision
                    normalize = torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]
                    )
                    normalized = normalize(crop.unsqueeze(0))
                    
                    # Get features
                    image_features = self.clip_model.encode_image(normalized)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    
                    # Compute similarities
                    similarities = (100.0 * image_features @ self.text_features.T)
                    score, idx = similarities.max(dim=1)
                    
                    pred_idx = idx[0].item()
                    confidence = self._calibrate_confidence(score[0].item())
            
            predictions.append(self.text_prompts[pred_idx])
            confidences.append(confidence)
        
        return {
            'predictions': predictions,
            'confidences': confidences,
            'metadata': [{'mask_idx': i} for i in range(len(sam_masks))]
        }
    
    def _build_semantic_map(self,
                          image_shape: Tuple[int, int],
                          sam_masks: List[Dict],
                          classification_results: Dict) -> Tuple[torch.Tensor, Dict]:
        """Build semantic instance mask and label map."""
        h, w = image_shape
        local_instance_mask = torch.zeros((h, w), dtype=torch.int64, device=self.device)
        local_id_to_class_map = {0: 'background'}
        
        current_id = 1
        
        for i, (mask, pred, conf) in enumerate(zip(
            sam_masks,
            classification_results['predictions'],
            classification_results['confidences']
        )):
            if conf > 25.0:  # Confidence threshold
                mask_tensor = torch.from_numpy(mask['segmentation']).to(self.device)
                valid_pixels = mask_tensor & (local_instance_mask == 0)
                
                if valid_pixels.any():
                    local_instance_mask[valid_pixels] = current_id
                    local_id_to_class_map[current_id] = pred
                    current_id += 1
        
        return local_instance_mask, local_id_to_class_map
    
    def _apply_post_processing(self,
                             labels: torch.Tensor,
                             confidences: List[float],
                             points_3d: Optional[torch.Tensor],
                             global_ids: torch.Tensor,
                             frame_id: int) -> Dict:
        """Apply temporal and spatial post-processing."""
        # Convert confidence list to tensor
        conf_tensor = torch.tensor(confidences, device=self.device)
        
        # Expand to match points if needed
        if conf_tensor.shape[0] != labels.shape[0]:
            # Map confidences to labels
            unique_labels = torch.unique(labels)
            label_conf = torch.zeros_like(labels, dtype=torch.float32)
            
            for label_id in unique_labels:
                if label_id > 0 and label_id <= len(confidences):
                    label_conf[labels == label_id] = confidences[label_id - 1]
        else:
            label_conf = conf_tensor
        
        # Apply post-processing
        return self.post_processor.process_frame(
            labels=labels,
            confidences=label_conf,
            points_3d=points_3d,
            global_ids=global_ids,
            frame_id=frame_id
        )
    
    def _extract_crop(self, image_tensor: torch.Tensor, mask: np.ndarray) -> Optional[torch.Tensor]:
        """Extract and preprocess crop from mask."""
        # Implementation similar to _extract_crop_optimized in batch processor
        mask_tensor = torch.from_numpy(mask).to(self.device)
        
        if not mask_tensor.any():
            return None
        
        # Find bounding box
        rows = torch.any(mask_tensor, dim=1)
        cols = torch.any(mask_tensor, dim=0)
        
        if not rows.any() or not cols.any():
            return None
        
        ymin, ymax = torch.where(rows)[0][[0, -1]]
        xmin, xmax = torch.where(cols)[0][[0, -1]]
        
        # Add adaptive padding
        h, w = mask_tensor.shape
        height = ymax - ymin + 1
        width = xmax - xmin + 1
        
        pad_ratio = 0.2 if height * width < 0.05 * h * w else 0.1
        pad_h = max(5, int(height * pad_ratio))
        pad_w = max(5, int(width * pad_ratio))
        
        ymin_pad = max(0, ymin - pad_h)
        ymax_pad = min(h - 1, ymax + pad_h)
        xmin_pad = max(0, xmin - pad_w)
        xmax_pad = min(w - 1, xmax + pad_w)
        
        # Extract and resize
        crop = image_tensor[:, ymin_pad:ymax_pad+1, xmin_pad:xmax_pad+1]
        
        crop_resized = torch.nn.functional.interpolate(
            crop.unsqueeze(0),
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        return crop_resized
    
    def _calibrate_confidence(self, raw_score: float) -> float:
        """Calibrate CLIP confidence score."""
        min_expected = 15.0
        max_expected = 40.0
        
        clamped = max(min_expected, min(raw_score, max_expected))
        normalized = (clamped - min_expected) / (max_expected - min_expected)
        calibrated = normalized * 80 + 20
        
        return calibrated
    
    def _empty_result(self, frame_id: int) -> Dict:
        """Return empty result when no masks found."""
        h, w = 480, 640  # Default size
        return {
            'local_instance_mask': torch.zeros((h, w), dtype=torch.int64, device=self.device),
            'local_id_to_class_map': {0: 'background'},
            'classification_results': {
                'predictions': [],
                'confidences': [],
                'metadata': []
            },
            'sam_masks': [],
            'processing_time': 0.0,
            'frame_id': frame_id
        }
    
    def get_performance_summary(self) -> Dict:
        """Get performance statistics."""
        summary = {}
        
        for key, times in self.timing_stats.items():
            if times:
                summary[key] = {
                    'mean_ms': np.mean(times) * 1000,
                    'std_ms': np.std(times) * 1000,
                    'total_s': np.sum(times),
                    'count': len(times)
                }
        
        # Add cache stats if enabled
        if self.enable_caching:
            summary['cache_efficiency'] = self.feature_cache.compute_cache_efficiency()
        
        # Add ensemble stats if enabled
        if self.enable_ensemble:
            summary['ensemble_performance'] = self.ensemble.get_performance_summary()
        
        return summary
    
    def should_run_full_segmentation(self, current_pose, last_keyframe_pose):
        """Decide if full segmentation needed based on pose change.
        
        Args:
            current_pose: Current frame's Sim3 pose (T_WC)
            last_keyframe_pose: Last keyframe's Sim3 pose
            
        Returns:
            bool: True if full segmentation needed
        """
        if last_keyframe_pose is None:
            return True
        
        # Compute relative transformation
        # T_rel = T_last^-1 * T_curr
        T_rel = last_keyframe_pose.inv() * current_pose
        
        # Extract metrics from Sim3 transformation matrix
        # Get the 4x4 matrix representation
        T_rel_matrix = T_rel.matrix()  # Shape: (1, 4, 4)
        
        # Translation distance (last column, first 3 elements)
        translation = T_rel_matrix[0, :3, 3]
        translation_dist = translation.norm().item()
        
        # Rotation angle from rotation matrix
        # Extract 3x3 rotation part
        R = T_rel_matrix[0, :3, :3]
        # Compute trace to get rotation angle
        trace = R.trace()
        # Clamp to avoid numerical issues
        cos_angle = (trace - 1.0) / 2.0
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        rotation_angle = torch.acos(cos_angle).item()
        
        # Scale change from Sim3
        # Scale is the determinant^(1/3) of the 3x3 part
        scale = torch.det(R).pow(1/3).item()
        scale_change = abs(scale - 1.0)
        
        # Adaptive thresholds
        translation_threshold = 0.5  # meters
        rotation_threshold = np.radians(30.0)  # 30 degrees in radians
        scale_threshold = 0.1  # 10% scale change
        
        # Check if significant motion
        needs_full_segmentation = (
            translation_dist > translation_threshold or
            rotation_angle > rotation_threshold or
            scale_change > scale_threshold
        )
        
        if not needs_full_segmentation:
            print(f"[SemanticProcessorV2] Skipping full segmentation - small motion: "
                  f"trans={translation_dist:.3f}m, rot={np.degrees(rotation_angle):.1f}°, scale={scale_change:.3f}")
        
        return needs_full_segmentation
    
    def shutdown(self):
        """Clean up resources."""
        if self.enable_batch_processing:
            self.batch_processor.shutdown()
        
        if self.enable_caching:
            self.feature_cache.save_persistent_cache()