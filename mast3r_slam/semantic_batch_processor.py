"""
Batch CLIP Processing Optimization for MASt3R-SLAM
Enables efficient processing of multiple frames simultaneously.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor
import queue


@dataclass
class BatchFrame:
    """Container for frame data in batch processing."""
    frame_id: int
    image_tensor: torch.Tensor
    sam_masks: List[Dict]
    confidence_threshold: float = 25.0


class SemanticBatchProcessor:
    """
    Optimized batch processor for CLIP inference across multiple frames.
    """
    
    def __init__(self,
                 batch_size: int = 32,
                 prefetch_size: int = 4,
                 num_workers: int = 2,
                 device: str = 'cuda'):
        """
        Initialize the batch processor.
        
        Args:
            batch_size: Maximum batch size for CLIP inference
            prefetch_size: Number of batches to prefetch
            num_workers: Number of workers for data preparation
            device: Device for computation
        """
        self.batch_size = batch_size
        self.prefetch_size = prefetch_size
        self.num_workers = num_workers
        self.device = device
        
        # Queues for async processing
        self.crop_queue = queue.Queue(maxsize=prefetch_size)
        self.result_queue = queue.Queue()
        
        # Thread pool for data preparation
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        
        # Cache for preprocessed crops
        self.crop_cache = {}
        
        # Performance metrics
        self.timing_stats = {
            'crop_preparation': [],
            'clip_inference': [],
            'post_processing': []
        }
        
    def prepare_crops_batch(self, frames: List[BatchFrame]) -> Dict:
        """
        Prepare crops from multiple frames in parallel.
        
        Args:
            frames: List of frames to process
            
        Returns:
            Dictionary with batched crops and metadata
        """
        start_time = time.time()
        
        all_crops = []
        all_metadata = []
        
        # Process frames in parallel
        def process_single_frame(frame):
            crops = []
            metadata = []
            
            for mask_idx, mask_data in enumerate(frame.sam_masks):
                # Extract crop using the mask
                crop = self._extract_crop_optimized(
                    frame.image_tensor, 
                    mask_data['segmentation']
                )
                
                if crop is not None:
                    crops.append(crop)
                    metadata.append({
                        'frame_id': frame.frame_id,
                        'mask_idx': mask_idx,
                        'mask_area': mask_data['area']
                    })
                    
            return crops, metadata
        
        # Submit all frames for processing
        futures = [self.executor.submit(process_single_frame, frame) for frame in frames]
        
        # Collect results
        for future in futures:
            crops, metadata = future.result()
            all_crops.extend(crops)
            all_metadata.extend(metadata)
        
        # Stack crops into batches
        batched_crops = []
        batched_metadata = []
        
        for i in range(0, len(all_crops), self.batch_size):
            batch_end = min(i + self.batch_size, len(all_crops))
            batch_crops = torch.stack(all_crops[i:batch_end])
            batched_crops.append(batch_crops)
            batched_metadata.append(all_metadata[i:batch_end])
        
        prep_time = time.time() - start_time
        self.timing_stats['crop_preparation'].append(prep_time)
        
        return {
            'batches': batched_crops,
            'metadata': batched_metadata,
            'total_crops': len(all_crops)
        }
    
    def _extract_crop_optimized(self, 
                               image_tensor: torch.Tensor,
                               mask: np.ndarray,
                               target_size: Tuple[int, int] = (224, 224)) -> Optional[torch.Tensor]:
        """
        Optimized crop extraction with GPU acceleration.
        
        Args:
            image_tensor: Input image tensor (C, H, W)
            mask: Binary mask array
            target_size: Target crop size
            
        Returns:
            Cropped and resized tensor or None
        """
        # Convert mask to tensor
        mask_tensor = torch.from_numpy(mask).to(self.device)
        
        if not mask_tensor.any():
            return None
        
        # Find bounding box using GPU operations
        rows = torch.any(mask_tensor, dim=1)
        cols = torch.any(mask_tensor, dim=0)
        
        if not rows.any() or not cols.any():
            return None
        
        ymin, ymax = torch.where(rows)[0][[0, -1]]
        xmin, xmax = torch.where(cols)[0][[0, -1]]
        
        # Add padding
        h, w = mask_tensor.shape
        height = ymax - ymin + 1
        width = xmax - xmin + 1
        
        # Adaptive padding based on object size
        pad_ratio = 0.2 if height * width < 0.05 * h * w else 0.1
        pad_h = max(5, int(height * pad_ratio))
        pad_w = max(5, int(width * pad_ratio))
        
        # Apply padding with bounds
        ymin_pad = max(0, ymin - pad_h)
        ymax_pad = min(h - 1, ymax + pad_h)
        xmin_pad = max(0, xmin - pad_w)
        xmax_pad = min(w - 1, xmax + pad_w)
        
        # Extract crop
        crop = image_tensor[:, ymin_pad:ymax_pad+1, xmin_pad:xmax_pad+1]
        
        # Resize to target size
        crop_resized = torch.nn.functional.interpolate(
            crop.unsqueeze(0),
            size=target_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        # Apply background darkening
        crop_mask = mask_tensor[ymin_pad:ymax_pad+1, xmin_pad:xmax_pad+1]
        crop_mask_resized = torch.nn.functional.interpolate(
            crop_mask.float().unsqueeze(0).unsqueeze(0),
            size=target_size,
            mode='nearest'
        ).squeeze().bool()
        
        crop_resized[:, ~crop_mask_resized] *= 0.3
        
        return crop_resized
    
    def batch_clip_inference(self,
                           clip_model,
                           batched_crops: List[torch.Tensor],
                           text_features: torch.Tensor,
                           normalize_transform) -> List[Dict]:
        """
        Perform batched CLIP inference with optimizations.
        
        Args:
            clip_model: CLIP model
            batched_crops: List of crop batches
            text_features: Precomputed text features
            normalize_transform: Normalization transform
            
        Returns:
            List of inference results
        """
        start_time = time.time()
        all_results = []
        
        with torch.no_grad():
            # Enable mixed precision for faster inference
            with torch.cuda.amp.autocast():
                for batch_crops in batched_crops:
                    # Move to device if not already
                    if batch_crops.device != self.device:
                        batch_crops = batch_crops.to(self.device)
                    
                    # Apply normalization
                    normalized_batch = normalize_transform(batch_crops)
                    
                    # Get image features
                    image_features = clip_model.encode_image(normalized_batch)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    
                    # Compute similarities
                    similarities = (100.0 * image_features @ text_features.T)
                    
                    # Get best matches
                    best_scores, best_indices = similarities.max(dim=1)
                    
                    # Store results
                    for i in range(batch_crops.shape[0]):
                        all_results.append({
                            'score': best_scores[i].item(),
                            'class_idx': best_indices[i].item(),
                            'all_scores': similarities[i].cpu()
                        })
        
        inference_time = time.time() - start_time
        self.timing_stats['clip_inference'].append(inference_time)
        
        return all_results
    
    def process_frames(self,
                      frames: List[BatchFrame],
                      clip_model,
                      text_features: torch.Tensor,
                      text_prompts: List[str],
                      normalize_transform) -> Dict[int, Dict]:
        """
        Process multiple frames with optimized batching.
        
        Args:
            frames: List of frames to process
            clip_model: CLIP model
            text_features: Precomputed text features
            text_prompts: List of text prompts
            normalize_transform: Normalization transform
            
        Returns:
            Dictionary mapping frame_id to results
        """
        # Prepare crops in batches
        crop_data = self.prepare_crops_batch(frames)
        
        if not crop_data['batches']:
            return {}
        
        # Run batch inference
        inference_results = self.batch_clip_inference(
            clip_model,
            crop_data['batches'],
            text_features,
            normalize_transform
        )
        
        # Post-process and organize results by frame
        start_time = time.time()
        frame_results = {}
        
        # Flatten metadata
        all_metadata = []
        for batch_meta in crop_data['metadata']:
            all_metadata.extend(batch_meta)
        
        # Match results with metadata
        for i, (result, metadata) in enumerate(zip(inference_results, all_metadata)):
            frame_id = metadata['frame_id']
            
            if frame_id not in frame_results:
                frame_results[frame_id] = {
                    'predictions': [],
                    'confidences': [],
                    'metadata': []
                }
            
            # Calibrate confidence score
            calibrated_score = self._calibrate_confidence(result['score'])
            
            frame_results[frame_id]['predictions'].append(text_prompts[result['class_idx']])
            frame_results[frame_id]['confidences'].append(calibrated_score)
            frame_results[frame_id]['metadata'].append(metadata)
        
        post_time = time.time() - start_time
        self.timing_stats['post_processing'].append(post_time)
        
        return frame_results
    
    def _calibrate_confidence(self, raw_score: float) -> float:
        """
        Calibrate CLIP confidence scores to useful range.
        
        Args:
            raw_score: Raw CLIP similarity score
            
        Returns:
            Calibrated confidence score (0-100)
        """
        # CLIP scores typically range from 15-40
        # Map to 0-100 range with adaptive scaling
        min_expected = 15.0
        max_expected = 40.0
        
        # Clamp to expected range
        clamped = max(min_expected, min(raw_score, max_expected))
        
        # Linear mapping to 0-100
        normalized = (clamped - min_expected) / (max_expected - min_expected)
        calibrated = normalized * 80 + 20  # Map to 20-100 range
        
        return calibrated
    
    def get_performance_stats(self) -> Dict:
        """
        Get performance statistics for optimization analysis.
        
        Returns:
            Dictionary with timing statistics
        """
        stats = {}
        
        for key, times in self.timing_stats.items():
            if times:
                stats[key] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'total': np.sum(times),
                    'count': len(times)
                }
        
        # Calculate speedup metrics
        if self.timing_stats['clip_inference']:
            total_crops = sum(len(m) for m in self.timing_stats['crop_preparation'])
            avg_time_per_crop = np.mean(self.timing_stats['clip_inference']) / self.batch_size
            stats['avg_ms_per_crop'] = avg_time_per_crop * 1000
            stats['crops_per_second'] = 1.0 / avg_time_per_crop if avg_time_per_crop > 0 else 0
        
        return stats
    
    def clear_cache(self):
        """Clear the crop cache to free memory."""
        self.crop_cache.clear()
        
    def shutdown(self):
        """Shutdown the thread pool executor."""
        self.executor.shutdown(wait=True)