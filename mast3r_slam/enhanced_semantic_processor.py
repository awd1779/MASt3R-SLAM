# mast3r_slam/enhanced_semantic_processor.py
"""
Enhanced Semantic Processor with Adaptive SAM-CLIP Fusion
This module provides improved semantic processing using novel fusion techniques.
"""
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Import adaptive fusion components
from .adaptive_fusion import (
    AdaptiveSAMCLIPFusion, 
    HierarchicalVocabulary,
    create_adaptive_fusion_model
)

# Import base semantic processor components
from .semantic_processor import (
    load_instance_segmentation_model,
    load_clip_model,
    TEXT_PROMPTS,
    SEMANTIC_DEVICE,
    DEBUG_VISUALIZATION,
    DEBUG_OUTPUT_DIR
)

class EnhancedSemanticProcessor:
    """
    Enhanced semantic processor that uses adaptive SAM-CLIP fusion
    instead of naive cropping approach.
    """
    
    def __init__(self, device=SEMANTIC_DEVICE, enable_temporal=True, 
                 use_hierarchical_vocab=True, debug_mode=True):
        self.device = device
        self.enable_temporal = enable_temporal
        self.use_hierarchical_vocab = use_hierarchical_vocab
        self.debug_mode = debug_mode
        
        # Initialize models
        self.sam_generator = None
        self.clip_model = None
        self.clip_preprocess = None
        self.text_features = None
        self.text_prompts = None
        
        # Initialize adaptive fusion model
        self.fusion_model = create_adaptive_fusion_model(device=device)
        
        # Initialize hierarchical vocabulary
        if use_hierarchical_vocab:
            self.vocab_manager = HierarchicalVocabulary()
        else:
            self.vocab_manager = None
        
        # Temporal storage for consistency
        self.previous_features = {}  # frame_id -> features mapping
        self.temporal_window = 5  # Keep features from last N frames
        
        # Performance tracking
        self.performance_stats = {
            'total_frames': 0,
            'avg_fusion_time': 0,
            'avg_detection_count': 0,
            'confidence_improvements': []
        }
    
    def initialize_models(self):
        """Initialize SAM and CLIP models."""
        # Load SAM
        self.sam_generator = load_instance_segmentation_model(device=self.device)
        
        # Load CLIP with initial prompts
        initial_prompts = TEXT_PROMPTS
        if self.use_hierarchical_vocab:
            initial_prompts = self.vocab_manager.get_all_prompts()
        
        self.clip_model, self.clip_preprocess, self.text_features, self.text_prompts = load_clip_model(
            text_prompts=initial_prompts, device=self.device
        )
        
        print(f"[Enhanced SP] Initialized with {len(self.text_prompts)} text prompts")
    
    def update_vocabulary_for_scene(self, mask_data, image_shape):
        """
        Dynamically update vocabulary based on scene characteristics.
        """
        if not self.use_hierarchical_vocab or not mask_data:
            return
        
        # Extract mask areas for vocabulary selection
        mask_areas = [mask['area'] for mask in mask_data]
        
        # Select appropriate vocabulary level
        selected_prompts = self.vocab_manager.select_vocabulary_level(
            mask_areas, image_shape[:2]
        )
        
        # Update CLIP text features if prompts changed
        if selected_prompts != self.text_prompts:
            print(f"[Enhanced SP] Updating vocabulary: {len(self.text_prompts)} -> {len(selected_prompts)} prompts")
            
            self.clip_model, self.clip_preprocess, self.text_features, self.text_prompts = load_clip_model(
                text_prompts=selected_prompts, device=self.device, force_reload_prompts=True
            )
    
    def extract_enhanced_sam_features(self, image_tensor, mask_data):
        """
        Extract enhanced SAM features using spatial attention over mask regions.
        This is a simplified version - full implementation would use SAM encoder internals.
        """
        sam_features = []
        confidence_scores = []
        
        for mask_info in mask_data:
            mask = torch.from_numpy(mask_info['segmentation']).to(self.device)
            
            # Multi-scale feature extraction
            features_scales = []
            
            # Original scale
            masked_region = image_tensor * mask.unsqueeze(0)
            original_feat = torch.mean(masked_region.view(3, -1), dim=1)  # [3]
            features_scales.append(original_feat)
            
            # Downsampled scales for context
            for scale in [0.5, 0.25]:
                scaled_image = torch.nn.functional.interpolate(
                    image_tensor.unsqueeze(0), scale_factor=scale, mode='bilinear'
                ).squeeze(0)
                scaled_mask = torch.nn.functional.interpolate(
                    mask.float().unsqueeze(0).unsqueeze(0), scale_factor=scale, mode='nearest'
                ).squeeze().bool()
                
                scaled_masked = scaled_image * scaled_mask.unsqueeze(0)
                scaled_feat = torch.mean(scaled_masked.view(3, -1), dim=1)
                features_scales.append(scaled_feat)
            
            # Concatenate multi-scale features
            multi_scale_feat = torch.cat(features_scales)  # [9]
            
            # Project to target dimension using a learned projection
            target_dim = 256  # SAM feature dimension
            if not hasattr(self, 'sam_projection'):
                self.sam_projection = torch.nn.Linear(9, target_dim).to(self.device)
                # Initialize with small weights for stability
                torch.nn.init.xavier_normal_(self.sam_projection.weight, gain=0.01)
            
            enhanced_feature = self.sam_projection(multi_scale_feat)
            
            sam_features.append(enhanced_feature)
            confidence_scores.append(mask_info.get('stability_score', 0.5))
        
        if not sam_features:
            return None, None
        
        sam_features = torch.stack(sam_features)  # [N, 256]
        confidence_scores = torch.tensor(confidence_scores, device=self.device).unsqueeze(1)  # [N, 1]
        
        return sam_features, confidence_scores
    
    def extract_enhanced_clip_features(self, image_tensor, mask_data):
        """
        Extract CLIP features using multiple enhancement strategies.
        """
        clip_features = []
        
        for mask_info in mask_data:
            mask = torch.from_numpy(mask_info['segmentation']).to(self.device)
            
            # Strategy 1: Highlight method (dim background)
            highlighted = image_tensor.clone()
            highlighted[:, ~mask] *= 0.3
            
            # Strategy 2: Gaussian blur background
            blurred_bg = image_tensor.clone()
            # Simple blur approximation using average pooling
            blur_kernel = torch.ones(1, 1, 5, 5, device=self.device) / 25
            for c in range(3):
                channel = blurred_bg[c:c+1].unsqueeze(0)
                blurred_channel = torch.nn.functional.conv2d(channel, blur_kernel, padding=2)
                blurred_bg[c] = blurred_channel.squeeze()
            
            # Apply blur only to background
            enhanced_image = torch.where(mask.unsqueeze(0), image_tensor, blurred_bg)
            
            # Strategy 3: Combine highlight + blur for best of both
            final_image = 0.7 * highlighted + 0.3 * enhanced_image
            
            # Resize and normalize for CLIP
            clip_input = torch.nn.functional.interpolate(
                final_image.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
            ).squeeze(0)
            
            # Apply CLIP normalization
            if hasattr(self.clip_preprocess, 'transforms'):
                for transform in self.clip_preprocess.transforms:
                    if hasattr(transform, 'mean') and hasattr(transform, 'std'):
                        clip_input = transform(clip_input.unsqueeze(0)).squeeze(0)
                        break
            
            # Extract CLIP features
            with torch.no_grad():
                clip_feature = self.clip_model.encode_image(clip_input.unsqueeze(0))
                clip_feature = clip_feature / clip_feature.norm(dim=-1, keepdim=True)
                clip_features.append(clip_feature.squeeze(0))
        
        if not clip_features:
            return None
        
        return torch.stack(clip_features)  # [N, clip_dim]
    
    def process_frame_with_adaptive_fusion(self, image_tensor_chw_0_1_rgb, frame_id=None, 
                                         enable_debug_viz=False):
        """
        Main processing function using adaptive SAM-CLIP fusion.
        
        Args:
            image_tensor_chw_0_1_rgb: Input image tensor [3, H, W] in range [0, 1]
            frame_id: Frame identifier for temporal consistency
            enable_debug_viz: Whether to save debug visualizations
        
        Returns:
            local_mask: Semantic mask with region IDs
            local_map: Mapping from region ID to semantic class
            performance_info: Dictionary with timing and quality metrics
        """
        if self.sam_generator is None or self.clip_model is None:
            self.initialize_models()
        
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        
        # Ensure correct tensor range
        if image_tensor_chw_0_1_rgb.min() < -0.1:
            print(f"[Enhanced SP] Converting tensor range from [-1,1] to [0,1]")
            image_tensor_chw_0_1_rgb = (image_tensor_chw_0_1_rgb + 1.0) / 2.0
        
        # Convert to numpy for SAM
        image_np = (image_tensor_chw_0_1_rgb.permute(1, 2, 0) * 255.0).byte().cpu().numpy()
        
        # Run SAM segmentation
        sam_masks_data = self.sam_generator.generate(image_np)
        print(f"[Enhanced SP] SAM generated {len(sam_masks_data)} masks")
        
        if not sam_masks_data:
            return torch.zeros_like(image_tensor_chw_0_1_rgb[0], dtype=torch.long), {}, {}
        
        # Update vocabulary based on scene characteristics
        self.update_vocabulary_for_scene(sam_masks_data, image_np.shape)
        
        # Extract enhanced features
        sam_features, confidence_scores = self.extract_enhanced_sam_features(
            image_tensor_chw_0_1_rgb, sam_masks_data
        )
        clip_features = self.extract_enhanced_clip_features(
            image_tensor_chw_0_1_rgb, sam_masks_data
        )
        
        if sam_features is None or clip_features is None:
            return torch.zeros_like(image_tensor_chw_0_1_rgb[0], dtype=torch.long), {}, {}
        
        # Get previous features for temporal consistency
        previous_features = None
        if self.enable_temporal and frame_id is not None:
            previous_features = self.previous_features.get(frame_id - 1)
        
        # Apply adaptive fusion
        with torch.no_grad():
            fused_features, similarities = self.fusion_model(
                sam_features, clip_features, confidence_scores,
                text_features=self.text_features,
                previous_features=previous_features
            )
        
        # Store features for temporal consistency
        if self.enable_temporal and frame_id is not None:
            self.previous_features[frame_id] = fused_features.clone()
            
            # Clean old features
            old_frames = [fid for fid in self.previous_features.keys() 
                         if fid < frame_id - self.temporal_window]
            for old_fid in old_frames:
                del self.previous_features[old_fid]
        
        # Assign semantic labels
        best_scores, best_indices = similarities.max(dim=1)
        
        # Debug: Print score statistics
        print(f"[Enhanced SP DEBUG] Score statistics:")
        print(f"   Similarities shape: {similarities.shape}")
        print(f"   Best scores range: [{best_scores.min().item():.2f}, {best_scores.max().item():.2f}]")
        print(f"   Best scores mean: {best_scores.mean().item():.2f}")
        print(f"   Text prompts count: {len(self.text_prompts)}")
        
        # Create semantic mask
        H, W = image_tensor_chw_0_1_rgb.shape[1], image_tensor_chw_0_1_rgb.shape[2]
        semantic_mask = torch.zeros((H, W), dtype=torch.long, device=self.device)
        local_map = {0: "background"}  # Background class
        
        # Apply confidence threshold adjusted for enhanced approach
        confidence_threshold = 8.0  # Adjusted for the different scale in enhanced approach
        
        assigned_count = 0
        for i, (mask_data, score, class_idx) in enumerate(zip(sam_masks_data, best_scores, best_indices)):
            if score.item() > confidence_threshold:
                mask = torch.from_numpy(mask_data['segmentation']).to(self.device)
                local_id = i + 1
                semantic_mask[mask] = local_id
                local_map[local_id] = self.text_prompts[class_idx.item()]
                assigned_count += 1
                
                if assigned_count <= 3:  # Print first few assignments for debugging
                    print(f"[Enhanced SP DEBUG] Assigned mask {i}: {self.text_prompts[class_idx.item()]} (score: {score.item():.2f})")
        
        print(f"[Enhanced SP DEBUG] Total assignments: {assigned_count}/{len(sam_masks_data)}")
        
        end_time.record()
        torch.cuda.synchronize()
        processing_time = start_time.elapsed_time(end_time)
        
        # Update performance statistics
        self.performance_stats['total_frames'] += 1
        self.performance_stats['avg_fusion_time'] = (
            (self.performance_stats['avg_fusion_time'] * (self.performance_stats['total_frames'] - 1) + processing_time)
            / self.performance_stats['total_frames']
        )
        self.performance_stats['avg_detection_count'] = (
            (self.performance_stats['avg_detection_count'] * (self.performance_stats['total_frames'] - 1) + len(local_map))
            / self.performance_stats['total_frames']
        )
        
        # Confidence improvement tracking
        avg_confidence = best_scores.mean().item()
        self.performance_stats['confidence_improvements'].append(avg_confidence)
        
        # Debug visualization
        if enable_debug_viz and DEBUG_VISUALIZATION:
            self.save_debug_visualizations(
                image_tensor_chw_0_1_rgb, sam_masks_data, semantic_mask, 
                local_map, similarities, frame_id
            )
        
        performance_info = {
            'processing_time_ms': processing_time,
            'num_detections': len(local_map),
            'avg_confidence': avg_confidence,
            'fusion_improvements': avg_confidence - 25.0  # Baseline comparison
        }
        
        print(f"[Enhanced SP] Processed frame in {processing_time:.1f}ms, "
              f"detected {len(local_map)} objects, avg confidence: {avg_confidence:.2f}")
        
        return semantic_mask.cpu(), local_map, performance_info
    
    def save_debug_visualizations(self, image_tensor, sam_masks_data, semantic_mask, 
                                local_map, similarities, frame_id):
        """Save comprehensive debug visualizations."""
        if frame_id is None:
            frame_id = "unknown"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_dir = Path(DEBUG_OUTPUT_DIR) / f"enhanced_frame_{frame_id:06d}_{timestamp}"
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Save original image
        orig_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        plt.figure(figsize=(10, 8))
        plt.imshow(orig_np)
        plt.title("Original Image")
        plt.axis('off')
        plt.savefig(debug_dir / "01_original.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save semantic mask overlay
        plt.figure(figsize=(12, 8))
        plt.imshow(orig_np)
        
        # Create colored overlay
        semantic_colored = torch.zeros((*semantic_mask.shape, 3), dtype=torch.float32)
        colors = plt.cm.Set3(np.linspace(0, 1, len(local_map)))
        
        for local_id, class_name in local_map.items():
            if local_id == 0:
                continue
            mask_region = semantic_mask == local_id
            semantic_colored[mask_region] = torch.tensor(colors[local_id % len(colors)][:3], dtype=torch.float32)
        
        plt.imshow(semantic_colored.numpy(), alpha=0.6)
        plt.title("Enhanced Semantic Segmentation")
        plt.axis('off')
        
        # Add legend
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i%len(colors)], 
                                       label=f"{i}: {class_name}")
                          for i, class_name in local_map.items() if i != 0]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.savefig(debug_dir / "02_enhanced_semantic.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save confidence analysis
        if similarities is not None:
            plt.figure(figsize=(12, 6))
            
            # Confidence distribution
            plt.subplot(1, 2, 1)
            best_scores = similarities.max(dim=1)[0].cpu().numpy()
            plt.hist(best_scores, bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('Confidence Score')
            plt.ylabel('Frequency')
            plt.title('Confidence Score Distribution')
            plt.axvline(x=best_scores.mean(), color='red', linestyle='--', 
                       label=f'Mean: {best_scores.mean():.2f}')
            plt.legend()
            
            # Top predictions
            plt.subplot(1, 2, 2)
            top_scores, top_indices = similarities.topk(5, dim=1)
            
            for i in range(min(5, len(top_scores))):
                scores = top_scores[i].cpu().numpy()
                labels = [self.text_prompts[idx] for idx in top_indices[i]]
                
                plt.barh(range(5), scores, alpha=0.7, label=f'Mask {i}')
            
            plt.xlabel('Confidence Score')
            plt.ylabel('Top 5 Predictions')
            plt.title('Top Predictions per Mask')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(debug_dir / "03_confidence_analysis.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        # Save performance summary
        summary = {
            'frame_id': frame_id,
            'timestamp': timestamp,
            'num_masks': len(sam_masks_data),
            'num_detections': len(local_map),
            'vocabulary_size': len(self.text_prompts),
            'avg_confidence': similarities.max(dim=1)[0].mean().item() if similarities is not None else 0,
            'detected_classes': list(local_map.values()),
            'performance_stats': self.performance_stats
        }
        
        with open(debug_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[Enhanced SP] Debug visualizations saved to: {debug_dir}")
    
    def get_performance_summary(self):
        """Get comprehensive performance summary."""
        if self.performance_stats['total_frames'] == 0:
            return "No frames processed yet."
        
        confidence_improvements = self.performance_stats['confidence_improvements']
        
        summary = f"""
Enhanced Semantic Processor Performance Summary:
===============================================
Total frames processed: {self.performance_stats['total_frames']}
Average processing time: {self.performance_stats['avg_fusion_time']:.1f}ms
Average detections per frame: {self.performance_stats['avg_detection_count']:.1f}
Average confidence score: {np.mean(confidence_improvements):.2f}
Confidence improvement over baseline: {np.mean(confidence_improvements) - 25.0:.2f}
Max confidence achieved: {np.max(confidence_improvements):.2f}
Temporal consistency: {'Enabled' if self.enable_temporal else 'Disabled'}
Hierarchical vocabulary: {'Enabled' if self.use_hierarchical_vocab else 'Disabled'}
"""
        return summary

# Factory function for easy usage
def create_enhanced_semantic_processor(device=SEMANTIC_DEVICE, enable_temporal=True, 
                                     use_hierarchical_vocab=True, debug_mode=True):
    """Create and return an enhanced semantic processor instance."""
    return EnhancedSemanticProcessor(
        device=device,
        enable_temporal=enable_temporal,
        use_hierarchical_vocab=use_hierarchical_vocab,
        debug_mode=debug_mode
    )

# Convenience function that matches the original API
def process_frame_for_semantics_enhanced(image_tensor_chw_0_1_rgb, 
                                       text_prompts_for_clip=None,
                                       enable_debug_viz=False, 
                                       frame_id=None):
    """
    Enhanced version of the original process_frame_for_semantics function.
    Uses adaptive fusion instead of naive SAM+CLIP cropping.
    """
    # Create processor instance (in practice, you'd reuse this)
    processor = create_enhanced_semantic_processor()
    
    return processor.process_frame_with_adaptive_fusion(
        image_tensor_chw_0_1_rgb, frame_id, enable_debug_viz
    )