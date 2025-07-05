# mast3r_slam/adaptive_fusion.py
"""
Adaptive SAM-CLIP Fusion Module for Enhanced Semantic Processing
This module implements novel fusion techniques to improve upon naive SAM+CLIP integration.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

class LearnableAdapter(nn.Module):
    """
    Learnable adapter module to transform features between SAM and CLIP domains.
    This enables better feature alignment and fusion.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = (input_dim + output_dim) // 2
        
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Initialize with identity-like mapping for stable training
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.adapter(x)

class ConfidenceWeightingModule(nn.Module):
    """
    Attention-based confidence weighting for adaptive feature fusion.
    Learns to weight SAM vs CLIP features based on mask quality and context.
    """
    def __init__(self, feature_dim: int, context_dim: int = 64):
        super().__init__()
        
        self.feature_proj = nn.Linear(feature_dim, context_dim)
        self.attention = nn.MultiheadAttention(context_dim, num_heads=8, batch_first=True)
        self.confidence_head = nn.Sequential(
            nn.Linear(context_dim, context_dim // 2),
            nn.ReLU(),
            nn.Linear(context_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, sam_features, clip_features, mask_confidence_scores):
        """
        Args:
            sam_features: [B, feature_dim] SAM-derived features
            clip_features: [B, feature_dim] CLIP-derived features  
            mask_confidence_scores: [B, 1] SAM mask confidence scores
        Returns:
            fusion_weights: [B, 1] weights for adaptive fusion
        """
        # Project features to common space
        sam_proj = self.feature_proj(sam_features)  # [B, context_dim]
        clip_proj = self.feature_proj(clip_features)  # [B, context_dim]
        
        # Stack for attention computation
        features_stack = torch.stack([sam_proj, clip_proj], dim=1)  # [B, 2, context_dim]
        
        # Self-attention to capture feature interactions
        attended_features, _ = self.attention(features_stack, features_stack, features_stack)
        
        # Use attended SAM features for confidence prediction
        attended_sam = attended_features[:, 0]  # [B, context_dim]
        
        # Combine with mask confidence for final weight
        confidence_input = attended_sam + mask_confidence_scores.expand(-1, attended_sam.size(1))
        fusion_weights = self.confidence_head(confidence_input)
        
        return fusion_weights

class AdaptiveSAMCLIPFusion(nn.Module):
    """
    Main adaptive fusion module that combines SAM and CLIP representations
    using learnable adapters and confidence-based weighting.
    """
    def __init__(self, sam_feature_dim: int = 256, clip_feature_dim: int = 512, 
                 unified_dim: int = 384, enable_temporal: bool = True):
        super().__init__()
        
        self.sam_feature_dim = sam_feature_dim
        self.clip_feature_dim = clip_feature_dim
        self.unified_dim = unified_dim
        self.enable_temporal = enable_temporal
        
        # Adapter modules for cross-domain feature transfer
        self.sam_to_unified = LearnableAdapter(sam_feature_dim, unified_dim)
        self.clip_to_unified = LearnableAdapter(clip_feature_dim, unified_dim)
        
        # Confidence weighting module
        self.confidence_weighting = ConfidenceWeightingModule(unified_dim)
        
        # Feature enhancement layers
        self.feature_enhancer = nn.Sequential(
            nn.Linear(unified_dim, unified_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(unified_dim, unified_dim),
            nn.LayerNorm(unified_dim)
        )
        
        # Temporal consistency module (if enabled)
        if enable_temporal:
            self.temporal_smoother = nn.LSTM(unified_dim, unified_dim // 2, 
                                           batch_first=True, bidirectional=True)
            self.temporal_alpha = nn.Parameter(torch.tensor(0.7))  # Learnable temporal weight
        
        # Classification head
        self.classifier = nn.Linear(unified_dim, unified_dim)  # For similarity computation
        
    def extract_sam_features(self, image_tensor, mask_data, sam_model):
        """
        Extract rich features from SAM encoder for each mask region.
        Instead of simple cropping, use SAM's internal representations.
        """
        sam_features = []
        confidence_scores = []
        
        # Get SAM encoder features (this requires access to SAM internals)
        with torch.no_grad():
            # For now, we'll use a spatial pooling approach over mask regions
            # In a full implementation, you'd extract from SAM's encoder layers
            for mask_info in mask_data:
                mask = torch.from_numpy(mask_info['segmentation']).to(image_tensor.device)
                
                # Spatial average pooling over mask region
                masked_features = image_tensor * mask.unsqueeze(0)  # [3, H, W]
                
                # Global average pool to get feature vector
                feature_vector = F.adaptive_avg_pool2d(masked_features.unsqueeze(0), (1, 1))
                feature_vector = feature_vector.flatten()  # [3] -> expand to sam_feature_dim
                
                # Expand to expected dimension (simplified - real implementation would use SAM encoder)
                expanded_features = F.linear(feature_vector, 
                                           torch.randn(self.sam_feature_dim, 3).to(image_tensor.device))
                
                sam_features.append(expanded_features)
                confidence_scores.append(mask_info.get('stability_score', 0.5))
        
        if not sam_features:
            return None, None
            
        sam_features = torch.stack(sam_features)  # [N, sam_feature_dim]
        confidence_scores = torch.tensor(confidence_scores, device=image_tensor.device).unsqueeze(1)  # [N, 1]
        
        return sam_features, confidence_scores
    
    def extract_clip_features(self, image_tensor, mask_data, clip_model, clip_preprocess):
        """
        Extract CLIP features using enhanced region processing.
        Uses highlight method instead of simple cropping.
        """
        clip_features = []
        
        for mask_info in mask_data:
            mask = torch.from_numpy(mask_info['segmentation']).to(image_tensor.device)
            
            # Create highlighted image (dim background, keep foreground)
            highlighted_image = image_tensor.clone()
            highlighted_image[:, ~mask] *= 0.3  # Dim background
            
            # Resize to CLIP input size
            clip_input = F.interpolate(highlighted_image.unsqueeze(0), 
                                     size=(224, 224), mode='bilinear', align_corners=False)
            clip_input = clip_input.squeeze(0)
            
            # Apply CLIP normalization
            if hasattr(clip_preprocess, 'transforms'):
                for transform in clip_preprocess.transforms:
                    if hasattr(transform, 'mean') and hasattr(transform, 'std'):
                        clip_input = transform(clip_input.unsqueeze(0)).squeeze(0)
                        break
            
            # Extract CLIP features
            with torch.no_grad():
                clip_feature = clip_model.encode_image(clip_input.unsqueeze(0))
                clip_feature = clip_feature / clip_feature.norm(dim=-1, keepdim=True)
                clip_features.append(clip_feature.squeeze(0))
        
        if not clip_features:
            return None
            
        return torch.stack(clip_features)  # [N, clip_feature_dim]
    
    def forward(self, sam_features, clip_features, confidence_scores, 
                text_features=None, previous_features=None):
        """
        Forward pass for adaptive fusion.
        
        Args:
            sam_features: [N, sam_feature_dim] SAM-derived features
            clip_features: [N, clip_feature_dim] CLIP-derived features
            confidence_scores: [N, 1] SAM mask confidence scores
            text_features: [M, text_feature_dim] Text embedding features
            previous_features: [N, unified_dim] Features from previous frame (for temporal consistency)
        
        Returns:
            fused_features: [N, unified_dim] Adaptively fused features
            similarities: [N, M] Similarity scores with text features (if provided)
        """
        batch_size = sam_features.size(0)
        
        # Transform features to unified space
        sam_unified = self.sam_to_unified(sam_features)  # [N, unified_dim]
        clip_unified = self.clip_to_unified(clip_features)  # [N, unified_dim]
        
        # Compute adaptive fusion weights
        fusion_weights = self.confidence_weighting(sam_unified, clip_unified, confidence_scores)  # [N, 1]
        
        # Adaptive fusion
        fused_features = fusion_weights * sam_unified + (1 - fusion_weights) * clip_unified
        
        # Feature enhancement
        fused_features = self.feature_enhancer(fused_features)
        
        # Temporal consistency (if enabled and previous features provided)
        if self.enable_temporal and previous_features is not None:
            # Simple exponential moving average for now
            # In full implementation, use LSTM for more sophisticated temporal modeling
            fused_features = self.temporal_alpha * fused_features + (1 - self.temporal_alpha) * previous_features
        
        # Compute similarities with text features if provided
        similarities = None
        if text_features is not None:
            # Get text feature dimension and adjust classifier if needed
            text_dim = text_features.shape[1]  # Should be 512 for CLIP
            
            # Create a projection layer to match text feature dimension
            if not hasattr(self, 'text_projection'):
                self.text_projection = torch.nn.Linear(self.unified_dim, text_dim).to(fused_features.device)
                # Initialize with identity-like mapping for better alignment
                with torch.no_grad():
                    # Initialize to preserve similarity scales
                    torch.nn.init.xavier_normal_(self.text_projection.weight, gain=1.0)
                    torch.nn.init.zeros_(self.text_projection.bias)
            
            # Project fused features to text feature space
            query_features = self.text_projection(fused_features)  # [N, text_dim]
            query_features = query_features / query_features.norm(dim=-1, keepdim=True)
            
            # Normalize text features
            text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarities (CLIP already uses temperature internally)
            similarities = torch.mm(query_features, text_features_norm.T) * 100.0  # [N, M]
        
        return fused_features, similarities

class HierarchicalVocabulary:
    """
    Hierarchical vocabulary system for dynamic prompt selection.
    Selects appropriate level of detail based on scene context and mask properties.
    """
    def __init__(self):
        self.vocabulary_hierarchy = {
            'coarse': [
                "background", "furniture", "electronics", "objects", "structural_elements"
            ],
            'medium': [
                "background", "wall", "floor", "ceiling", "door", "window",
                "chair", "table", "desk", "shelf", "cabinet",
                "monitor", "computer", "lamp", "decoration"
            ],
            'fine': [
                # Your current detailed prompts
                "background", "wall surface", "wooden floor", "white ceiling", "room corner", "empty space",
                "office chair", "wooden chair", "computer desk", "wooden table", "dining table", "work surface",
                "bookshelf", "storage cabinet", "file drawer", "furniture leg",
                "computer monitor", "laptop computer", "desktop computer", "television screen", "electronic display",
                "computer keyboard", "computer mouse", "electronic device",
                "coffee cup", "drinking mug", "book spine", "stack of books", "paper document",
                "picture frame", "wall art", "decorative object", "storage box", "container object",
                "interior door", "glass window", "ceiling light", "desk lamp", "table lamp", "light fixture",
                "light switch", "wall outlet", "door frame", "window frame"
            ]
        }
    
    def select_vocabulary_level(self, mask_areas, image_resolution, scene_complexity=None):
        """
        Dynamically select vocabulary detail level based on context.
        
        Args:
            mask_areas: List of mask areas (in pixels)
            image_resolution: Tuple of (H, W)
            scene_complexity: Optional complexity metric
        
        Returns:
            selected_prompts: List of text prompts for CLIP
        """
        # For now, always use fine-grained vocabulary for better performance
        # The hierarchical selection can be refined later
        return self.vocabulary_hierarchy['fine']
        
        # Original logic (commented out for now)
        # total_pixels = image_resolution[0] * image_resolution[1]
        # avg_mask_ratio = np.mean(mask_areas) / total_pixels if mask_areas else 0
        # 
        # # Selection logic
        # if avg_mask_ratio > 0.1:  # Large objects -> use fine-grained
        #     return self.vocabulary_hierarchy['fine']
        # elif avg_mask_ratio > 0.02:  # Medium objects -> use medium detail
        #     return self.vocabulary_hierarchy['medium']
        # else:  # Small objects -> use coarse categories
        #     return self.vocabulary_hierarchy['coarse']
    
    def get_all_prompts(self):
        """Get all prompts from all levels."""
        all_prompts = []
        for level_prompts in self.vocabulary_hierarchy.values():
            all_prompts.extend(level_prompts)
        return list(set(all_prompts))  # Remove duplicates

def create_adaptive_fusion_model(device='cuda'):
    """
    Factory function to create and initialize the adaptive fusion model.
    """
    model = AdaptiveSAMCLIPFusion(
        sam_feature_dim=256,
        clip_feature_dim=512,
        unified_dim=384,
        enable_temporal=True
    )
    
    model = model.to(device)
    model.eval()  # Start in eval mode for inference
    
    return model

# Export main components
__all__ = [
    'AdaptiveSAMCLIPFusion',
    'LearnableAdapter', 
    'ConfidenceWeightingModule',
    'HierarchicalVocabulary',
    'create_adaptive_fusion_model'
]