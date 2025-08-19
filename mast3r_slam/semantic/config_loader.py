"""
Configuration loader for semantic processing modules.

This module provides utilities to load and access semantic configuration
parameters with proper defaults for backward compatibility.
"""

from typing import Dict, Any, Optional


class SemanticConfig:
    """Semantic configuration wrapper with safe access and defaults."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize semantic configuration.
        
        Args:
            config: Full configuration dictionary containing 'semantic_segmentation' section
        """
        self.config = config or {}
        self.semantic_config = self.config.get('semantic_segmentation', {})
    
    # Processing parameters
    def get_min_depth(self) -> float:
        """Get minimum depth threshold."""
        return self.semantic_config.get('processing', {}).get('min_depth', 0.1)
    
    def get_max_depth(self) -> float:
        """Get maximum depth threshold."""
        return self.semantic_config.get('processing', {}).get('max_depth', 50.0)
    
    def get_c_conf_threshold(self) -> float:
        """Get SLAM confidence threshold."""
        return self.semantic_config.get('processing', {}).get('c_conf_threshold', 1.5)
    
    def get_max_keyframes(self) -> int:
        """Get maximum number of keyframes."""
        return self.semantic_config.get('processing', {}).get('max_keyframes', 1000)
    
    def get_default_image_height(self) -> int:
        """Get default image height."""
        return self.semantic_config.get('processing', {}).get('default_image_height', 480)
    
    def get_default_image_width(self) -> int:
        """Get default image width."""
        return self.semantic_config.get('processing', {}).get('default_image_width', 640)
    
    def get_global_id_multiplier(self) -> int:
        """Get global ID multiplier."""
        return self.semantic_config.get('processing', {}).get('global_id_multiplier', 10000)
    
    # Mask processing parameters
    def get_deduplication_iou_threshold(self) -> float:
        """Get deduplication IoU threshold."""
        return self.semantic_config.get('mask_processing', {}).get('deduplication_iou_threshold', 0.9)
    
    def get_overlap_iou_threshold(self) -> float:
        """Get overlap IoU threshold."""
        return self.semantic_config.get('mask_processing', {}).get('overlap_iou_threshold', 0.8)
    
    def get_containment_threshold(self) -> float:
        """Get containment threshold."""
        return self.semantic_config.get('mask_processing', {}).get('containment_threshold', 0.9)
    
    def get_large_mask_threshold(self) -> int:
        """Get large mask threshold percentage."""
        return self.semantic_config.get('mask_processing', {}).get('large_mask_threshold', 50)
    
    # Priority weights
    def get_size_priority(self) -> float:
        """Get size priority weight."""
        return self.semantic_config.get('priority_weights', {}).get('size_priority', 1.0)
    
    def get_label_priority(self) -> float:
        """Get label priority weight."""
        return self.semantic_config.get('priority_weights', {}).get('label_priority', 0.5)
    
    def get_confidence_priority(self) -> float:
        """Get confidence priority weight."""
        return self.semantic_config.get('priority_weights', {}).get('confidence_priority', 0.3)
    
    # Clustering configuration
    def get_clustering_enabled(self) -> bool:
        """Check if clustering is enabled."""
        return self.semantic_config.get('clustering', {}).get('enabled', True)
    
    def get_clustering_config(self) -> Optional[Dict]:
        """Get clustering configuration (legacy support)."""
        return self.semantic_config.get('clustering', {}).get('config', None)
    
    def get_clustering_pipeline_config(self) -> Dict:
        """Get clustering pipeline configuration."""
        default_pipeline = {
            'use_adaptive_params': True,
            'use_stacking_detection': True,
            'use_post_merge': True,
            'debug': False
        }
        return self.semantic_config.get('clustering', {}).get('pipeline', default_pipeline)
    
    def get_clustering_base_params(self) -> Dict:
        """Get base clustering parameters."""
        default_base = {
            'spatial_threshold': 1.0,
            'temporal_threshold': 25,
            'movement_threshold': 0.8,
            'min_samples': 1
        }
        return self.semantic_config.get('clustering', {}).get('base_params', default_base)
    
    def get_clustering_adaptive_params(self) -> Dict:
        """Get adaptive clustering parameters."""
        default_adaptive = {
            'spatial_base': 0.4,
            'temporal_base': 10,
            'movement_base': 0.3,
            'merge_base': 0.8,
            'size_thresholds': {
                'large': {'volume': 2.0, 'extent': 2.0},
                'medium': {'volume': 0.1, 'extent': 0.8}
            },
            'confidence_weights': {
                'large': 0.6,
                'medium': 0.8,
                'small': 0.9
            }
        }
        return self.semantic_config.get('clustering', {}).get('adaptive_params', default_adaptive)
    
    def get_clustering_merge_params(self) -> Dict:
        """Get post-clustering merge parameters."""
        default_merge = {
            'max_spatial_distance': 2.0,
            'min_temporal_overlap': 0.2,
            'max_centroid_distance': 3.0,
            'confidence_threshold': 0.4,
            'confidence_factors': {
                'temporal_weight': 0.3,
                'spatial_weight': 0.4,
                'extent_weight': 0.2,
                'geometric_weight': 0.1
            }
        }
        return self.semantic_config.get('clustering', {}).get('merge_params', default_merge)
    
    def get_stacking_detection_params(self) -> Dict:
        """Get stacking detection parameters."""
        default_stacking = {
            'min_confidence': 0.3,
            'proximity_threshold': 0.05,
            'overlap_ratio_threshold': 0.3
        }
        return self.semantic_config.get('clustering', {}).get('stacking_detection', default_stacking)
    
    def get_temporal_consistency_params(self) -> Dict:
        """Get temporal consistency parameters."""
        default_temporal = {
            'max_gap': 15,
            'movement_detection': {
                'enabled': True,
                'threshold': 0.5
            }
        }
        return self.semantic_config.get('clustering', {}).get('temporal_consistency', default_temporal)
    
    # Visualization settings
    def get_use_semantic_colors(self) -> bool:
        """Check if semantic colors should be used."""
        return self.semantic_config.get('visualization', {}).get('use_semantic_colors', True)
    
    def get_track_color_saturation(self) -> float:
        """Get track color saturation."""
        return self.semantic_config.get('visualization', {}).get('track_color_saturation', 0.8)
    
    def get_track_color_value(self) -> float:
        """Get track color value/brightness."""
        return self.semantic_config.get('visualization', {}).get('track_color_value', 0.9)
    
    def get_golden_ratio(self) -> float:
        """Get golden ratio for color distribution."""
        return self.semantic_config.get('visualization', {}).get('golden_ratio', 0.618033988749895)
    
    # Grounded SAM2 runtime settings
    def get_device(self) -> str:
        """Get processing device."""
        return self.semantic_config.get('grounded_sam2', {}).get('runtime', {}).get('device', 'cuda:0')
    
    def get_dtype(self) -> str:
        """Get model dtype."""
        return self.semantic_config.get('grounded_sam2', {}).get('runtime', {}).get('dtype', 'float32')
    
    def get_confidence_threshold(self) -> float:
        """Get detection confidence threshold."""
        return self.semantic_config.get('grounded_sam2', {}).get('runtime', {}).get('confidence_threshold', 0.35)
    
    def get_text_threshold(self) -> float:
        """Get text detection threshold."""
        return self.semantic_config.get('grounded_sam2', {}).get('runtime', {}).get('text_threshold', 0.25)
    
    # Label filtering
    def get_filter_empty_labels(self) -> bool:
        """Check if empty labels should be filtered."""
        return self.semantic_config.get('grounded_sam2', {}).get('label_filtering', {}).get('filter_empty_labels', True)
    
    def get_min_phrase_length(self) -> int:
        """Get minimum phrase length."""
        return self.semantic_config.get('grounded_sam2', {}).get('label_filtering', {}).get('min_phrase_length', 2)
    
    # Debug settings
    def get_debug_mode(self) -> bool:
        """Check if debug mode is enabled."""
        return self.semantic_config.get('grounded_sam2', {}).get('debug', {}).get('debug_mode', False)
    
    def get_save_debug_visualizations(self) -> bool:
        """Check if debug visualizations should be saved."""
        return self.semantic_config.get('grounded_sam2', {}).get('debug', {}).get('save_debug_visualizations', False)
    
    # Model selection
    def get_model_configs(self) -> Dict:
        """Get model configurations."""
        return self.semantic_config.get('grounded_sam2', {})
    
    def get_model_selection(self) -> Dict:
        """Get model selection."""
        return self.semantic_config.get('grounded_sam2', {}).get('model_selection', {
            'model_type': 'hiera_large',
            'grounding_model': 'grounding_dino_swin-b'
        })
    
    # Get raw config sections
    def get_processing_config(self) -> Dict:
        """Get full processing configuration section."""
        return self.semantic_config.get('processing', {})
    
    def get_mask_processing_config(self) -> Dict:
        """Get full mask processing configuration section."""
        return self.semantic_config.get('mask_processing', {})
    
    def get_grounded_sam2_config(self) -> Dict:
        """Get full Grounded SAM2 configuration section."""
        return self.semantic_config.get('grounded_sam2', {})