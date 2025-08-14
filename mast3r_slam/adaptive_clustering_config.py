"""
Adaptive Clustering Configuration for Semantic SLAM

This module provides object-specific clustering parameters to handle different
types of objects appropriately. It addresses the one-size-fits-all limitation
of the original clustering approach by adapting parameters based on object
semantic categories.

Key improvements:
- Larger spatial thresholds for large surfaces (walls, ceilings)
- Tighter thresholds for small objects to prevent false merges  
- Object-specific temporal thresholds based on expected persistence
- Confidence-based parameter adjustment

Author: Generated for MASt3R SLAM semantic extension
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Object semantic categories with their characteristics
OBJECT_CATEGORIES = {
    'large_surfaces': {
        'keywords': [
            'wall', 'ceiling', 'floor', 'door', 'window', 'blinds', 'vent',
            'a wall', 'a ceiling', 'a floor', 'a door', 'a window', 'a blinds', 'a vent'
        ],
        'characteristics': {
            'spatial_extent': 'very_large',      # Can span several meters
            'detection_stability': 'unstable',    # Often fragmented across views
            'movement_expectation': 'static',     # Never move
            'typical_count': 'few'                # Usually 1-3 per scene
        }
    },
    'furniture': {
        'keywords': [
            'table', 'chair', 'sofa', 'cabinet', 'stool', 'rug', 'blanket',
            'a table', 'a chair', 'a sofa', 'a cabinet', 'a stool', 'a rug', 'a blanket'
        ],
        'characteristics': {
            'spatial_extent': 'medium',           # 0.5-2 meters typically
            'detection_stability': 'stable',      # Consistently detected
            'movement_expectation': 'static',     # Usually don't move
            'typical_count': 'multiple'           # Can have several similar items
        }
    },
    'small_objects': {
        'keywords': [
            'book', 'plate', 'vase', 'candle', 'switch', 'wall plug', 'pot', 
            'cushion', 'basket', 'lamp', 'a book', 'a plate', 'a vase', 
            'a candle', 'a switch', 'a wall plug', 'a pot', 'a cushion', 
            'a basket', 'a lamp'
        ],
        'characteristics': {
            'spatial_extent': 'small',            # Usually < 0.5 meters
            'detection_stability': 'variable',    # Can be missed/fragmented
            'movement_expectation': 'movable',    # Could be moved
            'typical_count': 'many'               # Often multiple similar items
        }
    },
    'plants_and_decor': {
        'keywords': [
            'indoor plant', 'plant stand', 'pillar', 'a indoor plant', 
            'a plant stand', 'a pillar'
        ],
        'characteristics': {
            'spatial_extent': 'small_medium',     # 0.3-1.5 meters
            'detection_stability': 'stable',      # Usually well-detected
            'movement_expectation': 'static',     # Rarely moved
            'typical_count': 'few'                # Usually 1-3 per scene
        }
    }
}

# Adaptive clustering parameters by category
ADAPTIVE_CLUSTERING_PARAMS = {
    'large_surfaces': {
        'spatial_threshold': 2.5,          # Very large - walls can be far apart
        'temporal_threshold': 30,          # Large temporal window - often fragmented
        'movement_threshold': 1.0,         # Allow some apparent "movement" due to detection variation
        'min_samples': 1,                  # Allow single detections
        'confidence_weight': 0.6,          # Lower confidence requirement
        'merge_aggressiveness': 'high'     # Aggressively merge large surfaces
    },
    'furniture': {
        'spatial_threshold': 1.2,          # Medium threshold
        'temporal_threshold': 20,          # Medium temporal window
        'movement_threshold': 0.8,         # Allow moderate movement
        'min_samples': 1,                  # Allow single detections
        'confidence_weight': 0.8,          # Standard confidence
        'merge_aggressiveness': 'medium'   # Standard merging
    },
    'small_objects': {
        'spatial_threshold': 0.6,          # Tight threshold - avoid false merges
        'temporal_threshold': 15,          # Smaller temporal window
        'movement_threshold': 0.4,         # Strict movement threshold
        'min_samples': 1,                  # Allow single detections
        'confidence_weight': 0.9,          # High confidence requirement
        'merge_aggressiveness': 'low'      # Conservative merging
    },
    'plants_and_decor': {
        'spatial_threshold': 1.0,          # Medium-small threshold
        'temporal_threshold': 25,          # Medium-large temporal window
        'movement_threshold': 0.6,         # Medium movement threshold
        'min_samples': 1,                  # Allow single detections
        'confidence_weight': 0.8,          # Standard confidence
        'merge_aggressiveness': 'medium'   # Standard merging
    },
    'default': {
        'spatial_threshold': 1.0,          # Conservative default
        'temporal_threshold': 20,          # Medium temporal window
        'movement_threshold': 0.6,         # Medium movement threshold
        'min_samples': 1,                  # Allow single detections
        'confidence_weight': 0.8,          # Standard confidence
        'merge_aggressiveness': 'medium'   # Standard merging
    }
}

# Global configuration that can be modified
GLOBAL_MULTIPLIERS = {
    'spatial_multiplier': 1.0,      # Global scaling for spatial thresholds
    'temporal_multiplier': 1.0,     # Global scaling for temporal thresholds
    'confidence_multiplier': 1.0    # Global scaling for confidence requirements
}


@dataclass
class ClusteringConfig:
    """Complete clustering configuration for a specific object type."""
    spatial_threshold: float
    temporal_threshold: int
    movement_threshold: float
    min_samples: int
    confidence_weight: float
    merge_aggressiveness: str
    object_category: str
    base_label: str


def categorize_object(label: str) -> str:
    """Determine the category of an object based on its label."""
    clean_label = label.lower().strip()
    
    # Remove common prefixes
    for prefix in ['a ', 'an ', 'the ']:
        if clean_label.startswith(prefix):
            clean_label = clean_label[len(prefix):]
            break
    
    # Check each category
    for category, info in OBJECT_CATEGORIES.items():
        for keyword in info['keywords']:
            keyword_clean = keyword.lower().strip()
            # Remove prefixes from keywords too
            for prefix in ['a ', 'an ', 'the ']:
                if keyword_clean.startswith(prefix):
                    keyword_clean = keyword_clean[len(prefix):]
                    break
            
            if keyword_clean == clean_label or keyword_clean in clean_label:
                return category
    
    return 'default'


def get_clustering_config(label: str) -> ClusteringConfig:
    """Get adaptive clustering configuration for a specific object label."""
    
    # Determine category
    category = categorize_object(label)
    
    # Get base parameters
    params = ADAPTIVE_CLUSTERING_PARAMS.get(category, ADAPTIVE_CLUSTERING_PARAMS['default'])
    
    # Apply global multipliers
    spatial_threshold = params['spatial_threshold'] * GLOBAL_MULTIPLIERS['spatial_multiplier']
    temporal_threshold = int(params['temporal_threshold'] * GLOBAL_MULTIPLIERS['temporal_multiplier'])
    confidence_weight = params['confidence_weight'] * GLOBAL_MULTIPLIERS['confidence_multiplier']
    
    # Clamp values to reasonable ranges
    spatial_threshold = max(0.1, min(spatial_threshold, 5.0))
    temporal_threshold = max(1, min(temporal_threshold, 100))
    confidence_weight = max(0.1, min(confidence_weight, 1.0))
    
    return ClusteringConfig(
        spatial_threshold=spatial_threshold,
        temporal_threshold=temporal_threshold,
        movement_threshold=params['movement_threshold'],
        min_samples=params['min_samples'],
        confidence_weight=confidence_weight,
        merge_aggressiveness=params['merge_aggressiveness'],
        object_category=category,
        base_label=label
    )


def get_unified_clustering_config(labels: List[str]) -> Dict:
    """
    Get a unified clustering configuration for hybrid clustering.
    
    When processing multiple object types together, this computes reasonable
    compromise parameters that work well across all object types present.
    
    Args:
        labels: List of object labels that will be clustered together
        
    Returns:
        Dictionary with unified clustering parameters
    """
    
    if not labels:
        return ADAPTIVE_CLUSTERING_PARAMS['default'].copy()
    
    # Get configs for all labels
    configs = [get_clustering_config(label) for label in labels]
    
    # Group by category to understand the mix
    category_counts = {}
    for config in configs:
        category_counts[config.object_category] = category_counts.get(config.object_category, 0) + 1
    
    logger.info(f"Unified config for {len(labels)} objects across categories: {category_counts}")
    
    # Determine dominant strategy based on object mix
    if category_counts.get('large_surfaces', 0) > len(configs) * 0.3:
        # Large surfaces dominate - use larger thresholds
        base_config = ADAPTIVE_CLUSTERING_PARAMS['large_surfaces'].copy()
        logger.info("Using large_surfaces-dominant configuration")
    elif category_counts.get('small_objects', 0) > len(configs) * 0.5:
        # Small objects dominate - use conservative thresholds
        base_config = ADAPTIVE_CLUSTERING_PARAMS['small_objects'].copy()
        logger.info("Using small_objects-dominant configuration")
    else:
        # Mixed or furniture-dominant - use balanced approach
        base_config = ADAPTIVE_CLUSTERING_PARAMS['furniture'].copy()
        logger.info("Using balanced/furniture-dominant configuration")
    
    # Apply global multipliers
    base_config['spatial_threshold'] *= GLOBAL_MULTIPLIERS['spatial_multiplier']
    base_config['temporal_threshold'] = int(base_config['temporal_threshold'] * GLOBAL_MULTIPLIERS['temporal_multiplier'])
    base_config['confidence_weight'] *= GLOBAL_MULTIPLIERS['confidence_multiplier']
    
    return base_config


def set_global_multipliers(spatial: float = 1.0, temporal: float = 1.0, confidence: float = 1.0):
    """
    Set global multipliers to scale all clustering parameters.
    
    Useful for quick adjustments without modifying individual category parameters.
    
    Args:
        spatial: Multiplier for spatial thresholds (>1 = more permissive)
        temporal: Multiplier for temporal thresholds (>1 = more permissive)  
        confidence: Multiplier for confidence requirements (>1 = more strict)
    """
    global GLOBAL_MULTIPLIERS
    
    GLOBAL_MULTIPLIERS['spatial_multiplier'] = spatial
    GLOBAL_MULTIPLIERS['temporal_multiplier'] = temporal
    GLOBAL_MULTIPLIERS['confidence_multiplier'] = confidence
    
    logger.info(f"Updated global multipliers: spatial={spatial}, temporal={temporal}, confidence={confidence}")


def analyze_object_distribution(labels: List[str]) -> Dict:
    """Analyze distribution of object types for configuration optimization."""
    
    category_stats = {}
    
    for label in labels:
        category = categorize_object(label)
        if category not in category_stats:
            category_stats[category] = {
                'count': 0,
                'labels': [],
                'recommended_config': ADAPTIVE_CLUSTERING_PARAMS.get(category, ADAPTIVE_CLUSTERING_PARAMS['default'])
            }
        
        category_stats[category]['count'] += 1
        category_stats[category]['labels'].append(label)
    
    return category_stats


def generate_config_report(labels: List[str]) -> str:
    """Generate a detailed configuration report for debugging and optimization."""
    
    analysis = analyze_object_distribution(labels)
    unified_config = get_unified_clustering_config(labels)
    
    report = []
    report.append("ADAPTIVE CLUSTERING CONFIGURATION REPORT")
    report.append("=" * 50)
    report.append(f"Total objects: {len(labels)}")
    report.append(f"Global multipliers: {GLOBAL_MULTIPLIERS}")
    report.append("")
    
    report.append("OBJECT CATEGORY BREAKDOWN:")
    for category, stats in analysis.items():
        report.append(f"  {category}: {stats['count']} objects")
        report.append(f"    Labels: {', '.join(stats['labels'][:5])}" + 
                     ("..." if len(stats['labels']) > 5 else ""))
        config = stats['recommended_config']
        report.append(f"    Config: spatial={config['spatial_threshold']:.1f}m, "
                     f"temporal={config['temporal_threshold']}kf")
        report.append("")
    
    report.append("UNIFIED CONFIGURATION:")
    report.append(f"  spatial_threshold: {unified_config['spatial_threshold']:.2f}m")
    report.append(f"  temporal_threshold: {unified_config['temporal_threshold']} keyframes")
    report.append(f"  movement_threshold: {unified_config['movement_threshold']:.2f}m")
    report.append(f"  confidence_weight: {unified_config['confidence_weight']:.2f}")
    report.append(f"  merge_aggressiveness: {unified_config['merge_aggressiveness']}")
    
    return "\n".join(report)


# Testing and demonstration
if __name__ == "__main__":
    
    # Test categorization
    test_labels = [
        "a wall", "table", "book", "indoor plant", "unknown_object",
        "ceiling", "a chair", "a vase", "switch"
    ]
    
    print("OBJECT CATEGORIZATION TEST:")
    for label in test_labels:
        category = categorize_object(label)
        config = get_clustering_config(label)
        print(f"  '{label}' → {category} (spatial: {config.spatial_threshold:.1f}m)")
    
    print("\nCONFIGURATION REPORT:")
    print(generate_config_report(test_labels))