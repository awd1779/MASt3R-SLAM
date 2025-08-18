"""
Semantic Processing Subsystem

This package provides semantic segmentation and processing for semantic SLAM,
including object detection, mask processing, and dense reconstruction.

Key Components:
- GroundedSAM2Processor: Object detection and segmentation
- SemanticFrame: Semantic frame data structure  
- Dense reconstruction: Final semantic point cloud generation
- Vocabulary loading: Dataset-specific object vocabularies
"""

# Core semantic processing
from .grounded_sam2_real import RealGroundedSAM2Processor

# Data structures
from .semantic_frame import SemanticFrame

# Dense reconstruction
from .dense_semantic_reconstruction_tracked_v2 import create_dense_semantic_reconstruction_tracked

# Utilities
from .mask_deduplicator import MaskDeduplicator
from .replica_vocabulary_loader import load_replica_vocabulary, ReplicaSemanticLoader
from .utils import (
    convert_cxcywh_to_xyxy,
    generate_track_color,
    ensure_debug_directory,
    validate_array_correspondence,
    convert_rle_to_binary_mask,
    convert_binary_mask_to_rle
)

__all__ = [
    'RealGroundedSAM2Processor',
    'SemanticFrame',
    'create_dense_semantic_reconstruction_tracked',
    'MaskDeduplicator',
    'load_replica_vocabulary',
    'ReplicaSemanticLoader',
    'convert_cxcywh_to_xyxy',
    'generate_track_color',
    'ensure_debug_directory',
    'validate_array_correspondence',
    'convert_rle_to_binary_mask',
    'convert_binary_mask_to_rle',
]