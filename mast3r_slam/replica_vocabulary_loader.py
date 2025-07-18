#!/usr/bin/env python3
"""Automatically load vocabulary from Replica info_semantic.json"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Set

logger = logging.getLogger(__name__)

def load_replica_vocabulary(dataset_path: str, use_only_present_objects: bool = True) -> List[str]:
    """
    Load vocabulary from Replica dataset's info_semantic.json.
    
    Args:
        dataset_path: Path to Replica dataset (e.g., datasets/room_0)
    
    Returns:
        List of unique class names present in the scene
    """
    dataset_path = Path(dataset_path)
    
    # Look for info_semantic.json in various locations
    possible_paths = [
        dataset_path / "info_semantic.json",
        dataset_path / "habitat" / "info_semantic.json",
    ]
    
    info_semantic_path = None
    for path in possible_paths:
        if path.exists():
            info_semantic_path = path
            break
    
    if not info_semantic_path:
        logger.warning(f"info_semantic.json not found in {dataset_path}")
        logger.warning("Falling back to default vocabulary")
        return None
    
    try:
        with open(info_semantic_path, 'r') as f:
            data = json.load(f)
        
        # Extract unique class names from objects in the scene
        class_names = set()
        undefined_count = 0
        
        for obj in data.get('objects', []):
            class_name = obj.get('class_name', '')
            if class_name and class_name != 'undefined':
                # Replace hyphens with spaces for better Grounding DINO compatibility
                class_name_clean = class_name.replace('-', ' ')
                class_names.add(class_name_clean)
            elif class_name == 'undefined':
                undefined_count += 1
        
        # Sort alphabetically for consistency
        vocabulary = sorted(list(class_names))
        
        logger.info(f"Loaded {len(vocabulary)} unique classes from {info_semantic_path}")
        if undefined_count > 0:
            logger.info(f"Skipped {undefined_count} undefined objects")
        logger.info(f"Vocabulary: {vocabulary}")
        
        return vocabulary
        
    except Exception as e:
        logger.error(f"Error loading vocabulary from {info_semantic_path}: {e}")
        return None


def get_replica_class_mapping(dataset_path: str) -> Dict[int, str]:
    """
    Get mapping from class ID to class name for Replica dataset.
    
    Returns:
        Dict mapping class_id to class_name
    """
    dataset_path = Path(dataset_path)
    info_semantic_path = dataset_path / "info_semantic.json"
    
    if not info_semantic_path.exists():
        return {}
    
    try:
        with open(info_semantic_path, 'r') as f:
            data = json.load(f)
        
        # Create class ID to name mapping
        class_mapping = {}
        for cls in data.get('classes', []):
            class_id = cls['id']
            class_name = cls['name']
            class_mapping[class_id] = class_name
        
        return class_mapping
        
    except Exception as e:
        logger.error(f"Error loading class mapping: {e}")
        return {}


def get_replica_instance_mapping(dataset_path: str) -> Dict[int, Dict]:
    """
    Get mapping from instance ID to class information for Replica dataset.
    
    Returns:
        Dict mapping instance_id to {class_id, class_name}
    """
    dataset_path = Path(dataset_path)
    info_semantic_path = dataset_path / "info_semantic.json"
    
    if not info_semantic_path.exists():
        return {}
    
    try:
        with open(info_semantic_path, 'r') as f:
            data = json.load(f)
        
        # Create instance ID to class mapping
        instance_mapping = {}
        for obj in data.get('objects', []):
            instance_id = obj['id']
            class_id = obj['class_id']
            class_name = obj.get('class_name', 'unknown')
            instance_mapping[instance_id] = {
                'class_id': class_id,
                'class_name': class_name
            }
        
        return instance_mapping
        
    except Exception as e:
        logger.error(f"Error loading instance mapping: {e}")
        return {}


def get_scene_specific_vocabulary(dataset_path: str) -> Dict[str, int]:
    """
    Get vocabulary of objects actually present in this specific scene with their counts.
    
    Returns:
        Dict mapping class_name to count
    """
    dataset_path = Path(dataset_path)
    info_semantic_path = dataset_path / "info_semantic.json"
    
    if not info_semantic_path.exists():
        logger.warning(f"info_semantic.json not found in {dataset_path}")
        return {}
    
    try:
        with open(info_semantic_path, 'r') as f:
            data = json.load(f)
        
        # Count objects by class name
        class_counts = {}
        undefined_count = 0
        
        for obj in data.get('objects', []):
            class_name = obj.get('class_name', '')
            if class_name and class_name != 'undefined':
                # Replace hyphens with spaces for better Grounding DINO compatibility
                class_name_clean = class_name.replace('-', ' ')
                class_counts[class_name_clean] = class_counts.get(class_name_clean, 0) + 1
            elif class_name == 'undefined':
                undefined_count += 1
        
        logger.info(f"Scene contains {len(class_counts)} unique object types")
        logger.info(f"Object counts: {dict(sorted(class_counts.items(), key=lambda x: -x[1]))}")
        if undefined_count > 0:
            logger.info(f"Skipped {undefined_count} undefined objects")
        
        return class_counts
        
    except Exception as e:
        logger.error(f"Error loading scene vocabulary: {e}")
        return {}