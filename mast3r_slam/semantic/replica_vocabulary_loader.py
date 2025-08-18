#!/usr/bin/env python3
"""Automatically load vocabulary from Replica info_semantic.json"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Set, Optional

logger = logging.getLogger(__name__)


class ReplicaSemanticLoader:
    """
    Unified loader for Replica semantic data with centralized file handling.
    
    This class eliminates duplication across the various loader functions by
    providing a single source of truth for data loading and caching.
    """
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self._data: Optional[Dict] = None
        self._info_semantic_path: Optional[Path] = None
    
    def _find_info_semantic_path(self) -> Path:
        """Find info_semantic.json with fallback locations."""
        if self._info_semantic_path:
            return self._info_semantic_path
            
        possible_paths = [
            self.dataset_path / "info_semantic.json",
            self.dataset_path / "habitat" / "info_semantic.json",
        ]
        
        for path in possible_paths:
            if path.exists():
                self._info_semantic_path = path
                return path
        
        raise FileNotFoundError(f"info_semantic.json not found in {self.dataset_path}")
    
    def _load_data(self) -> Dict:
        """Load and cache JSON data."""
        if self._data is None:
            path = self._find_info_semantic_path()
            with open(path, 'r') as f:
                self._data = json.load(f)
        return self._data
    
    @staticmethod
    def _clean_class_name(class_name: str) -> str:
        """Clean class name for better compatibility."""
        return class_name.replace('-', ' ')
    
    def get_vocabulary(self) -> List[str]:
        """Get sorted list of unique class names."""
        try:
            data = self._load_data()
            class_names = set()
            undefined_count = 0
            
            for obj in data.get('objects', []):
                class_name = obj.get('class_name', '')
                if class_name and class_name != 'undefined':
                    clean_name = self._clean_class_name(class_name)
                    class_names.add(clean_name)
                else:
                    undefined_count += 1
            
            if undefined_count > 0:
                logger.info(f"Skipped {undefined_count} undefined objects")
            
            vocabulary = sorted(list(class_names))
            logger.info(f"Loaded vocabulary with {len(vocabulary)} unique classes: {vocabulary}")
            return vocabulary
            
        except FileNotFoundError:
            logger.warning(f"info_semantic.json not found in {self.dataset_path}")
            return []
        except Exception as e:
            logger.error(f"Error loading vocabulary: {e}")
            return []
    
    def get_class_mapping(self) -> Dict[int, str]:
        """Get mapping from class ID to class name."""
        try:
            data = self._load_data()
            class_mapping = {}
            
            for cls in data.get('classes', []):
                class_id = cls['id']
                class_name = cls['name']
                class_mapping[class_id] = class_name
            
            return class_mapping
            
        except Exception as e:
            logger.error(f"Error loading class mapping: {e}")
            return {}
    
    def get_instance_mapping(self) -> Dict[int, Dict]:
        """Get mapping from instance ID to instance data."""
        try:
            data = self._load_data()
            instance_mapping = {}
            
            for obj in data.get('objects', []):
                instance_id = obj.get('id')
                if instance_id is not None:
                    instance_mapping[instance_id] = {
                        'class_name': obj.get('class_name', ''),
                        'class_id': obj.get('class_id', -1),
                        'room_id': obj.get('room_id', -1),
                        'size': obj.get('size', []),
                    }
            
            return instance_mapping
            
        except Exception as e:
            logger.error(f"Error loading instance mapping: {e}")
            return {}
    
    def get_class_counts(self) -> Dict[str, int]:
        """Get object counts by class name."""
        try:
            data = self._load_data()
            class_counts = {}
            
            for obj in data.get('objects', []):
                class_name = obj.get('class_name', '')
                if class_name and class_name != 'undefined':
                    clean_name = self._clean_class_name(class_name)
                    class_counts[clean_name] = class_counts.get(clean_name, 0) + 1
            
            return class_counts
            
        except Exception as e:
            logger.error(f"Error loading class counts: {e}")
            return {}

def load_replica_vocabulary(dataset_path: str, use_only_present_objects: bool = True) -> List[str]:
    """
    Load vocabulary from Replica dataset's info_semantic.json.
    
    Args:
        dataset_path: Path to Replica dataset (e.g., datasets/room_0)
        use_only_present_objects: Compatibility parameter (unused)
    
    Returns:
        List of unique class names present in the scene, or None if file not found
    """
    try:
        loader = ReplicaSemanticLoader(dataset_path)
        vocabulary = loader.get_vocabulary()
        return vocabulary if vocabulary else None
    except FileNotFoundError:
        logger.warning(f"info_semantic.json not found in {dataset_path}")
        logger.warning("Falling back to default vocabulary")
        return None
    except Exception as e:
        logger.error(f"Error loading vocabulary: {e}")
        return None


def get_replica_class_mapping(dataset_path: str) -> Dict[int, str]:
    """
    Get mapping from class ID to class name for Replica dataset.
    
    Args:
        dataset_path: Path to Replica dataset
        
    Returns:
        Dict mapping class_id to class_name
    """
    try:
        loader = ReplicaSemanticLoader(dataset_path)
        return loader.get_class_mapping()
    except Exception as e:
        logger.error(f"Error loading class mapping: {e}")
        return {}


def get_replica_instance_mapping(dataset_path: str) -> Dict[int, Dict]:
    """
    Get mapping from instance ID to class information for Replica dataset.
    
    Args:
        dataset_path: Path to Replica dataset
        
    Returns:
        Dict mapping instance_id to {class_id, class_name, room_id, size}
    """
    try:
        loader = ReplicaSemanticLoader(dataset_path)
        return loader.get_instance_mapping()
    except Exception as e:
        logger.error(f"Error loading instance mapping: {e}")
        return {}


def get_scene_specific_vocabulary(dataset_path: str) -> Dict[str, int]:
    """
    Get vocabulary of objects actually present in this specific scene with their counts.
    
    Args:
        dataset_path: Path to Replica dataset
        
    Returns:
        Dict mapping class_name to count
    """
    try:
        loader = ReplicaSemanticLoader(dataset_path)
        return loader.get_class_counts()
    except Exception as e:
        logger.error(f"Error loading scene vocabulary: {e}")
        return {}