"""Dynamic Semantic Query API for MAST3R-SLAM"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import re
from mast3r_slam.semantic_integration import SemanticSLAMBackend
from mast3r_slam.track_manager import GlobalTrackManager


@dataclass
class SemanticObject:
    """Represents a semantic object in the map."""
    instance_id: int
    class_name: str
    confidence: float
    num_observations: int
    first_seen_frame: int
    last_seen_frame: int
    centroid: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    
    def volume(self) -> float:
        """Get bounding box volume."""
        dims = self.bbox_max - self.bbox_min
        return float(np.prod(dims))
    
    def contains_point(self, point: np.ndarray) -> bool:
        """Check if point is inside bounding box."""
        return np.all(point >= self.bbox_min) and np.all(point <= self.bbox_max)


class SemanticSLAMAPI:
    """
    High-level API for semantic SLAM queries and dynamic vocabulary.
    
    Provides natural language queries, object search, and runtime vocabulary updates.
    """
    
    def __init__(self, 
                 semantic_backend: SemanticSLAMBackend,
                 semantic_processor = None):
        self.semantic_backend = semantic_backend
        self.semantic_processor = semantic_processor
        self.vocabulary = []
        
        # Spatial relationship keywords
        self.spatial_keywords = {
            'near': self._find_near,
            'far': self._find_far,
            'above': self._find_above,
            'below': self._find_below,
            'left': self._find_left_of,
            'right': self._find_right_of,
            'between': self._find_between,
            'closest': self._find_closest,
            'largest': self._find_largest,
            'smallest': self._find_smallest
        }
        
    def add_object_class(self, text_prompt: str) -> bool:
        """
        Add a new object class to the vocabulary at runtime.
        
        Args:
            text_prompt: Object class to add (e.g., "laptop", "plant")
            
        Returns:
            Success status
        """
        if self.semantic_processor is None:
            print("Warning: No semantic processor available for dynamic vocabulary")
            return False
            
        # Normalize prompt
        text_prompt = text_prompt.strip().lower()
        
        if text_prompt in self.vocabulary:
            print(f"'{text_prompt}' already in vocabulary")
            return True
            
        # Add to processor vocabulary
        if hasattr(self.semantic_processor, 'vocabulary'):
            self.semantic_processor.vocabulary.append(text_prompt)
            self.vocabulary.append(text_prompt)
            print(f"Added '{text_prompt}' to vocabulary")
            return True
            
        return False
        
    def remove_object_class(self, text_prompt: str) -> bool:
        """Remove an object class from vocabulary."""
        text_prompt = text_prompt.strip().lower()
        
        if text_prompt in self.vocabulary:
            self.vocabulary.remove(text_prompt)
            if hasattr(self.semantic_processor, 'vocabulary'):
                self.semantic_processor.vocabulary.remove(text_prompt)
            print(f"Removed '{text_prompt}' from vocabulary")
            return True
            
        return False
        
    def find_objects(self, query: str) -> List[SemanticObject]:
        """
        Natural language object search.
        
        Examples:
            - "all chairs"
            - "person near table"
            - "largest chair"
            - "objects between person and door"
            
        Args:
            query: Natural language query
            
        Returns:
            List of matching semantic objects
        """
        query = query.lower().strip()
        
        # Extract all objects from the map first
        all_objects = self._extract_all_objects()
        
        # Simple keyword-based parsing
        tokens = query.split()
        
        # Check for "all" keyword
        if "all" in tokens:
            # Find class name after "all"
            idx = tokens.index("all")
            if idx + 1 < len(tokens):
                class_name = tokens[idx + 1].rstrip('s')  # Remove plural
                return [obj for obj in all_objects if obj.class_name == class_name]
                
        # Check for spatial relationships
        for keyword, func in self.spatial_keywords.items():
            if keyword in query:
                return func(query, all_objects)
                
        # Default: search by class name
        results = []
        for obj in all_objects:
            if obj.class_name in query:
                results.append(obj)
                
        return results
        
    def get_object_by_id(self, instance_id: int) -> Optional[SemanticObject]:
        """Get object by instance ID."""
        all_objects = self._extract_all_objects()
        for obj in all_objects:
            if obj.instance_id == instance_id:
                return obj
        return None
        
    def get_object_trajectory(self, instance_id: int) -> Dict[int, np.ndarray]:
        """
        Get trajectory of an object across frames.
        
        Returns:
            Dict mapping frame_id to object centroid position
        """
        trajectory = {}
        
        # Get track history
        track_info = self.semantic_backend.track_manager.get_track_history(instance_id)
        if not track_info:
            return trajectory
            
        # For each frame where object appears
        for frame_id in track_info['frames']:
            # Get keyframe index (simplified - assumes 1:1 mapping)
            kf_idx = frame_id
            
            # Get semantic data
            semantics = self.semantic_backend.keyframe_semantics.get(kf_idx)
            if semantics is None:
                continue
                
            labels, _ = semantics
            mask = labels == instance_id
            
            if mask.any():
                # Get 3D points for this instance
                kf = self.semantic_backend.keyframes.get_frame(kf_idx)
                if kf and kf.X_canon is not None:
                    instance_points = kf.T_WC.act(kf.X_canon[mask])
                    centroid = instance_points.mean(dim=0).cpu().numpy()
                    trajectory[frame_id] = centroid
                    
        return trajectory
        
    def get_semantic_map_summary(self) -> Dict:
        """Get summary statistics of the semantic map."""
        all_objects = self._extract_all_objects()
        
        summary = {
            'total_instances': len(all_objects),
            'unique_classes': len(set(obj.class_name for obj in all_objects)),
            'class_counts': {},
            'average_confidence': 0.0,
            'total_observations': 0
        }
        
        # Count by class
        for obj in all_objects:
            if obj.class_name not in summary['class_counts']:
                summary['class_counts'][obj.class_name] = 0
            summary['class_counts'][obj.class_name] += 1
            summary['average_confidence'] += obj.confidence
            summary['total_observations'] += obj.num_observations
            
        if all_objects:
            summary['average_confidence'] /= len(all_objects)
            
        return summary
        
    def _extract_all_objects(self) -> List[SemanticObject]:
        """Extract all semantic objects from the map."""
        objects = []
        processed_instances = set()
        
        # Process each keyframe
        for kf_idx in range(len(self.semantic_backend.keyframes)):
            semantics = self.semantic_backend.keyframe_semantics.get(kf_idx)
            if semantics is None:
                continue
                
            labels, confidences = semantics
            unique_labels = torch.unique(labels[labels > 0])
            
            for label in unique_labels:
                instance_id = label.item()
                if instance_id in processed_instances:
                    continue
                    
                processed_instances.add(instance_id)
                
                # Get object info
                obj = self._create_semantic_object(instance_id)
                if obj:
                    objects.append(obj)
                    
        return objects
        
    def _create_semantic_object(self, instance_id: int) -> Optional[SemanticObject]:
        """Create SemanticObject from instance ID."""
        # Get track info
        track_info = self.semantic_backend.track_manager.get_track_history(instance_id)
        if not track_info:
            return None
            
        # Collect all 3D points for this instance
        all_points = []
        all_confidences = []
        
        for frame_id in track_info['frames']:
            kf_idx = frame_id  # Simplified
            semantics = self.semantic_backend.keyframe_semantics.get(kf_idx)
            if semantics is None:
                continue
                
            labels, confidences = semantics
            mask = labels == instance_id
            
            if mask.any():
                kf = self.semantic_backend.keyframes.get_frame(kf_idx)
                if kf and kf.X_canon is not None:
                    world_points = kf.T_WC.act(kf.X_canon[mask])
                    all_points.append(world_points.cpu().numpy())
                    all_confidences.extend(confidences[mask].cpu().numpy())
                    
        if not all_points:
            return None
            
        # Compute statistics
        all_points = np.concatenate(all_points, axis=0)
        centroid = all_points.mean(axis=0)
        bbox_min = all_points.min(axis=0)
        bbox_max = all_points.max(axis=0)
        avg_confidence = np.mean(all_confidences)
        
        return SemanticObject(
            instance_id=instance_id,
            class_name=track_info['label'],
            confidence=avg_confidence,
            num_observations=track_info['num_observations'],
            first_seen_frame=min(track_info['frames']),
            last_seen_frame=max(track_info['frames']),
            centroid=centroid,
            bbox_min=bbox_min,
            bbox_max=bbox_max
        )
        
    # Spatial relationship functions
    def _find_near(self, query: str, objects: List[SemanticObject]) -> List[SemanticObject]:
        """Find objects near another object."""
        # Parse "X near Y"
        match = re.search(r'(\w+)\s+near\s+(\w+)', query)
        if not match:
            return []
            
        target_class = match.group(1)
        reference_class = match.group(2)
        
        # Find reference objects
        ref_objects = [obj for obj in objects if obj.class_name == reference_class]
        if not ref_objects:
            return []
            
        # Find target objects near any reference
        results = []
        threshold = 2.0  # meters
        
        for obj in objects:
            if obj.class_name != target_class:
                continue
                
            for ref in ref_objects:
                dist = np.linalg.norm(obj.centroid - ref.centroid)
                if dist < threshold:
                    results.append(obj)
                    break
                    
        return results
        
    def _find_largest(self, query: str, objects: List[SemanticObject]) -> List[SemanticObject]:
        """Find largest object of a class."""
        # Extract class name
        tokens = query.split()
        if "largest" not in tokens:
            return []
            
        idx = tokens.index("largest")
        if idx + 1 >= len(tokens):
            return []
            
        class_name = tokens[idx + 1]
        
        # Filter by class and sort by volume
        class_objects = [obj for obj in objects if obj.class_name == class_name]
        if not class_objects:
            return []
            
        return [max(class_objects, key=lambda x: x.volume())]
        
    def _find_far(self, query: str, objects: List[SemanticObject]) -> List[SemanticObject]:
        """Find objects far from another object."""
        # Similar to _find_near but with larger threshold
        # Implementation similar to _find_near
        return []
        
    def _find_above(self, query: str, objects: List[SemanticObject]) -> List[SemanticObject]:
        """Find objects above another object."""
        # Check Y coordinate
        return []
        
    def _find_below(self, query: str, objects: List[SemanticObject]) -> List[SemanticObject]:
        """Find objects below another object."""
        return []
        
    def _find_left_of(self, query: str, objects: List[SemanticObject]) -> List[SemanticObject]:
        """Find objects to the left of another object."""
        return []
        
    def _find_right_of(self, query: str, objects: List[SemanticObject]) -> List[SemanticObject]:
        """Find objects to the right of another object."""
        return []
        
    def _find_between(self, query: str, objects: List[SemanticObject]) -> List[SemanticObject]:
        """Find objects between two other objects."""
        return []
        
    def _find_closest(self, query: str, objects: List[SemanticObject]) -> List[SemanticObject]:
        """Find closest object to another object."""
        return []
        
    def _find_smallest(self, query: str, objects: List[SemanticObject]) -> List[SemanticObject]:
        """Find smallest object of a class."""
        # Similar to _find_largest but with min
        return []