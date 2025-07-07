"""
Semantic Caching System for MASt3R-SLAM
Caches CLIP features and semantic predictions to avoid redundant processing.
"""
import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
import hashlib
from collections import OrderedDict
import pickle
import lz4.frame
from pathlib import Path
import time


class SemanticFeatureCache:
    """
    Caches semantic features and predictions with similarity-based retrieval.
    """
    
    def __init__(self,
                 max_cache_size: int = 1000,
                 similarity_threshold: float = 0.95,
                 feature_dim: int = 768,
                 device: str = 'cuda',
                 persistent_cache_dir: Optional[Path] = None):
        """
        Initialize the semantic cache.
        
        Args:
            max_cache_size: Maximum number of entries to cache
            similarity_threshold: Threshold for considering features similar
            feature_dim: Dimension of CLIP features
            device: Device for computations
            persistent_cache_dir: Directory for persistent cache storage
        """
        self.max_cache_size = max_cache_size
        self.similarity_threshold = similarity_threshold
        self.feature_dim = feature_dim
        self.device = device
        self.persistent_cache_dir = persistent_cache_dir
        
        # LRU cache for features and predictions
        self.feature_cache = OrderedDict()
        self.prediction_cache = OrderedDict()
        
        # Index for fast similarity search
        self.feature_index = None
        self.index_keys = []
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.partial_hits = 0
        
        # Load persistent cache if available
        if persistent_cache_dir:
            self._load_persistent_cache()
    
    def _compute_hash(self, image_crop: torch.Tensor) -> str:
        """
        Compute a hash for an image crop.
        
        Args:
            image_crop: Image tensor
            
        Returns:
            Hash string
        """
        # Convert to bytes and compute hash
        crop_bytes = image_crop.cpu().numpy().tobytes()
        return hashlib.md5(crop_bytes).hexdigest()
    
    def _build_feature_index(self):
        """Build or update the feature index for fast similarity search."""
        if not self.feature_cache:
            self.feature_index = None
            self.index_keys = []
            return
        
        # Stack all features
        features_list = []
        keys_list = []
        
        for key, feature in self.feature_cache.items():
            features_list.append(feature)
            keys_list.append(key)
        
        self.feature_index = torch.stack(features_list)
        self.index_keys = keys_list
    
    def get_cached_features(self, 
                          image_crops: List[torch.Tensor],
                          return_partial: bool = True) -> Tuple[Dict[int, torch.Tensor], List[int]]:
        """
        Retrieve cached features for image crops.
        
        Args:
            image_crops: List of image crop tensors
            return_partial: Whether to return partial matches based on similarity
            
        Returns:
            Tuple of (cached_features_dict, missing_indices)
        """
        cached_features = {}
        missing_indices = []
        
        for idx, crop in enumerate(image_crops):
            crop_hash = self._compute_hash(crop)
            
            # Check exact match
            if crop_hash in self.feature_cache:
                cached_features[idx] = self.feature_cache[crop_hash]
                self.hits += 1
                # Move to end (LRU)
                self.feature_cache.move_to_end(crop_hash)
            elif return_partial and self.feature_index is not None:
                # Check for similar features
                similar_idx = self._find_similar_feature(crop)
                if similar_idx is not None:
                    similar_key = self.index_keys[similar_idx]
                    cached_features[idx] = self.feature_cache[similar_key]
                    self.partial_hits += 1
                else:
                    missing_indices.append(idx)
                    self.misses += 1
            else:
                missing_indices.append(idx)
                self.misses += 1
        
        return cached_features, missing_indices
    
    def _find_similar_feature(self, crop: torch.Tensor) -> Optional[int]:
        """
        Find similar feature in cache using cosine similarity.
        
        Args:
            crop: Image crop tensor
            
        Returns:
            Index of similar feature or None
        """
        # This would require computing CLIP features for the crop
        # For now, return None (would need CLIP model access)
        return None
    
    def add_features(self, 
                    image_crops: List[torch.Tensor],
                    features: torch.Tensor,
                    predictions: Optional[Dict] = None):
        """
        Add features and predictions to cache.
        
        Args:
            image_crops: List of image crop tensors
            features: Corresponding CLIP features
            predictions: Optional prediction results
        """
        for idx, (crop, feature) in enumerate(zip(image_crops, features)):
            crop_hash = self._compute_hash(crop)
            
            # Add to cache
            self.feature_cache[crop_hash] = feature.detach().cpu()
            
            if predictions and idx in predictions:
                self.prediction_cache[crop_hash] = predictions[idx]
            
            # Maintain cache size
            if len(self.feature_cache) > self.max_cache_size:
                # Remove oldest (first) item
                oldest_key = next(iter(self.feature_cache))
                del self.feature_cache[oldest_key]
                if oldest_key in self.prediction_cache:
                    del self.prediction_cache[oldest_key]
        
        # Rebuild index periodically
        if len(self.feature_cache) % 100 == 0:
            self._build_feature_index()
    
    def get_cached_predictions(self, crop_hashes: List[str]) -> Dict[str, Dict]:
        """
        Retrieve cached predictions for given hashes.
        
        Args:
            crop_hashes: List of crop hashes
            
        Returns:
            Dictionary of predictions
        """
        predictions = {}
        for hash_val in crop_hashes:
            if hash_val in self.prediction_cache:
                predictions[hash_val] = self.prediction_cache[hash_val]
                # Move to end (LRU)
                self.prediction_cache.move_to_end(hash_val)
        
        return predictions
    
    def compute_cache_efficiency(self) -> Dict:
        """
        Compute cache efficiency statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.hits + self.partial_hits + self.misses
        
        if total_requests == 0:
            return {
                'hit_rate': 0.0,
                'partial_hit_rate': 0.0,
                'miss_rate': 0.0,
                'total_requests': 0,
                'cache_size': len(self.feature_cache)
            }
        
        return {
            'hit_rate': self.hits / total_requests,
            'partial_hit_rate': self.partial_hits / total_requests,
            'miss_rate': self.misses / total_requests,
            'total_requests': total_requests,
            'cache_size': len(self.feature_cache),
            'exact_hits': self.hits,
            'partial_hits': self.partial_hits,
            'misses': self.misses
        }
    
    def save_persistent_cache(self):
        """Save cache to disk for persistence across runs."""
        if not self.persistent_cache_dir:
            return
        
        self.persistent_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Save feature cache
        feature_cache_path = self.persistent_cache_dir / 'feature_cache.lz4'
        with lz4.frame.open(str(feature_cache_path), 'wb') as f:
            pickle.dump(dict(self.feature_cache), f)
        
        # Save prediction cache
        prediction_cache_path = self.persistent_cache_dir / 'prediction_cache.lz4'
        with lz4.frame.open(str(prediction_cache_path), 'wb') as f:
            pickle.dump(dict(self.prediction_cache), f)
        
        # Save statistics
        stats_path = self.persistent_cache_dir / 'cache_stats.pkl'
        with open(stats_path, 'wb') as f:
            pickle.dump({
                'hits': self.hits,
                'partial_hits': self.partial_hits,
                'misses': self.misses
            }, f)
    
    def _load_persistent_cache(self):
        """Load cache from disk if available."""
        if not self.persistent_cache_dir or not self.persistent_cache_dir.exists():
            return
        
        try:
            # Load feature cache
            feature_cache_path = self.persistent_cache_dir / 'feature_cache.lz4'
            if feature_cache_path.exists():
                with lz4.frame.open(str(feature_cache_path), 'rb') as f:
                    loaded_features = pickle.load(f)
                    self.feature_cache = OrderedDict(loaded_features)
            
            # Load prediction cache
            prediction_cache_path = self.persistent_cache_dir / 'prediction_cache.lz4'
            if prediction_cache_path.exists():
                with lz4.frame.open(str(prediction_cache_path), 'rb') as f:
                    loaded_predictions = pickle.load(f)
                    self.prediction_cache = OrderedDict(loaded_predictions)
            
            # Load statistics
            stats_path = self.persistent_cache_dir / 'cache_stats.pkl'
            if stats_path.exists():
                with open(stats_path, 'rb') as f:
                    stats = pickle.load(f)
                    self.hits = stats.get('hits', 0)
                    self.partial_hits = stats.get('partial_hits', 0)
                    self.misses = stats.get('misses', 0)
            
            # Build index
            self._build_feature_index()
            
            print(f"[SemanticCache] Loaded {len(self.feature_cache)} cached features")
            
        except Exception as e:
            print(f"[SemanticCache] Failed to load persistent cache: {e}")
            self.clear_cache()
    
    def clear_cache(self):
        """Clear all cached data."""
        self.feature_cache.clear()
        self.prediction_cache.clear()
        self.feature_index = None
        self.index_keys = []
        self.hits = 0
        self.partial_hits = 0
        self.misses = 0


class ViewpointCache:
    """
    Cache semantic results based on camera viewpoint similarity.
    """
    
    def __init__(self,
                 position_threshold: float = 0.5,
                 rotation_threshold: float = 15.0,  # degrees
                 max_cache_size: int = 500):
        """
        Initialize viewpoint-based cache.
        
        Args:
            position_threshold: Maximum position difference for cache hit (meters)
            rotation_threshold: Maximum rotation difference for cache hit (degrees)
            max_cache_size: Maximum number of viewpoints to cache
        """
        self.position_threshold = position_threshold
        self.rotation_threshold = np.radians(rotation_threshold)
        self.max_cache_size = max_cache_size
        
        # Cache indexed by viewpoint
        self.viewpoint_cache = OrderedDict()
        
    def _compute_viewpoint_hash(self, pose: torch.Tensor) -> str:
        """
        Compute hash for a camera pose.
        
        Args:
            pose: 4x4 transformation matrix
            
        Returns:
            Hash string
        """
        # Extract position and rotation
        position = pose[:3, 3].cpu().numpy()
        rotation = pose[:3, :3].cpu().numpy()
        
        # Quantize for hashing
        pos_quantized = np.round(position / self.position_threshold).astype(int)
        rot_quantized = np.round(rotation * 10).astype(int)
        
        # Combine and hash
        combined = np.concatenate([pos_quantized.flatten(), rot_quantized.flatten()])
        return hashlib.md5(combined.tobytes()).hexdigest()
    
    def get_cached_semantics(self, pose: torch.Tensor) -> Optional[Dict]:
        """
        Retrieve cached semantics for similar viewpoint.
        
        Args:
            pose: Current camera pose
            
        Returns:
            Cached semantic data or None
        """
        # First try exact hash match
        pose_hash = self._compute_viewpoint_hash(pose)
        if pose_hash in self.viewpoint_cache:
            self.viewpoint_cache.move_to_end(pose_hash)  # LRU
            return self.viewpoint_cache[pose_hash]
        
        # Then check for similar viewpoints
        current_pos = pose[:3, 3].cpu().numpy()
        current_rot = pose[:3, :3].cpu().numpy()
        
        for cached_hash, cached_data in self.viewpoint_cache.items():
            cached_pose = cached_data['pose']
            cached_pos = cached_pose[:3, 3].cpu().numpy()
            cached_rot = cached_pose[:3, :3].cpu().numpy()
            
            # Check position distance
            pos_dist = np.linalg.norm(current_pos - cached_pos)
            if pos_dist > self.position_threshold:
                continue
            
            # Check rotation difference (using Frobenius norm as approximation)
            rot_diff = np.linalg.norm(current_rot - cached_rot, 'fro')
            if rot_diff < self.rotation_threshold:
                self.viewpoint_cache.move_to_end(cached_hash)  # LRU
                return cached_data
        
        return None
    
    def add_semantics(self, pose: torch.Tensor, semantic_data: Dict):
        """
        Add semantic data for a viewpoint.
        
        Args:
            pose: Camera pose
            semantic_data: Semantic processing results
        """
        pose_hash = self._compute_viewpoint_hash(pose)
        
        # Add pose to semantic data
        semantic_data['pose'] = pose.detach().cpu()
        semantic_data['timestamp'] = time.time()
        
        self.viewpoint_cache[pose_hash] = semantic_data
        
        # Maintain cache size
        if len(self.viewpoint_cache) > self.max_cache_size:
            self.viewpoint_cache.popitem(last=False)  # Remove oldest
    
    def get_nearby_semantics(self, pose: torch.Tensor, radius: float = 2.0) -> List[Dict]:
        """
        Get all cached semantics within a radius of current pose.
        
        Args:
            pose: Current camera pose
            radius: Search radius in meters
            
        Returns:
            List of nearby semantic data
        """
        current_pos = pose[:3, 3].cpu().numpy()
        nearby = []
        
        for cached_data in self.viewpoint_cache.values():
            cached_pos = cached_data['pose'][:3, 3].cpu().numpy()
            dist = np.linalg.norm(current_pos - cached_pos)
            
            if dist <= radius:
                nearby.append({
                    'data': cached_data,
                    'distance': dist
                })
        
        # Sort by distance
        nearby.sort(key=lambda x: x['distance'])
        
        return nearby