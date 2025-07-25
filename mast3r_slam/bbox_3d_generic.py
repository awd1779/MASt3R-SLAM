"""Generic 3D Bounding Box Creation - No hardcoded values, works for any object."""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger('mast3r_slam.bbox_3d')


@dataclass
class BBox3DConfig:
    """Configuration for 3D bounding box creation - all adaptive."""
    # Point filtering
    min_depth: float = 0.1  # meters
    max_depth: float = 50.0  # meters
    
    # Adaptive thresholds based on statistics
    min_points_ratio: float = 0.001  # Min 0.1% of mask pixels must have valid 3D
    outlier_std_factor: float = 3.0  # Remove points > 3 std from mean
    
    # Bounding box method selection
    elongation_threshold: float = 2.5  # If max_dim/min_dim > this, use OBB
    planarity_threshold: float = 0.1  # If smallest eigenvalue < this * largest, it's planar
    
    # Size validation (adaptive based on scene statistics)
    size_outlier_factor: float = 10.0  # Reject if bbox > 10x median size for category


class AdaptiveBBoxEstimator:
    """Estimates appropriate bbox parameters from scene statistics."""
    
    def __init__(self):
        self.object_sizes = {}  # Store size statistics per category
        self.scene_scale = None
        self.depth_range = None
        
    def update_statistics(self, label: str, bbox_size: torch.Tensor):
        """Update running statistics for object sizes."""
        if label not in self.object_sizes:
            self.object_sizes[label] = []
        self.object_sizes[label].append(bbox_size.cpu().numpy())
        
    def get_expected_size_range(self, label: str) -> Optional[Tuple[float, float]]:
        """Get expected size range for object category."""
        if label not in self.object_sizes or len(self.object_sizes[label]) < 3:
            return None  # Not enough data
            
        sizes = np.array(self.object_sizes[label])
        volumes = np.prod(sizes, axis=1)
        
        # Use robust statistics (median and MAD)
        median_vol = np.median(volumes)
        mad = np.median(np.abs(volumes - median_vol))
        
        # Adaptive range based on variance
        min_vol = median_vol - 3 * mad
        max_vol = median_vol + 3 * mad
        
        return (max(min_vol, 0.0001), max_vol)  # Ensure positive


class Generic3DBoundingBox:
    """Generic 3D bounding box computation without hardcoded values."""
    
    def __init__(self, config: BBox3DConfig = None, estimator: AdaptiveBBoxEstimator = None):
        self.config = config or BBox3DConfig()
        self.estimator = estimator or AdaptiveBBoxEstimator()
        
    def extract_valid_3d_points(self, 
                               keyframe,
                               mask: Union[torch.Tensor, np.ndarray],
                               return_stats: bool = False) -> Tuple[Optional[torch.Tensor], Dict]:
        """Extract valid 3D points for masked object - fully adaptive."""
        
        # Ensure mask is tensor
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        
        # Get dimensions
        h, w = keyframe.img_shape[0, 0].item(), keyframe.img_shape[0, 1].item()
        
        # Flatten mask
        mask_flat = mask.reshape(-1)
        num_mask_pixels = mask_flat.sum().item()
        
        if num_mask_pixels == 0:
            return None, {"error": "Empty mask"}
        
        # Get 3D points
        X_cam = keyframe.X_canon  # (H*W, 3)
        masked_points = X_cam[mask_flat]
        
        # Get depths for filtering
        depths = masked_points[:, 2]
        
        # Adaptive depth filtering based on scene
        if self.estimator.depth_range is None:
            # First frame - estimate depth range from all points
            all_depths = X_cam[:, 2]
            valid_depths = all_depths[(all_depths > 0) & torch.isfinite(all_depths)]
            if len(valid_depths) > 0:
                self.estimator.depth_range = (
                    torch.quantile(valid_depths, 0.01).item(),
                    torch.quantile(valid_depths, 0.99).item()
                )
            else:
                self.estimator.depth_range = (self.config.min_depth, self.config.max_depth)
        
        min_d, max_d = self.estimator.depth_range
        
        # Filter by depth and validity
        valid_mask = (depths > min_d) & (depths < max_d)
        valid_mask &= torch.isfinite(masked_points).all(dim=1)
        
        valid_points = masked_points[valid_mask]
        
        # Check minimum points adaptively
        min_points = max(10, int(num_mask_pixels * self.config.min_points_ratio))
        
        if len(valid_points) < min_points:
            return None, {
                "error": "Too few valid points",
                "valid_points": len(valid_points),
                "min_required": min_points,
                "mask_pixels": num_mask_pixels
            }
        
        # Remove outliers using robust statistics
        if len(valid_points) > 50:  # Only for sufficient points
            valid_points = self._remove_outliers(valid_points)
        
        # Transform to world coordinates
        T_WC = keyframe.T_WC
        
        # Handle different types of T_WC (tensor or lietorch object)
        if hasattr(T_WC, 'act'):
            # T_WC is a lietorch SE3/Sim3 object
            world_points = T_WC.act(valid_points)
        else:
            # T_WC is a regular tensor/matrix
            if T_WC.dim() == 2:
                T_WC_matrix = T_WC
            else:
                T_WC_matrix = T_WC.squeeze() if T_WC.dim() > 2 else T_WC
            
            # Apply transformation manually
            valid_points_homo = torch.cat([valid_points, torch.ones(valid_points.shape[0], 1, device=valid_points.device)], dim=1)
            world_points_homo = (T_WC_matrix @ valid_points_homo.T).T
            world_points = world_points_homo[:, :3]
        
        stats = {
            "mask_pixels": num_mask_pixels,
            "valid_3d_points": len(world_points),
            "validity_ratio": len(world_points) / num_mask_pixels,
            "depth_range": (depths[valid_mask].min().item(), depths[valid_mask].max().item())
        }
        
        return world_points, stats
    
    def _remove_outliers(self, points: torch.Tensor) -> torch.Tensor:
        """Remove outliers using robust statistics - no hardcoded values."""
        # Compute robust center (median)
        center = torch.median(points, dim=0)[0]
        
        # Compute distances from center
        distances = torch.norm(points - center, dim=1)
        
        # Robust scale estimate (MAD)
        mad = torch.median(torch.abs(distances - torch.median(distances)))
        
        # Keep points within adaptive threshold
        threshold = torch.median(distances) + self.config.outlier_std_factor * mad
        inliers = distances < threshold
        
        return points[inliers]
    
    def compute_adaptive_bbox(self, 
                            points: torch.Tensor,
                            label: str = "unknown") -> Dict:
        """Compute bounding box with automatic method selection."""
        
        if len(points) < 4:
            # Not enough points for oriented bbox
            return self._compute_aabb(points)
        
        # Analyze point distribution
        analysis = self._analyze_point_distribution(points)
        
        # Choose method based on shape
        if analysis['is_planar']:
            # Planar object (wall, floor, picture)
            bbox = self._compute_planar_bbox(points, analysis)
        elif analysis['elongation_ratio'] > self.config.elongation_threshold:
            # Elongated object - use OBB
            bbox = self._compute_obb(points)
        else:
            # Roughly cubic - use AABB
            bbox = self._compute_aabb(points)
        
        # Add metadata
        bbox['shape_type'] = analysis['shape_type']
        bbox['confidence'] = self._compute_bbox_confidence(points, bbox)
        
        # Validate size if we have statistics
        bbox['size_valid'] = self._validate_bbox_size(bbox, label)
        
        return bbox
    
    def _analyze_point_distribution(self, points: torch.Tensor) -> Dict:
        """Analyze point cloud shape without hardcoded assumptions."""
        # Center points
        centroid = torch.mean(points, dim=0)
        centered = points - centroid
        
        # PCA analysis
        cov = torch.mm(centered.T, centered) / len(points)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        
        # Sort eigenvalues
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Compute shape metrics
        total_var = eigenvalues.sum()
        var_ratios = eigenvalues / total_var
        
        # Adaptive shape detection
        elongation = eigenvalues[0] / (eigenvalues[2] + 1e-6)
        planarity = 1.0 - (eigenvalues[2] / eigenvalues[0])
        
        # Classify shape
        if var_ratios[2] < self.config.planarity_threshold:
            shape_type = "planar"
        elif elongation > self.config.elongation_threshold:
            shape_type = "elongated"
        else:
            shape_type = "compact"
        
        return {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'elongation_ratio': elongation.item(),
            'planarity': planarity.item(),
            'is_planar': var_ratios[2] < self.config.planarity_threshold,
            'shape_type': shape_type,
            'variance_ratios': var_ratios
        }
    
    def _compute_aabb(self, points: torch.Tensor) -> Dict:
        """Compute axis-aligned bounding box."""
        min_coords = torch.min(points, dim=0)[0]
        max_coords = torch.max(points, dim=0)[0]
        
        center = (min_coords + max_coords) / 2
        dimensions = max_coords - min_coords
        
        return {
            'type': 'aabb',
            'center': center,
            'dimensions': dimensions,
            'min': min_coords,
            'max': max_coords,
            'volume': torch.prod(dimensions).item(),
            'surface_area': 2 * (dimensions[0]*dimensions[1] + 
                               dimensions[1]*dimensions[2] + 
                               dimensions[0]*dimensions[2]).item()
        }
    
    def _compute_obb(self, points: torch.Tensor) -> Dict:
        """Compute oriented bounding box."""
        analysis = self._analyze_point_distribution(points)
        
        # Use PCA axes
        centroid = torch.mean(points, dim=0)
        rotation = analysis['eigenvectors']
        
        # Transform to principal axes
        centered = points - centroid
        rotated = torch.mm(centered, rotation)
        
        # Compute AABB in rotated space
        min_coords = torch.min(rotated, dim=0)[0]
        max_coords = torch.max(rotated, dim=0)[0]
        dimensions = max_coords - min_coords
        
        return {
            'type': 'obb',
            'center': centroid,
            'dimensions': dimensions,
            'rotation': rotation,
            'eigenvalues': analysis['eigenvalues'],
            'volume': torch.prod(dimensions).item(),
            'corners': self._compute_obb_corners(centroid, dimensions, rotation)
        }
    
    def _compute_planar_bbox(self, points: torch.Tensor, analysis: Dict) -> Dict:
        """Special handling for planar objects."""
        # Use the two main axes and minimal thickness
        bbox = self._compute_obb(points)
        
        # Ensure minimum thickness for planar objects
        min_dim_idx = torch.argmin(bbox['dimensions'])
        min_thickness = torch.max(bbox['dimensions']) * 0.01  # 1% of max dimension
        
        if bbox['dimensions'][min_dim_idx] < min_thickness:
            bbox['dimensions'][min_dim_idx] = min_thickness
        
        bbox['type'] = 'planar_obb'
        return bbox
    
    def _compute_obb_corners(self, center, dimensions, rotation):
        """Compute 8 corners of oriented bounding box."""
        # Create corners in local space
        half_dims = dimensions / 2
        corners_local = torch.tensor([
            [-half_dims[0], -half_dims[1], -half_dims[2]],
            [+half_dims[0], -half_dims[1], -half_dims[2]],
            [-half_dims[0], +half_dims[1], -half_dims[2]],
            [+half_dims[0], +half_dims[1], -half_dims[2]],
            [-half_dims[0], -half_dims[1], +half_dims[2]],
            [+half_dims[0], -half_dims[1], +half_dims[2]],
            [-half_dims[0], +half_dims[1], +half_dims[2]],
            [+half_dims[0], +half_dims[1], +half_dims[2]]
        ], device=center.device)
        
        # Transform to world space
        corners_world = torch.mm(corners_local, rotation.T) + center
        return corners_world
    
    def _compute_bbox_confidence(self, points: torch.Tensor, bbox: Dict) -> float:
        """Compute confidence score for bbox quality."""
        # Multiple factors contribute to confidence
        scores = []
        
        # 1. Point coverage - how well points fill the bbox
        if bbox['type'] == 'aabb':
            # Count points in each octant
            center = bbox['center']
            octant_counts = torch.zeros(8)
            for p in points:
                idx = 0
                if p[0] > center[0]: idx += 1
                if p[1] > center[1]: idx += 2
                if p[2] > center[2]: idx += 4
                octant_counts[idx] += 1
            
            coverage = (octant_counts > 0).sum().item() / 8
            scores.append(coverage)
        
        # 2. Point density uniformity
        if len(points) > 10:
            # Subsample points if too many to avoid memory issues
            max_points_for_density = 1000
            if len(points) > max_points_for_density:
                # Random subsample
                indices = torch.randperm(len(points))[:max_points_for_density]
                sampled_points = points[indices]
            else:
                sampled_points = points
            
            # Compute local density variance
            k = min(5, len(sampled_points) // 2)
            distances = torch.cdist(sampled_points, sampled_points)
            k_nearest = torch.topk(distances, k=k+1, largest=False)[0][:, 1:]  # Exclude self
            mean_dists = k_nearest.mean(dim=1)
            density_uniformity = 1.0 / (1.0 + mean_dists.std() / (mean_dists.mean() + 1e-6))
            scores.append(density_uniformity.item())
        
        # 3. Bbox compactness (not too thin)
        dims = bbox['dimensions']
        compactness = dims.min() / dims.max()
        scores.append(compactness.item())
        
        # Combined confidence
        return np.mean(scores) if scores else 0.5
    
    def _validate_bbox_size(self, bbox: Dict, label: str) -> bool:
        """Validate bbox size based on learned statistics."""
        expected_range = self.estimator.get_expected_size_range(label)
        
        if expected_range is None:
            # No statistics yet - accept and learn
            self.estimator.update_statistics(label, bbox['dimensions'])
            return True
        
        min_vol, max_vol = expected_range
        volume = bbox['volume']
        
        # Check if within expected range
        is_valid = min_vol <= volume <= max_vol
        
        if is_valid:
            # Update statistics with valid bbox
            self.estimator.update_statistics(label, bbox['dimensions'])
        
        return is_valid
    
    def compute_3d_iou(self, bbox1: Dict, bbox2: Dict) -> float:
        """Compute IoU between any two bboxes."""
        if bbox1['type'] == 'aabb' and bbox2['type'] == 'aabb':
            return self._compute_aabb_iou(bbox1, bbox2)
        else:
            # For mixed or OBB types, use sampling
            return self._compute_iou_sampling(bbox1, bbox2)
    
    def _compute_aabb_iou(self, bbox1: Dict, bbox2: Dict) -> float:
        """Fast AABB IoU computation."""
        inter_min = torch.max(bbox1['min'], bbox2['min'])
        inter_max = torch.min(bbox1['max'], bbox2['max'])
        
        if torch.any(inter_min >= inter_max):
            return 0.0
        
        inter_vol = torch.prod(inter_max - inter_min).item()
        union_vol = bbox1['volume'] + bbox2['volume'] - inter_vol
        
        return inter_vol / union_vol if union_vol > 0 else 0.0
    
    def _compute_iou_sampling(self, bbox1: Dict, bbox2: Dict, n_samples: int = 1000) -> float:
        """Approximate IoU using point sampling for complex bbox types."""
        # Sample points in each bbox
        points1 = self._sample_bbox_points(bbox1, n_samples)
        points2 = self._sample_bbox_points(bbox2, n_samples)
        
        # Check containment
        in1 = self._points_in_bbox(points2, bbox1).sum().item()
        in2 = self._points_in_bbox(points1, bbox2).sum().item()
        
        # Approximate volumes and intersection
        intersection = (in1 + in2) / (2 * n_samples)
        union = 2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _sample_bbox_points(self, bbox: Dict, n_samples: int) -> torch.Tensor:
        """Sample random points within bbox."""
        if bbox['type'] == 'aabb':
            # Uniform sampling in AABB
            mins = bbox['min'].unsqueeze(0)
            maxs = bbox['max'].unsqueeze(0)
            rand = torch.rand(n_samples, 3, device=mins.device)
            return mins + rand * (maxs - mins)
        else:
            # For OBB, sample in local space then transform
            dims = bbox['dimensions']
            rand = (torch.rand(n_samples, 3, device=dims.device) - 0.5) * dims
            return torch.mm(rand, bbox['rotation'].T) + bbox['center']
    
    def _points_in_bbox(self, points: torch.Tensor, bbox: Dict) -> torch.Tensor:
        """Check which points are inside bbox."""
        if bbox['type'] == 'aabb':
            above_min = torch.all(points >= bbox['min'], dim=1)
            below_max = torch.all(points <= bbox['max'], dim=1)
            return above_min & below_max
        else:
            # Transform to local space
            local = torch.mm(points - bbox['center'], bbox['rotation'])
            half_dims = bbox['dimensions'] / 2
            return torch.all(torch.abs(local) <= half_dims, dim=1)


# Convenience function
def create_generic_3d_bbox(keyframe, mask, label="unknown", 
                          config: BBox3DConfig = None,
                          estimator: AdaptiveBBoxEstimator = None):
    """Simple interface to create 3D bbox."""
    
    bbox_creator = Generic3DBoundingBox(config, estimator)
    
    # Extract points
    points, stats = bbox_creator.extract_valid_3d_points(keyframe, mask, return_stats=True)
    
    if points is None:
        logger.warning(f"Failed to create bbox for {label}: {stats.get('error', 'Unknown error')}")
        return None
    
    # Compute bbox
    bbox = bbox_creator.compute_adaptive_bbox(points, label)
    bbox['extraction_stats'] = stats
    bbox['label'] = label
    
    return bbox