"""
Clustering Parameter Optimization Module for MAST3R SLAM

This module provides comprehensive parameter optimization for the object clustering system,
implementing grid search, cross-validation, sensitivity analysis, and adaptive optimization
to find optimal clustering parameters for different scene types and conditions.

Author: Generated for MASt3R SLAM semantic extension
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
import itertools
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.model_selection import KFold
import json
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from .object_clustering import (
    ObjectInstance, ObjectCluster, hybrid_cluster_objects,
    get_clustering_config, set_clustering_config
)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class OptimizationResult:
    """Container for optimization results."""
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Dict[str, Any]]
    optimization_time: float
    convergence_history: List[float] = field(default_factory=list)
    stability_scores: Dict[str, float] = field(default_factory=dict)


@dataclass 
class ValidationMetrics:
    """Container for clustering validation metrics."""
    silhouette_score: float
    compactness_score: float
    separation_score: float
    temporal_consistency: float
    size_consistency: float
    num_clusters: int
    num_noise_points: int
    avg_cluster_size: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'silhouette_score': self.silhouette_score,
            'compactness_score': self.compactness_score,
            'separation_score': self.separation_score,
            'temporal_consistency': self.temporal_consistency,
            'size_consistency': self.size_consistency,
            'num_clusters': self.num_clusters,
            'num_noise_points': self.num_noise_points,
            'avg_cluster_size': self.avg_cluster_size
        }


def define_parameter_space(scene_type: str = 'indoor') -> Dict[str, List]:
    """
    Define parameter ranges for optimization based on scene type.
    
    Args:
        scene_type: Type of scene ('indoor', 'outdoor', 'mixed')
        
    Returns:
        Dictionary mapping parameter names to lists of values to test
    """
    if scene_type == 'indoor':
        param_space = {
            'spatial_threshold': [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0],
            'temporal_threshold': [5, 10, 15, 20, 25, 30],
            'movement_threshold': [0.1, 0.2, 0.3, 0.5, 0.8],
            'min_samples': [1, 2, 3]
        }
    elif scene_type == 'outdoor':
        param_space = {
            'spatial_threshold': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
            'temporal_threshold': [5, 10, 15, 20, 25, 30],
            'movement_threshold': [0.5, 1.0, 1.5, 2.0, 3.0],
            'min_samples': [1, 2, 3, 4]
        }
    else:  # mixed
        param_space = {
            'spatial_threshold': [0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5],
            'temporal_threshold': [5, 10, 15, 20, 25, 30],
            'movement_threshold': [0.2, 0.3, 0.5, 0.8, 1.0, 1.5],
            'min_samples': [1, 2, 3]
        }
    
    logger.info(f"Defined parameter space for {scene_type} scenes:")
    for param, values in param_space.items():
        logger.info(f"  {param}: {values}")
    
    return param_space


def compute_silhouette_score(clusters: List[ObjectCluster]) -> float:
    """
    Compute silhouette score for clustering quality.
    
    Args:
        clusters: List of object clusters
        
    Returns:
        Silhouette score (higher is better)
    """
    if len(clusters) < 2:
        return 0.0
    
    try:
        # Extract all centroids and cluster labels
        centroids = []
        labels = []
        
        for cluster_idx, cluster in enumerate(clusters):
            for instance in cluster.instances:
                centroids.append(instance.centroid_3d)
                labels.append(cluster_idx)
        
        if len(set(labels)) < 2 or len(centroids) < 2:
            return 0.0
            
        centroids = np.array(centroids)
        score = silhouette_score(centroids, labels)
        return max(0.0, score)  # Normalize to [0, 1]
        
    except Exception as e:
        logger.debug(f"Error computing silhouette score: {e}")
        return 0.0


def compute_compactness_score(clusters: List[ObjectCluster]) -> float:
    """
    Compute cluster compactness (lower intra-cluster variance is better).
    
    Args:
        clusters: List of object clusters
        
    Returns:
        Compactness score (higher is better, inverted from variance)
    """
    if not clusters:
        return 0.0
        
    total_variance = 0.0
    total_instances = 0
    
    for cluster in clusters:
        if len(cluster.instances) <= 1:
            continue
            
        centroids = np.array([inst.centroid_3d for inst in cluster.instances])
        cluster_center = np.mean(centroids, axis=0)
        
        # Compute within-cluster sum of squares
        distances = [np.linalg.norm(centroid - cluster_center) 
                    for centroid in centroids]
        cluster_variance = np.mean(distances) if distances else 0.0
        
        total_variance += cluster_variance * len(cluster.instances)
        total_instances += len(cluster.instances)
    
    if total_instances == 0:
        return 0.0
        
    avg_variance = total_variance / total_instances
    # Convert to score (lower variance = higher score)
    return 1.0 / (1.0 + avg_variance)


def compute_separation_score(clusters: List[ObjectCluster]) -> float:
    """
    Compute cluster separation (higher inter-cluster distance is better).
    
    Args:
        clusters: List of object clusters
        
    Returns:
        Separation score (higher is better)
    """
    if len(clusters) < 2:
        return 1.0
        
    cluster_centers = [cluster.avg_centroid for cluster in clusters]
    
    min_distance = float('inf')
    for i in range(len(cluster_centers)):
        for j in range(i + 1, len(cluster_centers)):
            distance = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
            min_distance = min(min_distance, distance)
    
    if min_distance == float('inf'):
        return 0.0
        
    # Normalize separation score
    return min(1.0, min_distance / 2.0)  # Assume 2m is good separation


def compute_temporal_consistency(clusters: List[ObjectCluster]) -> float:
    """
    Compute temporal consistency score.
    
    Args:
        clusters: List of object clusters
        
    Returns:
        Temporal consistency score (higher is better)
    """
    if not clusters:
        return 0.0
        
    consistency_scores = []
    
    for cluster in clusters:
        if len(cluster.instances) <= 1:
            continue
            
        keyframes = sorted([inst.keyframe_idx for inst in cluster.instances])
        
        # Check for temporal continuity
        gaps = []
        for i in range(1, len(keyframes)):
            gap = keyframes[i] - keyframes[i-1]
            gaps.append(gap)
        
        if not gaps:
            consistency_scores.append(1.0)
            continue
            
        # Penalize large temporal gaps
        max_gap = max(gaps)
        avg_gap = np.mean(gaps)
        
        # Score based on continuity (lower gaps = higher score)
        gap_score = 1.0 / (1.0 + avg_gap / 5.0)  # 5 keyframes is reasonable gap
        max_gap_penalty = 1.0 / (1.0 + max_gap / 20.0)  # 20 keyframes is large gap
        
        consistency_scores.append(gap_score * max_gap_penalty)
    
    return np.mean(consistency_scores) if consistency_scores else 0.0


def compute_size_consistency(clusters: List[ObjectCluster]) -> float:
    """
    Compute cluster size consistency score.
    
    Args:
        clusters: List of object clusters
        
    Returns:
        Size consistency score (higher is better)
    """
    if not clusters:
        return 0.0
        
    cluster_sizes = [len(cluster.instances) for cluster in clusters]
    
    if len(cluster_sizes) <= 1:
        return 1.0
        
    # Penalize extreme size variations
    size_std = np.std(cluster_sizes)
    size_mean = np.mean(cluster_sizes)
    
    if size_mean == 0:
        return 0.0
        
    coefficient_of_variation = size_std / size_mean
    
    # Lower variation = higher consistency score
    return 1.0 / (1.0 + coefficient_of_variation)


def evaluate_clustering_quality(clusters: List[ObjectCluster]) -> ValidationMetrics:
    """
    Comprehensive evaluation of clustering quality.
    
    Args:
        clusters: List of object clusters to evaluate
        
    Returns:
        ValidationMetrics object with all computed scores
    """
    silhouette = compute_silhouette_score(clusters)
    compactness = compute_compactness_score(clusters)
    separation = compute_separation_score(clusters)
    temporal_consistency = compute_temporal_consistency(clusters)
    size_consistency = compute_size_consistency(clusters)
    
    # Compute cluster statistics
    num_clusters = len(clusters)
    num_noise_points = sum(1 for cluster in clusters if len(cluster.instances) == 1)
    total_instances = sum(len(cluster.instances) for cluster in clusters)
    avg_cluster_size = total_instances / num_clusters if num_clusters > 0 else 0.0
    
    return ValidationMetrics(
        silhouette_score=silhouette,
        compactness_score=compactness,
        separation_score=separation,
        temporal_consistency=temporal_consistency,
        size_consistency=size_consistency,
        num_clusters=num_clusters,
        num_noise_points=num_noise_points,
        avg_cluster_size=avg_cluster_size
    )


def compute_clustering_objective(clusters: List[ObjectCluster], 
                                weights: Optional[Dict[str, float]] = None) -> float:
    """
    Weighted combination of multiple clustering metrics into single objective score.
    
    Args:
        clusters: List of object clusters to evaluate
        weights: Optional weights for different metrics
        
    Returns:
        Combined objective score (higher is better)
    """
    if weights is None:
        weights = {
            'silhouette_score': 1.0,
            'compactness_score': 1.0,
            'separation_score': 1.0,
            'temporal_consistency': 1.5,  # Emphasize temporal consistency
            'size_consistency': 0.5
        }
    
    metrics = evaluate_clustering_quality(clusters)
    
    # Compute weighted sum
    objective = (
        weights.get('silhouette_score', 0.0) * metrics.silhouette_score +
        weights.get('compactness_score', 0.0) * metrics.compactness_score +
        weights.get('separation_score', 0.0) * metrics.separation_score +
        weights.get('temporal_consistency', 0.0) * metrics.temporal_consistency +
        weights.get('size_consistency', 0.0) * metrics.size_consistency
    )
    
    # Normalize by total weight
    total_weight = sum(weights.values())
    if total_weight > 0:
        objective /= total_weight
    
    # Apply penalties for extreme cases
    if metrics.num_clusters == 0:
        objective = 0.0
    elif metrics.num_clusters == len([inst for cluster in clusters for inst in cluster.instances]):
        # Every instance is its own cluster (over-segmentation)
        objective *= 0.1
    
    return objective


def optimize_clustering_parameters(instances: List[ObjectInstance],
                                 param_space: Dict[str, List],
                                 weights: Optional[Dict[str, float]] = None,
                                 max_combinations: int = 1000) -> OptimizationResult:
    """
    Test all parameter combinations and find optimal settings using grid search.
    
    Args:
        instances: List of object instances to cluster
        param_space: Dictionary of parameters and their possible values
        weights: Weights for objective function components
        max_combinations: Maximum number of combinations to test (for efficiency)
        
    Returns:
        OptimizationResult with best parameters and performance analysis
    """
    logger.info("Starting grid search parameter optimization...")
    start_time = time.time()
    
    # Generate all parameter combinations
    param_names = list(param_space.keys())
    param_values = list(param_space.values())
    all_combinations = list(itertools.product(*param_values))
    
    # Limit combinations if too many
    if len(all_combinations) > max_combinations:
        logger.warning(f"Too many combinations ({len(all_combinations)}), "
                      f"sampling {max_combinations} randomly")
        indices = np.random.choice(len(all_combinations), max_combinations, replace=False)
        all_combinations = [all_combinations[i] for i in indices]
    
    logger.info(f"Testing {len(all_combinations)} parameter combinations")
    
    best_score = -1.0
    best_params = None
    all_results = []
    
    # Test each combination
    for i, combination in enumerate(tqdm(all_combinations, desc="Grid search")):
        params = dict(zip(param_names, combination))
        
        try:
            # Apply clustering with current parameters
            clusters = hybrid_cluster_objects(
                instances,
                spatial_threshold=params['spatial_threshold'],
                temporal_threshold=params['temporal_threshold'], 
                movement_threshold=params['movement_threshold'],
                min_samples=params['min_samples']
            )
            
            # Evaluate clustering quality
            objective_score = compute_clustering_objective(clusters, weights)
            metrics = evaluate_clustering_quality(clusters)
            
            # Store result
            result = {
                'params': params.copy(),
                'objective_score': objective_score,
                'metrics': metrics.to_dict()
            }
            all_results.append(result)
            
            # Update best if better
            if objective_score > best_score:
                best_score = objective_score
                best_params = params.copy()
                
        except Exception as e:
            logger.debug(f"Error with params {params}: {e}")
            continue
    
    optimization_time = time.time() - start_time
    
    logger.info(f"Grid search completed in {optimization_time:.2f}s")
    logger.info(f"Best objective score: {best_score:.4f}")
    logger.info(f"Best parameters: {best_params}")
    
    return OptimizationResult(
        best_params=best_params,
        best_score=best_score,
        all_results=all_results,
        optimization_time=optimization_time
    )


def cross_validate_clustering(instances: List[ObjectInstance],
                             params: Dict[str, Any],
                             k_folds: int = 5,
                             split_method: str = 'keyframes') -> Dict[str, float]:
    """
    K-fold cross-validation for clustering parameters.
    
    Args:
        instances: List of object instances
        params: Parameters to validate
        k_folds: Number of cross-validation folds
        split_method: How to split data ('keyframes', 'spatial', 'random')
        
    Returns:
        Dictionary with cross-validation scores and confidence intervals
    """
    logger.info(f"Running {k_folds}-fold cross-validation with {split_method} split")
    
    if len(instances) < k_folds:
        logger.warning("Not enough instances for proper cross-validation")
        return {'mean_score': 0.0, 'std_score': 0.0, 'confidence_interval': (0.0, 0.0)}
    
    # Create splits based on method
    if split_method == 'keyframes':
        # Split by keyframes to test temporal generalization
        keyframes = sorted(list(set(inst.keyframe_idx for inst in instances)))
        kf_splits = np.array_split(keyframes, k_folds)
        
        folds = []
        for kf_split in kf_splits:
            fold_instances = [inst for inst in instances if inst.keyframe_idx in kf_split]
            folds.append(fold_instances)
            
    elif split_method == 'spatial':
        # Split by spatial regions
        centroids = np.array([inst.centroid_3d for inst in instances])
        # Use k-means to create spatial regions
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k_folds, random_state=42, n_init=10)
        spatial_labels = kmeans.fit_predict(centroids)
        
        folds = []
        for i in range(k_folds):
            fold_instances = [inst for j, inst in enumerate(instances) 
                            if spatial_labels[j] == i]
            folds.append(fold_instances)
            
    else:  # random
        # Random split
        shuffled_instances = instances.copy()
        np.random.shuffle(shuffled_instances)
        folds = np.array_split(shuffled_instances, k_folds)
    
    # Perform cross-validation
    scores = []
    for i in range(k_folds):
        # Use fold i as test set, rest as training set
        test_instances = folds[i]
        train_instances = []
        for j in range(k_folds):
            if j != i:
                train_instances.extend(folds[j])
        
        if len(train_instances) == 0 or len(test_instances) == 0:
            continue
        
        try:
            # Cluster training data
            train_clusters = hybrid_cluster_objects(
                train_instances,
                spatial_threshold=params['spatial_threshold'],
                temporal_threshold=params['temporal_threshold'],
                movement_threshold=params['movement_threshold'],
                min_samples=params['min_samples']
            )
            
            # Cluster test data with same parameters
            test_clusters = hybrid_cluster_objects(
                test_instances,
                spatial_threshold=params['spatial_threshold'],
                temporal_threshold=params['temporal_threshold'],
                movement_threshold=params['movement_threshold'],
                min_samples=params['min_samples']
            )
            
            # Evaluate test clustering
            test_score = compute_clustering_objective(test_clusters)
            scores.append(test_score)
            
        except Exception as e:
            logger.debug(f"Error in fold {i}: {e}")
            continue
    
    if not scores:
        return {'mean_score': 0.0, 'std_score': 0.0, 'confidence_interval': (0.0, 0.0)}
    
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    # 95% confidence interval
    confidence_interval = (
        mean_score - 1.96 * std_score / np.sqrt(len(scores)),
        mean_score + 1.96 * std_score / np.sqrt(len(scores))
    )
    
    logger.info(f"Cross-validation results: {mean_score:.4f} ± {std_score:.4f}")
    logger.info(f"95% confidence interval: {confidence_interval}")
    
    return {
        'mean_score': mean_score,
        'std_score': std_score,
        'confidence_interval': confidence_interval,
        'fold_scores': scores
    }


def analyze_parameter_sensitivity(instances: List[ObjectInstance],
                                base_params: Dict[str, Any],
                                param_ranges: Optional[Dict[str, List]] = None) -> Dict[str, Dict]:
    """
    Analyze how sensitive clustering is to parameter changes.
    
    Args:
        instances: List of object instances
        base_params: Baseline parameter configuration
        param_ranges: Parameter ranges to test around baseline
        
    Returns:
        Dictionary with sensitivity analysis results for each parameter
    """
    logger.info("Running parameter sensitivity analysis...")
    
    if param_ranges is None:
        # Default ranges around baseline values
        param_ranges = {
            'spatial_threshold': np.linspace(
                base_params['spatial_threshold'] * 0.5,
                base_params['spatial_threshold'] * 2.0, 10
            ),
            'temporal_threshold': np.arange(
                max(5, base_params['temporal_threshold'] - 10),
                base_params['temporal_threshold'] + 15, 5
            ),
            'movement_threshold': np.linspace(
                base_params['movement_threshold'] * 0.5,
                base_params['movement_threshold'] * 2.0, 10
            ),
            'min_samples': [1, 2, 3, 4] if base_params['min_samples'] <= 2 else [1, 2, 3, 4, 5]
        }
    
    sensitivity_results = {}
    
    for param_name, param_values in param_ranges.items():
        logger.info(f"Analyzing sensitivity to {param_name}")
        
        param_scores = []
        param_configs = []
        
        for param_value in tqdm(param_values, desc=f"Testing {param_name}"):
            # Create parameter configuration with varied parameter
            test_params = base_params.copy()
            test_params[param_name] = param_value
            
            try:
                # Cluster with test parameters
                clusters = hybrid_cluster_objects(
                    instances,
                    spatial_threshold=test_params['spatial_threshold'],
                    temporal_threshold=test_params['temporal_threshold'],
                    movement_threshold=test_params['movement_threshold'],
                    min_samples=test_params['min_samples']
                )
                
                # Evaluate clustering
                score = compute_clustering_objective(clusters)
                param_scores.append(score)
                param_configs.append(param_value)
                
            except Exception as e:
                logger.debug(f"Error with {param_name}={param_value}: {e}")
                param_scores.append(0.0)
                param_configs.append(param_value)
        
        # Analyze sensitivity
        if param_scores:
            score_std = np.std(param_scores)
            score_range = max(param_scores) - min(param_scores)
            optimal_value = param_configs[np.argmax(param_scores)]
            
            sensitivity_results[param_name] = {
                'values': list(param_configs),
                'scores': param_scores,
                'std_deviation': score_std,
                'score_range': score_range,
                'optimal_value': optimal_value,
                'baseline_value': base_params[param_name]
            }
            
            logger.info(f"{param_name} sensitivity: std={score_std:.4f}, "
                       f"range={score_range:.4f}, optimal={optimal_value}")
    
    # Rank parameters by sensitivity
    sensitivity_ranking = sorted(
        sensitivity_results.items(),
        key=lambda x: x[1]['score_range'],
        reverse=True
    )
    
    logger.info("Parameter sensitivity ranking (most sensitive first):")
    for i, (param_name, results) in enumerate(sensitivity_ranking):
        logger.info(f"  {i+1}. {param_name}: range={results['score_range']:.4f}")
    
    return sensitivity_results


def test_clustering_stability(instances: List[ObjectInstance],
                             params: Dict[str, Any],
                             noise_levels: List[float] = [0.01, 0.05, 0.1],
                             n_trials: int = 10) -> Dict[str, float]:
    """
    Test robustness to noise and measurement errors.
    
    Args:
        instances: List of object instances
        params: Clustering parameters to test
        noise_levels: List of noise standard deviations (as fraction of scene scale)
        n_trials: Number of trials per noise level
        
    Returns:
        Dictionary with stability scores for different noise levels
    """
    logger.info("Testing clustering stability under noise...")
    
    # Compute scene scale for noise normalization
    centroids = np.array([inst.centroid_3d for inst in instances])
    scene_scale = np.std(centroids, axis=0).mean()
    
    stability_results = {}
    
    # Test each noise level
    for noise_level in noise_levels:
        logger.info(f"Testing noise level {noise_level:.3f}")
        
        trial_scores = []
        
        for trial in range(n_trials):
            # Add noise to instance centroids
            noisy_instances = []
            for inst in instances:
                noise = np.random.normal(0, noise_level * scene_scale, 3)
                noisy_centroid = inst.centroid_3d + noise
                
                # Create noisy instance (deep copy with modified centroid)
                noisy_instance = ObjectInstance(
                    global_id=inst.global_id,
                    local_id=inst.local_id,
                    keyframe_idx=inst.keyframe_idx,
                    label=inst.label,
                    confidence=inst.confidence,
                    centroid_3d=noisy_centroid,
                    num_points=inst.num_points,
                    point_indices=inst.point_indices.copy()
                )
                noisy_instances.append(noisy_instance)
            
            try:
                # Cluster noisy data
                noisy_clusters = hybrid_cluster_objects(
                    noisy_instances,
                    spatial_threshold=params['spatial_threshold'],
                    temporal_threshold=params['temporal_threshold'],
                    movement_threshold=params['movement_threshold'],
                    min_samples=params['min_samples']
                )
                
                # Evaluate clustering
                score = compute_clustering_objective(noisy_clusters)
                trial_scores.append(score)
                
            except Exception as e:
                logger.debug(f"Error in stability trial: {e}")
                trial_scores.append(0.0)
        
        if trial_scores:
            stability_score = np.mean(trial_scores)
            stability_std = np.std(trial_scores)
            
            stability_results[f'noise_{noise_level:.3f}'] = {
                'mean_score': stability_score,
                'std_score': stability_std,
                'trials': trial_scores
            }
            
            logger.info(f"Noise {noise_level:.3f}: stability={stability_score:.4f} ± {stability_std:.4f}")
    
    # Compute overall stability score
    if stability_results:
        overall_stability = np.mean([
            result['mean_score'] for result in stability_results.values()
        ])
        stability_results['overall_stability'] = overall_stability
        
        logger.info(f"Overall stability score: {overall_stability:.4f}")
    
    return stability_results


def optimize_for_scene_type(instances: List[ObjectInstance],
                           scene_type: str = 'indoor') -> OptimizationResult:
    """
    Optimize parameters for specific scene types.
    
    Args:
        instances: List of object instances
        scene_type: Type of scene ('indoor', 'outdoor', 'mixed')
        
    Returns:
        OptimizationResult optimized for the scene type
    """
    logger.info(f"Optimizing parameters for {scene_type} scenes")
    
    # Define scene-specific parameter space
    param_space = define_parameter_space(scene_type)
    
    # Define scene-specific weights
    if scene_type == 'indoor':
        weights = {
            'silhouette_score': 1.0,
            'compactness_score': 1.2,  # Indoor objects are more compact
            'separation_score': 1.0,
            'temporal_consistency': 1.5,
            'size_consistency': 0.8
        }
    elif scene_type == 'outdoor':
        weights = {
            'silhouette_score': 1.0,
            'compactness_score': 0.8,  # Outdoor objects may be more spread
            'separation_score': 1.2,   # Need better separation outdoors
            'temporal_consistency': 1.3,
            'size_consistency': 1.0
        }
    else:  # mixed
        weights = {
            'silhouette_score': 1.0,
            'compactness_score': 1.0,
            'separation_score': 1.0,
            'temporal_consistency': 1.4,
            'size_consistency': 0.9
        }
    
    # Run optimization with scene-specific settings
    result = optimize_clustering_parameters(
        instances, param_space, weights, max_combinations=500
    )
    
    logger.info(f"Optimization completed for {scene_type} scenes")
    
    return result


def adaptive_parameter_selection(instances: List[ObjectInstance],
                                current_params: Dict[str, Any],
                                performance_history: List[float],
                                adaptation_rate: float = 0.1) -> Dict[str, Any]:
    """
    Automatically adjust parameters based on performance history.
    
    Args:
        instances: Current object instances
        current_params: Current parameter configuration
        performance_history: List of recent performance scores
        adaptation_rate: Rate of parameter adaptation (0.0 to 1.0)
        
    Returns:
        Updated parameter configuration
    """
    logger.info("Running adaptive parameter selection...")
    
    if len(performance_history) < 5:
        logger.info("Not enough history for adaptation, keeping current parameters")
        return current_params
    
    # Analyze performance trend
    recent_scores = performance_history[-5:]
    performance_trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
    
    logger.info(f"Performance trend: {performance_trend:.4f}")
    
    # If performance is declining, try to adapt parameters
    if performance_trend < -0.01:  # Significant decline
        logger.info("Declining performance detected, adapting parameters...")
        
        # Test small parameter variations
        param_deltas = {
            'spatial_threshold': [-0.1, 0.0, 0.1],
            'temporal_threshold': [-2, 0, 2],
            'movement_threshold': [-0.05, 0.0, 0.05],
            'min_samples': [-1, 0, 1]
        }
        
        best_score = -1.0
        best_params = current_params.copy()
        
        # Test each parameter variation
        for param_name, deltas in param_deltas.items():
            for delta in deltas:
                test_params = current_params.copy()
                
                if param_name == 'min_samples':
                    test_params[param_name] = max(1, current_params[param_name] + int(delta))
                elif param_name == 'temporal_threshold':
                    test_params[param_name] = max(5, current_params[param_name] + int(delta))
                else:
                    test_params[param_name] = max(0.1, current_params[param_name] + delta)
                
                try:
                    # Test clustering with modified parameters
                    clusters = hybrid_cluster_objects(
                        instances,
                        spatial_threshold=test_params['spatial_threshold'],
                        temporal_threshold=test_params['temporal_threshold'],
                        movement_threshold=test_params['movement_threshold'],
                        min_samples=test_params['min_samples']
                    )
                    
                    score = compute_clustering_objective(clusters)
                    
                    if score > best_score:
                        best_score = score
                        best_params = test_params.copy()
                        
                except Exception as e:
                    logger.debug(f"Error testing params {test_params}: {e}")
                    continue
        
        # Apply adaptation with rate limiting
        adapted_params = {}
        for param_name in current_params:
            current_val = current_params[param_name]
            best_val = best_params[param_name]
            
            # Gradual adaptation
            if param_name in ['min_samples', 'temporal_threshold']:
                # Integer parameters
                adapted_val = int(current_val + adaptation_rate * (best_val - current_val))
            else:
                # Float parameters
                adapted_val = current_val + adaptation_rate * (best_val - current_val)
            
            adapted_params[param_name] = adapted_val
        
        logger.info(f"Adapted parameters: {adapted_params}")
        return adapted_params
    
    else:
        logger.info("Performance stable, keeping current parameters")
        return current_params


def save_optimization_results(result: OptimizationResult, 
                             filename: str = "clustering_optimization_results.json"):
    """Save optimization results to JSON file."""
    try:
        # Convert numpy arrays to lists for JSON serialization
        serializable_result = {
            'best_params': result.best_params,
            'best_score': result.best_score,
            'optimization_time': result.optimization_time,
            'convergence_history': result.convergence_history,
            'stability_scores': result.stability_scores,
            'num_combinations_tested': len(result.all_results),
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(filename, 'w') as f:
            json.dump(serializable_result, f, indent=2)
        
        logger.info(f"Optimization results saved to {filename}")
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")


def load_optimization_results(filename: str = "clustering_optimization_results.json") -> Dict:
    """Load optimization results from JSON file."""
    try:
        with open(filename, 'r') as f:
            results = json.load(f)
        
        logger.info(f"Loaded optimization results from {filename}")
        return results
        
    except Exception as e:
        logger.error(f"Error loading results: {e}")
        return {}


# Main optimization pipeline
def run_complete_optimization(instances: List[ObjectInstance],
                             scene_type: str = 'indoor',
                             enable_cross_validation: bool = True,
                             enable_sensitivity_analysis: bool = True,
                             enable_stability_testing: bool = True) -> Dict[str, Any]:
    """
    Run complete optimization pipeline with all analysis methods.
    
    Args:
        instances: List of object instances to optimize for
        scene_type: Scene type for optimization
        enable_cross_validation: Whether to run cross-validation
        enable_sensitivity_analysis: Whether to run sensitivity analysis  
        enable_stability_testing: Whether to test stability
        
    Returns:
        Dictionary with all optimization results
    """
    logger.info("Starting complete clustering parameter optimization pipeline...")
    start_time = time.time()
    
    results = {}
    
    # 1. Primary optimization
    logger.info("Phase 1: Grid search optimization")
    optimization_result = optimize_for_scene_type(instances, scene_type)
    results['optimization'] = optimization_result
    
    best_params = optimization_result.best_params
    logger.info(f"Best parameters found: {best_params}")
    
    # 2. Cross-validation
    if enable_cross_validation:
        logger.info("Phase 2: Cross-validation")
        cv_results = cross_validate_clustering(instances, best_params)
        results['cross_validation'] = cv_results
    
    # 3. Sensitivity analysis
    if enable_sensitivity_analysis:
        logger.info("Phase 3: Sensitivity analysis")
        sensitivity_results = analyze_parameter_sensitivity(instances, best_params)
        results['sensitivity'] = sensitivity_results
    
    # 4. Stability testing
    if enable_stability_testing:
        logger.info("Phase 4: Stability testing")
        stability_results = test_clustering_stability(instances, best_params)
        results['stability'] = stability_results
    
    total_time = time.time() - start_time
    results['total_optimization_time'] = total_time
    
    logger.info(f"Complete optimization pipeline finished in {total_time:.2f}s")
    
    # Save results
    save_optimization_results(optimization_result, 
                            f"optimization_{scene_type}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    
    return results


if __name__ == "__main__":
    # Example usage and testing
    import sys
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample instances for testing
    np.random.seed(42)
    sample_instances = []
    
    for kf in range(5):
        for obj_id in range(3):
            # Create some clustered objects with noise
            base_pos = np.array([obj_id * 2.0, kf * 0.1, 0.5])
            noise = np.random.normal(0, 0.1, 3)
            
            instance = ObjectInstance(
                global_id=kf * 10000 + obj_id,
                local_id=obj_id,
                keyframe_idx=kf,
                label=f"object_{obj_id}",
                confidence=0.8 + np.random.random() * 0.2,
                centroid_3d=base_pos + noise,
                num_points=50 + np.random.randint(50),
                point_indices=np.arange(100)
            )
            sample_instances.append(instance)
    
    logger.info(f"Created {len(sample_instances)} sample instances for testing")
    
    # Run optimization
    results = run_complete_optimization(
        sample_instances,
        scene_type='indoor',
        enable_cross_validation=True,
        enable_sensitivity_analysis=True,
        enable_stability_testing=True
    )
    
    print("\nOptimization Results Summary:")
    print("=" * 50)
    print(f"Best parameters: {results['optimization'].best_params}")
    print(f"Best score: {results['optimization'].best_score:.4f}")
    
    if 'cross_validation' in results:
        cv = results['cross_validation']
        print(f"Cross-validation: {cv['mean_score']:.4f} ± {cv['std_score']:.4f}")
    
    if 'stability' in results and 'overall_stability' in results['stability']:
        print(f"Stability score: {results['stability']['overall_stability']:.4f}")
    
    print(f"Total optimization time: {results['total_optimization_time']:.2f}s")