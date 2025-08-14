"""
Comprehensive Clustering Validation Module for Semantic SLAM

This module implements quantitative metrics to assess clustering quality without requiring 
ground truth data. It provides various validation metrics to identify over-segmentation,
under-segmentation, and clustering quality issues.

The validation module analyzes clustering results across multiple dimensions:
- Spatial clustering quality (silhouette analysis, compactness, separation)
- Temporal consistency and continuity
- Size consistency and coherence
- Statistical clustering indices

Author: Generated for MASt3R SLAM semantic extension
"""

import numpy as np
import logging
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import pdist, squareform
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from .object_clustering import ObjectCluster, ObjectInstance

# Configure logging for the validation module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Default validation configuration
VALIDATION_CONFIG = {
    "silhouette_sample_size": 1000,    # Max samples for silhouette computation
    "temporal_gap_threshold": 10,       # Threshold for temporal gaps
    "size_variation_threshold": 2.0,    # Max acceptable size variation ratio
    "min_cluster_size": 3,             # Minimum cluster size for analysis
    "outlier_std_threshold": 2.0,      # Standard deviations for outlier detection
}


@dataclass
class ValidationMetrics:
    """
    Container for all clustering validation metrics.
    
    Attributes:
        silhouette_score: Overall silhouette coefficient (-1 to +1)
        silhouette_samples: Per-sample silhouette scores
        avg_intra_cluster_distance: Average distance within clusters
        avg_inter_cluster_distance: Average distance between clusters
        temporal_consistency_score: Temporal validation score (0 to 1)
        davies_bouldin_index: Davies-Bouldin clustering quality index (lower is better)
        calinski_harabasz_index: Calinski-Harabasz clustering quality index (higher is better)
        cluster_size_statistics: Statistics about cluster sizes
        semantic_coherence_score: Percentage of semantically coherent clusters
        spatial_outliers: List of cluster IDs with spatial outliers
        temporal_outliers: List of cluster IDs with temporal issues
        overall_quality_score: Combined quality assessment (0 to 1)
    """
    silhouette_score: float
    silhouette_samples: Dict[int, float]
    avg_intra_cluster_distance: float
    avg_inter_cluster_distance: float
    temporal_consistency_score: float
    davies_bouldin_index: float
    calinski_harabasz_index: float
    cluster_size_statistics: Dict[str, float]
    semantic_coherence_score: float
    spatial_outliers: List[int]
    temporal_outliers: List[int]
    overall_quality_score: float


def compute_silhouette_score(clusters: List[ObjectCluster]) -> Tuple[float, Dict[int, float]]:
    """
    Compute silhouette coefficient for spatial clustering quality.
    
    The silhouette coefficient measures how well objects fit their clusters versus other clusters.
    Score ranges from -1 (poor clustering) to +1 (excellent clustering).
    
    Args:
        clusters: List of ObjectCluster instances
        
    Returns:
        overall_score: Mean silhouette coefficient across all instances
        sample_scores: Dictionary mapping cluster_id to average silhouette score
    """
    if len(clusters) < 2:
        logger.warning("Need at least 2 clusters for silhouette analysis")
        return 0.0, {}
    
    # Collect all centroids and cluster labels
    centroids = []
    cluster_labels = []
    cluster_id_map = {}
    
    for i, cluster in enumerate(clusters):
        for instance in cluster.instances:
            centroids.append(instance.centroid_3d)
            cluster_labels.append(i)  # Use sequential cluster labels for sklearn
            cluster_id_map[i] = cluster.cluster_id
    
    if len(centroids) < 2:
        logger.warning("Need at least 2 instances for silhouette analysis")
        return 0.0, {}
    
    centroids = np.array(centroids)
    cluster_labels = np.array(cluster_labels)
    
    try:
        # Compute overall silhouette score
        overall_score = silhouette_score(centroids, cluster_labels, metric='euclidean')
        
        # Compute per-sample silhouette scores
        sample_scores = silhouette_samples(centroids, cluster_labels, metric='euclidean')
        
        # Average per cluster
        cluster_silhouette_scores = {}
        for i, cluster in enumerate(clusters):
            cluster_mask = cluster_labels == i
            if np.any(cluster_mask):
                avg_cluster_score = np.mean(sample_scores[cluster_mask])
                cluster_silhouette_scores[cluster.cluster_id] = avg_cluster_score
        
        logger.info(f"Silhouette analysis: overall_score={overall_score:.3f}, "
                   f"clusters analyzed={len(cluster_silhouette_scores)}")
        
        return overall_score, cluster_silhouette_scores
        
    except Exception as e:
        logger.error(f"Error computing silhouette score: {e}")
        return 0.0, {}


def compute_intra_cluster_distances(clusters: List[ObjectCluster]) -> float:
    """
    Measure spatial compactness within each cluster.
    
    Computes the average distance between instances within the same cluster.
    Lower values indicate better spatial compactness.
    
    Args:
        clusters: List of ObjectCluster instances
        
    Returns:
        avg_intra_distance: Average intra-cluster distance across all clusters
    """
    if not clusters:
        return 0.0
    
    total_intra_distance = 0.0
    total_pairs = 0
    
    for cluster in clusters:
        if len(cluster.instances) < 2:
            continue  # Skip single-instance clusters
        
        # Compute pairwise distances within cluster
        centroids = np.array([inst.centroid_3d for inst in cluster.instances])
        distances = pdist(centroids, metric='euclidean')
        
        total_intra_distance += np.sum(distances)
        total_pairs += len(distances)
    
    if total_pairs == 0:
        logger.warning("No valid cluster pairs found for intra-cluster distance computation")
        return 0.0
    
    avg_intra_distance = total_intra_distance / total_pairs
    logger.debug(f"Average intra-cluster distance: {avg_intra_distance:.3f}m")
    
    return avg_intra_distance


def compute_inter_cluster_distances(clusters: List[ObjectCluster]) -> float:
    """
    Measure separation between different clusters of the same semantic class.
    
    Computes the minimum distance between different clusters that have the same
    semantic label. Higher values indicate better separation.
    
    Args:
        clusters: List of ObjectCluster instances
        
    Returns:
        avg_inter_distance: Average inter-cluster distance for same-class clusters
    """
    if len(clusters) < 2:
        return float('inf')  # Perfect separation if only one cluster
    
    # Group clusters by semantic class
    semantic_groups = {}
    for cluster in clusters:
        base_label = cluster.label.split('_obj_')[0] if '_obj_' in cluster.label else cluster.label
        if base_label not in semantic_groups:
            semantic_groups[base_label] = []
        semantic_groups[base_label].append(cluster)
    
    total_inter_distance = 0.0
    total_comparisons = 0
    
    for semantic_class, class_clusters in semantic_groups.items():
        if len(class_clusters) < 2:
            continue  # Need at least 2 clusters of same class
        
        # Compute pairwise distances between clusters of same semantic class
        for i in range(len(class_clusters)):
            for j in range(i + 1, len(class_clusters)):
                distance = np.linalg.norm(
                    class_clusters[i].avg_centroid - class_clusters[j].avg_centroid
                )
                total_inter_distance += distance
                total_comparisons += 1
    
    if total_comparisons == 0:
        logger.info("No same-class cluster pairs found for inter-cluster distance")
        return float('inf')  # Perfect separation
    
    avg_inter_distance = total_inter_distance / total_comparisons
    logger.debug(f"Average inter-cluster distance (same class): {avg_inter_distance:.3f}m")
    
    return avg_inter_distance


def validate_temporal_consistency(clusters: List[ObjectCluster]) -> float:
    """
    Check if clusters make temporal sense.
    
    Analyzes temporal span, gaps, and continuity to identify clusters with
    suspicious temporal patterns that might indicate clustering errors.
    
    Args:
        clusters: List of ObjectCluster instances
        
    Returns:
        consistency_score: Temporal consistency score (0 to 1, higher is better)
    """
    if not clusters:
        return 1.0
    
    total_score = 0.0
    scored_clusters = 0
    
    for cluster in clusters:
        if len(cluster.keyframes) < 2:
            continue  # Single keyframe clusters are trivially consistent
        
        keyframes = sorted(cluster.keyframes)
        cluster_score = 1.0
        
        # Check for large temporal gaps
        gaps = [keyframes[i] - keyframes[i-1] for i in range(1, len(keyframes))]
        max_gap = max(gaps) if gaps else 0
        
        # Penalize large gaps (threshold from config)
        gap_threshold = VALIDATION_CONFIG["temporal_gap_threshold"]
        if max_gap > gap_threshold:
            gap_penalty = min(1.0, (max_gap - gap_threshold) / gap_threshold)
            cluster_score -= 0.5 * gap_penalty
        
        # Check temporal span vs number of detections
        temporal_span = keyframes[-1] - keyframes[0] + 1
        detection_density = len(cluster.instances) / temporal_span
        
        # Penalize very sparse detections (might indicate false associations)
        if detection_density < 0.1:  # Less than 10% coverage
            cluster_score -= 0.3
        
        # Ensure score is non-negative
        cluster_score = max(0.0, cluster_score)
        
        total_score += cluster_score
        scored_clusters += 1
    
    if scored_clusters == 0:
        return 1.0  # No multi-keyframe clusters to validate
    
    consistency_score = total_score / scored_clusters
    logger.debug(f"Temporal consistency score: {consistency_score:.3f}")
    
    return consistency_score


def analyze_cluster_sizes(clusters: List[ObjectCluster]) -> Dict[str, float]:
    """
    Analyze point count and spatial extent consistency.
    
    Checks if cluster instances have similar sizes and spatial extents,
    which can help identify clustering errors.
    
    Args:
        clusters: List of ObjectCluster instances
        
    Returns:
        size_statistics: Dictionary with size analysis metrics
    """
    if not clusters:
        return {"mean_size": 0.0, "std_size": 0.0, "cv_size": 0.0, 
                "mean_extent": 0.0, "std_extent": 0.0, "cv_extent": 0.0}
    
    # Collect size metrics for all clusters
    cluster_sizes = [cluster.total_points for cluster in clusters]
    cluster_extents = [cluster.get_spatial_extent() for cluster in clusters]
    
    # Basic statistics
    size_stats = {
        "mean_size": np.mean(cluster_sizes),
        "std_size": np.std(cluster_sizes),
        "cv_size": np.std(cluster_sizes) / np.mean(cluster_sizes) if np.mean(cluster_sizes) > 0 else 0.0,
        "mean_extent": np.mean(cluster_extents),
        "std_extent": np.std(cluster_extents),
        "cv_extent": np.std(cluster_extents) / np.mean(cluster_extents) if np.mean(cluster_extents) > 0 else 0.0,
        "min_size": np.min(cluster_sizes),
        "max_size": np.max(cluster_sizes),
        "min_extent": np.min(cluster_extents),
        "max_extent": np.max(cluster_extents),
    }
    
    logger.debug(f"Cluster size analysis: mean_size={size_stats['mean_size']:.1f}, "
                f"cv_size={size_stats['cv_size']:.3f}, "
                f"mean_extent={size_stats['mean_extent']:.3f}m")
    
    return size_stats


def validate_semantic_coherence(clusters: List[ObjectCluster]) -> float:
    """
    Ensure all instances in each cluster have the same semantic label.
    
    This should be 100% for our system since we cluster within semantic classes,
    but it's a good sanity check.
    
    Args:
        clusters: List of ObjectCluster instances
        
    Returns:
        coherence_score: Percentage of semantically coherent clusters (0 to 1)
    """
    if not clusters:
        return 1.0
    
    coherent_clusters = 0
    
    for cluster in clusters:
        if not cluster.instances:
            continue
        
        # Extract base labels from all instances
        base_labels = set()
        for instance in cluster.instances:
            base_label = instance.label.split('_kf')[0] if '_kf' in instance.label else instance.label
            base_labels.add(base_label)
        
        # Cluster is coherent if all instances have the same base label
        if len(base_labels) == 1:
            coherent_clusters += 1
        else:
            logger.warning(f"Cluster {cluster.cluster_id} has mixed labels: {base_labels}")
    
    coherence_score = coherent_clusters / len(clusters)
    logger.debug(f"Semantic coherence: {coherent_clusters}/{len(clusters)} = {coherence_score:.3f}")
    
    return coherence_score


def compute_davies_bouldin_index(clusters: List[ObjectCluster]) -> float:
    """
    Compute Davies-Bouldin index for clustering quality assessment.
    
    Lower values indicate better clustering (better separation and compactness).
    
    Args:
        clusters: List of ObjectCluster instances
        
    Returns:
        db_index: Davies-Bouldin index (lower is better)
    """
    if len(clusters) < 2:
        return 0.0  # Perfect score for single cluster
    
    # Collect centroids and labels
    centroids = []
    labels = []
    
    for i, cluster in enumerate(clusters):
        for instance in cluster.instances:
            centroids.append(instance.centroid_3d)
            labels.append(i)
    
    if len(centroids) < 2:
        return 0.0
    
    try:
        centroids = np.array(centroids)
        labels = np.array(labels)
        db_index = davies_bouldin_score(centroids, labels)
        logger.debug(f"Davies-Bouldin index: {db_index:.3f}")
        return db_index
    except Exception as e:
        logger.error(f"Error computing Davies-Bouldin index: {e}")
        return float('inf')


def compute_calinski_harabasz_index(clusters: List[ObjectCluster]) -> float:
    """
    Compute Calinski-Harabasz index for clustering quality assessment.
    
    Higher values indicate better clustering (better defined clusters).
    
    Args:
        clusters: List of ObjectCluster instances
        
    Returns:
        ch_index: Calinski-Harabasz index (higher is better)
    """
    if len(clusters) < 2:
        return 0.0  # Cannot compute for single cluster
    
    # Collect centroids and labels
    centroids = []
    labels = []
    
    for i, cluster in enumerate(clusters):
        for instance in cluster.instances:
            centroids.append(instance.centroid_3d)
            labels.append(i)
    
    if len(centroids) < 2:
        return 0.0
    
    try:
        centroids = np.array(centroids)
        labels = np.array(labels)
        ch_index = calinski_harabasz_score(centroids, labels)
        logger.debug(f"Calinski-Harabasz index: {ch_index:.3f}")
        return ch_index
    except Exception as e:
        logger.error(f"Error computing Calinski-Harabasz index: {e}")
        return 0.0


def detect_spatial_outliers(clusters: List[ObjectCluster]) -> List[int]:
    """
    Identify clusters with spatial outliers or unusual patterns.
    
    Args:
        clusters: List of ObjectCluster instances
        
    Returns:
        outlier_cluster_ids: List of cluster IDs with spatial issues
    """
    outlier_clusters = []
    threshold = VALIDATION_CONFIG["outlier_std_threshold"]
    
    for cluster in clusters:
        if len(cluster.instances) < 3:
            continue  # Need at least 3 instances for outlier detection
        
        centroids = np.array([inst.centroid_3d for inst in cluster.instances])
        
        # Compute distances from cluster average centroid
        distances = [np.linalg.norm(centroid - cluster.avg_centroid) for centroid in centroids]
        
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        # Check for outliers (instances far from cluster center)
        if std_dist > 0:
            z_scores = [(d - mean_dist) / std_dist for d in distances]
            max_z_score = max(np.abs(z_scores))
            
            if max_z_score > threshold:
                outlier_clusters.append(cluster.cluster_id)
                logger.warning(f"Cluster {cluster.cluster_id} has spatial outlier "
                             f"(max z-score: {max_z_score:.2f})")
    
    return outlier_clusters


def detect_temporal_outliers(clusters: List[ObjectCluster]) -> List[int]:
    """
    Identify clusters with temporal inconsistencies.
    
    Args:
        clusters: List of ObjectCluster instances
        
    Returns:
        outlier_cluster_ids: List of cluster IDs with temporal issues
    """
    outlier_clusters = []
    gap_threshold = VALIDATION_CONFIG["temporal_gap_threshold"]
    
    for cluster in clusters:
        if len(cluster.keyframes) < 2:
            continue
        
        keyframes = sorted(cluster.keyframes)
        gaps = [keyframes[i] - keyframes[i-1] for i in range(1, len(keyframes))]
        
        # Check for abnormally large gaps
        max_gap = max(gaps)
        if max_gap > gap_threshold:
            outlier_clusters.append(cluster.cluster_id)
            logger.warning(f"Cluster {cluster.cluster_id} has large temporal gap: {max_gap}")
        
        # Check for very sparse temporal coverage
        temporal_span = keyframes[-1] - keyframes[0] + 1
        coverage = len(cluster.instances) / temporal_span
        
        if coverage < 0.05:  # Less than 5% temporal coverage
            if cluster.cluster_id not in outlier_clusters:
                outlier_clusters.append(cluster.cluster_id)
            logger.warning(f"Cluster {cluster.cluster_id} has sparse temporal coverage: {coverage:.3f}")
    
    return outlier_clusters


def compute_overall_quality_score(metrics: ValidationMetrics) -> float:
    """
    Compute an overall clustering quality score from individual metrics.
    
    Args:
        metrics: ValidationMetrics object with computed metrics
        
    Returns:
        quality_score: Overall quality score (0 to 1, higher is better)
    """
    score_components = []
    
    # Silhouette score (convert from [-1,1] to [0,1])
    if metrics.silhouette_score is not None:
        silhouette_normalized = (metrics.silhouette_score + 1) / 2
        score_components.append(silhouette_normalized)
    
    # Temporal consistency (already 0-1)
    if metrics.temporal_consistency_score is not None:
        score_components.append(metrics.temporal_consistency_score)
    
    # Semantic coherence (already 0-1)
    if metrics.semantic_coherence_score is not None:
        score_components.append(metrics.semantic_coherence_score)
    
    # Davies-Bouldin penalty (lower is better, convert to 0-1 scale)
    if metrics.davies_bouldin_index is not None and metrics.davies_bouldin_index < float('inf'):
        db_penalty = max(0, 1 - metrics.davies_bouldin_index / 5.0)  # Assume 5.0 as "bad" threshold
        score_components.append(db_penalty)
    
    # Outlier penalty
    total_outliers = len(metrics.spatial_outliers) + len(metrics.temporal_outliers)
    if total_outliers == 0:
        score_components.append(1.0)
    else:
        # Assume up to 20% outliers is acceptable
        total_samples = len(metrics.silhouette_samples) if metrics.silhouette_samples else 1
        outlier_penalty = max(0, 1 - total_outliers / (0.2 * total_samples))
        score_components.append(outlier_penalty)
    
    if not score_components:
        return 0.5  # Neutral score if no valid metrics
    
    overall_score = np.mean(score_components)
    logger.info(f"Overall quality score: {overall_score:.3f} "
               f"(from {len(score_components)} components)")
    
    return overall_score


def validate_clustering_quality(clusters: List[ObjectCluster], 
                              output_dir: Optional[str] = None) -> ValidationMetrics:
    """
    Comprehensive clustering validation with all implemented metrics.
    
    This is the main validation function that computes all quality metrics
    and provides a comprehensive assessment of clustering performance.
    
    Args:
        clusters: List of ObjectCluster instances to validate
        output_dir: Optional directory to save validation report and visualizations
        
    Returns:
        metrics: ValidationMetrics object with all computed metrics
    """
    logger.info(f"Starting comprehensive clustering validation for {len(clusters)} clusters")
    
    # Compute all validation metrics
    silhouette_overall, silhouette_samples = compute_silhouette_score(clusters)
    intra_distance = compute_intra_cluster_distances(clusters)
    inter_distance = compute_inter_cluster_distances(clusters)
    temporal_score = validate_temporal_consistency(clusters)
    db_index = compute_davies_bouldin_index(clusters)
    ch_index = compute_calinski_harabasz_index(clusters)
    size_stats = analyze_cluster_sizes(clusters)
    coherence_score = validate_semantic_coherence(clusters)
    spatial_outliers = detect_spatial_outliers(clusters)
    temporal_outliers = detect_temporal_outliers(clusters)
    
    # Create metrics object
    metrics = ValidationMetrics(
        silhouette_score=silhouette_overall,
        silhouette_samples=silhouette_samples,
        avg_intra_cluster_distance=intra_distance,
        avg_inter_cluster_distance=inter_distance,
        temporal_consistency_score=temporal_score,
        davies_bouldin_index=db_index,
        calinski_harabasz_index=ch_index,
        cluster_size_statistics=size_stats,
        semantic_coherence_score=coherence_score,
        spatial_outliers=spatial_outliers,
        temporal_outliers=temporal_outliers,
        overall_quality_score=0.0  # Will be computed below
    )
    
    # Compute overall quality score
    metrics.overall_quality_score = compute_overall_quality_score(metrics)
    
    # Save validation report if output directory specified
    if output_dir:
        save_validation_report(metrics, clusters, output_dir)
    
    # Log summary
    logger.info("Clustering validation complete:")
    logger.info(f"  Overall quality score: {metrics.overall_quality_score:.3f}")
    logger.info(f"  Silhouette score: {metrics.silhouette_score:.3f}")
    logger.info(f"  Temporal consistency: {metrics.temporal_consistency_score:.3f}")
    logger.info(f"  Semantic coherence: {metrics.semantic_coherence_score:.3f}")
    logger.info(f"  Spatial outliers: {len(spatial_outliers)} clusters")
    logger.info(f"  Temporal outliers: {len(temporal_outliers)} clusters")
    
    return metrics


def save_validation_report(metrics: ValidationMetrics, 
                          clusters: List[ObjectCluster], 
                          output_dir: str):
    """
    Save comprehensive validation report with metrics and visualizations.
    
    Args:
        metrics: ValidationMetrics object with computed metrics
        clusters: List of ObjectCluster instances
        output_dir: Directory to save report files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics as JSON
    metrics_dict = asdict(metrics)
    # Convert numpy arrays and other non-serializable types to JSON-compatible formats
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    metrics_dict = convert_for_json(metrics_dict)
    
    metrics_file = os.path.join(output_dir, "validation_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    # Save detailed cluster analysis
    cluster_analysis = []
    for cluster in clusters:
        analysis = {
            "cluster_id": int(cluster.cluster_id),
            "label": str(cluster.label),
            "num_instances": int(len(cluster.instances)),
            "keyframes": [int(kf) for kf in cluster.keyframes],
            "temporal_span": [int(x) for x in cluster.get_temporal_span()],
            "spatial_extent": float(cluster.get_spatial_extent()),
            "total_points": int(cluster.total_points),
            "avg_centroid": cluster.avg_centroid.tolist(),
            "silhouette_score": float(metrics.silhouette_samples.get(cluster.cluster_id, 0.0)) if metrics.silhouette_samples.get(cluster.cluster_id) is not None else None,
            "spatial_outlier": bool(cluster.cluster_id in metrics.spatial_outliers),
            "temporal_outlier": bool(cluster.cluster_id in metrics.temporal_outliers),
        }
        cluster_analysis.append(analysis)
    
    cluster_file = os.path.join(output_dir, "cluster_analysis.json")
    with open(cluster_file, 'w') as f:
        json.dump(cluster_analysis, f, indent=2)
    
    # Generate summary report
    report_file = os.path.join(output_dir, "validation_report.txt")
    with open(report_file, 'w') as f:
        f.write("CLUSTERING VALIDATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total clusters analyzed: {len(clusters)}\n")
        f.write(f"Total instances: {sum(len(c.instances) for c in clusters)}\n\n")
        
        f.write("QUALITY METRICS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Overall Quality Score: {metrics.overall_quality_score:.3f}\n")
        f.write(f"Silhouette Score: {metrics.silhouette_score:.3f}\n")
        f.write(f"Davies-Bouldin Index: {metrics.davies_bouldin_index:.3f}\n")
        f.write(f"Calinski-Harabasz Index: {metrics.calinski_harabasz_index:.3f}\n\n")
        
        f.write("SPATIAL ANALYSIS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Avg Intra-cluster Distance: {metrics.avg_intra_cluster_distance:.3f}m\n")
        f.write(f"Avg Inter-cluster Distance: {metrics.avg_inter_cluster_distance:.3f}m\n")
        f.write(f"Spatial Outliers: {len(metrics.spatial_outliers)} clusters\n\n")
        
        f.write("TEMPORAL ANALYSIS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Temporal Consistency Score: {metrics.temporal_consistency_score:.3f}\n")
        f.write(f"Temporal Outliers: {len(metrics.temporal_outliers)} clusters\n\n")
        
        f.write("SIZE ANALYSIS\n")
        f.write("-" * 20 + "\n")
        size_stats = metrics.cluster_size_statistics
        f.write(f"Mean Cluster Size: {size_stats['mean_size']:.1f} points\n")
        f.write(f"Size Coefficient of Variation: {size_stats['cv_size']:.3f}\n")
        f.write(f"Mean Spatial Extent: {size_stats['mean_extent']:.3f}m\n")
        f.write(f"Extent Coefficient of Variation: {size_stats['cv_extent']:.3f}\n\n")
        
        f.write(f"Semantic Coherence: {metrics.semantic_coherence_score:.3f}\n\n")
        
        if metrics.spatial_outliers:
            f.write("SPATIAL OUTLIER CLUSTERS\n")
            f.write("-" * 25 + "\n")
            for cluster_id in metrics.spatial_outliers:
                f.write(f"  Cluster {cluster_id}\n")
            f.write("\n")
        
        if metrics.temporal_outliers:
            f.write("TEMPORAL OUTLIER CLUSTERS\n")
            f.write("-" * 26 + "\n")
            for cluster_id in metrics.temporal_outliers:
                f.write(f"  Cluster {cluster_id}\n")
            f.write("\n")
    
    logger.info(f"Validation report saved to {output_dir}")


def create_validation_visualizations(metrics: ValidationMetrics, 
                                   clusters: List[ObjectCluster], 
                                   output_dir: str):
    """
    Create visualization plots for clustering validation results.
    
    Args:
        metrics: ValidationMetrics object with computed metrics
        clusters: List of ObjectCluster instances
        output_dir: Directory to save visualization files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style for better plots
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Silhouette scores per cluster
    if metrics.silhouette_samples:
        plt.figure(figsize=(12, 6))
        cluster_ids = list(metrics.silhouette_samples.keys())
        scores = list(metrics.silhouette_samples.values())
        
        bars = plt.bar(range(len(cluster_ids)), scores)
        plt.axhline(y=metrics.silhouette_score, color='red', linestyle='--', 
                   label=f'Overall Score: {metrics.silhouette_score:.3f}')
        plt.xlabel('Cluster Index')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Scores per Cluster')
        plt.legend()
        
        # Color bars based on score quality
        for i, (bar, score) in enumerate(zip(bars, scores)):
            if score < 0:
                bar.set_color('red')
            elif score < 0.5:
                bar.set_color('orange')
            else:
                bar.set_color('green')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'silhouette_scores.png'), dpi=300)
        plt.close()
    
    # 2. Cluster size distribution
    if clusters:
        plt.figure(figsize=(10, 6))
        
        sizes = [len(c.instances) for c in clusters]
        extents = [c.get_spatial_extent() for c in clusters]
        
        plt.subplot(1, 2, 1)
        plt.hist(sizes, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Number of Instances')
        plt.ylabel('Number of Clusters')
        plt.title('Cluster Size Distribution')
        
        plt.subplot(1, 2, 2)
        plt.hist(extents, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Spatial Extent (m)')
        plt.ylabel('Number of Clusters')
        plt.title('Spatial Extent Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cluster_distributions.png'), dpi=300)
        plt.close()
    
    # 3. Quality metrics summary
    plt.figure(figsize=(10, 8))
    
    metric_names = ['Overall\nQuality', 'Silhouette\nScore', 'Temporal\nConsistency', 
                   'Semantic\nCoherence']
    metric_values = [metrics.overall_quality_score, 
                    (metrics.silhouette_score + 1) / 2,  # Normalize to 0-1
                    metrics.temporal_consistency_score,
                    metrics.semantic_coherence_score]
    
    bars = plt.bar(metric_names, metric_values)
    plt.ylim(0, 1)
    plt.ylabel('Score (0-1, higher is better)')
    plt.title('Clustering Quality Metrics Summary')
    
    # Color bars based on quality
    for bar, value in zip(bars, metric_values):
        if value < 0.3:
            bar.set_color('red')
        elif value < 0.7:
            bar.set_color('orange')
        else:
            bar.set_color('green')
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'quality_summary.png'), dpi=300)
    plt.close()
    
    logger.info(f"Validation visualizations saved to {output_dir}")


# Configuration functions
def get_validation_config() -> Dict:
    """Return current validation configuration."""
    return VALIDATION_CONFIG.copy()


def set_validation_config(config: Dict):
    """Update validation configuration."""
    global VALIDATION_CONFIG
    VALIDATION_CONFIG.update(config)
    logger.info(f"Updated validation configuration: {VALIDATION_CONFIG}")


if __name__ == "__main__":
    # Basic testing and demonstration
    logger.setLevel(logging.DEBUG)
    
    # Create sample clusters for testing
    from .object_clustering import ObjectInstance, create_object_cluster
    
    # Sample instances with varying quality
    instances = [
        ObjectInstance(1, 1, 0, "table", 0.9, np.array([1.0, 2.0, 0.5]), 100, np.arange(100)),
        ObjectInstance(2, 1, 1, "table", 0.8, np.array([1.1, 2.1, 0.5]), 120, np.arange(120)),
        ObjectInstance(3, 1, 2, "table", 0.85, np.array([1.05, 1.95, 0.5]), 110, np.arange(110)),
        ObjectInstance(4, 1, 5, "chair", 0.7, np.array([3.0, 1.0, 0.4]), 80, np.arange(80)),
        ObjectInstance(5, 1, 6, "chair", 0.75, np.array([3.1, 0.9, 0.4]), 85, np.arange(85)),
        ObjectInstance(6, 1, 20, "table", 0.6, np.array([1.2, 2.2, 0.5]), 90, np.arange(90)),  # Temporal outlier
    ]
    
    # Create test clusters
    table_cluster = create_object_cluster(instances[:3], 1)
    chair_cluster = create_object_cluster(instances[3:5], 2)
    outlier_cluster = create_object_cluster([instances[5]], 3)
    
    test_clusters = [table_cluster, chair_cluster, outlier_cluster]
    
    # Run validation
    print("Running clustering validation test...")
    metrics = validate_clustering_quality(test_clusters, output_dir="/tmp/clustering_validation_test")
    
    print(f"\nValidation Results:")
    print(f"Overall Quality Score: {metrics.overall_quality_score:.3f}")
    print(f"Silhouette Score: {metrics.silhouette_score:.3f}")
    print(f"Temporal Consistency: {metrics.temporal_consistency_score:.3f}")
    print(f"Spatial Outliers: {len(metrics.spatial_outliers)}")
    print(f"Temporal Outliers: {len(metrics.temporal_outliers)}")