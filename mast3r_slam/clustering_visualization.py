"""
Comprehensive 3D Visualization Module for Clustering Analysis

This module provides interactive visualizations and detailed analysis tools 
for manual inspection of object clustering results in semantic SLAM.

Key Features:
- Interactive 3D cluster visualization with plotly
- Temporal distribution analysis
- Spatial distribution analysis
- Size distribution plots
- Semantic class breakdowns
- Individual cluster detailed analysis
- Comparative analysis between different clustering parameters
- HTML report generation

Author: Generated for MASt3R SLAM semantic extension
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import json
import logging
from dataclasses import asdict
from datetime import datetime

# Import clustering data structures
from .object_clustering import ObjectInstance, ObjectCluster, get_clustering_config

# Configure module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Color palettes for visualization
PLOTLY_SEMANTIC_COLORS = px.colors.qualitative.Set3
PLOTLY_CLUSTER_COLORS = px.colors.qualitative.Plotly
MATPLOTLIB_SEMANTIC_COLORS = plt.cm.Set3
MATPLOTLIB_CLUSTER_COLORS = plt.cm.tab10
TEMPORAL_COLORMAP = plt.cm.viridis
SPATIAL_COLORMAP = plt.cm.plasma

# Default visualization settings
VIZ_CONFIG = {
    "figure_size": (12, 8),
    "dpi": 150,
    "font_size": 10,
    "title_size": 14,
    "save_format": "png",
    "plotly_theme": "plotly_white",
    "interactive_height": 600,
    "interactive_width": 800,
}


class ClusteringMetrics:
    """Container for clustering evaluation metrics."""
    
    def __init__(self, clusters: List[ObjectCluster]):
        self.clusters = clusters
        self.total_instances = sum(len(cluster.instances) for cluster in clusters)
        self.total_clusters = len(clusters)
        self.semantic_classes = self._get_semantic_classes()
        self.temporal_span = self._get_temporal_span()
        self.spatial_extent = self._get_spatial_extent()
        
    def _get_semantic_classes(self) -> Dict[str, int]:
        """Count clusters per semantic class."""
        classes = {}
        for cluster in self.clusters:
            base_label = cluster.label.split('_obj_')[0] if '_obj_' in cluster.label else cluster.label
            classes[base_label] = classes.get(base_label, 0) + 1
        return classes
        
    def _get_temporal_span(self) -> Tuple[int, int]:
        """Get overall temporal span of all clusters."""
        if not self.clusters:
            return (0, 0)
        all_keyframes = []
        for cluster in self.clusters:
            all_keyframes.extend(cluster.keyframes)
        return (min(all_keyframes), max(all_keyframes)) if all_keyframes else (0, 0)
        
    def _get_spatial_extent(self) -> float:
        """Get maximum spatial extent across all clusters."""
        if not self.clusters:
            return 0.0
        all_centroids = []
        for cluster in self.clusters:
            all_centroids.append(cluster.avg_centroid)
        
        if len(all_centroids) < 2:
            return 0.0
            
        centroids = np.array(all_centroids)
        max_dist = 0.0
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                dist = np.linalg.norm(centroids[i] - centroids[j])
                max_dist = max(max_dist, dist)
        return max_dist


def visualize_clusters_3d(clusters: List[ObjectCluster], 
                         save_dir: str = "cluster_analysis/",
                         show_interactive: bool = True) -> Dict[str, str]:
    """
    Create interactive 3D visualizations using plotly.
    
    Args:
        clusters: List of ObjectCluster objects to visualize
        save_dir: Directory to save visualizations
        show_interactive: Whether to show interactive plots
        
    Returns:
        Dictionary mapping plot names to file paths
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    if not clusters:
        logger.warning("No clusters to visualize")
        return saved_files
        
    logger.info(f"Creating 3D visualizations for {len(clusters)} clusters")
    
    # Prepare data for visualization
    plot_data = []
    for cluster in clusters:
        base_label = cluster.label.split('_obj_')[0] if '_obj_' in cluster.label else cluster.label
        for instance in cluster.instances:
            plot_data.append({
                'x': instance.centroid_3d[0],
                'y': instance.centroid_3d[1], 
                'z': instance.centroid_3d[2],
                'cluster_id': cluster.cluster_id,
                'cluster_label': cluster.label,
                'semantic_class': base_label,
                'keyframe': instance.keyframe_idx,
                'confidence': instance.confidence,
                'num_points': instance.num_points,
                'instance_id': instance.global_id
            })
    
    df = pd.DataFrame(plot_data)
    
    # 1. Overall 3D scatter plot colored by cluster
    fig = px.scatter_3d(
        df, x='x', y='y', z='z',
        color='cluster_label',
        size='num_points',
        hover_data=['keyframe', 'confidence', 'semantic_class'],
        title="3D Object Clusters - Colored by Cluster ID",
        color_discrete_sequence=PLOTLY_CLUSTER_COLORS
    )
    
    fig.update_layout(
        scene=dict(
            xaxis_title="X (meters)",
            yaxis_title="Y (meters)", 
            zaxis_title="Z (meters)"
        ),
        width=VIZ_CONFIG["interactive_width"],
        height=VIZ_CONFIG["interactive_height"],
        template=VIZ_CONFIG["plotly_theme"]
    )
    
    cluster_3d_path = save_path / "clusters_3d_overview.html"
    fig.write_html(str(cluster_3d_path))
    saved_files["clusters_3d_overview"] = str(cluster_3d_path)
    
    if show_interactive:
        fig.show()
    
    # 2. 3D scatter plot colored by semantic class
    fig_semantic = px.scatter_3d(
        df, x='x', y='y', z='z',
        color='semantic_class',
        size='num_points',
        hover_data=['cluster_label', 'keyframe', 'confidence'],
        title="3D Object Clusters - Colored by Semantic Class",
        color_discrete_sequence=PLOTLY_SEMANTIC_COLORS
    )
    
    fig_semantic.update_layout(
        scene=dict(
            xaxis_title="X (meters)",
            yaxis_title="Y (meters)",
            zaxis_title="Z (meters)"
        ),
        width=VIZ_CONFIG["interactive_width"],
        height=VIZ_CONFIG["interactive_height"],
        template=VIZ_CONFIG["plotly_theme"]
    )
    
    semantic_3d_path = save_path / "clusters_3d_semantic.html"
    fig_semantic.write_html(str(semantic_3d_path))
    saved_files["clusters_3d_semantic"] = str(semantic_3d_path)
    
    if show_interactive:
        fig_semantic.show()
        
    # 3. Temporal 3D visualization
    fig_temporal = px.scatter_3d(
        df, x='x', y='y', z='z',
        color='keyframe',
        size='num_points',
        hover_data=['cluster_label', 'semantic_class', 'confidence'],
        title="3D Object Clusters - Colored by Keyframe",
        color_continuous_scale='viridis'
    )
    
    fig_temporal.update_layout(
        scene=dict(
            xaxis_title="X (meters)",
            yaxis_title="Y (meters)",
            zaxis_title="Z (meters)"
        ),
        width=VIZ_CONFIG["interactive_width"],
        height=VIZ_CONFIG["interactive_height"],
        template=VIZ_CONFIG["plotly_theme"]
    )
    
    temporal_3d_path = save_path / "clusters_3d_temporal.html"
    fig_temporal.write_html(str(temporal_3d_path))
    saved_files["clusters_3d_temporal"] = str(temporal_3d_path)
    
    if show_interactive:
        fig_temporal.show()
    
    # 4. Create separate plots for each semantic class
    semantic_classes = df['semantic_class'].unique()
    for sem_class in semantic_classes:
        class_df = df[df['semantic_class'] == sem_class]
        
        fig_class = px.scatter_3d(
            class_df, x='x', y='y', z='z',
            color='cluster_label',
            size='num_points',
            hover_data=['keyframe', 'confidence'],
            title=f"3D Clusters for '{sem_class}' Objects",
            color_discrete_sequence=PLOTLY_CLUSTER_COLORS
        )
        
        fig_class.update_layout(
            scene=dict(
                xaxis_title="X (meters)",
                yaxis_title="Y (meters)",
                zaxis_title="Z (meters)"
            ),
            width=VIZ_CONFIG["interactive_width"],
            height=VIZ_CONFIG["interactive_height"],
            template=VIZ_CONFIG["plotly_theme"]
        )
        
        class_path = save_path / f"clusters_3d_{sem_class.replace(' ', '_')}.html"
        fig_class.write_html(str(class_path))
        saved_files[f"clusters_3d_{sem_class}"] = str(class_path)
    
    logger.info(f"Saved {len(saved_files)} 3D visualization files to {save_dir}")
    return saved_files


def plot_temporal_distribution(clusters: List[ObjectCluster], 
                              save_dir: str = "cluster_analysis/") -> str:
    """
    Visualize when each cluster appears across keyframes.
    
    Args:
        clusters: List of ObjectCluster objects
        save_dir: Directory to save plots
        
    Returns:
        Path to saved temporal distribution plot
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    if not clusters:
        logger.warning("No clusters to analyze temporally")
        return ""
    
    logger.info(f"Creating temporal distribution plots for {len(clusters)} clusters")
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
    
    # 1. Timeline showing cluster activity
    cluster_data = []
    for cluster in clusters:
        base_label = cluster.label.split('_obj_')[0] if '_obj_' in cluster.label else cluster.label
        for keyframe in cluster.keyframes:
            cluster_data.append({
                'cluster_id': cluster.cluster_id,
                'cluster_label': cluster.label,
                'semantic_class': base_label,
                'keyframe': keyframe,
                'num_instances': len([inst for inst in cluster.instances if inst.keyframe_idx == keyframe])
            })
    
    df = pd.DataFrame(cluster_data)
    
    # Plot 1: Activity timeline
    pivot_data = df.pivot_table(
        index='cluster_label', 
        columns='keyframe', 
        values='num_instances', 
        fill_value=0
    )
    
    sns.heatmap(
        pivot_data, 
        ax=ax1, 
        cmap='YlOrRd', 
        cbar_kws={'label': 'Number of Instances'},
        xticklabels=True,
        yticklabels=True
    )
    ax1.set_title('Cluster Activity Timeline', fontsize=VIZ_CONFIG["title_size"])
    ax1.set_xlabel('Keyframe Index')
    ax1.set_ylabel('Cluster')
    
    # Plot 2: Temporal gaps analysis
    gap_data = []
    for cluster in clusters:
        keyframes = sorted(cluster.keyframes)
        if len(keyframes) > 1:
            gaps = [keyframes[i+1] - keyframes[i] for i in range(len(keyframes)-1)]
            max_gap = max(gaps)
            avg_gap = np.mean(gaps)
            gap_data.append({
                'cluster_label': cluster.label,
                'max_gap': max_gap,
                'avg_gap': avg_gap,
                'num_gaps': len(gaps)
            })
    
    if gap_data:
        gap_df = pd.DataFrame(gap_data)
        
        # Scatter plot of temporal gaps
        scatter = ax2.scatter(
            gap_df['avg_gap'], 
            gap_df['max_gap'],
            s=gap_df['num_gaps'] * 20,
            alpha=0.6,
            c=range(len(gap_df)),
            cmap='viridis'
        )
        
        ax2.set_xlabel('Average Gap Between Keyframes')
        ax2.set_ylabel('Maximum Gap Between Keyframes')
        ax2.set_title('Temporal Gap Analysis', fontsize=VIZ_CONFIG["title_size"])
        ax2.grid(True, alpha=0.3)
        
        # Add cluster labels to points
        for i, row in gap_df.iterrows():
            if row['max_gap'] > 10:  # Only label problematic clusters
                ax2.annotate(
                    row['cluster_label'], 
                    (row['avg_gap'], row['max_gap']),
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.7
                )
    
    # Plot 3: Temporal patterns by semantic class
    semantic_temporal = df.groupby(['semantic_class', 'keyframe'])['num_instances'].sum().reset_index()
    
    for sem_class in semantic_temporal['semantic_class'].unique():
        class_data = semantic_temporal[semantic_temporal['semantic_class'] == sem_class]
        ax3.plot(
            class_data['keyframe'], 
            class_data['num_instances'],
            marker='o',
            label=sem_class,
            linewidth=2,
            markersize=4
        )
    
    ax3.set_xlabel('Keyframe Index')
    ax3.set_ylabel('Total Instances')
    ax3.set_title('Temporal Patterns by Semantic Class', fontsize=VIZ_CONFIG["title_size"])
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    temporal_path = save_path / "temporal_distribution.png"
    plt.savefig(temporal_path, dpi=VIZ_CONFIG["dpi"], bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved temporal distribution plot to {temporal_path}")
    return str(temporal_path)


def plot_spatial_distribution(clusters: List[ObjectCluster], 
                             save_dir: str = "cluster_analysis/") -> str:
    """
    2D top-down view of cluster spatial layout.
    
    Args:
        clusters: List of ObjectCluster objects
        save_dir: Directory to save plots
        
    Returns:
        Path to saved spatial distribution plot
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    if not clusters:
        logger.warning("No clusters to analyze spatially")
        return ""
        
    logger.info(f"Creating spatial distribution plots for {len(clusters)} clusters")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract spatial data
    cluster_centroids = []
    for cluster in clusters:
        base_label = cluster.label.split('_obj_')[0] if '_obj_' in cluster.label else cluster.label
        cluster_centroids.append({
            'x': cluster.avg_centroid[0],
            'y': cluster.avg_centroid[1],
            'z': cluster.avg_centroid[2],
            'cluster_id': cluster.cluster_id,
            'cluster_label': cluster.label,
            'semantic_class': base_label,
            'num_instances': len(cluster.instances),
            'spatial_extent': cluster.get_spatial_extent(),
            'temporal_span': cluster.get_temporal_span()[1] - cluster.get_temporal_span()[0]
        })
    
    df = pd.DataFrame(cluster_centroids)
    
    # Plot 1: Top-down view (X-Y plane)
    semantic_classes = df['semantic_class'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(semantic_classes)))
    color_map = dict(zip(semantic_classes, colors))
    
    for sem_class in semantic_classes:
        class_data = df[df['semantic_class'] == sem_class]
        ax1.scatter(
            class_data['x'], 
            class_data['y'],
            c=[color_map[sem_class]], 
            s=class_data['num_instances'] * 20,
            alpha=0.7,
            label=sem_class,
            edgecolors='black',
            linewidth=0.5
        )
        
        # Add cluster extent circles
        for _, row in class_data.iterrows():
            if row['spatial_extent'] > 0:
                circle = patches.Circle(
                    (row['x'], row['y']), 
                    row['spatial_extent'] / 2,
                    fill=False, 
                    edgecolor=color_map[sem_class],
                    linestyle='--',
                    alpha=0.5
                )
                ax1.add_patch(circle)
    
    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    ax1.set_title('Top-Down View (Floor Plan)', fontsize=VIZ_CONFIG["title_size"])
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    
    # Plot 2: Side view (X-Z plane)
    for sem_class in semantic_classes:
        class_data = df[df['semantic_class'] == sem_class]
        ax2.scatter(
            class_data['x'], 
            class_data['z'],
            c=[color_map[sem_class]], 
            s=class_data['num_instances'] * 20,
            alpha=0.7,
            label=sem_class,
            edgecolors='black',
            linewidth=0.5
        )
    
    ax2.set_xlabel('X (meters)')
    ax2.set_ylabel('Z (meters)')
    ax2.set_title('Side View (Elevation)', fontsize=VIZ_CONFIG["title_size"])
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Cluster density heatmap
    if len(df) > 1:
        # Create 2D histogram of cluster positions
        hist, xedges, yedges = np.histogram2d(df['x'], df['y'], bins=20)
        
        im = ax3.imshow(
            hist.T, 
            origin='lower',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap='hot',
            alpha=0.8
        )
        
        ax3.set_xlabel('X (meters)')
        ax3.set_ylabel('Y (meters)') 
        ax3.set_title('Cluster Density Heatmap', fontsize=VIZ_CONFIG["title_size"])
        plt.colorbar(im, ax=ax3, label='Cluster Count')
    
    # Plot 4: Spatial extent vs temporal span
    scatter = ax4.scatter(
        df['spatial_extent'],
        df['temporal_span'],
        s=df['num_instances'] * 20,
        c=df['cluster_id'],
        cmap='viridis',
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )
    
    ax4.set_xlabel('Spatial Extent (meters)')
    ax4.set_ylabel('Temporal Span (keyframes)')
    ax4.set_title('Spatial vs Temporal Characteristics', fontsize=VIZ_CONFIG["title_size"])
    ax4.grid(True, alpha=0.3)
    
    # Add cluster labels to problematic points
    for _, row in df.iterrows():
        if row['spatial_extent'] > 2.0 or row['temporal_span'] > 20:
            ax4.annotate(
                row['cluster_label'], 
                (row['spatial_extent'], row['temporal_span']),
                xytext=(5, 5), 
                textcoords='offset points',
                fontsize=8,
                alpha=0.7
            )
    
    plt.tight_layout()
    
    # Save plot
    spatial_path = save_path / "spatial_distribution.png"
    plt.savefig(spatial_path, dpi=VIZ_CONFIG["dpi"], bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved spatial distribution plot to {spatial_path}")
    return str(spatial_path)


def plot_size_distributions(clusters: List[ObjectCluster], 
                           save_dir: str = "cluster_analysis/") -> str:
    """
    Analyze cluster size consistency.
    
    Args:
        clusters: List of ObjectCluster objects
        save_dir: Directory to save plots
        
    Returns:
        Path to saved size distribution plot
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    if not clusters:
        logger.warning("No clusters to analyze for size distribution")
        return ""
    
    logger.info(f"Creating size distribution plots for {len(clusters)} clusters")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract size data
    size_data = []
    for cluster in clusters:
        base_label = cluster.label.split('_obj_')[0] if '_obj_' in cluster.label else cluster.label
        
        # Per-instance data
        for instance in cluster.instances:
            size_data.append({
                'cluster_id': cluster.cluster_id,
                'cluster_label': cluster.label,
                'semantic_class': base_label,
                'instance_points': instance.num_points,
                'cluster_total_points': cluster.total_points,
                'cluster_instances': len(cluster.instances),
                'spatial_extent': cluster.get_spatial_extent()
            })
    
    df = pd.DataFrame(size_data)
    
    # Cluster-level aggregation
    cluster_summary = df.groupby('cluster_label').agg({
        'instance_points': ['mean', 'std', 'min', 'max'],
        'cluster_total_points': 'first',
        'cluster_instances': 'first',
        'spatial_extent': 'first',
        'semantic_class': 'first'
    }).round(2)
    
    cluster_summary.columns = ['_'.join(col).strip() for col in cluster_summary.columns.values]
    cluster_summary = cluster_summary.reset_index()
    
    # Plot 1: Point count distributions per cluster
    semantic_classes = df['semantic_class'].unique()
    
    for i, sem_class in enumerate(semantic_classes):
        class_data = df[df['semantic_class'] == sem_class]
        ax1.hist(
            class_data['instance_points'], 
            bins=20, 
            alpha=0.7, 
            label=sem_class,
            edgecolor='black',
            linewidth=0.5
        )
    
    ax1.set_xlabel('Points per Instance')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Point Count Distribution by Semantic Class', fontsize=VIZ_CONFIG["title_size"])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box plot of point counts by semantic class
    semantic_classes = df['semantic_class'].unique()
    box_data = [df[df['semantic_class'] == cls]['instance_points'].values for cls in semantic_classes]
    
    box_plot = ax2.boxplot(box_data, patch_artist=True, labels=semantic_classes)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(semantic_classes)))
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    ax2.set_xlabel('Semantic Class')
    ax2.set_ylabel('Points per Instance')
    ax2.set_title('Point Count Distribution by Class', fontsize=VIZ_CONFIG["title_size"])
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 3: Spatial extent distributions
    for i, sem_class in enumerate(semantic_classes):
        class_summary = cluster_summary[cluster_summary['semantic_class_first'] == sem_class]
        if len(class_summary) > 0:
            ax3.hist(
                class_summary['spatial_extent_first'], 
                bins=15, 
                alpha=0.7, 
                label=sem_class,
                edgecolor='black',
                linewidth=0.5
            )
    
    ax3.set_xlabel('Spatial Extent (meters)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Spatial Extent Distribution by Semantic Class', fontsize=VIZ_CONFIG["title_size"])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Correlation between size metrics
    scatter = ax4.scatter(
        cluster_summary['cluster_total_points_first'],
        cluster_summary['spatial_extent_first'], 
        s=cluster_summary['cluster_instances_first'] * 30,
        c=cluster_summary.index,
        cmap='viridis',
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )
    
    ax4.set_xlabel('Total Points in Cluster')
    ax4.set_ylabel('Spatial Extent (meters)')
    ax4.set_title('Size vs Spatial Extent Correlation', fontsize=VIZ_CONFIG["title_size"])
    ax4.grid(True, alpha=0.3)
    
    # Identify outliers
    q75_points = cluster_summary['cluster_total_points_first'].quantile(0.75)
    q75_extent = cluster_summary['spatial_extent_first'].quantile(0.75)
    
    outliers = cluster_summary[
        (cluster_summary['cluster_total_points_first'] > q75_points * 2) |
        (cluster_summary['spatial_extent_first'] > q75_extent * 2)
    ]
    
    for _, row in outliers.iterrows():
        ax4.annotate(
            row['cluster_label'], 
            (row['cluster_total_points_first'], row['spatial_extent_first']),
            xytext=(5, 5), 
            textcoords='offset points',
            fontsize=8,
            alpha=0.7
        )
    
    plt.tight_layout()
    
    # Save plot
    size_path = save_path / "size_distributions.png"
    plt.savefig(size_path, dpi=VIZ_CONFIG["dpi"], bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved size distribution plot to {size_path}")
    return str(size_path)


def plot_semantic_breakdown(clusters: List[ObjectCluster], 
                           save_dir: str = "cluster_analysis/") -> str:
    """
    Breakdown clustering by semantic categories.
    
    Args:
        clusters: List of ObjectCluster objects
        save_dir: Directory to save plots
        
    Returns:
        Path to saved semantic breakdown plot
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    if not clusters:
        logger.warning("No clusters to analyze for semantic breakdown")
        return ""
    
    logger.info(f"Creating semantic breakdown plots for {len(clusters)} clusters")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract semantic data
    semantic_data = []
    for cluster in clusters:
        base_label = cluster.label.split('_obj_')[0] if '_obj_' in cluster.label else cluster.label
        temporal_span = cluster.get_temporal_span()
        
        semantic_data.append({
            'semantic_class': base_label,
            'cluster_id': cluster.cluster_id,
            'cluster_label': cluster.label,
            'num_instances': len(cluster.instances),
            'total_points': cluster.total_points,
            'avg_points_per_instance': cluster.total_points / len(cluster.instances),
            'spatial_extent': cluster.get_spatial_extent(),
            'temporal_span': temporal_span[1] - temporal_span[0],
            'num_keyframes': len(cluster.keyframes)
        })
    
    df = pd.DataFrame(semantic_data)
    
    # Semantic class summary
    semantic_summary = df.groupby('semantic_class').agg({
        'cluster_id': 'count',  # Number of clusters
        'num_instances': ['sum', 'mean'],
        'total_points': ['sum', 'mean'], 
        'spatial_extent': ['mean', 'std'],
        'temporal_span': ['mean', 'std'],
        'num_keyframes': ['mean', 'std']
    }).round(2)
    
    semantic_summary.columns = ['_'.join(col).strip() for col in semantic_summary.columns.values]
    semantic_summary = semantic_summary.reset_index()
    
    # Plot 1: Number of clusters per semantic class
    cluster_counts = semantic_summary['cluster_id_count'].values
    class_names = semantic_summary['semantic_class'].values
    
    colors = MATPLOTLIB_SEMANTIC_COLORS(np.linspace(0, 1, len(class_names)))
    bars = ax1.bar(
        range(len(class_names)), 
        cluster_counts,
        color=colors,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5
    )
    
    ax1.set_xlabel('Semantic Class')
    ax1.set_ylabel('Number of Clusters')
    ax1.set_title('Clusters per Semantic Class', fontsize=VIZ_CONFIG["title_size"])
    ax1.set_xticks(range(len(class_names)))
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, count in zip(bars, cluster_counts):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2., 
            height + 0.1,
            str(int(count)), 
            ha='center', 
            va='bottom'
        )
    
    # Plot 2: Average cluster sizes by class
    avg_sizes = semantic_summary['num_instances_mean'].values
    
    bars2 = ax2.bar(
        range(len(class_names)), 
        avg_sizes,
        color=colors,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5
    )
    
    ax2.set_xlabel('Semantic Class')
    ax2.set_ylabel('Average Instances per Cluster')
    ax2.set_title('Average Cluster Size by Class', fontsize=VIZ_CONFIG["title_size"])
    ax2.set_xticks(range(len(class_names)))
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Temporal patterns by class
    temporal_means = semantic_summary['temporal_span_mean'].values
    temporal_stds = semantic_summary['temporal_span_std'].values
    
    bars3 = ax3.bar(
        range(len(class_names)), 
        temporal_means,
        yerr=temporal_stds,
        color=colors,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5,
        capsize=5
    )
    
    ax3.set_xlabel('Semantic Class')
    ax3.set_ylabel('Average Temporal Span (keyframes)')
    ax3.set_title('Temporal Characteristics by Class', fontsize=VIZ_CONFIG["title_size"])
    ax3.set_xticks(range(len(class_names)))
    ax3.set_xticklabels(class_names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Spatial extent comparison
    spatial_means = semantic_summary['spatial_extent_mean'].values
    spatial_stds = semantic_summary['spatial_extent_std'].values
    
    bars4 = ax4.bar(
        range(len(class_names)), 
        spatial_means,
        yerr=spatial_stds,
        color=colors,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5,
        capsize=5
    )
    
    ax4.set_xlabel('Semantic Class')
    ax4.set_ylabel('Average Spatial Extent (meters)')
    ax4.set_title('Spatial Characteristics by Class', fontsize=VIZ_CONFIG["title_size"])
    ax4.set_xticks(range(len(class_names)))
    ax4.set_xticklabels(class_names, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    semantic_path = save_path / "semantic_breakdown.png"
    plt.savefig(semantic_path, dpi=VIZ_CONFIG["dpi"], bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved semantic breakdown plot to {semantic_path}")
    return str(semantic_path)


def analyze_individual_clusters(clusters: List[ObjectCluster], 
                               save_dir: str = "cluster_analysis/") -> Dict[int, str]:
    """
    Detailed analysis of each cluster.
    
    Args:
        clusters: List of ObjectCluster objects
        save_dir: Directory to save individual cluster analyses
        
    Returns:
        Dictionary mapping cluster IDs to file paths
    """
    save_path = Path(save_dir) / "individual_clusters"
    save_path.mkdir(parents=True, exist_ok=True)
    
    if not clusters:
        logger.warning("No clusters to analyze individually")
        return {}
    
    logger.info(f"Creating individual cluster analysis for {len(clusters)} clusters")
    
    saved_files = {}
    
    for cluster in clusters:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract cluster instance data
        instance_data = []
        for instance in cluster.instances:
            instance_data.append({
                'keyframe': instance.keyframe_idx,
                'x': instance.centroid_3d[0],
                'y': instance.centroid_3d[1],
                'z': instance.centroid_3d[2],
                'confidence': instance.confidence,
                'num_points': instance.num_points,
                'instance_id': instance.global_id
            })
        
        df = pd.DataFrame(instance_data)
        
        # Plot 1: Spatial distribution of instances
        scatter1 = ax1.scatter(
            df['x'], 
            df['y'],
            s=df['num_points'] * 0.1,
            c=df['keyframe'],
            cmap='viridis',
            alpha=0.8,
            edgecolors='black',
            linewidth=0.5
        )
        
        ax1.set_xlabel('X (meters)')
        ax1.set_ylabel('Y (meters)')
        ax1.set_title(f'Spatial Distribution - {cluster.label}', fontsize=VIZ_CONFIG["title_size"])
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', adjustable='box')
        plt.colorbar(scatter1, ax=ax1, label='Keyframe')
        
        # Add centroid
        ax1.scatter(
            cluster.avg_centroid[0], 
            cluster.avg_centroid[1],
            s=200, 
            c='red', 
            marker='x', 
            linewidth=3,
            label='Cluster Centroid'
        )
        ax1.legend()
        
        # Plot 2: Temporal activity
        keyframe_counts = df['keyframe'].value_counts().sort_index()
        
        ax2.bar(
            keyframe_counts.index, 
            keyframe_counts.values,
            alpha=0.8,
            edgecolor='black',
            linewidth=0.5
        )
        
        ax2.set_xlabel('Keyframe Index')
        ax2.set_ylabel('Number of Instances')
        ax2.set_title(f'Temporal Activity - {cluster.label}', fontsize=VIZ_CONFIG["title_size"])
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Confidence and point count evolution
        ax3_twin = ax3.twinx()
        
        line1 = ax3.plot(
            df['keyframe'], 
            df['confidence'], 
            'bo-', 
            label='Confidence',
            linewidth=2,
            markersize=6
        )
        
        line2 = ax3_twin.plot(
            df['keyframe'], 
            df['num_points'], 
            'ro-', 
            label='Point Count',
            linewidth=2,
            markersize=6
        )
        
        ax3.set_xlabel('Keyframe Index')
        ax3.set_ylabel('Detection Confidence', color='blue')
        ax3_twin.set_ylabel('Number of Points', color='red')
        ax3.set_title(f'Quality Metrics - {cluster.label}', fontsize=VIZ_CONFIG["title_size"])
        ax3.grid(True, alpha=0.3)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='upper left')
        
        # Plot 4: Statistical summary table
        ax4.axis('off')
        
        # Calculate cluster statistics
        stats_data = [
            ['Cluster ID', cluster.cluster_id],
            ['Cluster Label', cluster.label],
            ['Total Instances', len(cluster.instances)],
            ['Total Points', cluster.total_points],
            ['Avg Points/Instance', f"{cluster.total_points / len(cluster.instances):.1f}"],
            ['Keyframes Span', f"{min(cluster.keyframes)}-{max(cluster.keyframes)}"],
            ['Temporal Duration', max(cluster.keyframes) - min(cluster.keyframes)],
            ['Spatial Extent', f"{cluster.get_spatial_extent():.2f}m"],
            ['Avg Confidence', f"{df['confidence'].mean():.3f}"],
            ['Confidence Std', f"{df['confidence'].std():.3f}"],
            ['Centroid X', f"{cluster.avg_centroid[0]:.2f}m"],
            ['Centroid Y', f"{cluster.avg_centroid[1]:.2f}m"],
            ['Centroid Z', f"{cluster.avg_centroid[2]:.2f}m"],
        ]
        
        # Create table
        table = ax4.table(
            cellText=stats_data,
            colLabels=['Metric', 'Value'],
            cellLoc='left',
            loc='center',
            colWidths=[0.4, 0.6]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        ax4.set_title(f'Cluster Statistics - {cluster.label}', 
                     fontsize=VIZ_CONFIG["title_size"], pad=20)
        
        plt.tight_layout()
        
        # Save individual cluster plot
        cluster_path = save_path / f"cluster_{cluster.cluster_id:03d}_{cluster.label.replace(' ', '_')}.png"
        plt.savefig(cluster_path, dpi=VIZ_CONFIG["dpi"], bbox_inches='tight')
        plt.close()
        
        saved_files[cluster.cluster_id] = str(cluster_path)
    
    logger.info(f"Saved {len(saved_files)} individual cluster analysis files to {save_path}")
    return saved_files


def compare_clustering_results(clusters_list: List[List[ObjectCluster]], 
                              labels: List[str],
                              save_dir: str = "cluster_analysis/") -> str:
    """
    Compare different clustering parameter sets.
    
    Args:
        clusters_list: List of cluster results from different parameter sets
        labels: List of labels describing each parameter set
        save_dir: Directory to save comparison plots
        
    Returns:
        Path to saved comparison plot
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    if not clusters_list or not labels:
        logger.warning("No clustering results to compare")
        return ""
    
    logger.info(f"Comparing {len(clusters_list)} different clustering results")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract comparison metrics
    comparison_data = []
    for i, (clusters, label) in enumerate(zip(clusters_list, labels)):
        metrics = ClusteringMetrics(clusters)
        
        comparison_data.append({
            'parameter_set': label,
            'total_clusters': metrics.total_clusters,
            'total_instances': metrics.total_instances,
            'avg_instances_per_cluster': metrics.total_instances / max(metrics.total_clusters, 1),
            'semantic_classes': len(metrics.semantic_classes),
            'temporal_span': metrics.temporal_span[1] - metrics.temporal_span[0],
            'spatial_extent': metrics.spatial_extent,
            'avg_spatial_extent': np.mean([c.get_spatial_extent() for c in clusters]) if clusters else 0
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Plot 1: Number of clusters comparison
    colors = MATPLOTLIB_CLUSTER_COLORS(np.linspace(0, 1, len(df)))
    bars1 = ax1.bar(
        df['parameter_set'], 
        df['total_clusters'],
        color=colors,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5
    )
    
    ax1.set_xlabel('Parameter Set')
    ax1.set_ylabel('Total Clusters')
    ax1.set_title('Cluster Count Comparison', fontsize=VIZ_CONFIG["title_size"])
    ax1.grid(True, alpha=0.3, axis='y')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add value labels
    for bar, count in zip(bars1, df['total_clusters']):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width()/2., 
            height + 0.5,
            str(int(count)), 
            ha='center', 
            va='bottom'
        )
    
    # Plot 2: Average instances per cluster
    bars2 = ax2.bar(
        df['parameter_set'], 
        df['avg_instances_per_cluster'],
        color=colors,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5
    )
    
    ax2.set_xlabel('Parameter Set')
    ax2.set_ylabel('Average Instances per Cluster')
    ax2.set_title('Cluster Quality Comparison', fontsize=VIZ_CONFIG["title_size"])
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 3: Spatial extent comparison
    bars3 = ax3.bar(
        df['parameter_set'], 
        df['avg_spatial_extent'],
        color=colors,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5
    )
    
    ax3.set_xlabel('Parameter Set')
    ax3.set_ylabel('Average Spatial Extent (meters)')
    ax3.set_title('Spatial Coherence Comparison', fontsize=VIZ_CONFIG["title_size"])
    ax3.grid(True, alpha=0.3, axis='y')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 4: Multi-metric radar chart
    categories = ['Total Clusters', 'Avg Instances/Cluster', 'Semantic Classes', 'Avg Spatial Extent']
    
    # Normalize metrics for radar plot
    metrics_normalized = []
    for _, row in df.iterrows():
        normalized = [
            row['total_clusters'] / df['total_clusters'].max() if df['total_clusters'].max() > 0 else 0,
            row['avg_instances_per_cluster'] / df['avg_instances_per_cluster'].max() if df['avg_instances_per_cluster'].max() > 0 else 0,
            row['semantic_classes'] / df['semantic_classes'].max() if df['semantic_classes'].max() > 0 else 0,
            row['avg_spatial_extent'] / df['avg_spatial_extent'].max() if df['avg_spatial_extent'].max() > 0 else 0
        ]
        metrics_normalized.append(normalized)
    
    # Create radar plot
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
    
    ax4 = plt.subplot(224, projection='polar')
    
    radar_colors = colors
    for i, (values, label) in enumerate(zip(metrics_normalized, labels)):
        values = np.concatenate((values, [values[0]]))  # Complete the circle
        color = radar_colors[i]
        ax4.plot(
            angles, 
            values, 
            'o-', 
            linewidth=2, 
            label=label, 
            color=color
        )
        ax4.fill(
            angles, 
            values, 
            alpha=0.1, 
            color=color
        )
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(categories)
    ax4.set_ylim(0, 1)
    ax4.set_title('Multi-Metric Comparison', fontsize=VIZ_CONFIG["title_size"], pad=20)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax4.grid(True)
    
    plt.tight_layout()
    
    # Save comparison plot
    comparison_path = save_path / "clustering_comparison.png"
    plt.savefig(comparison_path, dpi=VIZ_CONFIG["dpi"], bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved clustering comparison plot to {comparison_path}")
    return str(comparison_path)


def generate_html_report(clusters: List[ObjectCluster], 
                        metrics: Optional[ClusteringMetrics] = None,
                        save_dir: str = "cluster_analysis/") -> str:
    """
    Create comprehensive HTML report.
    
    Args:
        clusters: List of ObjectCluster objects
        metrics: Pre-computed metrics (optional)
        save_dir: Directory to save HTML report
        
    Returns:
        Path to saved HTML report
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    if not clusters:
        logger.warning("No clusters to generate report for")
        return ""
    
    logger.info(f"Generating comprehensive HTML report for {len(clusters)} clusters")
    
    # Compute metrics if not provided
    if metrics is None:
        metrics = ClusteringMetrics(clusters)
    
    # Generate all visualizations
    viz_files = visualize_clusters_3d(clusters, save_dir, show_interactive=False)
    temporal_file = plot_temporal_distribution(clusters, save_dir)
    spatial_file = plot_spatial_distribution(clusters, save_dir)
    size_file = plot_size_distributions(clusters, save_dir)
    semantic_file = plot_semantic_breakdown(clusters, save_dir)
    individual_files = analyze_individual_clusters(clusters, save_dir)
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Clustering Analysis Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
                text-align: center;
            }}
            .header h1 {{
                margin: 0;
                font-size: 2.5em;
            }}
            .header p {{
                margin: 10px 0 0 0;
                font-size: 1.1em;
                opacity: 0.9;
            }}
            .summary {{
                background: white;
                padding: 25px;
                border-radius: 10px;
                margin-bottom: 30px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }}
            .metric-card {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                border-left: 4px solid #667eea;
            }}
            .metric-card h3 {{
                margin: 0 0 10px 0;
                color: #667eea;
                font-size: 2em;
            }}
            .metric-card p {{
                margin: 0;
                color: #666;
            }}
            .section {{
                background: white;
                padding: 25px;
                border-radius: 10px;
                margin-bottom: 30px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .section h2 {{
                color: #333;
                border-bottom: 2px solid #667eea;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }}
            .viz-container {{
                text-align: center;
                margin: 20px 0;
            }}
            .viz-container img {{
                max-width: 100%;
                height: auto;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .iframe-container {{
                text-align: center;
                margin: 20px 0;
            }}
            .iframe-container iframe {{
                border: none;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .cluster-table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }}
            .cluster-table th,
            .cluster-table td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            .cluster-table th {{
                background-color: #667eea;
                color: white;
            }}
            .cluster-table tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            .recommendations {{
                background: #e7f3ff;
                border-left: 4px solid #2196F3;
                padding: 20px;
                margin: 20px 0;
                border-radius: 5px;
            }}
            .issues {{
                background: #fff3e0;
                border-left: 4px solid #ff9800;
                padding: 20px;
                margin: 20px 0;
                border-radius: 5px;
            }}
            .footer {{
                text-align: center;
                color: #666;
                margin-top: 50px;
                padding: 20px;
                border-top: 1px solid #ddd;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🎯 Clustering Analysis Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="summary">
            <h2>📊 Executive Summary</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>{metrics.total_clusters}</h3>
                    <p>Total Clusters</p>
                </div>
                <div class="metric-card">
                    <h3>{metrics.total_instances}</h3>
                    <p>Total Instances</p>
                </div>
                <div class="metric-card">
                    <h3>{len(metrics.semantic_classes)}</h3>
                    <p>Semantic Classes</p>
                </div>
                <div class="metric-card">
                    <h3>{metrics.spatial_extent:.2f}m</h3>
                    <p>Max Spatial Extent</p>
                </div>
                <div class="metric-card">
                    <h3>{metrics.temporal_span[1] - metrics.temporal_span[0]}</h3>
                    <p>Temporal Span</p>
                </div>
                <div class="metric-card">
                    <h3>{metrics.total_instances / max(metrics.total_clusters, 1):.1f}</h3>
                    <p>Avg Instances/Cluster</p>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>🌐 Interactive 3D Visualizations</h2>
            <p>Explore your clustering results in interactive 3D space. Use mouse to rotate, zoom, and hover for details.</p>
    """
    
    # Add interactive 3D visualizations
    for viz_name, viz_path in viz_files.items():
        if viz_path.endswith('.html'):
            html_content += f"""
            <div class="iframe-container">
                <h3>{viz_name.replace('_', ' ').title()}</h3>
                <iframe src="{Path(viz_path).name}" width="900" height="600"></iframe>
            </div>
            """
    
    html_content += """
        </div>
        
        <div class="section">
            <h2>📈 Static Analysis Plots</h2>
    """
    
    # Add static plots
    analysis_plots = [
        (temporal_file, "Temporal Distribution Analysis"),
        (spatial_file, "Spatial Distribution Analysis"), 
        (size_file, "Size Distribution Analysis"),
        (semantic_file, "Semantic Class Breakdown")
    ]
    
    for plot_path, plot_title in analysis_plots:
        if plot_path and Path(plot_path).exists():
            html_content += f"""
            <div class="viz-container">
                <h3>{plot_title}</h3>
                <img src="{Path(plot_path).name}" alt="{plot_title}">
            </div>
            """
    
    # Add cluster details table
    html_content += f"""
        </div>
        
        <div class="section">
            <h2>📋 Cluster Details</h2>
            <table class="cluster-table">
                <thead>
                    <tr>
                        <th>Cluster ID</th>
                        <th>Label</th>
                        <th>Instances</th>
                        <th>Keyframes</th>
                        <th>Temporal Span</th>
                        <th>Spatial Extent</th>
                        <th>Total Points</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    for cluster in sorted(clusters, key=lambda x: x.cluster_id):
        temporal_span = cluster.get_temporal_span()
        keyframes_str = f"{min(cluster.keyframes)}-{max(cluster.keyframes)}" if cluster.keyframes else "N/A"
        
        html_content += f"""
                    <tr>
                        <td>{cluster.cluster_id}</td>
                        <td>{cluster.label}</td>
                        <td>{len(cluster.instances)}</td>
                        <td>{keyframes_str}</td>
                        <td>{temporal_span[1] - temporal_span[0]}</td>
                        <td>{cluster.get_spatial_extent():.2f}m</td>
                        <td>{cluster.total_points:,}</td>
                    </tr>
        """
    
    # Add recommendations and issues
    html_content += """
                </tbody>
            </table>
        </div>
        
        <div class="section">
            <h2>💡 Analysis & Recommendations</h2>
    """
    
    # Identify potential issues and recommendations
    issues = []
    recommendations = []
    
    # Check for oversized clusters
    oversized_clusters = [c for c in clusters if len(c.instances) > 10]
    if oversized_clusters:
        issues.append(f"Found {len(oversized_clusters)} clusters with >10 instances - may indicate under-clustering")
        recommendations.append("Consider reducing spatial_threshold or increasing min_samples parameters")
    
    # Check for temporal gaps
    large_temporal_gaps = [c for c in clusters if max([
        c.keyframes[i+1] - c.keyframes[i] for i in range(len(c.keyframes)-1)
    ] if len(c.keyframes) > 1 else [0]) > 15]
    
    if large_temporal_gaps:
        issues.append(f"Found {len(large_temporal_gaps)} clusters with large temporal gaps")
        recommendations.append("Consider reducing temporal_threshold parameter")
    
    # Check for spatial extent issues
    large_spatial_extent = [c for c in clusters if c.get_spatial_extent() > 3.0]
    if large_spatial_extent:
        issues.append(f"Found {len(large_spatial_extent)} clusters with large spatial extent (>3m)")
        recommendations.append("Consider reducing movement_threshold or improving spatial clustering")
    
    # Check singleton clusters
    singleton_clusters = [c for c in clusters if len(c.instances) == 1]
    if len(singleton_clusters) > len(clusters) * 0.5:
        issues.append(f"High number of singleton clusters ({len(singleton_clusters)}/{len(clusters)})")
        recommendations.append("Consider increasing spatial_threshold or reducing min_samples")
    
    if issues:
        html_content += '<div class="issues"><h3>⚠️ Identified Issues</h3><ul>'
        for issue in issues:
            html_content += f'<li>{issue}</li>'
        html_content += '</ul></div>'
    
    if recommendations:
        html_content += '<div class="recommendations"><h3>🎯 Recommendations</h3><ul>'
        for rec in recommendations:
            html_content += f'<li>{rec}</li>'
        html_content += '</ul></div>'
    
    # Add semantic class breakdown
    html_content += f"""
            <h3>📊 Semantic Class Distribution</h3>
            <ul>
    """
    
    for class_name, count in metrics.semantic_classes.items():
        percentage = (count / metrics.total_clusters) * 100
        html_content += f"<li><strong>{class_name}</strong>: {count} clusters ({percentage:.1f}%)</li>"
    
    html_content += """
            </ul>
        </div>
        
        <div class="footer">
            <p>Report generated by MASt3R SLAM Clustering Visualization Module</p>
            <p>For detailed individual cluster analysis, check the individual_clusters/ folder</p>
        </div>
    </body>
    </html>
    """
    
    # Save HTML report
    report_path = save_path / "clustering_analysis_report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # Copy visualization files to report directory for relative linking
    for viz_path in viz_files.values():
        if viz_path.endswith('.html'):
            import shutil
            src_path = Path(viz_path)
            dest_path = save_path / src_path.name
            if src_path != dest_path:  # Only copy if different paths
                shutil.copy2(src_path, dest_path)
    
    logger.info(f"Generated comprehensive HTML report: {report_path}")
    return str(report_path)


def run_complete_analysis(clusters: List[ObjectCluster], 
                         save_dir: str = "cluster_analysis/",
                         show_interactive: bool = True) -> Dict[str, Any]:
    """
    Run complete clustering analysis pipeline.
    
    Args:
        clusters: List of ObjectCluster objects
        save_dir: Directory to save all analysis outputs
        show_interactive: Whether to show interactive plots
        
    Returns:
        Dictionary containing all generated files and metrics
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting complete clustering analysis for {len(clusters)} clusters")
    logger.info(f"Output directory: {save_path.absolute()}")
    
    if not clusters:
        logger.warning("No clusters provided for analysis")
        return {}
    
    # Compute metrics
    metrics = ClusteringMetrics(clusters)
    
    # Run all analysis functions
    results = {
        'metrics': metrics,
        'save_directory': str(save_path.absolute()),
        'generated_files': {}
    }
    
    try:
        # 3D visualizations
        logger.info("Generating 3D visualizations...")
        viz_files = visualize_clusters_3d(clusters, save_dir, show_interactive)
        results['generated_files']['3d_visualizations'] = viz_files
        
        # Temporal analysis
        logger.info("Generating temporal analysis...")
        temporal_file = plot_temporal_distribution(clusters, save_dir)
        results['generated_files']['temporal_analysis'] = temporal_file
        
        # Spatial analysis  
        logger.info("Generating spatial analysis...")
        spatial_file = plot_spatial_distribution(clusters, save_dir)
        results['generated_files']['spatial_analysis'] = spatial_file
        
        # Size analysis
        logger.info("Generating size analysis...")
        size_file = plot_size_distributions(clusters, save_dir)
        results['generated_files']['size_analysis'] = size_file
        
        # Semantic analysis
        logger.info("Generating semantic analysis...")
        semantic_file = plot_semantic_breakdown(clusters, save_dir)
        results['generated_files']['semantic_analysis'] = semantic_file
        
        # Individual cluster analysis
        logger.info("Generating individual cluster analyses...")
        individual_files = analyze_individual_clusters(clusters, save_dir)
        results['generated_files']['individual_analyses'] = individual_files
        
        # HTML report
        logger.info("Generating comprehensive HTML report...")
        report_file = generate_html_report(clusters, metrics, save_dir)
        results['generated_files']['html_report'] = report_file
        
        # Save clustering data as JSON
        logger.info("Saving clustering data...")
        data_file = save_path / "clustering_data.json"
        cluster_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_clusters': len(clusters),
                'total_instances': sum(len(c.instances) for c in clusters),
                'clustering_config': get_clustering_config()
            },
            'clusters': []
        }
        
        for cluster in clusters:
            cluster_dict = {
                'cluster_id': cluster.cluster_id,
                'label': cluster.label,
                'avg_centroid': cluster.avg_centroid.tolist(),
                'keyframes': cluster.keyframes,
                'total_points': cluster.total_points,
                'spatial_extent': cluster.get_spatial_extent(),
                'temporal_span': cluster.get_temporal_span(),
                'instances': []
            }
            
            for instance in cluster.instances:
                instance_dict = {
                    'global_id': instance.global_id,
                    'local_id': instance.local_id,
                    'keyframe_idx': instance.keyframe_idx,
                    'label': instance.label,
                    'confidence': instance.confidence,
                    'centroid_3d': instance.centroid_3d.tolist(),
                    'num_points': instance.num_points
                }
                cluster_dict['instances'].append(instance_dict)
            
            cluster_data['clusters'].append(cluster_dict)
        
        with open(data_file, 'w') as f:
            json.dump(cluster_data, f, indent=2)
        
        results['generated_files']['data_export'] = str(data_file)
        
        logger.info("Complete clustering analysis finished successfully!")
        logger.info(f"Generated files:")
        for category, files in results['generated_files'].items():
            if isinstance(files, dict):
                logger.info(f"  {category}: {len(files)} files")
            elif isinstance(files, str):
                logger.info(f"  {category}: {Path(files).name}")
        
        logger.info(f"View the complete analysis report: {report_file}")
        
    except Exception as e:
        logger.error(f"Error during clustering analysis: {e}")
        results['error'] = str(e)
    
    return results


if __name__ == "__main__":
    # Demo usage
    from .object_clustering import ObjectInstance, hybrid_cluster_objects
    
    # Create sample data for testing
    logger.info("Creating sample clustering data for demo...")
    
    sample_instances = [
        ObjectInstance(
            global_id=1, local_id=1, keyframe_idx=0, label="table",
            confidence=0.9, centroid_3d=np.array([1.0, 2.0, 0.5]),
            num_points=100, point_indices=np.arange(100)
        ),
        ObjectInstance(
            global_id=10001, local_id=1, keyframe_idx=1, label="table", 
            confidence=0.8, centroid_3d=np.array([1.1, 2.1, 0.5]),
            num_points=120, point_indices=np.arange(120)
        ),
        ObjectInstance(
            global_id=20001, local_id=1, keyframe_idx=2, label="chair",
            confidence=0.7, centroid_3d=np.array([3.0, 1.0, 0.4]),
            num_points=80, point_indices=np.arange(80)
        ),
        ObjectInstance(
            global_id=30001, local_id=1, keyframe_idx=3, label="chair",
            confidence=0.6, centroid_3d=np.array([3.2, 1.1, 0.4]),
            num_points=90, point_indices=np.arange(90)
        ),
    ]
    
    # Run clustering
    clusters = hybrid_cluster_objects(sample_instances)
    
    # Run complete analysis
    results = run_complete_analysis(
        clusters, 
        save_dir="demo_cluster_analysis/",
        show_interactive=False
    )
    
    print(f"\nDemo analysis complete!")
    print(f"Results saved to: {results['save_directory']}")
    print(f"View HTML report: {results['generated_files'].get('html_report', 'N/A')}")