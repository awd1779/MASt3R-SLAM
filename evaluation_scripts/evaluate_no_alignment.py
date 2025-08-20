#!/usr/bin/env python3
"""
3D reconstruction evaluation WITHOUT alignment.
Compares point clouds in their original coordinate systems.
"""

import numpy as np
import open3d as o3d
import pandas as pd
from pathlib import Path
from scipy.spatial import KDTree
import argparse

def compute_metrics(pred_pcd, gt_pcd):
    """Compute all evaluation metrics without any alignment."""
    print("📊 Computing evaluation metrics (NO ALIGNMENT)...")
    
    pred_points = np.asarray(pred_pcd.points)
    gt_points = np.asarray(gt_pcd.points)
    
    print(f"  Predicted points: {len(pred_points):,}")
    print(f"  GT points: {len(gt_points):,}")
    
    # Check point cloud extents
    pred_min, pred_max = pred_points.min(0), pred_points.max(0)
    gt_min, gt_max = gt_points.min(0), gt_points.max(0)
    
    pred_extent = np.linalg.norm(pred_max - pred_min)
    gt_extent = np.linalg.norm(gt_max - gt_min)
    
    print(f"  Predicted extent: {pred_extent:.3f} m")
    print(f"  GT extent: {gt_extent:.3f} m")
    print(f"  Scale ratio: {pred_extent/gt_extent:.3f}")
    
    print(f"  Predicted center: [{pred_points.mean(0)[0]:.3f}, {pred_points.mean(0)[1]:.3f}, {pred_points.mean(0)[2]:.3f}]")
    print(f"  GT center: [{gt_points.mean(0)[0]:.3f}, {gt_points.mean(0)[1]:.3f}, {gt_points.mean(0)[2]:.3f}]")
    
    # Build KD-trees
    pred_tree = KDTree(pred_points)
    gt_tree = KDTree(gt_points)
    
    # Chamfer distance components
    dist_pred_to_gt, _ = gt_tree.query(pred_points)
    dist_gt_to_pred, _ = pred_tree.query(gt_points)
    
    accuracy = np.mean(dist_pred_to_gt)
    completeness = np.mean(dist_gt_to_pred)
    chamfer = (accuracy + completeness) / 2
    hausdorff = max(np.max(dist_pred_to_gt), np.max(dist_gt_to_pred))
    
    # F-scores at different thresholds
    thresholds = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
    f_scores = {}
    
    for thresh in thresholds:
        precision = np.mean(dist_pred_to_gt < thresh)
        recall = np.mean(dist_gt_to_pred < thresh)
        if precision + recall > 0:
            f_score = 2 * precision * recall / (precision + recall)
        else:
            f_score = 0
        f_scores[f'F@{thresh:.3f}'] = f_score
    
    # Distance statistics
    stats = {
        'pred_to_gt': {
            'mean': np.mean(dist_pred_to_gt),
            'median': np.median(dist_pred_to_gt),
            'std': np.std(dist_pred_to_gt),
            'min': np.min(dist_pred_to_gt),
            'max': np.max(dist_pred_to_gt),
            'p95': np.percentile(dist_pred_to_gt, 95)
        },
        'gt_to_pred': {
            'mean': np.mean(dist_gt_to_pred),
            'median': np.median(dist_gt_to_pred),
            'std': np.std(dist_gt_to_pred),
            'min': np.min(dist_gt_to_pred),
            'max': np.max(dist_gt_to_pred),
            'p95': np.percentile(dist_gt_to_pred, 95)
        }
    }
    
    return {
        'points': len(pred_points),
        'gt_points': len(gt_points),
        'pred_extent': pred_extent,
        'gt_extent': gt_extent,
        'scale_ratio': pred_extent / gt_extent,
        'pred_center': pred_points.mean(0),
        'gt_center': gt_points.mean(0),
        'chamfer': chamfer,
        'accuracy': accuracy,
        'completeness': completeness,
        'hausdorff': hausdorff,
        'f_scores': f_scores,
        'distance_stats': stats
    }

def evaluate_single_method(pred_ply, gt_ply, method_name):
    """Evaluate single method against ground truth."""
    print(f"\n{'='*60}")
    print(f"EVALUATING: {method_name} (NO ALIGNMENT)")
    print(f"{'='*60}")
    
    # Load point clouds
    print(f"Loading {pred_ply.name}...")
    pred_pcd = o3d.io.read_point_cloud(str(pred_ply))
    
    print(f"Loading {gt_ply.name}...")
    gt_pcd = o3d.io.read_point_cloud(str(gt_ply))
    
    # Compute metrics
    metrics = compute_metrics(pred_pcd, gt_pcd)
    
    # Print results
    print(f"\n📈 Results for {method_name}:")
    print(f"  Points: {metrics['points']:,} (GT: {metrics['gt_points']:,})")
    print(f"  Extent: {metrics['pred_extent']:.3f} m (GT: {metrics['gt_extent']:.3f} m)")
    print(f"  Scale Ratio: {metrics['scale_ratio']:.3f}")
    print(f"  Chamfer: {metrics['chamfer']:.4f} m")
    print(f"  Accuracy: {metrics['accuracy']:.4f} m")
    print(f"  Completeness: {metrics['completeness']:.4f} m")
    print(f"  Hausdorff: {metrics['hausdorff']:.4f} m")
    
    print(f"\n  F-scores:")
    for thresh, score in metrics['f_scores'].items():
        print(f"    {thresh}: {score:.3f}")
    
    print(f"\n  Distance Statistics:")
    print(f"    Pred→GT: mean={metrics['distance_stats']['pred_to_gt']['mean']:.4f}, "
          f"median={metrics['distance_stats']['pred_to_gt']['median']:.4f}, "
          f"95%={metrics['distance_stats']['pred_to_gt']['p95']:.4f}")
    print(f"    GT→Pred: mean={metrics['distance_stats']['gt_to_pred']['mean']:.4f}, "
          f"median={metrics['distance_stats']['gt_to_pred']['median']:.4f}, "
          f"95%={metrics['distance_stats']['gt_to_pred']['p95']:.4f}")
    
    return metrics, pred_pcd

def create_comparison_table(results, output_dir):
    """Create comparison table in multiple formats."""
    print(f"\n📋 Creating comparison tables...")
    
    # Create DataFrame
    rows = []
    for method_name, metrics in results.items():
        row = {
            'Method': method_name,
            'Points': f"{metrics['points']:,}",
            'GT Points': f"{metrics['gt_points']:,}",
            'Extent (m)': f"{metrics['pred_extent']:.3f}",
            'GT Extent (m)': f"{metrics['gt_extent']:.3f}",
            'Scale Ratio': f"{metrics['scale_ratio']:.3f}",
            'Chamfer (m)': f"{metrics['chamfer']:.4f}",
            'Accuracy (m)': f"{metrics['accuracy']:.4f}",
            'Completeness (m)': f"{metrics['completeness']:.4f}",
            'Hausdorff (m)': f"{metrics['hausdorff']:.4f}",
        }
        
        # Add key F-scores
        key_f_scores = ['F@0.010', 'F@0.020', 'F@0.050', 'F@0.100', 'F@0.200', 'F@0.500']
        for thresh in key_f_scores:
            if thresh in metrics['f_scores']:
                row[thresh] = f"{metrics['f_scores'][thresh]:.3f}"
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Print to console
    print(f"\n{'='*120}")
    print("🏆 FINAL COMPARISON (NO ALIGNMENT)")
    print(f"{'='*120}")
    print(df.to_string(index=False))
    
    # Save files
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # CSV
    df.to_csv(output_path / "no_alignment_results.csv", index=False)
    
    # Detailed analysis
    with open(output_path / "no_alignment_analysis.txt", 'w') as f:
        f.write("# 3D Reconstruction Evaluation (No Alignment)\n\n")
        
        for method_name, metrics in results.items():
            f.write(f"## {method_name}\n\n")
            f.write(f"Point Counts:\n")
            f.write(f"  Predicted: {metrics['points']:,}\n")
            f.write(f"  Ground Truth: {metrics['gt_points']:,}\n\n")
            
            f.write(f"Spatial Properties:\n")
            f.write(f"  Predicted Extent: {metrics['pred_extent']:.3f} m\n")
            f.write(f"  GT Extent: {metrics['gt_extent']:.3f} m\n")
            f.write(f"  Scale Ratio: {metrics['scale_ratio']:.3f}\n")
            f.write(f"  Predicted Center: [{metrics['pred_center'][0]:.3f}, {metrics['pred_center'][1]:.3f}, {metrics['pred_center'][2]:.3f}]\n")
            f.write(f"  GT Center: [{metrics['gt_center'][0]:.3f}, {metrics['gt_center'][1]:.3f}, {metrics['gt_center'][2]:.3f}]\n\n")
            
            f.write(f"Distance Metrics:\n")
            f.write(f"  Chamfer Distance: {metrics['chamfer']:.4f} m\n")
            f.write(f"  Accuracy: {metrics['accuracy']:.4f} m\n")
            f.write(f"  Completeness: {metrics['completeness']:.4f} m\n")
            f.write(f"  Hausdorff Distance: {metrics['hausdorff']:.4f} m\n\n")
            
            f.write(f"F-scores:\n")
            for thresh, score in metrics['f_scores'].items():
                f.write(f"  {thresh}: {score:.4f}\n")
            f.write("\n")
            
            f.write(f"Distance Statistics:\n")
            f.write(f"  Predicted→GT:\n")
            for stat_name, stat_val in metrics['distance_stats']['pred_to_gt'].items():
                f.write(f"    {stat_name}: {stat_val:.4f} m\n")
            f.write(f"  GT→Predicted:\n")
            for stat_name, stat_val in metrics['distance_stats']['gt_to_pred'].items():
                f.write(f"    {stat_name}: {stat_val:.4f} m\n")
            f.write("\n\n")
    
    # Markdown
    with open(output_path / "no_alignment_results.md", 'w') as f:
        f.write("# 3D Reconstruction Comparison (No Alignment)\n\n")
        f.write("## Results\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Key Insights\n\n")
        f.write("- **Scale Ratio**: How much larger/smaller predicted vs GT\n")
        f.write("- **Extent**: Diagonal of bounding box\n")
        f.write("- **No alignment applied** - shows raw reconstruction quality\n")
        f.write("- Lower distance metrics are better\n")
        f.write("- Higher F-scores are better\n")
    
    print(f"\n💾 Results saved to: {output_path}")
    print(f"  • CSV: no_alignment_results.csv")
    print(f"  • Markdown: no_alignment_results.md") 
    print(f"  • Analysis: no_alignment_analysis.txt")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="3D reconstruction evaluation without alignment")
    parser.add_argument("--gt", required=True, help="Ground truth PLY file")
    parser.add_argument("--monocular2map", required=True, help="Monocular2Map method's PLY file")
    parser.add_argument("--hovsg", help="HOV-SG PLY file (optional)")
    parser.add_argument("--output", default="./no_alignment_results", help="Output directory")
    
    args = parser.parse_args()
    
    print("🚀 3D RECONSTRUCTION EVALUATION (NO ALIGNMENT)")
    print("="*60)
    print(f"Ground Truth: {args.gt}")
    print(f"Monocular2Map: {args.monocular2map}")
    if args.hovsg:
        print(f"HOV-SG: {args.hovsg}")
    print(f"Output: {args.output}")
    print("\n⚠️  NO ALIGNMENT WILL BE APPLIED - RAW COMPARISON")
    
    # Evaluate methods
    results = {}
    
    # Evaluate Monocular2Map method
    metrics, pred_pcd = evaluate_single_method(
        Path(args.monocular2map), Path(args.gt), "Monocular2Map"
    )
    results["Monocular2Map"] = metrics
    
    # Evaluate HOV-SG if provided
    if args.hovsg:
        metrics, pred_pcd = evaluate_single_method(
            Path(args.hovsg), Path(args.gt), "HOV-SG"
        )
        results["HOV-SG"] = metrics
    
    # Create comparison table
    df = create_comparison_table(results, args.output)
    
    # Analysis summary
    print(f"\n{'='*80}")
    print("✅ NO-ALIGNMENT EVALUATION COMPLETE!")
    print(f"{'='*80}")
    
    # Show scale issues clearly
    for method_name, metrics in results.items():
        print(f"\n📐 {method_name} Scale Analysis:")
        scale_ratio = metrics['scale_ratio']
        if scale_ratio > 2.0:
            print(f"  ⚠️  Reconstruction is {scale_ratio:.1f}x LARGER than GT")
        elif scale_ratio < 0.5:
            print(f"  ⚠️  Reconstruction is {1/scale_ratio:.1f}x SMALLER than GT")
        else:
            print(f"  ✅ Scale looks reasonable ({scale_ratio:.2f}x)")
        
        # Check coordinate alignment
        pred_center = metrics['pred_center']
        gt_center = metrics['gt_center']
        center_dist = np.linalg.norm(pred_center - gt_center)
        print(f"  📍 Center distance: {center_dist:.3f} m")
        if center_dist > metrics['gt_extent']:
            print(f"  ⚠️  Centers are very far apart - coordinate system mismatch")
    
    print(f"\n📁 All results saved to: {args.output}")

if __name__ == "__main__":
    main()