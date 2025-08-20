#!/usr/bin/env python3
"""
Single script for complete 3D reconstruction evaluation.
Handles alignment, evaluation, and comparison in one command.
"""

import numpy as np
import open3d as o3d
import pandas as pd
from pathlib import Path
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
import argparse
import copy

def align_to_ground_truth(source_ply, target_ply, method='icp'):
    """Align source PLY to target PLY."""
    print(f"🔧 Aligning {source_ply.name} to ground truth...")
    
    source_pcd = o3d.io.read_point_cloud(str(source_ply))
    target_pcd = o3d.io.read_point_cloud(str(target_ply))
    
    # Downsample for faster processing
    source_down = source_pcd.voxel_down_sample(0.02)
    target_down = target_pcd.voxel_down_sample(0.02)
    
    if method == 'icp':
        # Estimate scale first
        source_points = np.asarray(source_down.points)
        target_points = np.asarray(target_down.points)
        
        source_extent = np.linalg.norm(source_points.max(0) - source_points.min(0))
        target_extent = np.linalg.norm(target_points.max(0) - target_points.min(0))
        scale = target_extent / source_extent
        
        # Apply scale
        source_scaled = copy.deepcopy(source_pcd)
        points = np.asarray(source_scaled.points)
        center = points.mean(0)
        source_scaled.points = o3d.utility.Vector3dVector((points - center) * scale + center)
        
        # Run ICP
        source_scaled_down = source_scaled.voxel_down_sample(0.02)
        source_scaled_down.estimate_normals()
        target_down.estimate_normals()
        
        icp_result = o3d.pipelines.registration.registration_icp(
            source_scaled_down, target_down, 0.1, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        
        source_scaled.transform(icp_result.transformation)
        print(f"  ✅ Scale: {scale:.3f}, ICP fitness: {icp_result.fitness:.3f}")
        return source_scaled
    
    return source_pcd

def compute_metrics(pred_pcd, gt_pcd):
    """Compute all evaluation metrics."""
    print("📊 Computing evaluation metrics...")
    
    pred_points = np.asarray(pred_pcd.points)
    gt_points = np.asarray(gt_pcd.points)
    
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
    thresholds = [0.01, 0.02, 0.05, 0.10]
    f_scores = {}
    
    for thresh in thresholds:
        precision = np.mean(dist_pred_to_gt < thresh)
        recall = np.mean(dist_gt_to_pred < thresh)
        if precision + recall > 0:
            f_score = 2 * precision * recall / (precision + recall)
        else:
            f_score = 0
        f_scores[f'F@{thresh:.3f}'] = f_score
    
    return {
        'points': len(pred_points),
        'chamfer': chamfer,
        'accuracy': accuracy,
        'completeness': completeness,
        'hausdorff': hausdorff,
        'f_scores': f_scores
    }

def evaluate_single_method(pred_ply, gt_ply, method_name):
    """Evaluate single method against ground truth."""
    print(f"\n{'='*60}")
    print(f"EVALUATING: {method_name}")
    print(f"{'='*60}")
    
    # Align
    aligned_pcd = align_to_ground_truth(pred_ply, gt_ply)
    
    # Load ground truth
    gt_pcd = o3d.io.read_point_cloud(str(gt_ply))
    
    # Compute metrics
    metrics = compute_metrics(aligned_pcd, gt_pcd)
    
    print(f"\n📈 Results for {method_name}:")
    print(f"  Points: {metrics['points']:,}")
    print(f"  Chamfer: {metrics['chamfer']:.4f} m")
    print(f"  Accuracy: {metrics['accuracy']:.4f} m")
    print(f"  Completeness: {metrics['completeness']:.4f} m")
    print(f"  Hausdorff: {metrics['hausdorff']:.4f} m")
    for thresh, score in metrics['f_scores'].items():
        print(f"  {thresh}: {score:.3f}")
    
    return metrics, aligned_pcd

def create_comparison_table(results, output_dir):
    """Create comparison table in multiple formats."""
    print(f"\n📋 Creating comparison tables...")
    
    # Create DataFrame
    rows = []
    for method_name, metrics in results.items():
        row = {
            'Method': method_name,
            'Points': f"{metrics['points']:,}",
            'Chamfer (m)': f"{metrics['chamfer']:.4f}",
            'Accuracy (m)': f"{metrics['accuracy']:.4f}",
            'Completeness (m)': f"{metrics['completeness']:.4f}",
            'Hausdorff (m)': f"{metrics['hausdorff']:.4f}",
        }
        
        # Add F-scores
        for thresh, score in metrics['f_scores'].items():
            row[thresh] = f"{score:.3f}"
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Print to console
    print(f"\n{'='*80}")
    print("🏆 FINAL COMPARISON")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    
    # Save files
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # CSV
    df.to_csv(output_path / "results.csv", index=False)
    
    # Markdown
    with open(output_path / "results.md", 'w') as f:
        f.write("# 3D Reconstruction Comparison\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n**Note:** Lower distance metrics are better. Higher F-scores are better.\n")
    
    # LaTeX
    with open(output_path / "results.tex", 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{3D Reconstruction Quality Comparison}\n")
        f.write("\\begin{tabular}{" + "l" + "c" * (len(df.columns) - 1) + "}\n")
        f.write("\\hline\\hline\n")
        
        # Header
        f.write(" & ".join(df.columns) + " \\\\\n")
        f.write("\\hline\n")
        
        # Data
        for _, row in df.iterrows():
            f.write(" & ".join(str(val) for val in row.values) + " \\\\\n")
        
        f.write("\\hline\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"\n💾 Results saved to: {output_path}")
    print(f"  • CSV: results.csv")
    print(f"  • Markdown: results.md") 
    print(f"  • LaTeX: results.tex")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Complete 3D reconstruction evaluation")
    parser.add_argument("--gt", required=True, help="Ground truth PLY file")
    parser.add_argument("--monocular2map", required=True, help="Monocular2Map method's PLY file")
    parser.add_argument("--hovsg", help="HOV-SG PLY file (optional)")
    parser.add_argument("--output", default="./evaluation_results", help="Output directory")
    
    args = parser.parse_args()
    
    print("🚀 3D RECONSTRUCTION EVALUATION")
    print("="*60)
    print(f"Ground Truth: {args.gt}")
    print(f"Monocular2Map: {args.monocular2map}")
    if args.hovsg:
        print(f"HOV-SG: {args.hovsg}")
    print(f"Output: {args.output}")
    
    # Evaluate methods
    results = {}
    aligned_plys = {}
    
    # Evaluate Monocular2Map method
    metrics, aligned_pcd = evaluate_single_method(
        Path(args.monocular2map), Path(args.gt), "Monocular2Map"
    )
    results["Monocular2Map"] = metrics
    aligned_plys["Monocular2Map"] = aligned_pcd
    
    # Evaluate HOV-SG if provided
    if args.hovsg:
        metrics, aligned_pcd = evaluate_single_method(
            Path(args.hovsg), Path(args.gt), "HOV-SG"
        )
        results["HOV-SG"] = metrics
        aligned_plys["HOV-SG"] = aligned_pcd
    
    # Create comparison table
    df = create_comparison_table(results, args.output)
    
    # Save aligned PLYs
    aligned_dir = Path(args.output) / "aligned"
    aligned_dir.mkdir(exist_ok=True)
    
    for method_name, pcd in aligned_plys.items():
        output_file = aligned_dir / f"{method_name.lower().replace(' ', '_')}_aligned.ply"
        o3d.io.write_point_cloud(str(output_file), pcd)
        print(f"💾 Saved aligned PLY: {output_file}")
    
    # Summary
    print(f"\n{'='*80}")
    print("✅ EVALUATION COMPLETE!")
    print(f"{'='*80}")
    
    if len(results) > 1:
        # Find best method for each metric
        print("\n🏆 Best Performance:")
        
        metrics_to_check = ['chamfer', 'accuracy', 'completeness', 'hausdorff']
        for metric in metrics_to_check:
            best_method = min(results.keys(), key=lambda x: results[x][metric])
            best_value = results[best_method][metric]
            print(f"  • {metric.title()}: {best_method} ({best_value:.4f})")
        
        # F-score comparison (higher is better)
        for thresh in results[list(results.keys())[0]]['f_scores'].keys():
            best_method = max(results.keys(), key=lambda x: results[x]['f_scores'][thresh])
            best_value = results[best_method]['f_scores'][thresh]
            print(f"  • {thresh}: {best_method} ({best_value:.3f})")
    
    print(f"\n📁 All results saved to: {args.output}")

if __name__ == "__main__":
    main()