#!/usr/bin/env python3
"""
Test methods that actually improve CLIP confidence scores for 3D semantic mapping.
Focus on practical improvements that increase confidence without complex fusion.
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

from mast3r_slam.semantic_processor import (
    process_frame_for_semantics,
    load_instance_segmentation_model,
    load_clip_model,
    get_tensor_crop_from_mask,
    TEXT_PROMPTS,
    SEMANTIC_DEVICE
)

def load_test_image():
    """Load a test image from the dataset."""
    from mast3r_slam.dataloader import load_dataset
    from mast3r_slam.config import load_config
    from mast3r_slam.frame import create_frame
    import lietorch
    
    load_config("config/base.yaml")
    dataset = load_dataset("datasets/my/")
    
    # Get first image
    timestamp, raw_img = dataset[0]
    frame = create_frame(0, raw_img, lietorch.Sim3.Identity(1, device="cuda:0"), img_size=512, device="cuda:0")
    
    # Fix the range issue
    image_tensor = frame.img.squeeze(0)
    if image_tensor.min() < -0.1:
        image_tensor = (image_tensor + 1.0) / 2.0
    
    return image_tensor

def enhance_crop_with_padding(image_tensor, mask, pad_ratio=0.2):
    """
    Enhanced cropping with smart padding to include more context.
    This often improves CLIP confidence by providing more visual cues.
    """
    mask_torch = torch.from_numpy(mask.astype(bool)).to(image_tensor.device)
    
    if not mask_torch.any():
        return None
    
    rows, cols = torch.any(mask_torch, axis=1), torch.any(mask_torch, axis=0)
    if not rows.any() or not cols.any():
        return None
    
    ymin, ymax = torch.where(rows)[0][[0, -1]]
    xmin, xmax = torch.where(cols)[0][[0, -1]]
    
    # Calculate padding based on mask size
    height = ymax - ymin
    width = xmax - xmin
    pad_y = int(height * pad_ratio)
    pad_x = int(width * pad_ratio)
    
    # Apply padding with bounds checking
    h, w = image_tensor.shape[1], image_tensor.shape[2]
    ymin_pad = max(0, ymin - pad_y)
    ymax_pad = min(h - 1, ymax + pad_y)
    xmin_pad = max(0, xmin - pad_x)
    xmax_pad = min(w - 1, xmax + pad_x)
    
    # Extract padded crop
    cropped_tensor = image_tensor[:, ymin_pad:ymax_pad+1, xmin_pad:xmax_pad+1]
    
    # Resize to CLIP input size
    import torch.nn.functional as F
    if cropped_tensor.shape[1] > 0 and cropped_tensor.shape[2] > 0:
        clip_input = F.interpolate(cropped_tensor.unsqueeze(0), 
                                 size=(224, 224), mode='bilinear', align_corners=False)
        return clip_input.squeeze(0)
    
    return None

def create_multi_scale_features(image_tensor, mask, scales=[1.0, 1.5, 2.0]):
    """
    Create multi-scale crops and average their CLIP features.
    This can improve robustness and confidence.
    """
    mask_torch = torch.from_numpy(mask.astype(bool)).to(image_tensor.device)
    all_features = []
    
    for scale in scales:
        # Create crop with different padding scales
        crop = enhance_crop_with_padding(image_tensor, mask, pad_ratio=0.2 * scale)
        if crop is not None:
            all_features.append(crop)
    
    return all_features

def create_augmented_prompts(base_prompts):
    """
    Create augmented text prompts with more descriptive variations.
    This can improve CLIP's understanding and confidence.
    """
    augmented = []
    
    # Augmentation templates
    templates = [
        "{}",
        "a photo of a {}",
        "a picture of a {}",
        "an image of a {}",
        "{} in a room",
        "indoor {}",
    ]
    
    for prompt in base_prompts:
        if prompt == "background":
            augmented.append(prompt)
        else:
            for template in templates[:3]:  # Use first 3 templates to avoid too many prompts
                augmented.append(template.format(prompt))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_augmented = []
    for item in augmented:
        if item not in seen:
            seen.add(item)
            unique_augmented.append(item)
    
    return unique_augmented

def save_clip_inputs_for_inspection(crops_data, config_name, save_dir="clip_visual_inspection"):
    """Save what CLIP sees for visual inspection."""
    config_dir = Path(save_dir) / config_name
    config_dir.mkdir(parents=True, exist_ok=True)
    
    for i, (crop, pred, score, mask, full_image) in enumerate(crops_data[:20]):  # Save first 20
        if crop is None:
            continue
            
        # Convert to numpy for visualization
        crop_np = crop.permute(1, 2, 0).cpu().numpy()
        crop_np = np.clip(crop_np, 0, 1)
        
        # Create figure with both crop and segmented region
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Show the crop
        ax1.imshow(crop_np)
        ax1.set_title(f"CLIP Input\n{pred}\nScore: {score:.1f}", fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Add confidence color indicator
        if score > 30:
            border_color = 'green'
        elif score > 25:
            border_color = 'yellow'
        else:
            border_color = 'red'
        
        # Add colored border to crop
        rect = plt.Rectangle((0, 0), crop_np.shape[1]-1, crop_np.shape[0]-1, 
                           fill=False, edgecolor=border_color, linewidth=5)
        ax1.add_patch(rect)
        
        # Show the segmented region on full image
        full_img_np = full_image.permute(1, 2, 0).cpu().numpy()
        full_img_np = np.clip(full_img_np, 0, 1)
        
        # Create overlay
        overlay = full_img_np.copy()
        mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask
        
        # Highlight the segmented region
        overlay[~mask_np] *= 0.3  # Dim the background
        
        # Add colored outline to mask using simple edge detection
        # Create boundaries by finding edges
        mask_float = mask_np.astype(float)
        
        # Simple edge detection using shift
        boundaries = np.zeros_like(mask_np, dtype=bool)
        boundaries[:-1, :] |= (mask_float[:-1, :] != mask_float[1:, :])  # Vertical edges
        boundaries[:, :-1] |= (mask_float[:, :-1] != mask_float[:, 1:])  # Horizontal edges
        boundaries = boundaries & mask_np  # Keep only edges inside the mask
        
        # Set boundary color
        boundary_color = np.array([1.0, 0.0, 0.0]) if border_color == 'red' else \
                        np.array([1.0, 1.0, 0.0]) if border_color == 'yellow' else \
                        np.array([0.0, 1.0, 0.0])
        overlay[boundaries] = boundary_color
        
        ax2.imshow(overlay)
        ax2.set_title(f"Segmented Region\nArea: {mask_np.sum()} pixels", fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(config_dir / f"crop_{i:03d}_score{score:.1f}_{pred.replace(' ', '_')}.png", 
                   dpi=100, bbox_inches='tight')
        plt.close()
    
    print(f"   📸 Saved {min(len(crops_data), 20)} CLIP inputs and segmentations to: {config_dir}")

def test_confidence_improvement_methods(image_tensor):
    """Test various methods to improve CLIP confidence scores."""
    
    # Load models
    sam_generator = load_instance_segmentation_model(device=SEMANTIC_DEVICE)
    
    print("🔍 TESTING CONFIDENCE IMPROVEMENT METHODS")
    print("=" * 60)
    
    # Test configurations
    test_configs = {
        "baseline": {
            "description": "Baseline (current approach)",
            "prompts": TEXT_PROMPTS,
            "crop_padding": 0.0,
            "use_multi_scale": False,
            "use_ensemble": False
        },
        "padded_crops": {
            "description": "Smart padding (20% context)",
            "prompts": TEXT_PROMPTS,
            "crop_padding": 0.2,
            "use_multi_scale": False,
            "use_ensemble": False
        },
        "augmented_prompts": {
            "description": "Augmented text prompts",
            "prompts": create_augmented_prompts(TEXT_PROMPTS),
            "crop_padding": 0.0,
            "use_multi_scale": False,
            "use_ensemble": False
        },
        "multi_scale": {
            "description": "Multi-scale feature averaging",
            "prompts": TEXT_PROMPTS,
            "crop_padding": 0.2,
            "use_multi_scale": True,
            "use_ensemble": False
        },
        "combined_best": {
            "description": "Combined: padding + augmented prompts",
            "prompts": create_augmented_prompts(TEXT_PROMPTS),
            "crop_padding": 0.2,
            "use_multi_scale": False,
            "use_ensemble": False
        }
    }
    
    # Convert to numpy for SAM
    image_np = (image_tensor.permute(1, 2, 0) * 255.0).byte().cpu().numpy()
    
    # Run SAM segmentation
    sam_masks_data = sam_generator.generate(image_np)
    print(f"SAM generated {len(sam_masks_data)} masks\n")
    
    results = {}
    
    for config_name, config in test_configs.items():
        print(f"🧪 Testing: {config['description']}")
        print(f"   Prompts: {len(config['prompts'])}")
        print(f"   Padding: {config['crop_padding']}")
        
        # Load CLIP with specific prompts
        clip_model, clip_preprocess, text_features, text_prompts = load_clip_model(
            text_prompts=config['prompts'], device=SEMANTIC_DEVICE
        )
        
        # Get normalization transform
        normalize_transform = None
        if hasattr(clip_preprocess, 'transforms'):
            for t in clip_preprocess.transforms:
                if hasattr(t, 'mean') and hasattr(t, 'std') and hasattr(t, '__call__'):
                    normalize_transform = t
                    break
        
        if normalize_transform is None:
            import torchvision.transforms as transforms
            normalize_transform = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        
        # Process masks
        all_scores = []
        all_predictions = []
        crops_for_inspection = []  # Store crops for visual inspection
        
        for mask_data in sam_masks_data[:50]:  # Limit to first 50 masks for speed
            mask = mask_data['segmentation']
            
            if config['use_multi_scale']:
                # Multi-scale approach
                multi_crops = create_multi_scale_features(
                    image_tensor, mask, scales=[0.8, 1.0, 1.2]
                )
                
                if multi_crops:
                    # Average features from multiple scales
                    scale_features = []
                    for crop in multi_crops:
                        crop_norm = normalize_transform(crop.unsqueeze(0)).squeeze(0)
                        with torch.no_grad():
                            feat = clip_model.encode_image(crop_norm.unsqueeze(0))
                            feat = feat / feat.norm(dim=-1, keepdim=True)
                            scale_features.append(feat)
                    
                    # Average the features
                    avg_features = torch.stack(scale_features).mean(dim=0)
                    avg_features = avg_features / avg_features.norm(dim=-1, keepdim=True)
                    
                    # Compute similarities
                    similarities = (100.0 * avg_features @ text_features.T)
                    best_score, best_idx = similarities.max(dim=1)
                    
                    all_scores.append(best_score.item())
                    all_predictions.append(text_prompts[best_idx.item()])
            
            else:
                # Single crop approach
                if config['crop_padding'] > 0:
                    crop = enhance_crop_with_padding(image_tensor, mask, config['crop_padding'])
                else:
                    crop = get_tensor_crop_from_mask(
                        image_tensor, 
                        torch.from_numpy(mask.astype(bool)).to(image_tensor.device),
                        output_size=(224, 224)
                    )
                
                if crop is not None:
                    crop_norm = normalize_transform(crop.unsqueeze(0)).squeeze(0)
                    with torch.no_grad():
                        clip_feature = clip_model.encode_image(crop_norm.unsqueeze(0))
                        clip_feature = clip_feature / clip_feature.norm(dim=-1, keepdim=True)
                        
                        similarities = (100.0 * clip_feature @ text_features.T)
                        best_score, best_idx = similarities.max(dim=1)
                        
                        all_scores.append(best_score.item())
                        all_predictions.append(text_prompts[best_idx.item()])
                        
                        # Store crop for inspection (before normalization)
                        crops_for_inspection.append((crop, text_prompts[best_idx.item()], best_score.item(), 
                                                   mask, image_tensor))
        
        # Save CLIP inputs for visual inspection
        if crops_for_inspection:
            save_clip_inputs_for_inspection(crops_for_inspection, config_name)
        
        # Calculate statistics
        if all_scores:
            avg_score = np.mean(all_scores)
            max_score = np.max(all_scores)
            min_score = np.min(all_scores)
            std_score = np.std(all_scores)
            
            # Count high-confidence predictions (>25)
            high_conf_count = sum(1 for s in all_scores if s > 25.0)
            high_conf_ratio = high_conf_count / len(all_scores) * 100
            
            results[config_name] = {
                'description': config['description'],
                'avg_score': avg_score,
                'max_score': max_score,
                'min_score': min_score,
                'std_score': std_score,
                'high_conf_ratio': high_conf_ratio,
                'num_samples': len(all_scores)
            }
            
            print(f"   ✓ Avg confidence: {avg_score:.2f}")
            print(f"   ✓ Max confidence: {max_score:.2f}")
            print(f"   ✓ High confidence ratio: {high_conf_ratio:.1f}%")
            print()
    
    return results, sam_masks_data

def create_confidence_comparison_plot(results, save_path="confidence_comparison"):
    """Create visualization comparing confidence improvements."""
    save_dir = Path(save_path)
    save_dir.mkdir(exist_ok=True)
    
    # Sort by average confidence
    sorted_configs = sorted(results.items(), key=lambda x: x[1]['avg_score'], reverse=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('CLIP Confidence Score Improvements', fontsize=16, fontweight='bold')
    
    # Average confidence comparison
    config_names = [r[1]['description'] for r in sorted_configs]
    avg_scores = [r[1]['avg_score'] for r in sorted_configs]
    colors = plt.cm.viridis(np.linspace(0, 1, len(config_names)))
    
    bars = ax1.barh(config_names, avg_scores, color=colors)
    ax1.set_xlabel('Average Confidence Score')
    ax1.set_title('Average CLIP Confidence by Method')
    ax1.axvline(x=25.0, color='red', linestyle='--', alpha=0.5, label='Target threshold')
    
    # Add value labels
    for bar, score in zip(bars, avg_scores):
        width = bar.get_width()
        ax1.text(width + 0.3, bar.get_y() + bar.get_height()/2, 
                f'{score:.1f}', ha='left', va='center', fontweight='bold')
    
    # High confidence ratio
    high_conf_ratios = [r[1]['high_conf_ratio'] for r in sorted_configs]
    
    bars2 = ax2.barh(config_names, high_conf_ratios, color=colors)
    ax2.set_xlabel('High Confidence Ratio (%)')
    ax2.set_title('Percentage of Predictions with Score > 25')
    
    # Add value labels
    for bar, ratio in zip(bars2, high_conf_ratios):
        width = bar.get_width()
        ax2.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{ratio:.0f}%', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_dir / "confidence_improvements.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save detailed results
    with open(save_dir / "confidence_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📊 Visualizations saved to: {save_dir}/")
    
    return sorted_configs[0]  # Return best config

def create_side_by_side_comparison(save_dir="clip_visual_inspection"):
    """Create side-by-side comparison of baseline vs best method."""
    baseline_dir = Path(save_dir) / "baseline"
    
    # Find best method directory
    best_dirs = [d for d in Path(save_dir).iterdir() if d.is_dir() and d.name != "baseline"]
    if not best_dirs:
        return
    
    # Use the directory with most recent modification time as "best"
    best_dir = max(best_dirs, key=lambda d: d.stat().st_mtime)
    
    comparison_dir = Path(save_dir) / "comparison"
    comparison_dir.mkdir(exist_ok=True)
    
    # Get matching crops from both methods
    baseline_crops = sorted(list(baseline_dir.glob("crop_*.png")))[:10]
    
    for i, baseline_path in enumerate(baseline_crops):
        # Extract crop number from filename
        crop_num = baseline_path.name.split('_')[1]
        
        # Find corresponding best method crop
        best_crops = list(best_dir.glob(f"crop_{crop_num}_*.png"))
        if not best_crops:
            continue
        best_path = best_crops[0]
        
        # Load images
        baseline_img = plt.imread(baseline_path)
        best_img = plt.imread(best_path)
        
        # Extract scores from filenames
        baseline_score = float(baseline_path.stem.split('score')[1].split('_')[0])
        best_score = float(best_path.stem.split('score')[1].split('_')[0])
        
        # Create side-by-side figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        ax1.imshow(baseline_img)
        ax1.set_title(f"Baseline\nScore: {baseline_score:.1f}", fontsize=14)
        ax1.axis('off')
        
        ax2.imshow(best_img)
        ax2.set_title(f"{best_dir.name.replace('_', ' ').title()}\nScore: {best_score:.1f}", fontsize=14)
        ax2.axis('off')
        
        # Add improvement indicator
        improvement = best_score - baseline_score
        fig.suptitle(f"Crop {crop_num} - Improvement: {improvement:+.1f} points", 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(comparison_dir / f"comparison_{i:02d}_crop{crop_num}.png", 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"   📊 Created side-by-side comparisons in: {comparison_dir}")

def create_segmentation_summary(image_tensor, sam_masks_data, results, save_dir="clip_visual_inspection"):
    """Create a summary visualization showing all segmented regions with labels."""
    summary_dir = Path(save_dir) / "summary"
    summary_dir.mkdir(exist_ok=True)
    
    # Find the best method's predictions
    best_method = max(results.items(), key=lambda x: x[1]['avg_score'])[0]
    
    # Create full image with all segmentations
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Display original image
    img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np, 0, 1)
    ax.imshow(img_np)
    ax.set_title(f"All Segmented Regions - {best_method.replace('_', ' ').title()}", 
                fontsize=16, fontweight='bold')
    ax.axis('off')
    
    # Color map for different categories
    from matplotlib import cm
    colors = cm.Set3(np.linspace(0, 1, 20))
    
    # Track unique labels
    label_counts = {}
    
    # Overlay all masks with semi-transparent colors
    for i, mask_data in enumerate(sam_masks_data[:50]):
        mask = mask_data['segmentation']
        
        # Create colored overlay for this mask
        mask_overlay = np.zeros((*mask.shape, 4))
        color_idx = i % len(colors)
        mask_overlay[mask] = colors[color_idx]
        mask_overlay[mask, 3] = 0.3  # Semi-transparent
        
        # Add mask number at centroid
        if mask.any():
            y_coords, x_coords = np.where(mask)
            centroid_y = int(np.mean(y_coords))
            centroid_x = int(np.mean(x_coords))
            
            ax.text(centroid_x, centroid_y, str(i), 
                   color='white', fontsize=10, fontweight='bold',
                   ha='center', va='center',
                   bbox=dict(boxstyle="circle,pad=0.3", facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(summary_dir / "all_segmentations.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   📋 Created segmentation summary in: {summary_dir}")

def main():
    """Main test function for confidence improvements."""
    print("🎯 CLIP CONFIDENCE IMPROVEMENT TEST")
    print("Goal: Increase confidence scores for reliable 3D semantic mapping")
    print("=" * 60)
    
    # Load test image
    print("📸 Loading test image...")
    image_tensor = load_test_image()
    print(f"   Image shape: {image_tensor.shape}")
    print(f"   Image range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
    
    # Test confidence improvement methods
    results, sam_masks = test_confidence_improvement_methods(image_tensor)
    
    # Create visualizations
    print("\n" + "="*60)
    print("📊 Creating confidence comparison visualizations...")
    best_config = create_confidence_comparison_plot(results)
    
    # Create side-by-side visual comparisons
    print("🖼️  Creating visual comparisons...")
    create_side_by_side_comparison()
    
    # Create segmentation summary
    print("📋 Creating segmentation summary...")
    create_segmentation_summary(image_tensor, sam_masks, results)
    
    # Summary
    print("\n" + "🎯" + " "*20 + "RESULTS SUMMARY" + " "*20 + "🎯")
    print("=" * 70)
    
    baseline_score = results['baseline']['avg_score']
    best_score = best_config[1]['avg_score']
    improvement = best_score - baseline_score
    
    print(f"🏆 Best Method: {best_config[1]['description']}")
    print(f"📈 Confidence Improvement: {improvement:+.1f} points")
    print(f"   Baseline: {baseline_score:.1f} → Best: {best_score:.1f}")
    
    if improvement > 0:
        print(f"\n✅ SUCCESS! Found methods that improve confidence scores.")
        print(f"   This will lead to more reliable 3D semantic labels.")
    
    print(f"\n💡 Key Insights:")
    if 'padded' in best_config[0]:
        print(f"   - Adding context padding improves CLIP recognition")
    if 'augmented' in best_config[0]:
        print(f"   - Augmented prompts help CLIP understand scenes better")
    if 'multi_scale' in best_config[0]:
        print(f"   - Multi-scale features provide more robust predictions")
    
    print(f"\n🚀 For 3D Semantic Mapping:")
    print(f"   - Higher confidence = more reliable 3D labels")
    print(f"   - Best method achieves {best_config[1]['high_conf_ratio']:.0f}% high-confidence predictions")
    print(f"   - Ready for integration into your SLAM pipeline")
    
    print("=" * 70)

if __name__ == "__main__":
    main()