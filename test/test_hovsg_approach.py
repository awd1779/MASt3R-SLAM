#!/usr/bin/env python3
"""
Test HOV-SG approach: combining full-image and masked-region CLIP features
Based on: https://github.com/hovsg/HOV-SG
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from mast3r_slam.semantic_processor import (
    load_instance_segmentation_model,
    load_clip_model,
    get_tensor_crop_from_mask,
    TEXT_PROMPTS,
    SEMANTIC_DEVICE
)

# HOV-SG parameters
HOVSG_MASKED_WEIGHT = 0.4418  # Weight for masked features vs full image
HOVSG_CLIP_MODEL = 'ViT-H-14'  # Model used by HOV-SG
HOVSG_CLIP_PRETRAINED = 'laion2b_s32b_b79k'

def extract_hovsg_features(image_tensor, mask, clip_model, clip_preprocess, normalize_transform):
    """
    Extract CLIP features using HOV-SG approach:
    1. Get features from the full image
    2. Get features from the masked region
    3. Combine with weighted average
    """
    
    # 1. Full image features
    full_image_resized = torch.nn.functional.interpolate(
        image_tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
    ).squeeze(0)
    
    full_image_norm = normalize_transform(full_image_resized.unsqueeze(0)).squeeze(0)
    
    with torch.no_grad():
        full_features = clip_model.encode_image(full_image_norm.unsqueeze(0))
        full_features = full_features / full_features.norm(dim=-1, keepdim=True)
    
    # 2. Masked region features (HOV-SG approach)
    # Create masked image by zeroing out background
    masked_image = image_tensor.clone()
    mask_torch = torch.from_numpy(mask.astype(bool)).to(image_tensor.device)
    masked_image[:, ~mask_torch] = 0  # Zero out background
    
    # Get bounding box of mask for cropping
    rows, cols = torch.any(mask_torch, axis=1), torch.any(mask_torch, axis=0)
    if not rows.any() or not cols.any():
        return full_features  # Return full features if mask is empty
    
    ymin, ymax = torch.where(rows)[0][[0, -1]]
    xmin, xmax = torch.where(cols)[0][[0, -1]]
    
    # Crop to bounding box
    cropped_masked = masked_image[:, ymin:ymax+1, xmin:xmax+1]
    
    # Resize to CLIP input size
    if cropped_masked.shape[1] > 0 and cropped_masked.shape[2] > 0:
        masked_resized = torch.nn.functional.interpolate(
            cropped_masked.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
        ).squeeze(0)
        
        masked_norm = normalize_transform(masked_resized.unsqueeze(0)).squeeze(0)
        
        with torch.no_grad():
            masked_features = clip_model.encode_image(masked_norm.unsqueeze(0))
            masked_features = masked_features / masked_features.norm(dim=-1, keepdim=True)
    else:
        masked_features = full_features
    
    # 3. Combine features with HOV-SG weighting
    combined_features = (1 - HOVSG_MASKED_WEIGHT) * full_features + HOVSG_MASKED_WEIGHT * masked_features
    combined_features = combined_features / combined_features.norm(dim=-1, keepdim=True)
    
    return combined_features, full_features, masked_features

def test_hovsg_approach(image_tensor, sam_masks_data, save_visualizations=True):
    """Test HOV-SG feature extraction approach."""
    
    print("🔍 TESTING HOV-SG APPROACH")
    print("=" * 60)
    print(f"Using CLIP model: {HOVSG_CLIP_MODEL}")
    print(f"Masked weight: {HOVSG_MASKED_WEIGHT}")
    
    # Load CLIP model (HOV-SG configuration)
    clip_model, clip_preprocess, text_features, text_prompts = load_clip_model(
        model_name=HOVSG_CLIP_MODEL,
        pretrained_dataset=HOVSG_CLIP_PRETRAINED,
        text_prompts=TEXT_PROMPTS,
        device=SEMANTIC_DEVICE
    )
    
    # Get normalization transform
    normalize_transform = None
    if hasattr(clip_preprocess, 'transforms'):
        for t in clip_preprocess.transforms:
            if hasattr(t, 'mean') and hasattr(t, 'std'):
                normalize_transform = t
                break
    
    if normalize_transform is None:
        import torchvision.transforms as transforms
        normalize_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    # Process masks with HOV-SG approach
    hovsg_scores = []
    baseline_scores = []
    predictions = []
    
    print("\nProcessing masks...")
    for i, mask_data in enumerate(sam_masks_data[:50]):  # Test first 50
        mask = mask_data['segmentation']
        
        # HOV-SG approach
        combined_features, full_feat, masked_feat = extract_hovsg_features(
            image_tensor, mask, clip_model, clip_preprocess, normalize_transform
        )
        
        # Get predictions
        with torch.no_grad():
            similarities = (100.0 * combined_features @ text_features.T)
            best_score, best_idx = similarities.max(dim=1)
            hovsg_scores.append(best_score.item())
            predictions.append(text_prompts[best_idx.item()])
        
        # Baseline approach (crop only)
        crop = get_tensor_crop_from_mask(
            image_tensor,
            torch.from_numpy(mask.astype(bool)).to(SEMANTIC_DEVICE),
            output_size=(224, 224)
        )
        
        if crop is not None:
            crop_norm = normalize_transform(crop.unsqueeze(0)).squeeze(0)
            with torch.no_grad():
                baseline_feat = clip_model.encode_image(crop_norm.unsqueeze(0))
                baseline_feat = baseline_feat / baseline_feat.norm(dim=-1, keepdim=True)
                baseline_sim = (100.0 * baseline_feat @ text_features.T)
                baseline_score, _ = baseline_sim.max(dim=1)
                baseline_scores.append(baseline_score.item())
        else:
            baseline_scores.append(0)
        
        # Print first few results
        if i < 5:
            print(f"  Mask {i}: {predictions[-1]}")
            print(f"    HOV-SG score: {hovsg_scores[-1]:.2f}")
            print(f"    Baseline score: {baseline_scores[-1]:.2f}")
            print(f"    Improvement: {hovsg_scores[-1] - baseline_scores[-1]:+.2f}")
    
    # Calculate statistics
    hovsg_avg = np.mean(hovsg_scores)
    baseline_avg = np.mean(baseline_scores)
    improvement = hovsg_avg - baseline_avg
    
    print(f"\n📊 RESULTS:")
    print(f"HOV-SG approach average: {hovsg_avg:.2f}")
    print(f"Baseline average: {baseline_avg:.2f}")
    print(f"Improvement: {improvement:+.2f} points")
    print(f"High confidence ratio (>25): {sum(1 for s in hovsg_scores if s > 25) / len(hovsg_scores) * 100:.1f}%")
    
    if save_visualizations:
        visualize_hovsg_results(hovsg_scores, baseline_scores, predictions)
    
    return {
        'hovsg_scores': hovsg_scores,
        'baseline_scores': baseline_scores,
        'improvement': improvement,
        'predictions': predictions
    }

def visualize_hovsg_results(hovsg_scores, baseline_scores, predictions, save_dir="hovsg_comparison"):
    """Create visualizations comparing HOV-SG approach with baseline."""
    
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Score distribution comparison
    ax1.hist(baseline_scores, bins=20, alpha=0.5, label='Baseline (crop)', color='blue')
    ax1.hist(hovsg_scores, bins=20, alpha=0.5, label='HOV-SG approach', color='red')
    ax1.axvline(x=25, color='green', linestyle='--', label='Threshold')
    ax1.set_xlabel('Confidence Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Score Distribution Comparison')
    ax1.legend()
    
    # Score improvement scatter plot
    improvements = [h - b for h, b in zip(hovsg_scores, baseline_scores)]
    ax2.scatter(baseline_scores, improvements, alpha=0.6)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Baseline Score')
    ax2.set_ylabel('Score Improvement (HOV-SG - Baseline)')
    ax2.set_title('Score Improvements by Baseline Score')
    
    # Add average improvement line
    avg_improvement = np.mean(improvements)
    ax2.axhline(y=avg_improvement, color='red', linestyle='--', 
                label=f'Avg improvement: {avg_improvement:+.2f}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path / "hovsg_comparison.png", dpi=150)
    plt.close()
    
    # Save detailed results
    results = {
        'method': 'HOV-SG',
        'model': HOVSG_CLIP_MODEL,
        'masked_weight': HOVSG_MASKED_WEIGHT,
        'hovsg_avg': float(np.mean(hovsg_scores)),
        'baseline_avg': float(np.mean(baseline_scores)),
        'improvement': float(np.mean(improvements)),
        'high_conf_ratio': float(sum(1 for s in hovsg_scores if s > 25) / len(hovsg_scores) * 100)
    }
    
    with open(save_path / "hovsg_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Visualizations saved to: {save_path}/")

def test_different_weights(image_tensor, sam_masks_data):
    """Test different masked weights to find optimal value."""
    
    print("\n🔬 TESTING DIFFERENT MASKED WEIGHTS")
    print("=" * 60)
    
    weights = [0.0, 0.2, 0.4418, 0.6, 0.8, 1.0]
    results = []
    
    # Load CLIP model once
    clip_model, clip_preprocess, text_features, text_prompts = load_clip_model(
        model_name=HOVSG_CLIP_MODEL,
        pretrained_dataset=HOVSG_CLIP_PRETRAINED,
        text_prompts=TEXT_PROMPTS,
        device=SEMANTIC_DEVICE
    )
    
    # Get normalization
    normalize_transform = None
    if hasattr(clip_preprocess, 'transforms'):
        for t in clip_preprocess.transforms:
            if hasattr(t, 'mean') and hasattr(t, 'std'):
                normalize_transform = t
                break
    
    # Test each weight
    for weight in weights:
        global HOVSG_MASKED_WEIGHT
        HOVSG_MASKED_WEIGHT = weight
        
        scores = []
        for mask_data in sam_masks_data[:30]:  # Test subset
            mask = mask_data['segmentation']
            
            combined_features, _, _ = extract_hovsg_features(
                image_tensor, mask, clip_model, clip_preprocess, normalize_transform
            )
            
            with torch.no_grad():
                similarities = (100.0 * combined_features @ text_features.T)
                best_score, _ = similarities.max(dim=1)
                scores.append(best_score.item())
        
        avg_score = np.mean(scores)
        results.append((weight, avg_score))
        print(f"  Weight {weight:.4f}: avg score {avg_score:.2f}")
    
    # Find optimal weight
    optimal_weight, optimal_score = max(results, key=lambda x: x[1])
    print(f"\n✨ Optimal weight: {optimal_weight} (score: {optimal_score:.2f})")
    
    return results

def main():
    """Main test function."""
    print("🚀 HOV-SG APPROACH TEST")
    print("Testing hierarchical open-vocabulary scene graph approach")
    print("=" * 60)
    
    # Load test data
    from mast3r_slam.dataloader import load_dataset
    from mast3r_slam.config import load_config
    from mast3r_slam.frame import create_frame
    import lietorch
    
    load_config("config/base.yaml")
    dataset = load_dataset("datasets/my/")
    
    timestamp, raw_img = dataset[0]
    frame = create_frame(0, raw_img, lietorch.Sim3.Identity(1, device="cuda:0"), img_size=512, device="cuda:0")
    
    image_tensor = frame.img.squeeze(0)
    if image_tensor.min() < -0.1:
        image_tensor = (image_tensor + 1.0) / 2.0
    
    # Generate SAM masks
    sam_generator = load_instance_segmentation_model(device=SEMANTIC_DEVICE)
    image_np = (image_tensor.permute(1, 2, 0) * 255.0).byte().cpu().numpy()
    sam_masks_data = sam_generator.generate(image_np)
    
    print(f"\n📸 Test image shape: {image_tensor.shape}")
    print(f"🎭 SAM generated {len(sam_masks_data)} masks")
    
    # Test HOV-SG approach
    results = test_hovsg_approach(image_tensor, sam_masks_data)
    
    # Test different weights
    print("\n" + "="*60)
    weight_results = test_different_weights(image_tensor, sam_masks_data)
    
    # Summary
    print("\n" + "🎯"*20)
    print("SUMMARY:")
    print(f"HOV-SG approach improvement: {results['improvement']:+.2f} points")
    print("This approach combines global context (full image) with local details (masked region)")
    print("Benefits: More robust to partial occlusions and better semantic understanding")
    
    print("\n✅ Test complete!")

if __name__ == "__main__":
    main()