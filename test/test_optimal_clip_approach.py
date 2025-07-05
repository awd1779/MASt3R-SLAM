#!/usr/bin/env python3
"""
Test to find the optimal CLIP approach for diverse object detection.
The goal is to avoid everything being classified as the same thing.
"""

import sys
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))

from mast3r_slam.semantic_processor import (
    load_instance_segmentation_model,
    load_clip_model,
    get_tensor_crop_from_mask,
    TEXT_PROMPTS,
    SEMANTIC_DEVICE
)

def calculate_diversity_score(predictions):
    """Calculate how diverse the predictions are (0=all same, 1=all different)."""
    unique_preds = len(set(predictions))
    total_preds = len(predictions)
    return (unique_preds - 1) / max(total_preds - 1, 1) if total_preds > 1 else 0

def test_clip_approaches(image_tensor, sam_masks_data):
    """Test different CLIP approaches to find optimal for object diversity."""
    
    # Load CLIP model
    clip_model, clip_preprocess, text_features, text_prompts = load_clip_model(
        model_name='ViT-H-14',
        pretrained_dataset='laion2b_s32b_b79k',
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
    
    approaches = {
        "crop_only": {"weight": 1.0, "description": "Only masked crop (original approach)"},
        "hovsg_0.8": {"weight": 0.8, "description": "HOV-SG with 0.8 masked weight"},
        "hovsg_0.6": {"weight": 0.6, "description": "HOV-SG with 0.6 masked weight"},
        "hovsg_0.4418": {"weight": 0.4418, "description": "HOV-SG original weight"},
        "hovsg_0.2": {"weight": 0.2, "description": "HOV-SG with 0.2 masked weight"},
        "full_only": {"weight": 0.0, "description": "Only full image (no masking)"}
    }
    
    results = {}
    
    # Get full image features once
    full_image_resized = torch.nn.functional.interpolate(
        image_tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
    ).squeeze(0)
    full_image_norm = normalize_transform(full_image_resized.unsqueeze(0)).squeeze(0)
    
    with torch.no_grad():
        full_features = clip_model.encode_image(full_image_norm.unsqueeze(0))
        full_features = full_features / full_features.norm(dim=-1, keepdim=True)
    
    for approach_name, config in approaches.items():
        predictions = []
        scores = []
        
        for i, mask_data in enumerate(sam_masks_data[:50]):  # Test first 50
            mask = mask_data['segmentation']
            
            if config["weight"] == 0.0:
                # Full image only
                features = full_features
            elif config["weight"] == 1.0:
                # Crop only
                crop = get_tensor_crop_from_mask(
                    image_tensor,
                    torch.from_numpy(mask.astype(bool)).to(SEMANTIC_DEVICE),
                    output_size=(224, 224)
                )
                if crop is None:
                    continue
                    
                crop_norm = normalize_transform(crop.unsqueeze(0)).squeeze(0)
                with torch.no_grad():
                    features = clip_model.encode_image(crop_norm.unsqueeze(0))
                    features = features / features.norm(dim=-1, keepdim=True)
            else:
                # HOV-SG approach
                crop = get_tensor_crop_from_mask(
                    image_tensor,
                    torch.from_numpy(mask.astype(bool)).to(SEMANTIC_DEVICE),
                    output_size=(224, 224)
                )
                if crop is None:
                    continue
                    
                crop_norm = normalize_transform(crop.unsqueeze(0)).squeeze(0)
                with torch.no_grad():
                    crop_features = clip_model.encode_image(crop_norm.unsqueeze(0))
                    crop_features = crop_features / crop_features.norm(dim=-1, keepdim=True)
                
                # Weighted combination
                features = (1 - config["weight"]) * full_features + config["weight"] * crop_features
                features = features / features.norm(dim=-1, keepdim=True)
            
            # Get prediction
            similarities = (100.0 * features @ text_features.T)
            best_score, best_idx = similarities.max(dim=1)
            
            predictions.append(text_prompts[best_idx.item()])
            scores.append(best_score.item())
        
        # Calculate metrics
        avg_score = np.mean(scores)
        diversity = calculate_diversity_score(predictions)
        unique_classes = set(predictions)
        
        results[approach_name] = {
            "avg_score": avg_score,
            "diversity": diversity,
            "unique_classes": len(unique_classes),
            "predictions": predictions[:10],  # First 10 for display
            "description": config["description"]
        }
        
        print(f"\n{config['description']}:")
        print(f"  Average score: {avg_score:.2f}")
        print(f"  Diversity: {diversity:.2%}")
        print(f"  Unique classes: {len(unique_classes)}")
        print(f"  First 5 predictions: {predictions[:5]}")
    
    return results

def visualize_results(results):
    """Create visualization of approach comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract data
    approaches = list(results.keys())
    avg_scores = [results[a]["avg_score"] for a in approaches]
    diversities = [results[a]["diversity"] * 100 for a in approaches]
    
    # Score vs Diversity scatter
    ax1.scatter(avg_scores, diversities, s=100)
    for i, approach in enumerate(approaches):
        ax1.annotate(approach, (avg_scores[i], diversities[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    ax1.set_xlabel('Average Confidence Score')
    ax1.set_ylabel('Prediction Diversity (%)')
    ax1.set_title('CLIP Approach Comparison: Score vs Diversity')
    ax1.grid(True, alpha=0.3)
    
    # Add ideal zone
    ax1.axhspan(50, 100, alpha=0.1, color='green', label='Good diversity')
    ax1.axvspan(25, 40, alpha=0.1, color='blue', label='Good confidence')
    ax1.legend()
    
    # Bar chart of unique classes
    unique_classes = [results[a]["unique_classes"] for a in approaches]
    bars = ax2.bar(approaches, unique_classes)
    ax2.set_xlabel('Approach')
    ax2.set_ylabel('Number of Unique Classes Detected')
    ax2.set_title('Object Detection Diversity by Approach')
    ax2.tick_params(axis='x', rotation=45)
    
    # Color bars by diversity
    for bar, div in zip(bars, diversities):
        bar.set_color(plt.cm.RdYlGn(div/100))
    
    plt.tight_layout()
    plt.savefig('clip_approach_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Visualization saved to: clip_approach_comparison.png")

def main():
    """Main test function."""
    print("🔍 FINDING OPTIMAL CLIP APPROACH")
    print("Goal: Avoid everything being classified as the same object")
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
    
    # Test approaches
    results = test_clip_approaches(image_tensor, sam_masks_data)
    
    # Find best approach
    best_approach = max(results.items(), 
                       key=lambda x: x[1]["diversity"] * 0.7 + x[1]["avg_score"]/50 * 0.3)
    
    print("\n" + "="*60)
    print(f"🏆 BEST APPROACH: {best_approach[0]}")
    print(f"   {best_approach[1]['description']}")
    print(f"   Diversity: {best_approach[1]['diversity']:.2%}")
    print(f"   Average score: {best_approach[1]['avg_score']:.2f}")
    print(f"   Detected {best_approach[1]['unique_classes']} different object types")
    
    # Visualize results
    visualize_results(results)
    
    print("\n💡 RECOMMENDATION:")
    if best_approach[0] == "crop_only":
        print("Use crop-only approach (no HOV-SG) for best object discrimination")
    else:
        print(f"Use HOV-SG with weight {approaches[best_approach[0]]['weight']}")

if __name__ == "__main__":
    main()