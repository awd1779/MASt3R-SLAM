#!/usr/bin/env python3
"""
Test CLIP directly on the full image without SAM segmentation.
This helps identify if confidence issues are from CLIP or the segmentation process.
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

from mast3r_slam.semantic_processor import (
    load_clip_model,
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

def test_clip_on_full_image(image_tensor, text_prompts=None, save_results=True):
    """Test CLIP directly on the full image without segmentation."""
    
    if text_prompts is None:
        text_prompts = TEXT_PROMPTS
    
    print("🔍 TESTING CLIP ON FULL IMAGE (No Segmentation)")
    print("=" * 60)
    print(f"Image shape: {image_tensor.shape}")
    print(f"Number of text prompts: {len(text_prompts)}")
    
    # Load CLIP model
    clip_model, clip_preprocess, text_features, _ = load_clip_model(
        text_prompts=text_prompts, device=SEMANTIC_DEVICE
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
    
    # Resize full image to CLIP input size
    import torch.nn.functional as F
    clip_input = F.interpolate(image_tensor.unsqueeze(0), 
                             size=(224, 224), mode='bilinear', align_corners=False)
    clip_input = clip_input.squeeze(0)
    
    # Normalize
    clip_input_norm = normalize_transform(clip_input.unsqueeze(0)).squeeze(0)
    
    # Get CLIP features for the full image
    with torch.no_grad():
        image_features = clip_model.encode_image(clip_input_norm.unsqueeze(0))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Compute similarities with all text prompts
        similarities = (100.0 * image_features @ text_features.T)
        similarities = similarities.squeeze(0)  # Remove batch dimension
    
    # Sort by similarity score
    scores, indices = similarities.sort(descending=True)
    
    # Print top predictions
    print("\n📊 TOP 15 PREDICTIONS FOR FULL IMAGE:")
    print("-" * 50)
    print(f"{'Rank':<5} {'Class':<30} {'Score':<10}")
    print("-" * 50)
    
    top_15_results = []
    for i in range(min(15, len(scores))):
        class_name = text_prompts[indices[i].item()]
        score = scores[i].item()
        print(f"{i+1:<5} {class_name:<30} {score:<10.2f}")
        top_15_results.append((class_name, score))
    
    # Calculate statistics
    avg_score = similarities.mean().item()
    max_score = similarities.max().item()
    min_score = similarities.min().item()
    
    print("\n📈 STATISTICS:")
    print(f"   Average score: {avg_score:.2f}")
    print(f"   Max score: {max_score:.2f}")
    print(f"   Min score: {min_score:.2f}")
    print(f"   Score range: {max_score - min_score:.2f}")
    
    if save_results:
        save_clip_analysis(image_tensor, clip_input, top_15_results, similarities, text_prompts)
    
    return {
        'top_predictions': top_15_results,
        'all_scores': similarities.cpu().numpy(),
        'avg_score': avg_score,
        'max_score': max_score,
        'min_score': min_score,
        'text_prompts': text_prompts
    }

def save_clip_analysis(original_image, clip_input, top_predictions, all_similarities, text_prompts):
    """Save visualizations of CLIP analysis."""
    save_dir = Path("clip_baseline_analysis")
    save_dir.mkdir(exist_ok=True)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Original image
    ax1 = plt.subplot(2, 3, 1)
    img_np = original_image.permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np, 0, 1)
    ax1.imshow(img_np)
    ax1.set_title("Original Image", fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 2. CLIP input (224x224)
    ax2 = plt.subplot(2, 3, 2)
    clip_np = clip_input.permute(1, 2, 0).cpu().numpy()
    clip_np = np.clip(clip_np, 0, 1)
    ax2.imshow(clip_np)
    ax2.set_title("CLIP Input (224x224)", fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # 3. Top predictions bar chart
    ax3 = plt.subplot(2, 3, 3)
    top_classes = [pred[0] for pred in top_predictions[:10]]
    top_scores = [pred[1] for pred in top_predictions[:10]]
    colors = plt.cm.viridis(np.linspace(0, 1, 10))
    
    bars = ax3.barh(range(len(top_classes)), top_scores, color=colors)
    ax3.set_yticks(range(len(top_classes)))
    ax3.set_yticklabels(top_classes)
    ax3.set_xlabel('Similarity Score')
    ax3.set_title('Top 10 Predictions', fontsize=14, fontweight='bold')
    ax3.axvline(x=25, color='red', linestyle='--', alpha=0.5, label='Typical threshold')
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, top_scores)):
        ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{score:.1f}', va='center', fontweight='bold')
    
    # 4. Score distribution histogram
    ax4 = plt.subplot(2, 3, 4)
    all_scores = all_similarities.cpu().numpy()
    ax4.hist(all_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.axvline(x=25, color='red', linestyle='--', alpha=0.5, label='Typical threshold')
    ax4.axvline(x=all_scores.mean(), color='green', linestyle='--', alpha=0.5, label='Mean')
    ax4.set_xlabel('Similarity Score')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Score Distribution', fontsize=14, fontweight='bold')
    ax4.legend()
    
    # 5. All scores sorted
    ax5 = plt.subplot(2, 3, 5)
    sorted_scores = np.sort(all_scores)[::-1]
    ax5.plot(sorted_scores, 'b-', linewidth=2)
    ax5.axhline(y=25, color='red', linestyle='--', alpha=0.5, label='Typical threshold')
    ax5.set_xlabel('Rank')
    ax5.set_ylabel('Similarity Score')
    ax5.set_title('All Scores (Sorted)', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # 6. Score comparison text
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    analysis_text = f"""CLIP Full Image Analysis
    
Top Prediction: {top_predictions[0][0]}
Top Score: {top_predictions[0][1]:.2f}

Average Score: {all_scores.mean():.2f}
Max Score: {all_scores.max():.2f}
Min Score: {all_scores.min():.2f}

Scores > 25: {(all_scores > 25).sum()} / {len(all_scores)}
Scores > 20: {(all_scores > 20).sum()} / {len(all_scores)}

Interpretation:
- CLIP sees the full scene context
- Top predictions reflect dominant elements
- Lower scores than cropped regions
  (full scenes are more complex)
"""
    
    ax6.text(0.1, 0.9, analysis_text, transform=ax6.transAxes, 
            fontsize=12, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_dir / "clip_full_image_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save detailed results
    results = {
        'top_15_predictions': top_predictions,
        'statistics': {
            'avg_score': float(all_scores.mean()),
            'max_score': float(all_scores.max()),
            'min_score': float(all_scores.min()),
            'std_score': float(all_scores.std()),
            'scores_above_25': int((all_scores > 25).sum()),
            'scores_above_20': int((all_scores > 20).sum()),
            'total_prompts': len(all_scores)
        }
    }
    
    with open(save_dir / "clip_baseline_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Analysis saved to: {save_dir}/")

def compare_with_segmented_approach():
    """Compare CLIP on full image vs typical segmented approach scores."""
    print("\n🔄 COMPARISON WITH SEGMENTED APPROACH")
    print("=" * 60)
    
    print("Typical SAM+CLIP (segmented) results:")
    print("   Average confidence: 25-28")
    print("   Best segments: 'room corner', 'work surface'")
    print("   High confidence ratio: 80-100%")
    
    print("\nCLIP on full image (no segmentation):")
    print("   Sees entire scene at once")
    print("   Lower scores expected (more complex)")
    print("   Top predictions show dominant scene elements")
    
    print("\n💡 KEY INSIGHTS:")
    print("1. If full image scores are very low (<15):")
    print("   → CLIP model might not be suitable for your data")
    print("   → Consider different CLIP model or fine-tuning")
    
    print("\n2. If full image scores are reasonable (15-25):")
    print("   → CLIP is working, segmentation helps focus attention")
    print("   → Current approach is good")
    
    print("\n3. If full image scores are high (>25):")
    print("   → Scene is very clear/simple")
    print("   → Might not need complex segmentation")

def main():
    """Main test function."""
    print("🎯 CLIP BASELINE TEST - Full Image Analysis")
    print("Testing CLIP directly without SAM segmentation")
    print("=" * 60)
    
    # Load test image
    print("📸 Loading test image...")
    image_tensor = load_test_image()
    
    # Test 1: Original prompts
    print("\n" + "="*60)
    print("TEST 1: With original prompts")
    results_original = test_clip_on_full_image(image_tensor, TEXT_PROMPTS)
    
    # Test 2: Augmented prompts
    print("\n" + "="*60)
    print("TEST 2: With augmented prompts")
    from test_confidence_improvements import create_augmented_prompts
    augmented_prompts = create_augmented_prompts(TEXT_PROMPTS)
    results_augmented = test_clip_on_full_image(
        image_tensor, 
        augmented_prompts, 
        save_results=False  # Don't save twice
    )
    
    # Compare results
    print("\n" + "="*60)
    print("📊 PROMPT COMPARISON:")
    print(f"Original prompts ({len(TEXT_PROMPTS)}): Max score = {results_original['max_score']:.2f}")
    print(f"Augmented prompts ({len(augmented_prompts)}): Max score = {results_augmented['max_score']:.2f}")
    
    # Final comparison
    compare_with_segmented_approach()
    
    print("\n" + "="*60)
    print("✅ Test complete! Check 'clip_baseline_analysis/' for visualizations")

if __name__ == "__main__":
    main()