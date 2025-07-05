#!/usr/bin/env python3
"""
Compare CLIP performance on cropped regions vs full image with masked boundaries.
This will help identify if the issue is with cropping or other preprocessing.
"""

import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

from mast3r_slam.semantic_processor import (
    load_instance_segmentation_model, 
    load_clip_model,
    get_tensor_crop_from_mask,
    TEXT_PROMPTS,
    SEMANTIC_DEVICE
)

def create_masked_full_image(image_tensor_chw, mask_torch_hw, background_type="black"):
    """Create full-size image with mask applied."""
    # image_tensor_chw: (3, H, W) in [0, 1]
    # mask_torch_hw: (H, W) boolean mask
    
    masked_image = image_tensor_chw.clone()
    
    if background_type == "black":
        # Set non-mask areas to black
        masked_image[:, ~mask_torch_hw] = 0.0
    elif background_type == "blur":
        # Blur non-mask areas
        import torch.nn.functional as F
        # Convert to BHWC for blur
        full_img = image_tensor_chw.unsqueeze(0)  # Add batch dim
        blurred = F.avg_pool2d(full_img, kernel_size=15, stride=1, padding=7)
        blurred = blurred.squeeze(0)  # Remove batch dim
        
        # Use blurred background, original for mask
        masked_image[:, ~mask_torch_hw] = blurred[:, ~mask_torch_hw]
    elif background_type == "gray":
        # Set non-mask areas to gray
        masked_image[:, ~mask_torch_hw] = 0.5
    elif background_type == "highlight":
        # Dim non-mask areas
        masked_image[:, ~mask_torch_hw] *= 0.3
    
    return masked_image

def test_clip_on_different_inputs(image_tensor_chw, sam_masks_data_list, max_masks=5):
    """Test CLIP on different input formats and compare results."""
    
    print("=" * 80)
    print("🧪 CLIP COMPARISON TEST: Crops vs Full Image with Masks")
    print("=" * 80)
    
    # Load CLIP model
    try:
        clip_model, clip_img_val_preprocess, text_features_tensor, cached_prompts_list = load_clip_model(
            text_prompts=TEXT_PROMPTS, device=SEMANTIC_DEVICE
        )
        print(f"✓ CLIP model loaded with {len(cached_prompts_list)} prompts")
    except Exception as e:
        print(f"✗ Failed to load CLIP model: {e}")
        return
    
    # Get normalization transform
    normalize_transform = None
    if hasattr(clip_img_val_preprocess, 'transforms'):
        for t in clip_img_val_preprocess.transforms:
            if hasattr(t, 'mean') and hasattr(t, 'std') and hasattr(t, '__call__'):
                normalize_transform = t
                break
    
    if normalize_transform is None:
        import torchvision.transforms as transforms
        normalize_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        print("⚠ Using fallback ImageNet normalization")
    else:
        print(f"✓ Using CLIP normalization: mean={normalize_transform.mean}, std={normalize_transform.std}")
    
    debug_dir = Path("clip_comparison_test")
    debug_dir.mkdir(exist_ok=True)
    
    # Save original image
    orig_img_np = image_tensor_chw.permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize=(8, 6))
    plt.imshow(orig_img_np)
    plt.title("Original Image")
    plt.axis('off')
    plt.savefig(debug_dir / "00_original_image.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    results_comparison = []
    
    for i, mask_data in enumerate(sam_masks_data_list[:max_masks]):
        print(f"\n--- Testing Mask {i} (area: {mask_data['area']}) ---")
        
        mask = mask_data['segmentation']
        mask_torch = torch.from_numpy(mask.astype(bool)).to(device=SEMANTIC_DEVICE)
        
        # Method 1: Cropped region (current approach)
        print("🔸 Method 1: Cropped region")
        crop_tensor = get_tensor_crop_from_mask(image_tensor_chw, mask_torch, output_size=(224, 224))
        
        if crop_tensor is not None:
            crop_normalized = normalize_transform(crop_tensor.unsqueeze(0)).squeeze(0)
            
            with torch.no_grad():
                crop_features = clip_model.encode_image(crop_normalized.unsqueeze(0))
                crop_features = crop_features / crop_features.norm(dim=-1, keepdim=True)
                crop_similarities = (100.0 * crop_features @ text_features_tensor.T)
                crop_scores, crop_indices = crop_similarities.topk(5, dim=1)
            
            print(f"   Top 5 predictions (crop):")
            for j in range(5):
                cls = cached_prompts_list[crop_indices[0, j].item()]
                score = crop_scores[0, j].item()
                print(f"     {j+1}. {cls}: {score:.2f}")
            
            # Save crop visualization
            crop_np = crop_tensor.permute(1, 2, 0).cpu().numpy()
            plt.figure(figsize=(6, 6))
            plt.imshow(np.clip(crop_np, 0, 1))
            plt.title(f"Mask {i}: Crop Method\nBest: {cached_prompts_list[crop_indices[0, 0].item()]} ({crop_scores[0, 0].item():.2f})")
            plt.axis('off')
            plt.savefig(debug_dir / f"mask_{i:02d}_crop.png", dpi=150, bbox_inches='tight')
            plt.close()
        else:
            print("   ✗ Could not extract crop")
            crop_scores = torch.tensor([[0.0]])
            crop_indices = torch.tensor([[0]])
        
        # Method 2: Full image with black background
        print("🔸 Method 2: Full image with black background")
        masked_black = create_masked_full_image(image_tensor_chw, mask_torch, "black")
        
        # Resize to 224x224 for CLIP
        import torch.nn.functional as F
        masked_black_resized = F.interpolate(
            masked_black.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
        ).squeeze(0)
        
        masked_black_normalized = normalize_transform(masked_black_resized.unsqueeze(0)).squeeze(0)
        
        with torch.no_grad():
            black_features = clip_model.encode_image(masked_black_normalized.unsqueeze(0))
            black_features = black_features / black_features.norm(dim=-1, keepdim=True)
            black_similarities = (100.0 * black_features @ text_features_tensor.T)
            black_scores, black_indices = black_similarities.topk(5, dim=1)
        
        print(f"   Top 5 predictions (black background):")
        for j in range(5):
            cls = cached_prompts_list[black_indices[0, j].item()]
            score = black_scores[0, j].item()
            print(f"     {j+1}. {cls}: {score:.2f}")
        
        # Save black background visualization
        masked_black_np = masked_black_resized.permute(1, 2, 0).cpu().numpy()
        plt.figure(figsize=(6, 6))
        plt.imshow(np.clip(masked_black_np, 0, 1))
        plt.title(f"Mask {i}: Black Background\nBest: {cached_prompts_list[black_indices[0, 0].item()]} ({black_scores[0, 0].item():.2f})")
        plt.axis('off')
        plt.savefig(debug_dir / f"mask_{i:02d}_black_bg.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Method 3: Full image with highlighted region
        print("🔸 Method 3: Full image with highlighted region")
        masked_highlight = create_masked_full_image(image_tensor_chw, mask_torch, "highlight")
        
        masked_highlight_resized = F.interpolate(
            masked_highlight.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
        ).squeeze(0)
        
        masked_highlight_normalized = normalize_transform(masked_highlight_resized.unsqueeze(0)).squeeze(0)
        
        with torch.no_grad():
            highlight_features = clip_model.encode_image(masked_highlight_normalized.unsqueeze(0))
            highlight_features = highlight_features / highlight_features.norm(dim=-1, keepdim=True)
            highlight_similarities = (100.0 * highlight_features @ text_features_tensor.T)
            highlight_scores, highlight_indices = highlight_similarities.topk(5, dim=1)
        
        print(f"   Top 5 predictions (highlighted):")
        for j in range(5):
            cls = cached_prompts_list[highlight_indices[0, j].item()]
            score = highlight_scores[0, j].item()
            print(f"     {j+1}. {cls}: {score:.2f}")
        
        # Save highlight visualization
        masked_highlight_np = masked_highlight_resized.permute(1, 2, 0).cpu().numpy()
        plt.figure(figsize=(6, 6))
        plt.imshow(np.clip(masked_highlight_np, 0, 1))
        plt.title(f"Mask {i}: Highlighted\nBest: {cached_prompts_list[highlight_indices[0, 0].item()]} ({highlight_scores[0, 0].item():.2f})")
        plt.axis('off')
        plt.savefig(debug_dir / f"mask_{i:02d}_highlight.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Method 4: Test with simple color patches (control test)
        print("🔸 Method 4: Control test with solid colors")
        test_colors = {
            "red": [1.0, 0.0, 0.0],
            "blue": [0.0, 0.0, 1.0], 
            "green": [0.0, 1.0, 0.0],
            "white": [1.0, 1.0, 1.0],
            "brown": [0.6, 0.3, 0.1]
        }
        
        best_color_score = 0
        best_color = "none"
        
        for color_name, rgb in test_colors.items():
            color_img = torch.tensor(rgb, device=SEMANTIC_DEVICE).view(3, 1, 1).expand(3, 224, 224)
            color_normalized = normalize_transform(color_img.unsqueeze(0)).squeeze(0)
            
            with torch.no_grad():
                color_features = clip_model.encode_image(color_normalized.unsqueeze(0))
                color_features = color_features / color_features.norm(dim=-1, keepdim=True)
                color_similarities = (100.0 * color_features @ text_features_tensor.T)
                color_score, color_idx = color_similarities.max(dim=1)
                
                if color_score.item() > best_color_score:
                    best_color_score = color_score.item()
                    best_color = f"{color_name} -> {cached_prompts_list[color_idx.item()]}"
        
        print(f"   Best color test: {best_color} ({best_color_score:.2f})")
        
        # Store results for comparison
        results_comparison.append({
            'mask_id': i,
            'area': mask_data['area'],
            'crop_score': crop_scores[0, 0].item() if crop_tensor is not None else 0,
            'crop_class': cached_prompts_list[crop_indices[0, 0].item()] if crop_tensor is not None else "none",
            'black_score': black_scores[0, 0].item(),
            'black_class': cached_prompts_list[black_indices[0, 0].item()],
            'highlight_score': highlight_scores[0, 0].item(),
            'highlight_class': cached_prompts_list[highlight_indices[0, 0].item()],
            'control_score': best_color_score,
            'control_result': best_color
        })
    
    # Create comparison summary
    print(f"\n" + "=" * 80)
    print("📊 COMPARISON SUMMARY")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Score comparison
    masks = [r['mask_id'] for r in results_comparison]
    crop_scores = [r['crop_score'] for r in results_comparison]
    black_scores = [r['black_score'] for r in results_comparison]
    highlight_scores = [r['highlight_score'] for r in results_comparison]
    control_scores = [r['control_score'] for r in results_comparison]
    
    x = np.arange(len(masks))
    width = 0.2
    
    axes[0, 0].bar(x - 1.5*width, crop_scores, width, label='Crop', alpha=0.8)
    axes[0, 0].bar(x - 0.5*width, black_scores, width, label='Black BG', alpha=0.8)
    axes[0, 0].bar(x + 0.5*width, highlight_scores, width, label='Highlight', alpha=0.8)
    axes[0, 0].bar(x + 1.5*width, control_scores, width, label='Control', alpha=0.8)
    axes[0, 0].set_xlabel('Mask ID')
    axes[0, 0].set_ylabel('Confidence Score')
    axes[0, 0].set_title('CLIP Confidence Scores by Method')
    axes[0, 0].legend()
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(masks)
    
    # Average scores
    avg_scores = [np.mean(crop_scores), np.mean(black_scores), np.mean(highlight_scores), np.mean(control_scores)]
    methods = ['Crop', 'Black BG', 'Highlight', 'Control']
    
    axes[0, 1].bar(methods, avg_scores, color=['blue', 'red', 'green', 'orange'], alpha=0.7)
    axes[0, 1].set_ylabel('Average Confidence Score')
    axes[0, 1].set_title('Average Performance by Method')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Detailed results table
    axes[1, 0].axis('off')
    table_data = []
    for r in results_comparison:
        table_data.append([
            f"Mask {r['mask_id']}", 
            f"{r['crop_class'][:8]} ({r['crop_score']:.1f})",
            f"{r['black_class'][:8]} ({r['black_score']:.1f})",
            f"{r['highlight_class'][:8]} ({r['highlight_score']:.1f})"
        ])
    
    table = axes[1, 0].table(cellText=table_data,
                           colLabels=['Mask', 'Crop', 'Black BG', 'Highlight'],
                           cellLoc='center',
                           loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    axes[1, 0].set_title('Detailed Results Comparison')
    
    # Score distribution
    all_scores = crop_scores + black_scores + highlight_scores + control_scores
    axes[1, 1].hist([crop_scores, black_scores, highlight_scores, control_scores], 
                   bins=10, alpha=0.7, label=['Crop', 'Black BG', 'Highlight', 'Control'])
    axes[1, 1].set_xlabel('Confidence Score')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Score Distribution by Method')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(debug_dir / "comparison_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print analysis
    print(f"\n🔍 ANALYSIS:")
    print(f"   Average Crop Score: {np.mean(crop_scores):.2f}")
    print(f"   Average Black BG Score: {np.mean(black_scores):.2f}")
    print(f"   Average Highlight Score: {np.mean(highlight_scores):.2f}")
    print(f"   Average Control Score: {np.mean(control_scores):.2f}")
    
    best_method = methods[np.argmax(avg_scores)]
    print(f"\n🏆 Best performing method: {best_method}")
    
    if np.mean(control_scores) > max(np.mean(crop_scores), np.mean(black_scores), np.mean(highlight_scores)):
        print("⚠️  Control test (solid colors) performs better than real images!")
        print("   This suggests a fundamental issue with image preprocessing or CLIP model.")
    
    print(f"\n📁 Detailed visualizations saved to: {debug_dir}/")
    print("=" * 80)

def main():
    """Main test function."""
    # Load a test image and run SAM
    from mast3r_slam.dataloader import load_dataset
    from mast3r_slam.config import load_config
    
    print("🚀 Starting CLIP comparison test...")
    
    # Load config and dataset  
    load_config("config/base.yaml")
    dataset = load_dataset("datasets/my/")
    
    # Get first image
    timestamp, raw_img = dataset[0]
    
    # Convert to proper format (mimic frame creation)
    from mast3r_slam.frame import create_frame
    import lietorch
    frame = create_frame(0, raw_img, lietorch.Sim3.Identity(1, device="cuda:0"), img_size=512, device="cuda:0")
    
    # Fix the range issue we identified earlier
    image_tensor = frame.img.squeeze(0)  # Remove batch dim
    if image_tensor.min() < -0.1:
        image_tensor = (image_tensor + 1.0) / 2.0
    
    print(f"Image tensor range: [{image_tensor.min():.3f}, {image_tensor.max():.3f}]")
    
    # Run SAM
    sam_generator = load_instance_segmentation_model(device=SEMANTIC_DEVICE)
    image_hwc_uint8_rgb = (image_tensor.permute(1, 2, 0) * 255.0).byte().cpu().numpy()
    sam_masks_data_list = sam_generator.generate(image_hwc_uint8_rgb)
    
    print(f"SAM found {len(sam_masks_data_list)} masks")
    
    if not sam_masks_data_list:
        print("No masks found! Creating a dummy mask for testing...")
        h, w = image_tensor.shape[1], image_tensor.shape[2]
        dummy_mask = np.zeros((h, w), dtype=bool)
        dummy_mask[h//4:3*h//4, w//4:3*w//4] = True
        sam_masks_data_list = [{'segmentation': dummy_mask, 'area': dummy_mask.sum()}]
    
    # Run the comparison test
    test_clip_on_different_inputs(image_tensor, sam_masks_data_list, max_masks=5)

if __name__ == "__main__":
    main()