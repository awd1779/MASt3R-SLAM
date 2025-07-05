#!/usr/bin/env python3
"""
Enhanced debugging script for semantic processing pipeline.
Traces issues with color spaces and CLIP preprocessing.
"""

import os
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

def save_debug_image(image_array, title, filepath, is_tensor=False):
    """Save an image with detailed info for debugging."""
    if is_tensor:
        if image_array.dim() == 3:  # CHW
            img_np = image_array.permute(1, 2, 0).cpu().numpy()
        else:  # HW
            img_np = image_array.cpu().numpy()
    else:
        img_np = image_array
    
    # Ensure proper range for display
    if img_np.dtype == np.float32 or img_np.dtype == np.float64:
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    
    plt.figure(figsize=(10, 8))
    if len(img_np.shape) == 3:
        plt.imshow(img_np)
    else:
        plt.imshow(img_np, cmap='gray')
    
    plt.title(f"{title}\nShape: {img_np.shape}, Dtype: {img_np.dtype}, Range: [{img_np.min():.3f}, {img_np.max():.3f}]")
    plt.axis('off')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {filepath}")

def test_color_space_conversion():
    """Test color space handling throughout the pipeline."""
    print("=" * 60)
    print("1. TESTING COLOR SPACE CONVERSION")
    print("=" * 60)
    
    debug_dir = Path("debug_pipeline_test")
    debug_dir.mkdir(exist_ok=True)
    
    # Load a real image if available
    test_images = [
        "test_image_for_semantic_processor.png",
        "semantic_processor_test_output.png"
    ]
    
    image_bgr = None
    for img_path in test_images:
        if Path(img_path).exists():
            image_bgr = cv2.imread(str(img_path))
            if image_bgr is not None:
                print(f"   Loading test image: {img_path}")
                break
    
    if image_bgr is None:
        print("   Creating synthetic test image...")
        image_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        image_bgr[:] = [128, 128, 128]  # Gray background
        image_bgr[200:280, 200:400] = [0, 0, 255]  # Red rectangle (BGR)
        image_bgr[300:380, 300:500] = [255, 0, 0]  # Blue rectangle (BGR)
        image_bgr[100:180, 400:600] = [0, 255, 0]  # Green rectangle (BGR)
    
    # Save original BGR
    save_debug_image(image_bgr, "Original BGR (from cv2.imread)", 
                    debug_dir / "01_original_bgr.png")
    
    # Convert to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    save_debug_image(image_rgb, "Converted to RGB", 
                    debug_dir / "02_converted_rgb.png")
    
    # Convert to tensor (CHW, 0-1)
    image_tensor = torch.from_numpy(image_rgb).float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
    save_debug_image(image_tensor, "Tensor CHW 0-1 RGB", 
                    debug_dir / "03_tensor_chw.png", is_tensor=True)
    
    # Convert back for SAM (HWC, 0-255, uint8)
    image_hwc_uint8_rgb = (image_tensor.permute(1, 2, 0) * 255.0).byte().cpu().numpy()
    save_debug_image(image_hwc_uint8_rgb, "Back to HWC uint8 for SAM", 
                    debug_dir / "04_sam_input.png")
    
    return image_tensor, image_hwc_uint8_rgb

def test_clip_preprocessing():
    """Test CLIP preprocessing pipeline step by step."""
    print("\n" + "=" * 60)
    print("2. TESTING CLIP PREPROCESSING PIPELINE")
    print("=" * 60)
    
    debug_dir = Path("debug_pipeline_test")
    
    # Get test image
    image_tensor, image_hwc_uint8_rgb = test_color_space_conversion()
    
    print("\n   Loading CLIP model...")
    try:
        clip_model, clip_img_val_preprocess, text_features_tensor, cached_prompts_list = load_clip_model(
            text_prompts=TEXT_PROMPTS, device=SEMANTIC_DEVICE
        )
        print(f"   ✓ CLIP model loaded, text features shape: {text_features_tensor.shape}")
    except Exception as e:
        print(f"   ✗ CLIP model loading failed: {e}")
        return
    
    print("\n   Loading SAM model...")
    try:
        sam_generator = load_instance_segmentation_model(device=SEMANTIC_DEVICE)
        print("   ✓ SAM model loaded")
    except Exception as e:
        print(f"   ✗ SAM model loading failed: {e}")
        return
    
    print("\n   Running SAM segmentation...")
    sam_masks_data_list = sam_generator.generate(image_hwc_uint8_rgb)
    print(f"   ✓ SAM found {len(sam_masks_data_list)} masks")
    
    if not sam_masks_data_list:
        print("   ⚠ No masks found, creating dummy mask for testing")
        h, w = image_hwc_uint8_rgb.shape[:2]
        dummy_mask = np.zeros((h, w), dtype=bool)
        dummy_mask[h//4:3*h//4, w//4:3*w//4] = True
        sam_masks_data_list = [{'segmentation': dummy_mask, 'area': dummy_mask.sum()}]
    
    # Test first few masks
    for i, mask_data in enumerate(sam_masks_data_list[:3]):
        print(f"\n   Processing mask {i} (area: {mask_data['area']})...")
        
        mask = mask_data['segmentation']
        segment_torch_bool = torch.from_numpy(mask.astype(bool)).to(device=SEMANTIC_DEVICE)
        
        # Save the mask
        save_debug_image(mask, f"SAM Mask {i}", 
                        debug_dir / f"05_sam_mask_{i}.png")
        
        # Get crop before CLIP processing
        print(f"      Getting crop from mask...")
        raw_crop = get_tensor_crop_from_mask(image_tensor, segment_torch_bool, output_size=None)
        if raw_crop is not None:
            save_debug_image(raw_crop, f"Raw Crop {i} (before resize)", 
                            debug_dir / f"06_raw_crop_{i}.png", is_tensor=True)
            
            # Get resized crop
            from mast3r_slam.semantic_processor import _clip_input_resolution
            resized_crop = get_tensor_crop_from_mask(image_tensor, segment_torch_bool, 
                                                   output_size=_clip_input_resolution)
            if resized_crop is not None:
                save_debug_image(resized_crop, f"Resized Crop {i} ({_clip_input_resolution})", 
                                debug_dir / f"07_resized_crop_{i}.png", is_tensor=True)
                
                # Test CLIP preprocessing manually
                print(f"      Testing CLIP preprocessing...")
                
                # Find normalization transform
                normalize_transform = None
                if hasattr(clip_img_val_preprocess, 'transforms'):
                    for t in clip_img_val_preprocess.transforms:
                        if hasattr(t, 'mean') and hasattr(t, 'std'):
                            normalize_transform = t
                            break
                
                if normalize_transform is not None:
                    print(f"         Found normalize: mean={normalize_transform.mean}, std={normalize_transform.std}")
                    
                    # Apply normalization
                    normalized_crop = normalize_transform(resized_crop.unsqueeze(0)).squeeze(0)
                    save_debug_image(normalized_crop, f"Normalized Crop {i}", 
                                    debug_dir / f"08_normalized_crop_{i}.png", is_tensor=True)
                    
                    # Test CLIP inference on single crop
                    with torch.no_grad():
                        crop_features = clip_model.encode_image(normalized_crop.unsqueeze(0))
                        crop_features = crop_features / crop_features.norm(dim=-1, keepdim=True)
                        
                        # Calculate similarities
                        similarities = (100.0 * crop_features @ text_features_tensor.T)
                        scores, indices = similarities.max(dim=1)
                        
                        best_class = cached_prompts_list[indices[0].item()]
                        best_score = scores[0].item()
                        
                        print(f"         CLIP result: '{best_class}' (confidence: {best_score:.2f})")
                        
                        # Get top 5 predictions
                        top5_scores, top5_indices = similarities[0].topk(5)
                        print(f"         Top 5 predictions:")
                        for j in range(5):
                            cls = cached_prompts_list[top5_indices[j].item()]
                            score = top5_scores[j].item()
                            print(f"           {j+1}. {cls}: {score:.2f}")
                else:
                    print(f"         ✗ Could not find normalization transform")
        else:
            print(f"      ✗ Could not extract crop from mask")
        
        if i >= 2:  # Limit to first 3 masks
            break

def test_clip_with_known_images():
    """Test CLIP with simple, known images to verify it's working."""
    print("\n" + "=" * 60)
    print("3. TESTING CLIP WITH KNOWN OBJECTS")
    print("=" * 60)
    
    debug_dir = Path("debug_pipeline_test")
    
    # Create simple test images for known objects
    test_objects = {
        "red_square": ([255, 0, 0], "red"),
        "blue_square": ([0, 0, 255], "blue"),  
        "green_square": ([0, 255, 0], "green"),
        "white_square": ([255, 255, 255], "white"),
        "black_square": ([0, 0, 0], "black")
    }
    
    print("\n   Loading CLIP model...")
    try:
        clip_model, clip_img_val_preprocess, text_features_tensor, cached_prompts_list = load_clip_model(
            text_prompts=["red", "blue", "green", "white", "black", "background", "square", "shape"], 
            device=SEMANTIC_DEVICE
        )
        print("   ✓ CLIP model loaded with color prompts")
    except Exception as e:
        print(f"   ✗ CLIP model loading failed: {e}")
        return
    
    for obj_name, (color, expected_class) in test_objects.items():
        print(f"\n   Testing {obj_name} (expecting '{expected_class}')...")
        
        # Create simple colored square
        img = np.full((224, 224, 3), color, dtype=np.uint8)
        img_tensor = torch.from_numpy(img).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
        
        save_debug_image(img_tensor, f"Test {obj_name}", 
                        debug_dir / f"09_test_{obj_name}.png", is_tensor=True)
        
        # Apply CLIP preprocessing
        try:
            # Find normalization
            normalize_transform = None
            if hasattr(clip_img_val_preprocess, 'transforms'):
                for t in clip_img_val_preprocess.transforms:
                    if hasattr(t, 'mean') and hasattr(t, 'std'):
                        normalize_transform = t
                        break
            
            if normalize_transform is not None:
                normalized_img = normalize_transform(img_tensor.unsqueeze(0)).squeeze(0)
                
                with torch.no_grad():
                    img_features = clip_model.encode_image(normalized_img.unsqueeze(0))
                    img_features = img_features / img_features.norm(dim=-1, keepdim=True)
                    
                    similarities = (100.0 * img_features @ text_features_tensor.T)
                    scores, indices = similarities.max(dim=1)
                    
                    best_class = cached_prompts_list[indices[0].item()]
                    best_score = scores[0].item()
                    
                    success = "✓" if expected_class.lower() in best_class.lower() else "✗"
                    print(f"      {success} Predicted: '{best_class}' (confidence: {best_score:.2f})")
                    
                    # Show top 3
                    top3_scores, top3_indices = similarities[0].topk(3)
                    for j in range(3):
                        cls = cached_prompts_list[top3_indices[j].item()]
                        score = top3_scores[j].item()
                        print(f"         {j+1}. {cls}: {score:.2f}")
            else:
                print(f"      ✗ Could not find normalization transform")
        except Exception as e:
            print(f"      ✗ CLIP inference failed: {e}")

def main():
    """Run all debugging tests."""
    print("🔍 SEMANTIC PROCESSING PIPELINE DEBUG")
    print("=" * 60)
    
    test_color_space_conversion()
    test_clip_preprocessing()
    test_clip_with_known_images()
    
    print("\n" + "=" * 60)
    print("🎯 DEBUG COMPLETE!")
    print("Check 'debug_pipeline_test/' directory for detailed visualizations.")
    print("Look for:")
    print("  - Color space consistency across conversions")
    print("  - CLIP confidence scores on simple test objects")
    print("  - Preprocessing artifacts in crops")
    print("=" * 60)

if __name__ == "__main__":
    main()