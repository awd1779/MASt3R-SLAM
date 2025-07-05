#!/usr/bin/env python3
"""
Test script for semantic processor debug visualization.
This script tests the SAM and CLIP visualization functionality.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

from mast3r_slam.semantic_processor import (
    process_frame_for_semantics, 
    enable_debug_visualization,
    set_debug_options,
    TEXT_PROMPTS
)

def create_test_image(height=480, width=640):
    """Create a simple test image with geometric shapes."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Background (gray)
    image[:] = [128, 128, 128]
    
    # Floor (brown rectangle at bottom)
    image[height-100:, :] = [139, 69, 19]
    
    # Wall (light gray upper part)
    image[:height-100, :] = [200, 200, 200]
    
    # Table (rectangular brown shape)
    table_y1, table_y2 = height//2, height//2 + 80
    table_x1, table_x2 = width//4, 3*width//4
    image[table_y1:table_y2, table_x1:table_x2] = [160, 82, 45]
    
    # Chair (smaller rectangle)
    chair_y1, chair_y2 = height//3, height//3 + 50
    chair_x1, chair_x2 = width//6, width//6 + 60
    image[chair_y1:chair_y2, chair_x1:chair_x2] = [101, 67, 33]
    
    # Window (light blue rectangle)
    window_y1, window_y2 = height//6, height//6 + 100
    window_x1, window_x2 = 3*width//4, 3*width//4 + 120
    image[window_y1:window_y2, window_x1:window_x2] = [173, 216, 230]
    
    # Convert to tensor format (C, H, W) and normalize to 0-1
    image_tensor = torch.from_numpy(image).float() / 255.0
    image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
    
    return image_tensor

def test_semantic_debug():
    """Test the semantic processor debug functionality."""
    print("=" * 60)
    print("Testing Semantic Processor Debug Visualization")
    print("=" * 60)
    
    # Configure debug options
    enable_debug_visualization(True, "test_debug_output")
    set_debug_options(save_sam=True, save_clips=True, save_classifications=True, max_masks=15)
    
    # Create test image
    print("\n1. Creating synthetic test image...")
    test_image = create_test_image()
    print(f"   Test image shape: {test_image.shape}")
    
    # Test with synthetic image
    print("\n2. Running semantic processing with debug visualization...")
    try:
        local_mask, local_map = process_frame_for_semantics(
            image_tensor_chw_0_1_rgb=test_image,
            text_prompts_for_clip=TEXT_PROMPTS,
            enable_debug_viz=True,
            frame_id=0
        )
        
        print(f"   ✓ Processing completed successfully")
        print(f"   ✓ Local mask shape: {local_mask.shape}")
        print(f"   ✓ Local map: {local_map}")
        
        # Check if debug output directory was created
        debug_dir = Path("test_debug_output")
        if debug_dir.exists():
            subdirs = list(debug_dir.iterdir())
            print(f"   ✓ Debug output directory created with {len(subdirs)} subdirectories")
            
            # List contents
            for subdir in subdirs:
                if subdir.is_dir():
                    sam_masks = list((subdir / "sam_masks").glob("*.png"))
                    clip_crops = list((subdir / "clip_crops").glob("*.png"))
                    classifications = list((subdir / "classifications").glob("*.png"))
                    json_files = list((subdir / "classifications").glob("*.json"))
                    
                    print(f"   📁 {subdir.name}:")
                    print(f"      - SAM masks: {len(sam_masks)} files")
                    print(f"      - CLIP crops: {len(clip_crops)} files")
                    print(f"      - Classification images: {len(classifications)} files")
                    print(f"      - Classification data: {len(json_files)} JSON files")
        else:
            print("   ⚠ Warning: Debug output directory not found")
            
    except Exception as e:
        print(f"   ✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n3. Testing with real image (if available)...")
    # Try to load an actual image for more realistic testing
    possible_images = [
        "test_image_for_semantic_processor.png",
        "test_image_for_profiling.png", 
        "semantic_processor_test_output.png"
    ]
    
    real_image_tested = False
    for img_path in possible_images:
        if Path(img_path).exists():
            try:
                import cv2
                img = cv2.imread(str(img_path))
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_tensor = torch.from_numpy(img_rgb).float() / 255.0
                    img_tensor = img_tensor.permute(2, 0, 1)
                    
                    print(f"   Loading real image: {img_path}")
                    local_mask, local_map = process_frame_for_semantics(
                        image_tensor_chw_0_1_rgb=img_tensor,
                        text_prompts_for_clip=TEXT_PROMPTS,
                        enable_debug_viz=True,
                        frame_id=1
                    )
                    print(f"   ✓ Real image processing completed")
                    real_image_tested = True
                    break
            except Exception as e:
                print(f"   ⚠ Could not process {img_path}: {e}")
    
    if not real_image_tested:
        print("   ℹ No real images found for testing, using synthetic image only")
    
    print("\n" + "=" * 60)
    print("🎉 Debug visualization test completed!")
    print("Check the 'test_debug_output' directory for generated visualizations.")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    test_semantic_debug()