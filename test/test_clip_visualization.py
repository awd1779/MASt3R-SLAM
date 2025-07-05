#!/usr/bin/env python3
"""
Test script to verify CLIP input visualization with HOV-SG approach.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from mast3r_slam.semantic_processor import process_frame_for_semantics
from mast3r_slam.dataloader import load_dataset
from mast3r_slam.config import load_config
from mast3r_slam.frame import create_frame
import lietorch

def main():
    print("🎨 TESTING CLIP INPUT VISUALIZATION")
    print("=" * 60)
    
    # Load dataset
    load_config("config/base.yaml")
    dataset = load_dataset("datasets/my/")
    
    # Get first image
    timestamp, raw_img = dataset[0]
    frame = create_frame(0, raw_img, lietorch.Sim3.Identity(1, device="cuda:0"), img_size=512, device="cuda:0")
    
    # Fix range if needed
    image_tensor = frame.img.squeeze(0)
    if image_tensor.min() < -0.1:
        image_tensor = (image_tensor + 1.0) / 2.0
    
    print(f"📸 Processing image with shape: {image_tensor.shape}")
    
    # Process with semantic processor (debug viz enabled by default)
    local_mask, local_map = process_frame_for_semantics(
        image_tensor_chw_0_1_rgb=image_tensor,
        enable_debug_viz=True,  # Ensure debug is enabled
        frame_id=1
    )
    
    print(f"\n✅ Semantic processing complete!")
    print(f"📊 Detected {len(local_map) - 1} objects")
    
    print("\n📁 Check debug outputs in:")
    print("  - debug_semantic_output/sam_masks/")
    print("  - debug_semantic_output/clip_crops/")
    print("  - debug_semantic_output/classification_results/")
    print("  - debug_semantic_output/hovsg_clip_inputs/")
    
    print("\n🎯 HOV-SG visualizations include:")
    print("  1. Full image input (224x224)")
    print("  2. Grid of masked regions with predictions")
    print("  3. Feature fusion visualization")

if __name__ == "__main__":
    main()