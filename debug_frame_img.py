#!/usr/bin/env python3
"""
Debug script to investigate frame.img format and preprocessing issues.
"""

import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

from mast3r_slam.frame import create_frame
from mast3r_slam.dataloader import load_dataset
from mast3r_slam.config import load_config, config

def save_debug_tensor(tensor, name, output_dir):
    """Save tensor as image with debug info."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if tensor.dim() == 4:  # BCHW
        tensor = tensor.squeeze(0)
    
    if tensor.dim() == 3:  # CHW
        img_np = tensor.permute(1, 2, 0).cpu().numpy()
    else:  # HW
        img_np = tensor.cpu().numpy()
    
    # Create debug plot
    plt.figure(figsize=(12, 8))
    
    # Main image
    plt.subplot(2, 2, 1)
    if len(img_np.shape) == 3:
        # Handle different ranges
        if img_np.min() < 0 or img_np.max() > 1:
            # Normalize for display
            display_img = img_np.copy()
            display_img = (display_img - display_img.min()) / (display_img.max() - display_img.min())
            plt.imshow(display_img)
            plt.title(f"{name}\n(Normalized for display)")
        else:
            plt.imshow(np.clip(img_np, 0, 1))
            plt.title(f"{name}")
    else:
        plt.imshow(img_np, cmap='gray')
        plt.title(f"{name}")
    plt.axis('off')
    
    # Histogram per channel
    plt.subplot(2, 2, 2)
    if len(img_np.shape) == 3:
        colors = ['red', 'green', 'blue']
        for i in range(3):
            plt.hist(img_np[:, :, i].flatten(), bins=50, alpha=0.7, 
                    color=colors[i], label=f'Channel {i}')
        plt.legend()
    else:
        plt.hist(img_np.flatten(), bins=50, alpha=0.7)
    plt.title("Pixel Value Distribution")
    plt.xlabel("Value")
    plt.ylabel("Count")
    
    # Statistics
    plt.subplot(2, 2, 3)
    plt.axis('off')
    stats_text = f"""
    Shape: {tensor.shape}
    Dtype: {tensor.dtype}
    Device: {tensor.device}
    Min: {tensor.min().item():.6f}
    Max: {tensor.max().item():.6f}
    Mean: {tensor.mean().item():.6f}
    Std: {tensor.std().item():.6f}
    """
    if len(img_np.shape) == 3:
        for i in range(3):
            channel_data = tensor[i] if tensor.dim() == 3 else tensor[:, :, i]
            stats_text += f"\nCh{i} range: [{channel_data.min():.3f}, {channel_data.max():.3f}]"
    
    plt.text(0.1, 0.9, stats_text, fontsize=10, verticalalignment='top',
             fontfamily='monospace', transform=plt.gca().transAxes)
    
    # Save raw values sample
    plt.subplot(2, 2, 4)
    plt.axis('off')
    if len(img_np.shape) == 3:
        h, w, c = img_np.shape
        sample_text = "Sample pixel values (top-left 5x5):\n"
        for i in range(min(5, h)):
            for j in range(min(5, w)):
                sample_text += f"({img_np[i,j,0]:.2f},{img_np[i,j,1]:.2f},{img_np[i,j,2]:.2f}) "
            sample_text += "\n"
    else:
        h, w = img_np.shape
        sample_text = "Sample pixel values (top-left 5x5):\n"
        for i in range(min(5, h)):
            for j in range(min(5, w)):
                sample_text += f"{img_np[i,j]:.2f} "
            sample_text += "\n"
    
    plt.text(0.1, 0.9, sample_text, fontsize=8, verticalalignment='top',
             fontfamily='monospace', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{name.replace(' ', '_').lower()}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved debug visualization: {output_dir / f'{name.replace(' ', '_').lower()}.png'}")

def debug_frame_creation():
    """Debug the frame creation process step by step."""
    print("=" * 60)
    print("DEBUGGING FRAME CREATION AND IMAGE PREPROCESSING")
    print("=" * 60)
    
    # Load config and dataset
    load_config("config/base.yaml")
    dataset = load_dataset("datasets/my/")
    
    print(f"\n1. Loading first image from dataset...")
    timestamp, raw_img = dataset[0]
    print(f"   Raw image shape: {raw_img.shape}")
    print(f"   Raw image dtype: {raw_img.dtype}")
    print(f"   Raw image range: [{raw_img.min():.3f}, {raw_img.max():.3f}]")
    
    # Save raw image
    debug_dir = Path("debug_frame_img")
    save_debug_tensor(torch.from_numpy(raw_img).permute(2, 0, 1), "01_Raw_Dataset_Image", debug_dir)
    
    print(f"\n2. Creating frame with create_frame()...")
    import lietorch
    frame = create_frame(0, raw_img, lietorch.Sim3.Identity(1, device="cuda:0"), img_size=512, device="cuda:0")
    
    print(f"   frame.img shape: {frame.img.shape}")
    print(f"   frame.img dtype: {frame.img.dtype}")
    print(f"   frame.img device: {frame.img.device}")
    print(f"   frame.img range: [{frame.img.min():.6f}, {frame.img.max():.6f}]")
    
    save_debug_tensor(frame.img, "02_Frame_img_Tensor", debug_dir)
    
    print(f"\n3. Checking frame.uimg...")
    print(f"   frame.uimg shape: {frame.uimg.shape}")
    print(f"   frame.uimg dtype: {frame.uimg.dtype}")
    print(f"   frame.uimg range: [{frame.uimg.min():.6f}, {frame.uimg.max():.6f}]")
    
    save_debug_tensor(frame.uimg.permute(2, 0, 1), "03_Frame_uimg_Tensor", debug_dir)
    
    print(f"\n4. Checking conversion for semantic processor...")
    # This is what semantic processor receives
    semantic_input = frame.img.squeeze(0)  # Remove batch dimension
    print(f"   semantic_input shape: {semantic_input.shape}")
    print(f"   semantic_input range: [{semantic_input.min():.6f}, {semantic_input.max():.6f}]")
    
    save_debug_tensor(semantic_input, "04_Semantic_Processor_Input", debug_dir)
    
    print(f"\n5. Checking what SAM receives...")
    # This is what gets converted for SAM
    sam_input = (semantic_input.permute(1, 2, 0) * 255.0).byte().cpu().numpy()
    print(f"   sam_input shape: {sam_input.shape}")
    print(f"   sam_input dtype: {sam_input.dtype}")
    print(f"   sam_input range: [{sam_input.min()}, {sam_input.max()}]")
    
    save_debug_tensor(torch.from_numpy(sam_input).permute(2, 0, 1), "05_SAM_Input_HWC_uint8", debug_dir)
    
    print(f"\n6. Testing manual RGB conversion...")
    # Test if we need to convert color space
    if raw_img.shape[-1] == 3:  # Has color channels
        # Try BGR to RGB conversion
        raw_img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        save_debug_tensor(torch.from_numpy(raw_img_rgb).permute(2, 0, 1), "06_Manual_BGR_to_RGB", debug_dir)
        
        print(f"   Raw BGR vs RGB difference: {np.abs(raw_img - raw_img_rgb).mean():.6f}")
    
    print(f"\n7. Investigating MASt3R resize_img function...")
    from mast3r_slam.mast3r_utils import resize_img
    
    # Check what resize_img does
    print(f"   Input to resize_img - shape: {raw_img.shape}, range: [{raw_img.min():.3f}, {raw_img.max():.3f}]")
    
    resized_result = resize_img(raw_img, 512)
    print(f"   resize_img output keys: {resized_result.keys()}")
    
    for key, value in resized_result.items():
        if isinstance(value, np.ndarray) and len(value.shape) >= 2:
            print(f"   {key}: shape={value.shape}, dtype={value.dtype}, range=[{value.min():.3f}, {value.max():.3f}]")
            if key == "img":
                save_debug_tensor(torch.from_numpy(value), f"07_ResizeImg_{key}", debug_dir)
        else:
            print(f"   {key}: {value}")
    
    print(f"\n" + "=" * 60)
    print(f"🔍 Debug complete! Check '{debug_dir}' for visualizations.")
    print(f"🎯 Key things to check:")
    print(f"   - Are colors consistent across all images?")
    print(f"   - Do any images have negative values?")
    print(f"   - Is there a BGR/RGB swap somewhere?")
    print(f"   - What does MASt3R's resize_img actually do?")
    print("=" * 60)

if __name__ == "__main__":
    debug_frame_creation()