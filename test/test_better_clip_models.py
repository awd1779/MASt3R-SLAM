#!/usr/bin/env python3
"""
Test different CLIP models to find the best one for semantic mapping.
Compare various CLIP architectures and their confidence scores.
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json
import open_clip

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

from mast3r_slam.semantic_processor import (
    process_frame_for_semantics,
    TEXT_PROMPTS,
    SEMANTIC_DEVICE,
    load_instance_segmentation_model,
    get_tensor_crop_from_mask
)

# Define CLIP models to test
CLIP_MODELS = [
    # Current model
    ("ViT-B-32", "laion2b_e16"),
    
    # Larger/Better models
    ("ViT-L-14", "laion2b_s32b_b82k"),
    ("ViT-H-14", "laion2b_s32b_b79k"),
    ("ViT-g-14", "laion2b_s12b_b42k"),
    
    # Newer architectures
    ("convnext_large_d_320", "laion2b_s29b_b131k_ft_soup"),
    ("convnext_xxlarge", "laion2b_s34b_b82k_augreg_soup"),
    
    # Good balance of speed/accuracy
    ("ViT-L-14-336", "openai"),
    ("ViT-B-16", "laion2b_s34b_b88k"),
]

def get_model_info(model_name, pretrained):
    """Get information about a CLIP model."""
    try:
        # Check if model is available
        available = open_clip.list_pretrained()
        model_available = any(m[0] == model_name and m[1] == pretrained for m in available)
        
        if not model_available:
            return None
            
        # Try to get model config
        model_cfg = open_clip.get_model_config(model_name)
        
        # Get approximate parameter count
        param_counts = {
            "ViT-B-32": "~150M",
            "ViT-B-16": "~150M", 
            "ViT-L-14": "~430M",
            "ViT-L-14-336": "~430M",
            "ViT-H-14": "~986M",
            "ViT-g-14": "~1.8B",
            "convnext_large_d_320": "~660M",
            "convnext_xxlarge": "~846M"
        }
        
        return {
            "available": True,
            "params": param_counts.get(model_name, "Unknown"),
            "input_size": model_cfg.get("vision_cfg", {}).get("image_size", 224) if model_cfg else 224
        }
    except:
        return None

def test_clip_model(image_tensor, sam_masks_data, model_name, pretrained, device=SEMANTIC_DEVICE):
    """Test a specific CLIP model configuration."""
    
    print(f"\n🔍 Testing {model_name} ({pretrained})")
    print("-" * 50)
    
    try:
        # Load CLIP model
        start_load = time.time()
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        model.eval()
        
        # Get text features
        tokenizer = open_clip.get_tokenizer(model_name)
        text_tokens = tokenizer(TEXT_PROMPTS).to(device)
        
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        load_time = time.time() - start_load
        print(f"   Model loaded in {load_time:.2f}s")
        
        # Get normalization transform
        normalize_transform = None
        if hasattr(preprocess, 'transforms'):
            for t in preprocess.transforms:
                if hasattr(t, 'mean') and hasattr(t, 'std'):
                    normalize_transform = t
                    break
        
        if normalize_transform is None:
            # Default normalization
            import torchvision.transforms as transforms
            normalize_transform = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        
        # Process masks and get scores
        all_scores = []
        all_predictions = []
        process_times = []
        
        # Test on subset of masks for speed
        test_masks = sam_masks_data[:50]
        
        for mask_data in test_masks:
            mask = mask_data['segmentation']
            
            # Get crop
            crop = get_tensor_crop_from_mask(
                image_tensor,
                torch.from_numpy(mask.astype(bool)).to(device),
                output_size=(224, 224)
            )
            
            if crop is None:
                continue
            
            # Process with CLIP
            start_process = time.time()
            
            crop_norm = normalize_transform(crop.unsqueeze(0)).squeeze(0)
            
            with torch.no_grad():
                image_features = model.encode_image(crop_norm.unsqueeze(0))
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                similarities = (100.0 * image_features @ text_features.T)
                best_score, best_idx = similarities.max(dim=1)
            
            process_times.append(time.time() - start_process)
            all_scores.append(best_score.item())
            all_predictions.append(TEXT_PROMPTS[best_idx.item()])
        
        # Calculate statistics
        if all_scores:
            results = {
                'model_name': model_name,
                'pretrained': pretrained,
                'load_time': load_time,
                'avg_score': np.mean(all_scores),
                'max_score': np.max(all_scores),
                'min_score': np.min(all_scores),
                'std_score': np.std(all_scores),
                'high_conf_ratio': sum(1 for s in all_scores if s > 25.0) / len(all_scores) * 100,
                'avg_process_time': np.mean(process_times) * 1000,  # ms
                'num_samples': len(all_scores)
            }
            
            print(f"   ✓ Avg confidence: {results['avg_score']:.2f}")
            print(f"   ✓ Max confidence: {results['max_score']:.2f}")
            print(f"   ✓ High conf ratio: {results['high_conf_ratio']:.1f}%")
            print(f"   ✓ Avg process time: {results['avg_process_time']:.1f}ms per crop")
            
            return results
        else:
            print("   ❌ No valid crops to test")
            return None
            
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return None

def create_comparison_plot(results, save_path="clip_model_comparison"):
    """Create visualization comparing different CLIP models."""
    save_dir = Path(save_path)
    save_dir.mkdir(exist_ok=True)
    
    # Filter out failed models
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        print("No valid results to plot")
        return
    
    # Sort by average score
    valid_results.sort(key=lambda x: x['avg_score'], reverse=True)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('CLIP Model Comparison for Semantic Mapping', fontsize=16, fontweight='bold')
    
    model_labels = [f"{r['model_name']}\n({r['pretrained'][:15]}...)" if len(r['pretrained']) > 15 
                   else f"{r['model_name']}\n({r['pretrained']})" for r in valid_results]
    colors = plt.cm.viridis(np.linspace(0, 1, len(valid_results)))
    
    # 1. Average confidence scores
    avg_scores = [r['avg_score'] for r in valid_results]
    bars1 = ax1.barh(model_labels, avg_scores, color=colors)
    ax1.set_xlabel('Average Confidence Score')
    ax1.set_title('Average CLIP Confidence by Model')
    ax1.axvline(x=25.0, color='red', linestyle='--', alpha=0.5, label='Target threshold')
    
    for bar, score in zip(bars1, avg_scores):
        ax1.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{score:.1f}', ha='left', va='center', fontweight='bold')
    
    # 2. High confidence ratio
    high_conf_ratios = [r['high_conf_ratio'] for r in valid_results]
    bars2 = ax2.barh(model_labels, high_conf_ratios, color=colors)
    ax2.set_xlabel('High Confidence Ratio (%)')
    ax2.set_title('Percentage of Predictions with Score > 25')
    
    for bar, ratio in zip(bars2, high_conf_ratios):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{ratio:.0f}%', ha='left', va='center', fontweight='bold')
    
    # 3. Processing speed
    process_times = [r['avg_process_time'] for r in valid_results]
    bars3 = ax3.barh(model_labels, process_times, color=colors)
    ax3.set_xlabel('Average Processing Time (ms per crop)')
    ax3.set_title('Inference Speed')
    ax3.invert_xaxis()  # Lower is better
    
    for bar, time_ms in zip(bars3, process_times):
        ax3.text(bar.get_width() - 0.5, bar.get_y() + bar.get_height()/2,
                f'{time_ms:.1f}ms', ha='right', va='center', fontweight='bold')
    
    # 4. Model comparison table
    ax4.axis('tight')
    ax4.axis('off')
    
    # Get model info
    table_data = []
    for r in valid_results[:5]:  # Top 5 models
        model_info = get_model_info(r['model_name'], r['pretrained'])
        params = model_info['params'] if model_info else "Unknown"
        
        improvement = r['avg_score'] - valid_results[-1]['avg_score']  # vs worst model
        
        table_data.append([
            r['model_name'],
            params,
            f"{r['avg_score']:.1f}",
            f"{r['high_conf_ratio']:.0f}%",
            f"{r['avg_process_time']:.1f}ms",
            f"+{improvement:.1f}" if improvement > 0 else f"{improvement:.1f}"
        ])
    
    table = ax4.table(cellText=table_data,
                     colLabels=['Model', 'Parameters', 'Avg Score', 'High Conf %', 'Speed', 'Improvement'],
                     cellLoc='center',
                     loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color best model row
    for i in range(len(table_data[0])):
        table[(1, i)].set_facecolor('lightgreen')
    
    ax4.set_title('Top 5 Models Summary', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_dir / "clip_model_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save detailed results
    with open(save_dir / "clip_model_results.json", 'w') as f:
        json.dump(valid_results, f, indent=2)
    
    print(f"\n📊 Comparison saved to: {save_dir}/")

def load_test_data():
    """Load test image and generate SAM masks."""
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
    
    # Generate SAM masks
    sam_generator = load_instance_segmentation_model(device=SEMANTIC_DEVICE)
    image_np = (image_tensor.permute(1, 2, 0) * 255.0).byte().cpu().numpy()
    sam_masks_data = sam_generator.generate(image_np)
    
    return image_tensor, sam_masks_data

def main():
    """Main test function."""
    print("🚀 BETTER CLIP MODELS TEST")
    print("Testing various CLIP models for improved semantic mapping")
    print("=" * 60)
    
    # Check available models
    print("\n📋 Checking available CLIP models...")
    available_models = []
    for model_name, pretrained in CLIP_MODELS:
        info = get_model_info(model_name, pretrained)
        if info:
            print(f"   ✓ {model_name} ({pretrained}): {info['params']} parameters")
            available_models.append((model_name, pretrained))
        else:
            print(f"   ❌ {model_name} ({pretrained}): Not available")
    
    if not available_models:
        print("\n❌ No CLIP models available for testing")
        return
    
    # Load test data
    print("\n📸 Loading test data...")
    image_tensor, sam_masks_data = load_test_data()
    print(f"   Image shape: {image_tensor.shape}")
    print(f"   SAM masks: {len(sam_masks_data)}")
    
    # Test each model
    print("\n🧪 Testing CLIP models...")
    results = []
    
    for model_name, pretrained in available_models:
        result = test_clip_model(image_tensor, sam_masks_data, model_name, pretrained)
        results.append(result)
    
    # Create comparison visualization
    print("\n📊 Creating comparison visualization...")
    create_comparison_plot(results)
    
    # Find best model
    valid_results = [r for r in results if r is not None]
    if valid_results:
        best_model = max(valid_results, key=lambda x: x['avg_score'])
        
        print("\n" + "🏆" + " "*20 + "BEST MODEL" + " "*20 + "🏆")
        print("=" * 60)
        print(f"Model: {best_model['model_name']} ({best_model['pretrained']})")
        print(f"Average Score: {best_model['avg_score']:.2f}")
        print(f"High Confidence Ratio: {best_model['high_conf_ratio']:.1f}%")
        print(f"Processing Speed: {best_model['avg_process_time']:.1f}ms per crop")
        
        baseline = next((r for r in valid_results if r['model_name'] == 'ViT-B-32'), None)
        if baseline and baseline != best_model:
            improvement = best_model['avg_score'] - baseline['avg_score']
            print(f"\nImprovement over current model: +{improvement:.2f} points")
        
        print("\n💡 RECOMMENDATION:")
        if best_model['avg_process_time'] < 10:  # Fast enough
            print(f"Switch to {best_model['model_name']} for better accuracy")
        else:
            print(f"Consider {best_model['model_name']} if processing speed is not critical")
            print("Or use it for offline processing to generate high-quality labels")
    
    print("\n✅ Test complete! Check 'clip_model_comparison/' for detailed results")

if __name__ == "__main__":
    main()