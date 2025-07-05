#!/usr/bin/env python3
"""
Test the CLIP-ViT-bigG-14 model from Hugging Face.
This is one of the largest CLIP models available.
"""

import sys
import torch
import numpy as np
import time
from pathlib import Path
import open_clip

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

from mast3r_slam.semantic_processor import (
    TEXT_PROMPTS,
    SEMANTIC_DEVICE,
    load_instance_segmentation_model,
    get_tensor_crop_from_mask
)

def test_clip_bigg_model():
    """Test the ViT-bigG-14 CLIP model."""
    
    print("🚀 TESTING CLIP-ViT-bigG-14-laion2B-39B-b160k")
    print("=" * 60)
    
    # Model configuration
    model_name = "ViT-bigG-14"
    pretrained = "laion2b_s39b_b160k"
    
    print(f"\n📊 Model Details:")
    print(f"   Name: {model_name}")
    print(f"   Pretrained: {pretrained}")
    print(f"   Parameters: ~1.8B")
    print(f"   Training data: 39B image-text pairs")
    print(f"   This is one of the largest publicly available CLIP models")
    
    # Load test data
    print("\n📸 Loading test data...")
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
    
    print(f"   Image shape: {image_tensor.shape}")
    print(f"   SAM masks: {len(sam_masks_data)}")
    
    # Load CLIP model
    print(f"\n🔄 Loading {model_name} model...")
    print("   Note: This model is ~5.5GB and may take time to download on first use")
    
    start_load = time.time()
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=SEMANTIC_DEVICE
        )
        model.eval()
        
        # Get text features
        tokenizer = open_clip.get_tokenizer(model_name)
        text_tokens = tokenizer(TEXT_PROMPTS).to(SEMANTIC_DEVICE)
        
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        load_time = time.time() - start_load
        print(f"   ✅ Model loaded in {load_time:.2f}s")
        
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return
    
    # Get normalization transform
    normalize_transform = None
    if hasattr(preprocess, 'transforms'):
        for t in preprocess.transforms:
            if hasattr(t, 'mean') and hasattr(t, 'std'):
                normalize_transform = t
                break
    
    if normalize_transform is None:
        import torchvision.transforms as transforms
        normalize_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    # Test on subset of masks
    print("\n🧪 Processing masks with CLIP...")
    all_scores = []
    all_predictions = []
    process_times = []
    
    test_masks = sam_masks_data[:50]  # Test first 50 masks
    
    for i, mask_data in enumerate(test_masks):
        mask = mask_data['segmentation']
        
        # Get crop
        crop = get_tensor_crop_from_mask(
            image_tensor,
            torch.from_numpy(mask.astype(bool)).to(SEMANTIC_DEVICE),
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
        
        # Print first few predictions
        if i < 5:
            print(f"   Mask {i}: {TEXT_PROMPTS[best_idx.item()]} (score: {best_score.item():.2f})")
    
    # Calculate statistics
    if all_scores:
        avg_score = np.mean(all_scores)
        max_score = np.max(all_scores)
        min_score = np.min(all_scores)
        std_score = np.std(all_scores)
        high_conf_ratio = sum(1 for s in all_scores if s > 25.0) / len(all_scores) * 100
        avg_process_time = np.mean(process_times) * 1000  # ms
        
        print(f"\n📊 RESULTS for {model_name}:")
        print("=" * 60)
        print(f"Average confidence: {avg_score:.2f}")
        print(f"Max confidence: {max_score:.2f}")
        print(f"Min confidence: {min_score:.2f}")
        print(f"Std deviation: {std_score:.2f}")
        print(f"High confidence ratio (>25): {high_conf_ratio:.1f}%")
        print(f"Average processing time: {avg_process_time:.1f}ms per crop")
        print(f"Total samples processed: {len(all_scores)}")
        
        # Compare with other models
        print("\n🔄 COMPARISON WITH OTHER MODELS:")
        print("-" * 60)
        print("Model                          | Avg Score | High Conf % | Speed")
        print("-" * 60)
        print(f"ViT-bigG-14 (current)         | {avg_score:9.2f} | {high_conf_ratio:10.1f}% | {avg_process_time:6.1f}ms")
        print(f"ConvNeXt-XXLarge              | {29.59:9.2f} | {100.0:10.1f}% | {11.3:6.1f}ms")
        print(f"ViT-B-32 (baseline)           | {28.04:9.2f} | {100.0:10.1f}% | {4.7:6.1f}ms")
        
        # Test with augmented prompts
        print("\n🔄 Testing with augmented prompts...")
        from test_confidence_improvements import create_augmented_prompts
        augmented_prompts = create_augmented_prompts(TEXT_PROMPTS)
        
        # Re-encode augmented text features
        aug_text_tokens = tokenizer(augmented_prompts).to(SEMANTIC_DEVICE)
        with torch.no_grad():
            aug_text_features = model.encode_text(aug_text_tokens)
            aug_text_features = aug_text_features / aug_text_features.norm(dim=-1, keepdim=True)
        
        # Test a few masks with augmented prompts
        aug_scores = []
        for mask_data in test_masks[:20]:
            mask = mask_data['segmentation']
            crop = get_tensor_crop_from_mask(
                image_tensor,
                torch.from_numpy(mask.astype(bool)).to(SEMANTIC_DEVICE),
                output_size=(224, 224)
            )
            
            if crop is None:
                continue
            
            crop_norm = normalize_transform(crop.unsqueeze(0)).squeeze(0)
            
            with torch.no_grad():
                image_features = model.encode_image(crop_norm.unsqueeze(0))
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                similarities = (100.0 * image_features @ aug_text_features.T)
                best_score, _ = similarities.max(dim=1)
            
            aug_scores.append(best_score.item())
        
        if aug_scores:
            aug_avg_score = np.mean(aug_scores)
            aug_improvement = aug_avg_score - avg_score
            print(f"\nAugmented prompts average: {aug_avg_score:.2f} (+{aug_improvement:.2f} improvement)")
        
        # Final recommendation
        print("\n💡 RECOMMENDATION:")
        if avg_score > 29.59:  # Better than ConvNeXt-XXLarge
            print(f"✅ {model_name} is the best model with {avg_score:.2f} average score!")
            print("   Consider using it for highest accuracy semantic mapping")
        else:
            print(f"⚠️  {model_name} scores {avg_score:.2f}, lower than ConvNeXt-XXLarge (29.59)")
            print("   Stick with ConvNeXt-XXLarge for better accuracy")
        
        print(f"\n⏱️  Processing speed: {avg_process_time:.1f}ms per crop")
        if avg_process_time > 15:
            print("   Note: This model is slower, best for offline processing")
        else:
            print("   Speed is acceptable for real-time processing")

def main():
    """Main function."""
    print("🔬 CLIP-ViT-bigG-14 MODEL TEST")
    print("Testing the largest publicly available CLIP model")
    print("=" * 60)
    
    test_clip_bigg_model()
    
    print("\n✅ Test complete!")

if __name__ == "__main__":
    main()