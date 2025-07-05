#!/usr/bin/env python3
"""
Test multiple CLIP models to find the best performing one for semantic segmentation.
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

# Import after path setup
try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False
    print("open_clip_torch not available")

def test_clip_model(model_name, pretrained_dataset, test_prompts, test_images_dict):
    """Test a specific CLIP model configuration."""
    if not OPEN_CLIP_AVAILABLE:
        return None
    
    print(f"\n🧪 Testing {model_name} with {pretrained_dataset}")
    
    try:
        # Load model
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained_dataset, device="cuda", jit=False
        )
        model.eval()
        
        # Tokenize prompts
        tokenizer = open_clip.get_tokenizer(model_name)
        text_tokens = tokenizer(test_prompts).to("cuda")
        
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
        results = {}
        total_scores = []
        
        # Test each image
        for img_name, img_tensor in test_images_dict.items():
            # Apply preprocessing
            if hasattr(preprocess, 'transforms'):
                for t in preprocess.transforms:
                    if hasattr(t, 'mean') and hasattr(t, 'std'):
                        img_preprocessed = t(img_tensor.unsqueeze(0)).squeeze(0)
                        break
                else:
                    # Fallback normalization
                    import torchvision.transforms as transforms
                    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    img_preprocessed = norm(img_tensor.unsqueeze(0)).squeeze(0)
            else:
                img_preprocessed = img_tensor
            
            with torch.no_grad():
                img_features = model.encode_image(img_preprocessed.unsqueeze(0))
                img_features /= img_features.norm(dim=-1, keepdim=True)
                
                similarities = (100.0 * img_features @ text_features.T)
                scores, indices = similarities.topk(3, dim=1)
            
            top_predictions = []
            for i in range(3):
                cls = test_prompts[indices[0, i].item()]
                score = scores[0, i].item()
                top_predictions.append((cls, score))
                total_scores.append(score)
            
            results[img_name] = {
                'top_score': scores[0, 0].item(),
                'top_class': test_prompts[indices[0, 0].item()],
                'predictions': top_predictions
            }
        
        avg_score = np.mean(total_scores)
        max_score = np.max(total_scores)
        
        print(f"   ✓ Average score: {avg_score:.2f}")
        print(f"   ✓ Max score: {max_score:.2f}")
        
        return {
            'model_name': model_name,
            'pretrained': pretrained_dataset,
            'avg_score': avg_score,
            'max_score': max_score,
            'results': results
        }
        
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return None

def create_test_images():
    """Create simple test images that should be easy for CLIP to classify."""
    test_images = {}
    
    # Simple colored objects on white background
    colors = {
        'red_square': [1.0, 0.0, 0.0],
        'blue_square': [0.0, 0.0, 1.0],
        'green_square': [0.0, 1.0, 0.0],
        'brown_table': [0.6, 0.3, 0.1],
        'black_text': [0.0, 0.0, 0.0],
        'yellow_lamp': [1.0, 1.0, 0.0],
        'gray_wall': [0.5, 0.5, 0.5],
        'white_ceiling': [1.0, 1.0, 1.0]
    }
    
    for name, rgb in colors.items():
        # Create 224x224 image
        img = torch.tensor(rgb, device="cuda").view(3, 1, 1).expand(3, 224, 224)
        test_images[name] = img
    
    # Create simple patterns
    # Chair-like pattern (brown with legs)
    chair_img = torch.ones(3, 224, 224, device="cuda") * 0.9  # Light background
    chair_img[:, 50:150, 50:170] = torch.tensor([0.6, 0.3, 0.1], device="cuda").view(3, 1, 1)  # Seat
    chair_img[:, 150:200, 75:85] = torch.tensor([0.6, 0.3, 0.1], device="cuda").view(3, 1, 1)  # Leg 1
    chair_img[:, 150:200, 135:145] = torch.tensor([0.6, 0.3, 0.1], device="cuda").view(3, 1, 1)  # Leg 2
    test_images['chair_pattern'] = chair_img
    
    # Table-like pattern (larger brown rectangle)
    table_img = torch.ones(3, 224, 224, device="cuda") * 0.9  # Light background
    table_img[:, 80:120, 30:190] = torch.tensor([0.6, 0.3, 0.1], device="cuda").view(3, 1, 1)  # Table top
    table_img[:, 120:180, 50:70] = torch.tensor([0.6, 0.3, 0.1], device="cuda").view(3, 1, 1)  # Leg 1
    table_img[:, 120:180, 150:170] = torch.tensor([0.6, 0.3, 0.1], device="cuda").view(3, 1, 1)  # Leg 2
    test_images['table_pattern'] = table_img
    
    return test_images

def main():
    """Test multiple CLIP models and compare performance."""
    print("🚀 Testing Multiple CLIP Models for Better Performance")
    print("=" * 80)
    
    # Test prompts
    test_prompts = [
        "background", "wall", "floor", "ceiling", "chair", "desk", "table", 
        "monitor", "red", "blue", "green", "brown", "black", "yellow", 
        "gray", "white", "lamp", "furniture"
    ]
    
    # Create test images
    test_images = create_test_images()
    print(f"Created {len(test_images)} test images")
    
    # CLIP models to test (from best to fastest)
    clip_models_to_test = [
        # OpenAI models (usually best)
        ('ViT-B-32', 'openai'),
        ('ViT-B-16', 'openai'), 
        ('ViT-L-14', 'openai'),
        
        # LAION models
        ('ViT-B-32', 'laion2b_s34b_b79k'),
        ('ViT-B-16', 'laion2b_s34b_b88k'),
        ('ViT-L-14', 'laion2b_s32b_b82k'),
        
        # Original (current) model
        ('ViT-H-14', 'laion2b_s32b_b79k'),
        
        # Smaller/faster models  
        ('ViT-B-32', 'laion2b_e16'),
        ('RN50', 'openai'),
    ]
    
    results = []
    
    for model_name, pretrained in clip_models_to_test:
        result = test_clip_model(model_name, pretrained, test_prompts, test_images)
        if result:
            results.append(result)
    
    if not results:
        print("❌ No models worked!")
        return
    
    # Sort by average score
    results.sort(key=lambda x: x['avg_score'], reverse=True)
    
    print(f"\n" + "=" * 80)
    print("📊 CLIP MODEL COMPARISON RESULTS")
    print("=" * 80)
    
    print(f"{'Rank':<4} {'Model':<15} {'Dataset':<20} {'Avg Score':<10} {'Max Score':<10}")
    print("-" * 80)
    
    for i, result in enumerate(results):
        print(f"{i+1:<4} {result['model_name']:<15} {result['pretrained']:<20} {result['avg_score']:<10.2f} {result['max_score']:<10.2f}")
    
    # Show best model details
    best = results[0]
    print(f"\n🏆 BEST MODEL: {best['model_name']} with {best['pretrained']}")
    print(f"   Average Score: {best['avg_score']:.2f}")
    print(f"   Max Score: {best['max_score']:.2f}")
    
    print(f"\n📋 Best model detailed results:")
    for img_name, result in best['results'].items():
        print(f"   {img_name}: {result['top_class']} ({result['top_score']:.2f})")
    
    # Create comparison chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Average scores
    model_labels = [f"{r['model_name']}\n{r['pretrained'][:10]}" for r in results]
    avg_scores = [r['avg_score'] for r in results]
    max_scores = [r['max_score'] for r in results]
    
    x = np.arange(len(results))
    ax1.bar(x, avg_scores, alpha=0.7, color='skyblue', label='Average')
    ax1.bar(x, max_scores, alpha=0.5, color='orange', label='Max')
    ax1.set_xlabel('Models')
    ax1.set_ylabel('CLIP Confidence Score')
    ax1.set_title('CLIP Model Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Score distribution for best 3 models
    top_3 = results[:3]
    for i, result in enumerate(top_3):
        all_scores = []
        for img_result in result['results'].values():
            for _, score in img_result['predictions']:
                all_scores.append(score)
        ax2.hist(all_scores, bins=10, alpha=0.6, label=f"{result['model_name']}", density=True)
    
    ax2.set_xlabel('CLIP Confidence Score')
    ax2.set_ylabel('Density')
    ax2.set_title('Score Distribution (Top 3 Models)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    debug_dir = Path("clip_model_comparison")
    debug_dir.mkdir(exist_ok=True)
    plt.savefig(debug_dir / "model_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save results to file
    with open(debug_dir / "model_results.txt", 'w') as f:
        f.write("CLIP Model Comparison Results\n")
        f.write("=" * 50 + "\n\n")
        for i, result in enumerate(results):
            f.write(f"{i+1}. {result['model_name']} ({result['pretrained']})\n")
            f.write(f"   Average: {result['avg_score']:.2f}, Max: {result['max_score']:.2f}\n\n")
    
    print(f"\n💾 Results saved to: {debug_dir}/")
    print(f"\n🔧 To use the best model, update semantic_processor.py:")
    print(f"   CLIP_MODEL_NAME = '{best['model_name']}'")
    print(f"   CLIP_PRETRAINED_DATASET = '{best['pretrained']}'")
    
    if best['avg_score'] > 50:
        print(f"\n✅ Found a good model! Scores > 50 indicate proper CLIP functionality.")
    else:
        print(f"\n⚠️  All models have low scores. This suggests a deeper issue with:")
        print(f"   - Image preprocessing pipeline")
        print(f"   - Tensor format/range issues") 
        print(f"   - CUDA/device compatibility")
        print(f"   - Text prompt selection")

if __name__ == "__main__":
    main()