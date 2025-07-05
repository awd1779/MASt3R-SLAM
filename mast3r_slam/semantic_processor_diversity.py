"""
Alternative semantic processor that promotes diversity in predictions.
Truly open-vocabulary without hard-coded biases.
"""

import torch
import numpy as np

def select_diverse_predictions(similarity_matrix, text_prompts, diversity_threshold=0.5):
    """
    Select predictions that promote diversity when top scores are similar.
    
    Args:
        similarity_matrix: Tensor of shape (N, num_classes) with similarity scores
        text_prompts: List of text prompts
        diversity_threshold: If top 2 predictions differ by less than this, consider 2nd
    
    Returns:
        best_indices: Selected class indices
        best_scores: Selected scores
        used_second: Boolean mask of where 2nd best was used
    """
    
    # Get top 2 predictions for each mask
    top2_scores, top2_indices = similarity_matrix.topk(2, dim=1)
    
    best_indices = []
    best_scores = []
    used_second = []
    
    # Track how many times each class has been predicted
    class_counts = {}
    
    for i in range(similarity_matrix.shape[0]):
        first_score = top2_scores[i, 0]
        second_score = top2_scores[i, 1]
        first_idx = top2_indices[i, 0]
        second_idx = top2_indices[i, 1]
        
        score_gap = first_score - second_score
        
        # Get labels
        first_label = text_prompts[first_idx] if first_idx < len(text_prompts) else "unknown"
        second_label = text_prompts[second_idx] if second_idx < len(text_prompts) else "unknown"
        
        # Count current predictions
        first_count = class_counts.get(first_label, 0)
        second_count = class_counts.get(second_label, 0)
        
        # Decision logic (no hard-coded labels!)
        use_second = False
        
        if score_gap < diversity_threshold:
            # Scores are close - consider diversity
            if first_count > second_count * 2:  # First is overrepresented
                use_second = True
        
        if use_second:
            best_indices.append(second_idx)
            best_scores.append(second_score)
            class_counts[second_label] = second_count + 1
            used_second.append(True)
        else:
            best_indices.append(first_idx)
            best_scores.append(first_score)
            class_counts[first_label] = first_count + 1
            used_second.append(False)
    
    return torch.tensor(best_indices), torch.tensor(best_scores), used_second

def analyze_prediction_distribution(similarity_matrix, text_prompts):
    """
    Analyze why certain classes dominate predictions.
    """
    
    # Get top predictions
    top_scores, top_indices = similarity_matrix.max(dim=1)
    
    # Count predictions
    pred_counts = {}
    for idx in top_indices:
        label = text_prompts[idx.item()]
        pred_counts[label] = pred_counts.get(label, 0) + 1
    
    # Analyze score distributions
    print("\n📊 Prediction Distribution Analysis:")
    print(f"Total objects: {len(top_indices)}")
    
    # Show top predicted classes
    sorted_preds = sorted(pred_counts.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 5 predicted classes:")
    for label, count in sorted_preds[:5]:
        percentage = (count / len(top_indices)) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")
    
    # Check for dominance
    if sorted_preds[0][1] > len(top_indices) * 0.5:
        print(f"\n⚠️  WARNING: '{sorted_preds[0][0]}' dominates {sorted_preds[0][1]/len(top_indices)*100:.1f}% of predictions!")
        print("   This suggests:")
        print("   - Image might be taken from this perspective")
        print("   - Model might have bias toward this class")
        print("   - Consider using diversity-promoting selection")
    
    # Analyze score gaps
    if similarity_matrix.shape[1] > 1:
        top2_scores = similarity_matrix.topk(2, dim=1)[0]
        gaps = top2_scores[:, 0] - top2_scores[:, 1]
        avg_gap = gaps.mean().item()
        print(f"\nAverage confidence gap (1st-2nd): {avg_gap:.2f}")
        if avg_gap < 1.0:
            print("   → Small gaps suggest model uncertainty")
            print("   → Good candidate for diversity selection")

def create_open_vocabulary_processor():
    """
    Create a truly open-vocabulary processor without hard-coded biases.
    """
    
    class OpenVocabularyProcessor:
        def __init__(self, use_diversity=True):
            self.use_diversity = use_diversity
            self.prediction_history = {}
        
        def process_predictions(self, similarity_matrix, text_prompts):
            """Process predictions with optional diversity promotion."""
            
            if self.use_diversity:
                # Use diversity-aware selection
                indices, scores, used_second = select_diverse_predictions(
                    similarity_matrix, text_prompts, diversity_threshold=0.5
                )
                
                # Report diversity actions
                if any(used_second):
                    print(f"[Diversity] Used 2nd best prediction for {sum(used_second)}/{len(used_second)} objects")
            else:
                # Standard selection
                scores, indices = similarity_matrix.max(dim=1)
            
            # Adaptive confidence calibration (no hard-coded labels)
            calibrated_scores = self.calibrate_scores(scores)
            
            return indices, calibrated_scores
        
        def calibrate_scores(self, scores):
            """Calibrate scores without label-specific adjustments."""
            
            # Simple percentile-based calibration
            if len(scores) > 1:
                # Use percentile mapping
                sorted_scores, _ = scores.sort()
                p10 = sorted_scores[int(len(scores) * 0.1)]
                p90 = sorted_scores[int(len(scores) * 0.9)]
                
                # Map [p10, p90] to [30, 70]
                calibrated = []
                for score in scores:
                    if p90 > p10:
                        normalized = (score - p10) / (p90 - p10)
                        cal_score = 30 + normalized * 40
                        cal_score = max(20, min(80, cal_score))  # Clamp to reasonable range
                    else:
                        cal_score = 50
                    calibrated.append(cal_score)
                
                return calibrated
            else:
                return [50.0] * len(scores)
    
    return OpenVocabularyProcessor()

if __name__ == "__main__":
    # Example usage
    print("🌐 Open-Vocabulary Semantic Processor")
    print("=" * 60)
    print("\nKey Features:")
    print("✅ No hard-coded label penalties")
    print("✅ Diversity promotion when confidence is low")
    print("✅ Adaptive calibration based on score distribution")
    print("✅ Works with ANY vocabulary")
    
    print("\n💡 Integration:")
    print("1. Replace hard-coded calibration with adaptive method")
    print("2. Use diversity selection when scores are close")
    print("3. Monitor prediction distribution for bias detection")