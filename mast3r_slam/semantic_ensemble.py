"""
Multi-Model CLIP Ensemble for Robust Semantic Classification
Combines predictions from multiple CLIP models for improved accuracy.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import open_clip
from enum import Enum


class CLIPModelConfig(Enum):
    """Available CLIP model configurations."""
    # Fast models
    VITB32_LAION = ('ViT-B-32', 'laion2b_e16')  # Fastest, good accuracy
    VITB16_LAION = ('ViT-B-16', 'laion2b_s34b_b88k')  # Slightly slower, better
    
    # Balanced models
    VITH14_LAION = ('ViT-H-14', 'laion2b_s32b_b79k')  # Good balance
    VITL14_LAION = ('ViT-L-14', 'laion2b_s32b_b82k')  # Also balanced
    
    # High accuracy models
    VITBIGG14_LAION = ('ViT-bigG-14', 'laion2b_s39b_b160k')  # Best accuracy
    CONVNEXT_XXL = ('convnext_xxlarge', 'laion2b_s34b_b82k_augreg_soup')  # Alternative architecture


@dataclass
class EnsembleModel:
    """Container for a single model in the ensemble."""
    name: str
    model: nn.Module
    preprocess: nn.Module
    text_features: torch.Tensor
    weight: float = 1.0
    confidence_offset: float = 0.0
    confidence_scale: float = 1.0


class SemanticEnsemble:
    """
    Ensemble of multiple CLIP models for robust semantic classification.
    """
    
    def __init__(self,
                 model_configs: List[CLIPModelConfig] = None,
                 ensemble_method: str = 'weighted_average',
                 temperature: float = 1.0,
                 device: str = 'cuda',
                 cache_text_features: bool = True):
        """
        Initialize the semantic ensemble.
        
        Args:
            model_configs: List of model configurations to use
            ensemble_method: Method for combining predictions
            temperature: Temperature for softmax scaling
            device: Device for computation
            cache_text_features: Whether to cache text features
        """
        self.device = device
        self.ensemble_method = ensemble_method
        self.temperature = temperature
        self.cache_text_features = cache_text_features
        
        # Default ensemble configuration
        if model_configs is None:
            model_configs = [
                CLIPModelConfig.VITB32_LAION,    # Fast
                CLIPModelConfig.VITH14_LAION,    # Balanced
                CLIPModelConfig.VITBIGG14_LAION  # Accurate
            ]
        
        self.model_configs = model_configs
        self.models = []
        self.text_features_cache = {}
        
        # Model-specific calibration parameters (learned from data)
        self.calibration_params = {
            'ViT-B-32': {'offset': 0.0, 'scale': 1.2, 'weight': 0.8},
            'ViT-H-14': {'offset': 0.0, 'scale': 1.0, 'weight': 1.0},
            'ViT-bigG-14': {'offset': 0.0, 'scale': 0.9, 'weight': 1.2},
            'convnext_xxlarge': {'offset': 0.0, 'scale': 1.1, 'weight': 1.1},
            'ViT-B-16': {'offset': 0.0, 'scale': 1.1, 'weight': 0.9},
            'ViT-L-14': {'offset': 0.0, 'scale': 1.0, 'weight': 1.0},
        }
        
        # Performance tracking
        self.model_stats = {config.value[0]: {
            'inference_times': [],
            'avg_confidence': [],
            'agreement_rate': 0.0
        } for config in model_configs}
        
    def load_models(self, text_prompts: List[str]):
        """
        Load all models in the ensemble.
        
        Args:
            text_prompts: Text prompts for classification
        """
        print(f"[SemanticEnsemble] Loading {len(self.model_configs)} models...")
        
        for config in self.model_configs:
            model_name, pretrained = config.value
            
            try:
                # Load model
                model, _, preprocess = open_clip.create_model_and_transforms(
                    model_name, 
                    pretrained=pretrained, 
                    device=self.device
                )
                model.eval()
                
                # Get calibration parameters
                calib = self.calibration_params.get(model_name, {
                    'offset': 0.0, 'scale': 1.0, 'weight': 1.0
                })
                
                # Encode text features
                text_features = self._encode_text_features(model, model_name, text_prompts)
                
                # Create ensemble model
                ensemble_model = EnsembleModel(
                    name=model_name,
                    model=model,
                    preprocess=preprocess,
                    text_features=text_features,
                    weight=calib['weight'],
                    confidence_offset=calib['offset'],
                    confidence_scale=calib['scale']
                )
                
                self.models.append(ensemble_model)
                print(f"[SemanticEnsemble] Loaded {model_name} (weight={calib['weight']:.2f})")
                
            except Exception as e:
                print(f"[SemanticEnsemble] Failed to load {model_name}: {e}")
    
    def _encode_text_features(self, 
                            model: nn.Module, 
                            model_name: str,
                            text_prompts: List[str]) -> torch.Tensor:
        """
        Encode text features with caching.
        
        Args:
            model: CLIP model
            model_name: Model identifier
            text_prompts: Text prompts
            
        Returns:
            Encoded text features
        """
        cache_key = f"{model_name}_{len(text_prompts)}"
        
        if self.cache_text_features and cache_key in self.text_features_cache:
            return self.text_features_cache[cache_key]
        
        # Get tokenizer and encode
        tokenizer = open_clip.get_tokenizer(model_name)
        text_tokens = tokenizer(text_prompts).to(self.device)
        
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
        if self.cache_text_features:
            self.text_features_cache[cache_key] = text_features
        
        return text_features
    
    def predict_ensemble(self, 
                        image_crops: torch.Tensor,
                        return_all_scores: bool = False) -> Dict:
        """
        Get ensemble predictions for image crops.
        
        Args:
            image_crops: Batch of image crops (B, C, H, W)
            return_all_scores: Whether to return individual model scores
            
        Returns:
            Dictionary with ensemble predictions
        """
        if not self.models:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        batch_size = image_crops.shape[0]
        num_classes = self.models[0].text_features.shape[0]
        
        # Collect predictions from all models
        all_similarities = []
        all_predictions = []
        model_times = []
        
        for ensemble_model in self.models:
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            
            # Preprocess for this model
            processed_crops = self._preprocess_for_model(
                image_crops, ensemble_model.preprocess
            )
            
            # Get features
            with torch.no_grad():
                image_features = ensemble_model.model.encode_image(processed_crops)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Compute similarities
            similarities = (100.0 * image_features @ ensemble_model.text_features.T)
            
            # Apply model-specific calibration
            similarities = (similarities + ensemble_model.confidence_offset) * ensemble_model.confidence_scale
            
            all_similarities.append(similarities)
            
            # Get predictions
            scores, indices = similarities.max(dim=1)
            all_predictions.append(indices)
            
            end_time.record()
            torch.cuda.synchronize()
            
            elapsed_time = start_time.elapsed_time(end_time)
            model_times.append(elapsed_time)
            
            # Update stats
            self.model_stats[ensemble_model.name]['inference_times'].append(elapsed_time)
            self.model_stats[ensemble_model.name]['avg_confidence'].append(scores.mean().item())
        
        # Combine predictions based on ensemble method
        if self.ensemble_method == 'weighted_average':
            ensemble_scores = self._weighted_average_ensemble(all_similarities)
        elif self.ensemble_method == 'voting':
            ensemble_scores = self._voting_ensemble(all_predictions, num_classes)
        elif self.ensemble_method == 'max_confidence':
            ensemble_scores = self._max_confidence_ensemble(all_similarities)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        # Get final predictions
        final_scores, final_indices = ensemble_scores.max(dim=1)
        
        # Compute agreement rate
        self._update_agreement_stats(all_predictions)
        
        # Calibrate final scores
        calibrated_scores = self._calibrate_ensemble_scores(final_scores)
        
        results = {
            'predictions': final_indices,
            'scores': calibrated_scores,
            'raw_scores': final_scores,
            'model_times': model_times,
            'agreement_rate': self._compute_agreement_rate(all_predictions)
        }
        
        if return_all_scores:
            results['all_model_scores'] = all_similarities
            results['all_model_predictions'] = all_predictions
        
        return results
    
    def _preprocess_for_model(self, 
                            image_crops: torch.Tensor,
                            preprocess: nn.Module) -> torch.Tensor:
        """
        Preprocess crops for a specific model.
        
        Args:
            image_crops: Input crops
            preprocess: Model-specific preprocessing
            
        Returns:
            Preprocessed crops
        """
        # Extract normalization from preprocess
        normalize = None
        for transform in preprocess.transforms:
            if hasattr(transform, 'mean') and hasattr(transform, 'std'):
                normalize = transform
                break
        
        if normalize is None:
            # Default normalization
            normalize = torch.nn.functional.normalize
        
        # Apply normalization
        return normalize(image_crops)
    
    def _weighted_average_ensemble(self, 
                                 all_similarities: List[torch.Tensor]) -> torch.Tensor:
        """
        Combine predictions using weighted average.
        
        Args:
            all_similarities: List of similarity matrices from each model
            
        Returns:
            Ensemble similarity scores
        """
        # Apply temperature scaling
        scaled_similarities = [
            torch.softmax(sim / self.temperature, dim=1) * self.models[i].weight
            for i, sim in enumerate(all_similarities)
        ]
        
        # Weighted average
        ensemble_scores = torch.stack(scaled_similarities).sum(dim=0)
        
        # Normalize weights
        total_weight = sum(m.weight for m in self.models)
        ensemble_scores /= total_weight
        
        return ensemble_scores
    
    def _voting_ensemble(self, 
                        all_predictions: List[torch.Tensor],
                        num_classes: int) -> torch.Tensor:
        """
        Combine predictions using majority voting.
        
        Args:
            all_predictions: List of prediction indices from each model
            num_classes: Number of classes
            
        Returns:
            Voting scores
        """
        batch_size = all_predictions[0].shape[0]
        vote_counts = torch.zeros(batch_size, num_classes, device=self.device)
        
        # Count votes with weights
        for i, predictions in enumerate(all_predictions):
            weight = self.models[i].weight
            vote_counts.scatter_add_(1, predictions.unsqueeze(1), 
                                   torch.ones_like(predictions, dtype=torch.float).unsqueeze(1) * weight)
        
        return vote_counts
    
    def _max_confidence_ensemble(self, 
                               all_similarities: List[torch.Tensor]) -> torch.Tensor:
        """
        Select prediction with highest confidence across models.
        
        Args:
            all_similarities: List of similarity matrices
            
        Returns:
            Max confidence scores
        """
        # Stack all similarities
        stacked = torch.stack(all_similarities)
        
        # Get max confidence across models for each class
        max_scores, _ = stacked.max(dim=0)
        
        return max_scores
    
    def _calibrate_ensemble_scores(self, raw_scores: torch.Tensor) -> torch.Tensor:
        """
        Calibrate ensemble scores to 0-100 range.
        
        Args:
            raw_scores: Raw ensemble scores
            
        Returns:
            Calibrated scores
        """
        # Adaptive calibration based on score distribution
        min_score = raw_scores.min().item()
        max_score = raw_scores.max().item()
        mean_score = raw_scores.mean().item()
        
        # Map to 0-100 with emphasis on relative differences
        if max_score > min_score:
            normalized = (raw_scores - min_score) / (max_score - min_score)
            # Non-linear mapping to emphasize high-confidence predictions
            calibrated = 20 + 60 * normalized + 20 * (normalized ** 2)
        else:
            calibrated = torch.full_like(raw_scores, 50.0)
        
        return torch.clamp(calibrated, 0, 100)
    
    def _compute_agreement_rate(self, all_predictions: List[torch.Tensor]) -> float:
        """
        Compute agreement rate between models.
        
        Args:
            all_predictions: Predictions from all models
            
        Returns:
            Agreement rate (0-1)
        """
        if len(all_predictions) < 2:
            return 1.0
        
        agreements = 0
        comparisons = 0
        
        for i in range(len(all_predictions)):
            for j in range(i + 1, len(all_predictions)):
                agreements += (all_predictions[i] == all_predictions[j]).sum().item()
                comparisons += all_predictions[i].shape[0]
        
        return agreements / (comparisons + 1e-8)
    
    def _update_agreement_stats(self, all_predictions: List[torch.Tensor]):
        """Update model agreement statistics."""
        agreement_rate = self._compute_agreement_rate(all_predictions)
        
        for model_name in self.model_stats:
            self.model_stats[model_name]['agreement_rate'] = agreement_rate
    
    def get_performance_summary(self) -> Dict:
        """
        Get performance summary for all models.
        
        Returns:
            Dictionary with performance metrics
        """
        summary = {}
        
        for model_name, stats in self.model_stats.items():
            if stats['inference_times']:
                summary[model_name] = {
                    'avg_inference_ms': np.mean(stats['inference_times']),
                    'std_inference_ms': np.std(stats['inference_times']),
                    'avg_confidence': np.mean(stats['avg_confidence']) if stats['avg_confidence'] else 0,
                    'agreement_rate': stats['agreement_rate']
                }
        
        return summary
    
    def adaptive_model_selection(self, 
                               target_fps: float = 10.0,
                               min_accuracy: float = 0.8) -> List[str]:
        """
        Select models based on performance requirements.
        
        Args:
            target_fps: Target frames per second
            min_accuracy: Minimum required accuracy
            
        Returns:
            List of selected model names
        """
        max_inference_time = 1000.0 / target_fps  # Convert to ms
        
        # Get model performance
        perf_summary = self.get_performance_summary()
        
        # Filter by speed
        fast_enough = [
            name for name, stats in perf_summary.items()
            if stats['avg_inference_ms'] < max_inference_time
        ]
        
        # Sort by accuracy (agreement rate as proxy)
        fast_enough.sort(key=lambda x: perf_summary[x]['agreement_rate'], reverse=True)
        
        # Select models that meet criteria
        selected = []
        for model in fast_enough:
            if perf_summary[model]['agreement_rate'] >= min_accuracy or len(selected) == 0:
                selected.append(model)
        
        return selected