"""Base class for all evaluation metrics."""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import json
import os
from datetime import datetime


class BaseMetrics(ABC):
    """Abstract base class for evaluation metrics."""
    
    def __init__(self, name: str, output_dir: Optional[str] = None):
        """Initialize base metrics.
        
        Args:
            name: Name of the metric
            output_dir: Directory to save metric outputs
        """
        self.name = name
        self.output_dir = output_dir
        self.results = {}
        self.history = []
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    @abstractmethod
    def compute(self, prediction: Any, ground_truth: Any, **kwargs) -> Dict[str, float]:
        """Compute metrics for given prediction and ground truth.
        
        Args:
            prediction: Predicted data
            ground_truth: Ground truth data
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of metric values
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset internal state of metrics."""
        pass
    
    def update(self, prediction: Any, ground_truth: Any, **kwargs):
        """Update metrics with new prediction-ground truth pair.
        
        Args:
            prediction: Predicted data
            ground_truth: Ground truth data
            **kwargs: Additional parameters
        """
        metrics = self.compute(prediction, ground_truth, **kwargs)
        self.results.update(metrics)
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        })
        return metrics
    
    def get_results(self) -> Dict[str, float]:
        """Get current metric results.
        
        Returns:
            Dictionary of metric values
        """
        return self.results.copy()
    
    def save_results(self, filename: Optional[str] = None):
        """Save metric results to file.
        
        Args:
            filename: Output filename (default: {name}_metrics.json)
        """
        if not self.output_dir:
            raise ValueError("Output directory not specified")
        
        if filename is None:
            filename = f"{self.name}_metrics.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        output_data = {
            'name': self.name,
            'results': self.results,
            'history': self.history
        }
        
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Saved {self.name} metrics to: {filepath}")
    
    def print_results(self):
        """Print formatted metric results."""
        print(f"\n=== {self.name} Metrics ===")
        for key, value in self.results.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
    
    @staticmethod
    def aggregate_metrics(metric_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics from multiple evaluations.
        
        Args:
            metric_list: List of metric dictionaries
            
        Returns:
            Aggregated metrics with mean and std
        """
        if not metric_list:
            return {}
        
        aggregated = {}
        all_keys = set()
        for metrics in metric_list:
            all_keys.update(metrics.keys())
        
        for key in all_keys:
            values = [m[key] for m in metric_list if key in m]
            if values and all(isinstance(v, (int, float)) for v in values):
                aggregated[f"{key}_mean"] = np.mean(values)
                aggregated[f"{key}_std"] = np.std(values)
                aggregated[f"{key}_min"] = np.min(values)
                aggregated[f"{key}_max"] = np.max(values)
        
        return aggregated