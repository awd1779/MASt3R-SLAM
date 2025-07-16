"""Language queryability evaluation metrics for semantic SLAM."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import re
from sklearn.metrics import average_precision_score

from .base_metrics import BaseMetrics


class LanguageQueryabilityMetrics(BaseMetrics):
    """Evaluate language queryability of semantic SLAM reconstructions."""
    
    def __init__(self,
                 distance_threshold: float = 0.5,
                 iou_threshold: float = 0.5,
                 output_dir: Optional[str] = None):
        """Initialize language queryability metrics.
        
        Args:
            distance_threshold: Distance threshold for spatial queries (meters)
            iou_threshold: IoU threshold for object matching
            output_dir: Directory to save outputs
        """
        super().__init__("LanguageQueryability", output_dir)
        
        self.distance_threshold = distance_threshold
        self.iou_threshold = iou_threshold
        
        self.reset()
    
    def reset(self):
        """Reset internal state."""
        self.results = {}
        self.query_results = []
        self.vocabulary_coverage = {}
        self.spatial_query_results = []
    
    def evaluate_single_query(self,
                            query: str,
                            retrieved_objects: List[Dict],
                            ground_truth_objects: List[Dict],
                            scene_objects: List[Dict]) -> Dict:
        """Evaluate a single language query.
        
        Args:
            query: Natural language query (e.g., "chair", "red chair near table")
            retrieved_objects: List of retrieved objects with scores
            ground_truth_objects: List of relevant objects for this query
            scene_objects: All objects in the scene
            
        Returns:
            Dictionary of query-specific metrics
        """
        # Parse query type
        query_type = self._classify_query(query)
        
        # Get relevant and retrieved IDs
        relevant_ids = {obj['instance_id'] for obj in ground_truth_objects}
        retrieved_ids = [obj['instance_id'] for obj in retrieved_objects]
        scores = [obj.get('score', 1.0) for obj in retrieved_objects]
        
        # Compute basic retrieval metrics
        metrics = {
            'query': query,
            'query_type': query_type,
            'num_relevant': len(relevant_ids),
            'num_retrieved': len(retrieved_ids)
        }
        
        # Precision and Recall
        if len(retrieved_ids) > 0:
            true_positives = sum(1 for id in retrieved_ids if id in relevant_ids)
            precision = true_positives / len(retrieved_ids)
            recall = true_positives / len(relevant_ids) if len(relevant_ids) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        else:
            precision = recall = f1 = 0.0
        
        metrics.update({
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })
        
        # Precision@K for different K values
        for k in [1, 3, 5, 10]:
            if k <= len(retrieved_ids):
                true_positives_at_k = sum(1 for id in retrieved_ids[:k] if id in relevant_ids)
                metrics[f'precision@{k}'] = true_positives_at_k / k
            else:
                metrics[f'precision@{k}'] = precision
        
        # Average Precision (AP)
        if len(relevant_ids) > 0 and len(retrieved_ids) > 0:
            # Create binary relevance array
            relevance = [1 if id in relevant_ids else 0 for id in retrieved_ids]
            
            # Compute AP
            if sum(relevance) > 0:
                ap = self._compute_average_precision(relevance, scores)
            else:
                ap = 0.0
        else:
            ap = 0.0
        
        metrics['average_precision'] = ap
        
        # Query success (at least one relevant object retrieved)
        metrics['success'] = any(id in relevant_ids for id in retrieved_ids)
        
        # Store detailed results
        self.query_results.append(metrics)
        
        return metrics
    
    def evaluate_spatial_query(self,
                             query: str,
                             retrieved_objects: List[Dict],
                             ground_truth_objects: List[Dict]) -> Dict:
        """Evaluate spatial relationship queries.
        
        Args:
            query: Spatial query (e.g., "chair near table")
            retrieved_objects: Retrieved objects with positions
            ground_truth_objects: Ground truth objects satisfying the query
            
        Returns:
            Dictionary of spatial query metrics
        """
        # Extract spatial components from query
        spatial_components = self._parse_spatial_query(query)
        
        metrics = {
            'query': query,
            'query_type': 'spatial',
            'spatial_relation': spatial_components.get('relation', 'unknown')
        }
        
        # Basic retrieval metrics
        relevant_ids = {obj['instance_id'] for obj in ground_truth_objects}
        retrieved_ids = {obj['instance_id'] for obj in retrieved_objects}
        
        true_positives = len(relevant_ids & retrieved_ids)
        false_positives = len(retrieved_ids - relevant_ids)
        false_negatives = len(relevant_ids - retrieved_ids)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        
        metrics.update({
            'precision': precision,
            'recall': recall,
            'f1_score': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0,
            'success': true_positives > 0
        })
        
        self.spatial_query_results.append(metrics)
        
        return metrics
    
    def evaluate_vocabulary_coverage(self,
                                   scene_objects: List[Dict],
                                   queryable_vocabulary: List[str]) -> Dict:
        """Evaluate what percentage of scene objects can be queried.
        
        Args:
            scene_objects: All objects in the scene with their labels
            queryable_vocabulary: List of queryable object classes
            
        Returns:
            Dictionary of vocabulary coverage metrics
        """
        # Get unique object classes in scene
        scene_classes = set()
        for obj in scene_objects:
            if 'class_name' in obj:
                scene_classes.add(obj['class_name'].lower())
        
        # Normalize vocabulary
        vocab_set = {v.lower() for v in queryable_vocabulary}
        
        # Compute coverage
        covered_classes = scene_classes & vocab_set
        uncovered_classes = scene_classes - vocab_set
        
        # Count objects by coverage
        covered_objects = 0
        total_objects = len(scene_objects)
        
        for obj in scene_objects:
            if obj.get('class_name', '').lower() in vocab_set:
                covered_objects += 1
        
        coverage_metrics = {
            'class_coverage': len(covered_classes) / len(scene_classes) if len(scene_classes) > 0 else 0.0,
            'object_coverage': covered_objects / total_objects if total_objects > 0 else 0.0,
            'covered_classes': list(covered_classes),
            'uncovered_classes': list(uncovered_classes),
            'num_scene_classes': len(scene_classes),
            'num_vocab_classes': len(vocab_set)
        }
        
        self.vocabulary_coverage = coverage_metrics
        
        return coverage_metrics
    
    def compute(self,
               queries: List[Dict],
               scene_reconstruction: Dict,
               ground_truth: Optional[Dict] = None,
               **kwargs) -> Dict[str, float]:
        """Compute all language queryability metrics.
        
        Args:
            queries: List of query dicts with 'query', 'retrieved', 'relevant' keys
            scene_reconstruction: Dict with 'objects' and 'vocabulary' keys
            ground_truth: Optional ground truth scene data
            
        Returns:
            Dictionary of aggregated metrics
        """
        # Process each query
        for query_data in queries:
            self.evaluate_single_query(
                query_data['query'],
                query_data.get('retrieved', []),
                query_data.get('relevant', []),
                scene_reconstruction.get('objects', [])
            )
        
        # Aggregate metrics
        if self.query_results:
            # Overall metrics
            self.results['num_queries'] = len(self.query_results)
            self.results['success_rate'] = np.mean([r['success'] for r in self.query_results])
            self.results['mean_precision'] = np.mean([r['precision'] for r in self.query_results])
            self.results['mean_recall'] = np.mean([r['recall'] for r in self.query_results])
            self.results['mean_f1'] = np.mean([r['f1_score'] for r in self.query_results])
            self.results['mean_ap'] = np.mean([r['average_precision'] for r in self.query_results])
            
            # Precision@K
            for k in [1, 3, 5, 10]:
                key = f'precision@{k}'
                values = [r[key] for r in self.query_results if key in r]
                if values:
                    self.results[f'mean_{key}'] = np.mean(values)
            
            # By query type
            for query_type in ['simple', 'attribute', 'spatial', 'complex']:
                type_results = [r for r in self.query_results if r['query_type'] == query_type]
                if type_results:
                    self.results[f'{query_type}_success_rate'] = np.mean([r['success'] for r in type_results])
                    self.results[f'{query_type}_mean_ap'] = np.mean([r['average_precision'] for r in type_results])
        
        # Spatial query metrics
        if self.spatial_query_results:
            self.results['spatial_precision'] = np.mean([r['precision'] for r in self.spatial_query_results])
            self.results['spatial_recall'] = np.mean([r['recall'] for r in self.spatial_query_results])
            self.results['spatial_success_rate'] = np.mean([r['success'] for r in self.spatial_query_results])
        
        # Vocabulary coverage
        if 'vocabulary' in scene_reconstruction and 'objects' in scene_reconstruction:
            coverage = self.evaluate_vocabulary_coverage(
                scene_reconstruction['objects'],
                scene_reconstruction['vocabulary']
            )
            self.results.update({
                f'vocabulary_{k}': v for k, v in coverage.items() 
                if isinstance(v, (int, float))
            })
        
        return self.results
    
    def _classify_query(self, query: str) -> str:
        """Classify query type."""
        query_lower = query.lower()
        
        # Check for spatial keywords
        spatial_keywords = ['near', 'far', 'above', 'below', 'left', 'right', 'between', 'next to']
        if any(keyword in query_lower for keyword in spatial_keywords):
            return 'spatial'
        
        # Check for attribute keywords
        attribute_keywords = ['red', 'blue', 'green', 'large', 'small', 'big', 'tall', 'short']
        if any(keyword in query_lower for keyword in attribute_keywords):
            return 'attribute'
        
        # Check for complex queries (multiple conditions)
        if ' and ' in query_lower or ' or ' in query_lower:
            return 'complex'
        
        # Simple object query
        return 'simple'
    
    def _parse_spatial_query(self, query: str) -> Dict:
        """Parse spatial relationship from query."""
        components = {
            'target': None,
            'reference': None,
            'relation': None
        }
        
        # Simple pattern matching for common spatial queries
        patterns = [
            r'(\w+)\s+(near|close to|next to)\s+(\w+)',
            r'(\w+)\s+(above|below|on top of|under)\s+(\w+)',
            r'(\w+)\s+(left of|right of|beside)\s+(\w+)',
            r'(\w+)\s+(between)\s+(\w+)\s+and\s+(\w+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                components['target'] = match.group(1)
                components['relation'] = match.group(2)
                components['reference'] = match.group(3)
                break
        
        return components
    
    def _compute_average_precision(self, relevance: List[int], scores: List[float]) -> float:
        """Compute average precision for a single query."""
        # Sort by scores in descending order
        sorted_indices = np.argsort(scores)[::-1]
        sorted_relevance = [relevance[i] for i in sorted_indices]
        
        # Compute precision at each relevant position
        precisions = []
        num_relevant = 0
        
        for i, is_relevant in enumerate(sorted_relevance):
            if is_relevant:
                num_relevant += 1
                precision_at_i = num_relevant / (i + 1)
                precisions.append(precision_at_i)
        
        # Average precision
        if precisions:
            return np.mean(precisions)
        else:
            return 0.0
    
    def print_results(self):
        """Print evaluation results in a formatted way."""
        print("\n=== Language Queryability Metrics ===")
        print(f"Total Queries: {self.results.get('num_queries', 0)}")
        print(f"Success Rate: {self.results.get('success_rate', 0):.3f}")
        print(f"Mean Average Precision: {self.results.get('mean_ap', 0):.3f}")
        print(f"Mean F1 Score: {self.results.get('mean_f1', 0):.3f}")
        
        print("\nPrecision@K:")
        for k in [1, 3, 5, 10]:
            key = f'mean_precision@{k}'
            if key in self.results:
                print(f"  P@{k}: {self.results[key]:.3f}")
        
        print("\nQuery Type Performance:")
        for query_type in ['simple', 'attribute', 'spatial', 'complex']:
            success_key = f'{query_type}_success_rate'
            ap_key = f'{query_type}_mean_ap'
            if success_key in self.results:
                print(f"  {query_type.capitalize()}: Success={self.results[success_key]:.3f}, mAP={self.results.get(ap_key, 0):.3f}")
        
        print("\nVocabulary Coverage:")
        print(f"  Class Coverage: {self.results.get('vocabulary_class_coverage', 0):.3f}")
        print(f"  Object Coverage: {self.results.get('vocabulary_object_coverage', 0):.3f}")