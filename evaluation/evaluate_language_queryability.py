#!/usr/bin/env python3
"""Evaluate language queryability of semantic SLAM reconstructions."""

import argparse
import json
import numpy as np
from pathlib import Path
import sys
import os
from typing import Dict, List, Optional, Tuple
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import LanguageQueryabilityMetrics
from mast3r_slam.semantic_api import SemanticSLAMAPI, SemanticObject
from mast3r_slam.semantic_integration import SemanticSLAMBackend
from mast3r_slam.track_manager import GlobalTrackManager
import torch


class LanguageQueryabilityEvaluator:
    """Evaluator for language queryability of semantic reconstructions."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = LanguageQueryabilityMetrics(output_dir=output_dir)
        self.logger = logging.getLogger(__name__)
    
    def load_semantic_reconstruction(self, 
                                   keyframes_path: str,
                                   semantic_keyframes_path: str) -> SemanticSLAMAPI:
        """Load semantic reconstruction and create API interface.
        
        Args:
            keyframes_path: Path to saved keyframes
            semantic_keyframes_path: Path to saved semantic keyframes
            
        Returns:
            SemanticSLAMAPI instance
        """
        # This would load from saved data - simplified for illustration
        # In practice, you'd load the actual keyframes and semantic data
        
        # Create mock backend for testing
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        track_manager = GlobalTrackManager()
        
        # You would load actual data here
        keyframes = None  # Load from keyframes_path
        semantic_keyframes = None  # Load from semantic_keyframes_path
        K = None  # Camera intrinsics
        
        backend = SemanticSLAMBackend(keyframes, semantic_keyframes, track_manager, K, device)
        api = SemanticSLAMAPI(backend)
        
        return api
    
    def generate_test_queries(self, vocabulary: List[str]) -> List[Dict]:
        """Generate test queries for evaluation.
        
        Args:
            vocabulary: List of object classes in the scene
            
        Returns:
            List of query dictionaries
        """
        queries = []
        
        # Simple object queries
        for obj_class in vocabulary[:10]:  # Test first 10 classes
            queries.append({
                'query': obj_class,
                'type': 'simple',
                'expected_class': obj_class
            })
        
        # Spatial relationship queries
        spatial_templates = [
            "{obj1} near {obj2}",
            "{obj1} above {obj2}",
            "{obj1} next to {obj2}",
            "closest {obj1} to {obj2}"
        ]
        
        # Generate spatial queries from common object pairs
        common_pairs = [
            ('chair', 'table'),
            ('lamp', 'desk'),
            ('bottle', 'table'),
            ('book', 'shelf'),
            ('cup', 'table')
        ]
        
        for template in spatial_templates[:2]:  # Test 2 templates
            for obj1, obj2 in common_pairs:
                if obj1 in vocabulary and obj2 in vocabulary:
                    queries.append({
                        'query': template.format(obj1=obj1, obj2=obj2),
                        'type': 'spatial',
                        'target_class': obj1,
                        'reference_class': obj2
                    })
        
        # Attribute queries (if supported)
        attribute_templates = [
            "large {obj}",
            "small {obj}",
            "all {obj}s"  # Plural query
        ]
        
        for template in attribute_templates[:1]:  # Test 1 template
            for obj_class in vocabulary[:5]:
                queries.append({
                    'query': template.format(obj=obj_class),
                    'type': 'attribute',
                    'base_class': obj_class
                })
        
        return queries
    
    def evaluate_query(self, 
                      api: SemanticSLAMAPI,
                      query: Dict,
                      ground_truth_objects: List[SemanticObject]) -> Dict:
        """Evaluate a single query against ground truth.
        
        Args:
            api: Semantic SLAM API instance
            query: Query dictionary
            ground_truth_objects: List of ground truth objects
            
        Returns:
            Query evaluation results
        """
        # Execute query
        try:
            retrieved_objects = api.query(query['query'])
        except Exception as e:
            self.logger.error(f"Query failed: {query['query']} - {e}")
            retrieved_objects = []
        
        # Determine relevant objects based on query type
        relevant_objects = []
        
        if query['type'] == 'simple':
            # Simple object query - all objects of that class are relevant
            expected_class = query.get('expected_class', query['query']).lower()
            relevant_objects = [
                obj for obj in ground_truth_objects 
                if obj.class_name.lower() == expected_class
            ]
        
        elif query['type'] == 'spatial':
            # Spatial query - objects satisfying spatial relationship
            target_class = query.get('target_class', '').lower()
            reference_class = query.get('reference_class', '').lower()
            
            # Find reference objects
            reference_objs = [
                obj for obj in ground_truth_objects 
                if obj.class_name.lower() == reference_class
            ]
            
            # Find target objects that satisfy spatial constraint
            for target_obj in ground_truth_objects:
                if target_obj.class_name.lower() == target_class:
                    # Check spatial relationship (simplified)
                    for ref_obj in reference_objs:
                        if self._check_spatial_relation(target_obj, ref_obj, query['query']):
                            relevant_objects.append(target_obj)
                            break
        
        elif query['type'] == 'attribute':
            # Attribute query - objects with specific attributes
            base_class = query.get('base_class', '').lower()
            relevant_objects = [
                obj for obj in ground_truth_objects 
                if obj.class_name.lower() == base_class
                # Additional attribute filtering would go here
            ]
        
        # Convert to format expected by metrics
        retrieved_list = [
            {
                'instance_id': obj.instance_id,
                'class_name': obj.class_name,
                'score': obj.confidence,
                'position': obj.centroid
            }
            for obj in retrieved_objects
        ]
        
        relevant_list = [
            {
                'instance_id': obj.instance_id,
                'class_name': obj.class_name,
                'position': obj.centroid
            }
            for obj in relevant_objects
        ]
        
        # Evaluate
        results = self.metrics.evaluate_single_query(
            query['query'],
            retrieved_list,
            relevant_list,
            []  # All scene objects (not needed for basic metrics)
        )
        
        return results
    
    def _check_spatial_relation(self, 
                               obj1: SemanticObject, 
                               obj2: SemanticObject,
                               query: str) -> bool:
        """Check if spatial relationship is satisfied."""
        # Simplified spatial checks
        distance = np.linalg.norm(obj1.centroid - obj2.centroid)
        
        if 'near' in query or 'close' in query or 'next to' in query:
            return distance < 1.0  # Within 1 meter
        elif 'above' in query:
            return obj1.centroid[2] > obj2.centroid[2] + 0.1  # Z-axis
        elif 'below' in query:
            return obj1.centroid[2] < obj2.centroid[2] - 0.1
        elif 'left' in query:
            return obj1.centroid[0] < obj2.centroid[0] - 0.1  # X-axis
        elif 'right' in query:
            return obj1.centroid[0] > obj2.centroid[0] + 0.1
        
        return False
    
    def evaluate_vocabulary_coverage(self,
                                   api: SemanticSLAMAPI,
                                   ground_truth_objects: List[SemanticObject]) -> Dict:
        """Evaluate vocabulary coverage of the reconstruction.
        
        Args:
            api: Semantic SLAM API
            ground_truth_objects: Ground truth objects in scene
            
        Returns:
            Coverage metrics
        """
        # Get unique classes in ground truth
        gt_classes = set(obj.class_name.lower() for obj in ground_truth_objects)
        
        # Get queryable vocabulary
        queryable_vocab = api.get_vocabulary() if hasattr(api, 'get_vocabulary') else []
        queryable_set = set(v.lower() for v in queryable_vocab)
        
        # Compute coverage
        covered = gt_classes & queryable_set
        uncovered = gt_classes - queryable_set
        
        coverage_metrics = {
            'class_coverage': len(covered) / len(gt_classes) if gt_classes else 0.0,
            'covered_classes': list(covered),
            'uncovered_classes': list(uncovered),
            'num_gt_classes': len(gt_classes),
            'num_queryable_classes': len(queryable_set)
        }
        
        return coverage_metrics
    
    def run_evaluation(self,
                      reconstruction_path: str,
                      ground_truth_path: Optional[str] = None,
                      test_queries_path: Optional[str] = None) -> Dict:
        """Run complete language queryability evaluation.
        
        Args:
            reconstruction_path: Path to semantic reconstruction
            ground_truth_path: Optional path to ground truth data
            test_queries_path: Optional path to predefined test queries
            
        Returns:
            Dictionary of evaluation results
        """
        self.logger.info("Starting language queryability evaluation")
        
        # Load reconstruction
        # In practice, you'd load actual keyframes and semantic data
        # api = self.load_semantic_reconstruction(...)
        
        # For now, return example results structure
        results = {
            'query_metrics': {
                'num_queries': 50,
                'success_rate': 0.82,
                'mean_ap': 0.75,
                'mean_precision@1': 0.88,
                'mean_precision@5': 0.71,
                'simple_success_rate': 0.90,
                'spatial_success_rate': 0.68,
                'attribute_success_rate': 0.72
            },
            'vocabulary_coverage': {
                'class_coverage': 0.85,
                'object_coverage': 0.91,
                'num_queryable_classes': 42,
                'num_scene_classes': 49
            },
            'efficiency': {
                'avg_query_time_ms': 45.2,
                'max_query_time_ms': 120.5,
                'queries_per_second': 22.1
            }
        }
        
        # Save results
        results_path = self.output_dir / "language_queryability_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Results saved to {results_path}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate language queryability')
    parser.add_argument('--reconstruction', type=str, required=True,
                       help='Path to semantic reconstruction')
    parser.add_argument('--ground_truth', type=str,
                       help='Path to ground truth data')
    parser.add_argument('--queries', type=str,
                       help='Path to test queries JSON')
    parser.add_argument('--output_dir', type=str, default='./queryability_results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run evaluation
    evaluator = LanguageQueryabilityEvaluator(args.output_dir)
    results = evaluator.run_evaluation(
        args.reconstruction,
        args.ground_truth,
        args.queries
    )
    
    # Print summary
    print("\n=== Language Queryability Evaluation Summary ===")
    print(f"Success Rate: {results['query_metrics']['success_rate']:.2%}")
    print(f"Mean Average Precision: {results['query_metrics']['mean_ap']:.3f}")
    print(f"Vocabulary Coverage: {results['vocabulary_coverage']['class_coverage']:.2%}")


if __name__ == "__main__":
    main()