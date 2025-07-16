#!/usr/bin/env python3
"""Generate test queries for language queryability evaluation."""

import json
import argparse
from pathlib import Path
from typing import List, Dict
import random


def generate_replica_queries() -> List[Dict]:
    """Generate test queries for Replica dataset vocabulary."""
    
    # Replica vocabulary (common objects)
    common_objects = [
        "chair", "table", "sofa", "bed", "desk", "lamp", "door",
        "window", "wall", "floor", "ceiling", "cabinet", "sink",
        "toilet", "bathtub", "refrigerator", "book", "bottle",
        "cup", "plate", "vase", "pillow", "picture", "rug"
    ]
    
    queries = []
    
    # 1. Simple object queries (30 queries)
    for obj in common_objects[:30]:
        queries.append({
            "id": f"simple_{obj}",
            "query": obj,
            "type": "simple",
            "expected_objects": [obj],
            "difficulty": "easy"
        })
    
    # 2. Spatial relationship queries (20 queries)
    spatial_pairs = [
        ("chair", "table", "near"),
        ("lamp", "desk", "on"),
        ("picture", "wall", "on"),
        ("rug", "floor", "on"),
        ("book", "desk", "on"),
        ("cup", "table", "on"),
        ("pillow", "bed", "on"),
        ("bottle", "table", "near"),
        ("chair", "desk", "next to"),
        ("sink", "cabinet", "above")
    ]
    
    for obj1, obj2, relation in spatial_pairs:
        queries.append({
            "id": f"spatial_{obj1}_{relation}_{obj2}",
            "query": f"{obj1} {relation} {obj2}",
            "type": "spatial",
            "target_object": obj1,
            "reference_object": obj2,
            "spatial_relation": relation,
            "difficulty": "medium"
        })
    
    # 3. Attribute queries (10 queries)
    attributes = ["large", "small", "wooden", "metal", "white", "black"]
    
    for attr in attributes[:5]:
        obj = random.choice(common_objects[:10])
        queries.append({
            "id": f"attribute_{attr}_{obj}",
            "query": f"{attr} {obj}",
            "type": "attribute",
            "base_object": obj,
            "attribute": attr,
            "difficulty": "hard"
        })
    
    # 4. Complex queries (10 queries)
    complex_templates = [
        "all chairs in the room",
        "furniture near the window",
        "objects on the table",
        "red objects",
        "kitchen appliances",
        "bathroom fixtures",
        "seating furniture",
        "storage furniture",
        "decorative objects",
        "lighting fixtures"
    ]
    
    for i, template in enumerate(complex_templates):
        queries.append({
            "id": f"complex_{i}",
            "query": template,
            "type": "complex",
            "difficulty": "hard"
        })
    
    # 5. Negative queries (10 queries) - objects not in scene
    negative_objects = ["car", "tree", "dog", "bicycle", "phone"]
    
    for obj in negative_objects:
        queries.append({
            "id": f"negative_{obj}",
            "query": obj,
            "type": "negative",
            "expected_objects": [],
            "difficulty": "easy"
        })
    
    return queries


def generate_scannet_queries() -> List[Dict]:
    """Generate test queries for ScanNet dataset vocabulary."""
    
    # ScanNet NYU40 classes (subset)
    nyu40_objects = [
        "wall", "floor", "cabinet", "bed", "chair", "sofa", "table",
        "door", "window", "bookshelf", "picture", "counter", "blinds",
        "desk", "shelves", "curtain", "dresser", "pillow", "mirror",
        "floor mat", "clothes", "ceiling", "books", "refrigerator",
        "television", "paper", "towel", "shower curtain", "box",
        "whiteboard", "person", "nightstand", "toilet", "sink",
        "lamp", "bathtub", "bag"
    ]
    
    queries = []
    
    # Similar structure to Replica queries but with ScanNet vocabulary
    for obj in nyu40_objects[:20]:
        queries.append({
            "id": f"simple_{obj.replace(' ', '_')}",
            "query": obj,
            "type": "simple",
            "expected_objects": [obj],
            "difficulty": "easy"
        })
    
    return queries


def generate_cross_dataset_queries() -> List[Dict]:
    """Generate queries that work across datasets."""
    
    # Common objects across datasets
    universal_objects = [
        "chair", "table", "door", "window", "wall", "floor",
        "ceiling", "bed", "desk", "lamp", "sink", "toilet"
    ]
    
    queries = []
    
    for obj in universal_objects:
        queries.append({
            "id": f"universal_{obj}",
            "query": obj,
            "type": "simple",
            "expected_objects": [obj],
            "dataset": "any",
            "difficulty": "easy"
        })
    
    # Universal spatial queries
    universal_spatial = [
        "chair near table",
        "lamp on desk",
        "door near wall",
        "window on wall"
    ]
    
    for i, query_text in enumerate(universal_spatial):
        queries.append({
            "id": f"universal_spatial_{i}",
            "query": query_text,
            "type": "spatial",
            "dataset": "any",
            "difficulty": "medium"
        })
    
    return queries


def main():
    parser = argparse.ArgumentParser(description='Generate test queries for evaluation')
    parser.add_argument('--dataset', type=str, default='replica',
                       choices=['replica', 'scannet', 'universal'])
    parser.add_argument('--output', type=str, default='test_queries.json')
    parser.add_argument('--num_queries', type=int, default=100,
                       help='Maximum number of queries to generate')
    
    args = parser.parse_args()
    
    # Generate queries based on dataset
    if args.dataset == 'replica':
        queries = generate_replica_queries()
    elif args.dataset == 'scannet':
        queries = generate_scannet_queries()
    else:
        queries = generate_cross_dataset_queries()
    
    # Limit number of queries
    queries = queries[:args.num_queries]
    
    # Add metadata
    output_data = {
        "dataset": args.dataset,
        "num_queries": len(queries),
        "query_types": {
            "simple": len([q for q in queries if q["type"] == "simple"]),
            "spatial": len([q for q in queries if q["type"] == "spatial"]),
            "attribute": len([q for q in queries if q["type"] == "attribute"]),
            "complex": len([q for q in queries if q["type"] == "complex"]),
            "negative": len([q for q in queries if q["type"] == "negative"])
        },
        "queries": queries
    }
    
    # Save to file
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Generated {len(queries)} test queries")
    print(f"Query type distribution:")
    for qtype, count in output_data["query_types"].items():
        print(f"  {qtype}: {count}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()