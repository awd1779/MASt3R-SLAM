#!/usr/bin/env python3
"""Convert Replica semantic info from JSON to TXT format that Habitat-Sim expects."""

import json
import os
from pathlib import Path


def convert_semantic_json_to_txt(json_path: str, txt_path: str):
    """Convert info_semantic.json to info_semantic.txt format.
    
    Habitat-Sim expects a specific text format for semantic descriptors.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    with open(txt_path, 'w') as f:
        # Write header
        f.write("# Replica semantic information\n")
        f.write("# Format: instance_id,class_id,class_name,region_id\n\n")
        
        # Process objects
        if "objects" in data:
            for obj in data["objects"]:
                instance_id = obj["id"]
                class_id = obj.get("class_id", -1)
                class_name = obj.get("class_name", "unknown")
                region_id = obj.get("region_id", -1)
                
                # Remove spaces from class names for parsing
                class_name = class_name.replace(" ", "_")
                
                # Write in format expected by Habitat
                f.write(f"{instance_id},{class_id},{class_name},{region_id}\n")
    
    print(f"Converted {json_path} -> {txt_path}")


def process_replica_scene(scene_path: str):
    """Process a single Replica scene."""
    habitat_dir = Path(scene_path) / "habitat"
    json_file = habitat_dir / "info_semantic.json"
    txt_file = habitat_dir / "info_semantic.txt"
    
    if json_file.exists():
        convert_semantic_json_to_txt(str(json_file), str(txt_file))
        
        # Also update stage config to use .txt
        stage_config = habitat_dir / "replica_stage.stage_config.json"
        if stage_config.exists():
            with open(stage_config, 'r') as f:
                config = json.load(f)
            
            config["semantic_descriptor_filename"] = "info_semantic.txt"
            
            with open(stage_config, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"Updated {stage_config}")
    else:
        print(f"Warning: {json_file} not found")


def main():
    """Convert semantic info for all Replica scenes."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert Replica semantic info to Habitat format')
    parser.add_argument('--scene', type=str, default=None,
                       help='Single scene to process (e.g., apartment_0)')
    parser.add_argument('--dataset_path', type=str, default='datasets/replica_dataset',
                       help='Path to Replica dataset')
    
    args = parser.parse_args()
    
    if args.scene:
        # Process single scene
        scene_path = os.path.join(args.dataset_path, args.scene)
        process_replica_scene(scene_path)
    else:
        # Process all scenes
        dataset_path = Path(args.dataset_path)
        for scene_dir in dataset_path.iterdir():
            if scene_dir.is_dir() and scene_dir.name.endswith('_0'):
                process_replica_scene(str(scene_dir))
    
    print("\nConversion complete!")
    print("Now the semantic information should load properly in Habitat-Sim.")


if __name__ == "__main__":
    main()