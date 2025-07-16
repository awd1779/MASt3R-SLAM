#!/usr/bin/env python3
"""Prepare Replica dataset for semantic rendering in Habitat-Sim."""

import json
import os
import argparse
from pathlib import Path


def create_semantic_descriptor_file(scene_path: str):
    """Create info_semantic.txt from info_semantic.json for Habitat-Sim.
    
    Habitat-Sim expects a specific format for semantic scene descriptors.
    This converts Replica's JSON format to the expected text format.
    """
    scene_path = Path(scene_path)
    json_path = scene_path / "habitat" / "info_semantic.json"
    txt_path = scene_path / "habitat" / "info_semantic.txt"
    
    if not json_path.exists():
        print(f"Error: {json_path} not found")
        return False
    
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create text file in Habitat-Sim format
    with open(txt_path, 'w') as f:
        # Write header
        f.write("# Semantic information for Replica dataset\n")
        f.write("# Generated from info_semantic.json\n\n")
        
        # Write class definitions
        if "classes" in data:
            f.write("# Class definitions\n")
            for cls in data["classes"]:
                class_id = cls["id"]
                class_name = cls["name"]
                # Habitat expects: class_id,class_name
                f.write(f"{class_id},{class_name}\n")
        
        f.write("\n# Object instances\n")
        
        # Write object instances
        if "objects" in data:
            for obj in data["objects"]:
                obj_id = obj["id"]
                class_id = obj["class_id"]
                class_name = obj.get("class_name", "unknown")
                # Write instance information
                f.write(f"{obj_id},{class_id},{class_name}\n")
    
    print(f"Created {txt_path}")
    return True


def update_stage_config(scene_path: str):
    """Update the stage config to reference the semantic descriptor."""
    scene_path = Path(scene_path)
    config_path = scene_path / "habitat" / "replica_stage.stage_config.json"
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Update semantic descriptor reference
        config["semantic_descriptor_filename"] = "info_semantic.txt"
        
        # Save updated config
        backup_path = config_path.with_suffix('.json.bak')
        os.rename(config_path, backup_path)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Updated {config_path}")
        print(f"Backup saved to {backup_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare Replica dataset for semantic rendering in Habitat-Sim'
    )
    parser.add_argument('scene_path', type=str,
                       help='Path to Replica scene (e.g., datasets/replica_dataset/apartment_0)')
    parser.add_argument('--update-config', action='store_true',
                       help='Update stage config file to reference semantic descriptor')
    
    args = parser.parse_args()
    
    print(f"Preparing semantic data for: {args.scene_path}")
    
    # Create semantic descriptor file
    success = create_semantic_descriptor_file(args.scene_path)
    
    if success and args.update_config:
        update_stage_config(args.scene_path)
    
    print("\nDone! You can now generate trajectories with semantic rendering.")
    
    # Print next steps
    print("\nNext steps:")
    print("1. Run trajectory generation:")
    print(f"   python evaluation/generate_replica_trajectory.py --scene_path {args.scene_path} --output_dir ./trajectories_semantic")
    print("\n2. Check that semantic images contain non-zero values")


if __name__ == "__main__":
    main()