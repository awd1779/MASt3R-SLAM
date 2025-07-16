#!/usr/bin/env python3
"""Test script to verify Replica viewer functionality."""

import os
import sys
import numpy as np

def test_habitat_import():
    """Test if Habitat-Sim can be imported."""
    try:
        import habitat_sim
        print(f"✓ Habitat-Sim version: {habitat_sim.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import habitat_sim: {e}")
        print("\nTo install Habitat-Sim:")
        print("conda install habitat-sim -c conda-forge -c aihabitat")
        return False

def test_viewer_basic():
    """Test basic viewer functionality."""
    try:
        from interactive_replica_viewer import InteractiveReplicaViewer
        print("✓ Viewer module imports successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import viewer: {e}")
        return False

def check_replica_scene(scene_path):
    """Check if Replica scene has required files."""
    from pathlib import Path
    
    scene_path = Path(scene_path)
    if not scene_path.exists():
        print(f"✗ Scene path does not exist: {scene_path}")
        return False
    
    print(f"\nChecking scene: {scene_path}")
    
    # Check for mesh files
    mesh_files = {
        "mesh.ply": False,
        "mesh_semantic.ply": False,
        "habitat/mesh.ply": False,
        "habitat/mesh_semantic.ply": False
    }
    
    for mesh_file, _ in mesh_files.items():
        full_path = scene_path / mesh_file
        if full_path.exists():
            mesh_files[mesh_file] = True
            print(f"  ✓ Found: {mesh_file}")
    
    # Check for semantic info
    semantic_files = [
        "info_semantic.json",
        "info_semantic.txt",
        "habitat/info_semantic.json",
        "habitat/info_semantic.txt"
    ]
    
    semantic_found = False
    for semantic_file in semantic_files:
        if (scene_path / semantic_file).exists():
            print(f"  ✓ Found semantic info: {semantic_file}")
            semantic_found = True
            break
    
    if not semantic_found:
        print("  ⚠ No semantic info file found")
    
    # Summary
    has_textured = any(mesh_files[k] for k in ["mesh.ply", "habitat/mesh.ply"])
    has_semantic = any(mesh_files[k] for k in ["mesh_semantic.ply", "habitat/mesh_semantic.ply"])
    
    if has_textured:
        print("  ✓ Textured mesh available")
    if has_semantic:
        print("  ✓ Semantic mesh available")
    
    return has_textured or has_semantic

def main():
    print("=== Replica Viewer Test ===\n")
    
    # Test imports
    if not test_habitat_import():
        sys.exit(1)
    
    if not test_viewer_basic():
        sys.exit(1)
    
    # Check scene if provided
    if len(sys.argv) > 1:
        scene_path = sys.argv[1]
        if check_replica_scene(scene_path):
            print(f"\n✓ Scene '{scene_path}' is ready for viewing")
            print("\nTo run the viewer:")
            print(f"python evaluation/interactive_replica_viewer.py {scene_path}")
        else:
            print(f"\n✗ Scene '{scene_path}' is missing required files")
    else:
        print("\nUsage: python test_replica_viewer.py [scene_path]")
        print("\nExample Replica scenes:")
        print("  /path/to/replica/apartment_0")
        print("  /path/to/replica/frl_apartment_0")
        print("  /path/to/replica/office_0")

if __name__ == "__main__":
    main()