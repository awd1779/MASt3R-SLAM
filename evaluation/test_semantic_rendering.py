#!/usr/bin/env python3
"""Test semantic rendering with Replica dataset in Habitat-Sim."""

import habitat_sim
import numpy as np
import cv2
import os
from pathlib import Path


def test_semantic_rendering(scene_path: str):
    """Test if semantic rendering works with Replica."""
    
    print(f"Testing semantic rendering for: {scene_path}")
    
    # Method 1: Try loading mesh_semantic.ply directly
    print("\n1. Testing direct mesh_semantic.ply loading...")
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = os.path.join(scene_path, "habitat", "mesh_semantic.ply")
    cfg.enable_physics = False
    cfg.load_semantic_mesh = True
    
    # Configure semantic sensor
    sensor_spec = habitat_sim.CameraSensorSpec()
    sensor_spec.uuid = "semantic"
    sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    sensor_spec.resolution = [480, 640]
    sensor_spec.hfov = 90.0
    
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor_spec]
    
    try:
        sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent_cfg]))
        
        # Get observation
        obs = sim.get_sensor_observations()
        semantic = obs["semantic"]
        
        unique_vals = np.unique(semantic)
        print(f"Unique semantic values: {unique_vals}")
        print(f"Max value: {semantic.max()}")
        print(f"Non-zero pixels: {(semantic > 0).sum()}")
        
        # Save test image
        cv2.imwrite("test_semantic_method1.png", semantic.astype(np.uint16))
        
        sim.close()
        
    except Exception as e:
        print(f"Method 1 failed: {e}")
    
    # Method 2: Try using stage config
    print("\n2. Testing with stage config...")
    cfg2 = habitat_sim.SimulatorConfiguration()
    stage_config = os.path.join(scene_path, "habitat", "replica_stage.stage_config.json")
    
    if os.path.exists(stage_config):
        cfg2.scene_id = stage_config
        cfg2.enable_physics = False
        cfg2.load_semantic_mesh = True
        
        try:
            sim2 = habitat_sim.Simulator(habitat_sim.Configuration(cfg2, [agent_cfg]))
            
            # Check if semantic scene is loaded
            semantic_scene = sim2.semantic_scene
            if semantic_scene is not None:
                print(f"Semantic scene loaded!")
                print(f"Number of levels: {len(semantic_scene.levels)}")
                print(f"Number of regions: {len(semantic_scene.regions)}")
                print(f"Number of objects: {len(semantic_scene.objects)}")
                
                # Print some object info
                for i, obj in enumerate(semantic_scene.objects[:5]):
                    print(f"  Object {i}: id={obj.id}, category={obj.category}")
            else:
                print("No semantic scene loaded")
            
            # Get observation
            obs2 = sim2.get_sensor_observations()
            semantic2 = obs2["semantic"]
            
            unique_vals2 = np.unique(semantic2)
            print(f"Unique semantic values: {unique_vals2}")
            print(f"Max value: {semantic2.max()}")
            print(f"Non-zero pixels: {(semantic2 > 0).sum()}")
            
            # Save test image
            cv2.imwrite("test_semantic_method2.png", semantic2.astype(np.uint16))
            
            sim2.close()
            
        except Exception as e:
            print(f"Method 2 failed: {e}")
    
    # Method 3: Check what files exist
    print("\n3. Checking available files...")
    habitat_dir = Path(scene_path) / "habitat"
    if habitat_dir.exists():
        for f in habitat_dir.iterdir():
            print(f"  {f.name}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        scene_path = sys.argv[1]
    else:
        scene_path = "datasets/replica_dataset/apartment_0"
    
    test_semantic_rendering(scene_path)