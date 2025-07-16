#!/usr/bin/env python3
"""Test proper loading of Replica scenes with semantic and navigation support."""

import habitat_sim
import numpy as np
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt


def test_replica_loading(scene_dataset_config: str, scene_name: str):
    """Test loading a Replica scene with proper configuration.
    
    Args:
        scene_dataset_config: Path to replica.scene_dataset_config.json
        scene_name: Name of the scene (e.g., "apartment_0")
    """
    print(f"\n=== Testing Replica Scene Loading ===")
    print(f"Config: {scene_dataset_config}")
    print(f"Scene: {scene_name}")
    
    # Create simulator configuration
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = scene_dataset_config
    sim_cfg.scene_id = scene_name  # Just the scene name, not full path
    sim_cfg.enable_physics = False
    sim_cfg.load_semantic_mesh = True
    
    # Configure sensors
    sensor_specs = []
    
    # RGB sensor
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "rgb"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [480, 640]
    rgb_sensor_spec.hfov = 90.0
    sensor_specs.append(rgb_sensor_spec)
    
    # Depth sensor
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [480, 640]
    depth_sensor_spec.hfov = 90.0
    sensor_specs.append(depth_sensor_spec)
    
    # Semantic sensor
    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [480, 640]
    semantic_sensor_spec.hfov = 90.0
    sensor_specs.append(semantic_sensor_spec)
    
    # Agent configuration
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    
    # Create simulator
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    
    try:
        sim = habitat_sim.Simulator(cfg)
        print("\n✓ Simulator created successfully")
        
        # Test 1: Check if navigation mesh is loaded
        print("\n--- Navigation Mesh Test ---")
        if sim.pathfinder.is_loaded:
            print("✓ Navigation mesh loaded")
            
            # Get a random navigable point
            nav_point = sim.pathfinder.get_random_navigable_point()
            print(f"  Random navigable point: {nav_point}")
            
            # Test pathfinding
            nav_point2 = sim.pathfinder.get_random_navigable_point()
            path = habitat_sim.ShortestPath()
            path.requested_start = nav_point
            path.requested_end = nav_point2
            found = sim.pathfinder.find_path(path)
            
            if found:
                print(f"✓ Path found with {len(path.points)} points")
                print(f"  Distance: {path.geodesic_distance:.2f}m")
            else:
                print("✗ Path not found")
        else:
            print("✗ Navigation mesh NOT loaded")
        
        # Test 2: Check semantic scene
        print("\n--- Semantic Scene Test ---")
        semantic_scene = sim.semantic_scene
        if semantic_scene is not None:
            print("✓ Semantic scene loaded")
            print(f"  Number of levels: {len(semantic_scene.levels)}")
            print(f"  Number of regions: {len(semantic_scene.regions)}")
            print(f"  Number of objects: {len(semantic_scene.objects)}")
            
            # Show first few objects
            if len(semantic_scene.objects) > 0:
                print("\n  First 5 objects:")
                for i, obj in enumerate(semantic_scene.objects[:5]):
                    if hasattr(obj, 'id'):
                        print(f"    Object {i}: id={obj.id}")
                        if hasattr(obj, 'category'):
                            cat = obj.category
                            if hasattr(cat, 'name'):
                                print(f"      Category: {cat.name()}")
        else:
            print("✗ Semantic scene NOT loaded")
        
        # Test 3: Render test images
        print("\n--- Rendering Test ---")
        
        # Move to a navigable point
        if sim.pathfinder.is_loaded:
            agent = sim.get_agent(0)
            agent_state = habitat_sim.AgentState()
            
            # Set position at a navigable point
            nav_point = sim.pathfinder.get_random_navigable_point()
            agent_state.position = np.array([nav_point[0], nav_point[1], nav_point[2] + 1.6])  # Eye level
            
            # Look horizontal
            agent_state.rotation = habitat_sim.utils.common.quat_from_angle_axis(0, np.array([0, 1, 0]))
            agent.set_state(agent_state)
        
        # Get observations
        observations = sim.get_sensor_observations()
        
        rgb = observations["rgb"][:, :, :3]
        depth = observations["depth"]
        semantic = observations["semantic"]
        
        # Check semantic values
        unique_semantic = np.unique(semantic)
        print(f"✓ Rendered images")
        print(f"  RGB shape: {rgb.shape}")
        print(f"  Depth range: [{depth.min():.2f}, {depth.max():.2f}]")
        print(f"  Semantic unique values: {len(unique_semantic)} (first 10: {unique_semantic[:10]})")
        print(f"  Non-zero semantic pixels: {(semantic > 0).sum()} / {semantic.size}")
        
        # Save test images
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # RGB
        axes[0].imshow(rgb)
        axes[0].set_title("RGB")
        axes[0].axis('off')
        
        # Depth
        axes[1].imshow(depth, cmap='viridis')
        axes[1].set_title(f"Depth (range: {depth.min():.1f}-{depth.max():.1f}m)")
        axes[1].axis('off')
        
        # Semantic (colored)
        # Create random colors for visualization
        np.random.seed(42)
        colors = np.random.randint(0, 255, (1000, 3))
        colors[0] = [0, 0, 0]  # Background black
        semantic_colored = colors[semantic % 1000]
        
        axes[2].imshow(semantic_colored)
        axes[2].set_title(f"Semantic ({len(unique_semantic)} unique IDs)")
        axes[2].axis('off')
        
        plt.tight_layout()
        output_path = f"test_replica_loading_{scene_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved test image to {output_path}")
        
        # Test 4: Navigation mesh quality
        print("\n--- Navigation Quality Test ---")
        if sim.pathfinder.is_loaded:
            # Test multiple random paths
            success_count = 0
            total_tests = 10
            
            for i in range(total_tests):
                start = sim.pathfinder.get_random_navigable_point()
                end = sim.pathfinder.get_random_navigable_point()
                
                path = habitat_sim.ShortestPath()
                path.requested_start = start
                path.requested_end = end
                
                if sim.pathfinder.find_path(path):
                    success_count += 1
            
            print(f"✓ Pathfinding success rate: {success_count}/{total_tests}")
        
        sim.close()
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Test Replica scene loading."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test proper Replica scene loading')
    parser.add_argument('--config', type=str, 
                       default='datasets/replica_dataset/replica.scene_dataset_config.json',
                       help='Path to replica.scene_dataset_config.json')
    parser.add_argument('--scene', type=str, default='apartment_0',
                       help='Scene name (e.g., apartment_0)')
    
    args = parser.parse_args()
    
    # Check if files exist
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {args.config}")
        return
    
    # Run test
    success = test_replica_loading(args.config, args.scene)
    
    if success:
        print("\n=== All tests passed! ===")
        print("\nNext steps:")
        print("1. Use this loading method in trajectory generation")
        print("2. Ensure navigation mesh is used for pathfinding")
        print("3. Verify semantic IDs map to classes correctly")
    else:
        print("\n=== Tests failed ===")
        print("Check the error messages above for details")


if __name__ == "__main__":
    main()