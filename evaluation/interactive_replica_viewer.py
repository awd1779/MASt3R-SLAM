#!/usr/bin/env python3
"""Interactive viewer for Replica dataset with texture/semantic toggle."""

import os
import numpy as np
import habitat_sim
from habitat_sim.utils import common as utils
import cv2
import argparse
from pathlib import Path


class InteractiveReplicaViewer:
    """Interactive viewer for Replica scenes with rendering mode toggle."""
    
    def __init__(self, scene_path: str, resolution: tuple = (640, 480)):
        """Initialize the viewer.
        
        Args:
            scene_path: Path to Replica scene directory
            resolution: Display resolution (width, height)
        """
        self.scene_path = Path(scene_path)
        self.resolution = resolution
        self.width, self.height = resolution
        
        # Find mesh files
        self.textured_mesh = self._find_mesh_file("mesh.ply")
        self.semantic_mesh = self._find_mesh_file("mesh_semantic.ply")
        
        if not self.textured_mesh and not self.semantic_mesh:
            raise ValueError(f"No mesh files found in {scene_path}")
        
        # Rendering modes
        self.modes = []
        if self.textured_mesh:
            self.modes.append("texture")
        if self.semantic_mesh:
            self.modes.append("semantic")
        
        self.current_mode_idx = 0
        self.current_mode = self.modes[0] if self.modes else None
        
        # Initialize simulator
        self.sim = None
        self.agent = None
        self._init_simulator()
        
        # Camera parameters
        self.camera_speed = 0.1
        self.rotation_speed = 2.0
        self.screenshot_count = 0
        
        # Display info
        self.show_info = True
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        print(f"Viewer initialized with modes: {self.modes}")
        print("Controls:")
        print("  WASD: Move camera")
        print("  QE: Move up/down")
        print("  Mouse: Look around")
        print("  T: Toggle rendering mode")
        print("  I: Toggle info display")
        print("  P: Take screenshot")
        print("  R: Reset camera")
        print("  ESC: Exit")
    
    def _find_mesh_file(self, filename: str) -> Path:
        """Find mesh file in scene directory."""
        # Check multiple possible locations
        possible_paths = [
            self.scene_path / filename,
            self.scene_path / "habitat" / filename,
            self.scene_path / "habitat" / "mesh_semantic" / filename,
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        return None
    
    def _make_cfg(self) -> habitat_sim.Configuration:
        """Create simulator configuration."""
        sim_cfg = habitat_sim.SimulatorConfiguration()
        
        # Set scene based on current mode
        if self.current_mode == "texture" and self.textured_mesh:
            sim_cfg.scene_id = str(self.textured_mesh)
            print(f"Loading textured mesh: {self.textured_mesh}")
        elif self.current_mode == "semantic" and self.semantic_mesh:
            sim_cfg.scene_id = str(self.semantic_mesh)
            print(f"Loading semantic mesh: {self.semantic_mesh}")
        else:
            # Fallback to first available mesh
            sim_cfg.scene_id = str(self.textured_mesh or self.semantic_mesh)
        
        # Try to load semantic scene descriptor for semantic mode
        if self.current_mode == "semantic":
            semantic_desc = self._find_semantic_descriptor()
            if semantic_desc:
                sim_cfg.scene_dataset_config_file = str(semantic_desc)
                print(f"Using semantic descriptor: {semantic_desc}")
        
        # Note: Removed enable_frustum_culling as it's not available in all versions
        sim_cfg.enable_physics = False
        
        # Sensor specifications
        sensor_specs = []
        
        # RGB sensor
        rgb_sensor_spec = habitat_sim.CameraSensorSpec()
        rgb_sensor_spec.uuid = "color_sensor"
        rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor_spec.resolution = [self.height, self.width]
        rgb_sensor_spec.position = [0.0, 0.0, 0.0]
        rgb_sensor_spec.orientation = [0.0, 0.0, 0.0]
        sensor_specs.append(rgb_sensor_spec)
        
        # Depth sensor
        depth_sensor_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_spec.uuid = "depth_sensor"
        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_spec.resolution = [self.height, self.width]
        depth_sensor_spec.position = [0.0, 0.0, 0.0]
        depth_sensor_spec.orientation = [0.0, 0.0, 0.0]
        sensor_specs.append(depth_sensor_spec)
        
        # Semantic sensor (only in semantic mode)
        if self.current_mode == "semantic":
            semantic_sensor_spec = habitat_sim.CameraSensorSpec()
            semantic_sensor_spec.uuid = "semantic_sensor"
            semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
            semantic_sensor_spec.resolution = [self.height, self.width]
            semantic_sensor_spec.position = [0.0, 0.0, 0.0]
            semantic_sensor_spec.orientation = [0.0, 0.0, 0.0]
            sensor_specs.append(semantic_sensor_spec)
        
        # Agent configuration
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
            ),
        }
        
        return habitat_sim.Configuration(sim_cfg, [agent_cfg])
    
    def _find_semantic_descriptor(self) -> Path:
        """Find semantic descriptor file."""
        possible_files = [
            "info_semantic.json",
            "info_semantic.txt",
            "semantic.scene_dataset_config.json"
        ]
        
        for filename in possible_files:
            # Check multiple locations
            for base in [self.scene_path, self.scene_path / "habitat"]:
                path = base / filename
                if path.exists():
                    return path
        
        return None
    
    def _init_simulator(self):
        """Initialize or reinitialize the simulator."""
        # Close existing simulator
        if self.sim:
            self.sim.close()
        
        # Create new simulator with current configuration
        cfg = self._make_cfg()
        self.sim = habitat_sim.Simulator(cfg)
        self.agent = self.sim.initialize_agent(0)
        
        # Set initial camera position
        self._reset_camera()
        
        # Load semantic information if available
        if self.current_mode == "semantic":
            self._load_semantic_info()
    
    def _load_semantic_info(self):
        """Load semantic class information."""
        self.semantic_classes = {}
        
        # Try to load from info_semantic.json
        info_path = self._find_semantic_descriptor()
        if info_path and info_path.suffix == ".json":
            import json
            try:
                with open(info_path, 'r') as f:
                    info = json.load(f)
                
                if "classes" in info:
                    for class_info in info["classes"]:
                        class_id = class_info.get("id", -1)
                        class_name = class_info.get("name", "unknown")
                        self.semantic_classes[class_id] = class_name
                
                print(f"Loaded {len(self.semantic_classes)} semantic classes")
            except Exception as e:
                print(f"Error loading semantic info: {e}")
    
    def _reset_camera(self):
        """Reset camera to initial position."""
        # Try to get navigation bounds
        if hasattr(self.sim, "pathfinder") and self.sim.pathfinder.is_loaded:
            bounds = self.sim.pathfinder.get_bounds()
            center = (bounds[0] + bounds[1]) / 2.0
            # Place camera at human eye level
            initial_position = center + np.array([0, 1.6, 0])
        else:
            # Default position
            initial_position = np.array([0.0, 1.6, 0.0])
        
        initial_rotation = np.quaternion(1, 0, 0, 0)  # Identity quaternion
        
        self.agent.set_state(habitat_sim.AgentState(initial_position, initial_rotation))
        print(f"Camera reset to position: {initial_position}")
    
    def toggle_mode(self):
        """Toggle between rendering modes."""
        if len(self.modes) <= 1:
            print("Only one rendering mode available")
            return
        
        self.current_mode_idx = (self.current_mode_idx + 1) % len(self.modes)
        self.current_mode = self.modes[self.current_mode_idx]
        
        print(f"Switching to {self.current_mode} mode...")
        
        # Reinitialize simulator with new mode
        current_state = self.agent.get_state()
        self._init_simulator()
        self.agent.set_state(current_state)
    
    def take_screenshot(self):
        """Save current view as screenshot."""
        observations = self.sim.get_sensor_observations()
        rgb = observations["color_sensor"]
        
        # Convert from RGB to BGR for OpenCV
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        filename = f"screenshot_{self.current_mode}_{self.screenshot_count:04d}.png"
        cv2.imwrite(filename, bgr)
        self.screenshot_count += 1
        
        print(f"Screenshot saved: {filename}")
    
    def render_frame(self) -> np.ndarray:
        """Render current frame with optional info overlay."""
        observations = self.sim.get_sensor_observations()
        
        # Get RGB image
        rgb = observations["color_sensor"]
        
        # Convert to BGR for OpenCV
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        if self.show_info:
            # Add mode info
            mode_text = f"Mode: {self.current_mode.upper()}"
            cv2.putText(frame, mode_text, (10, 30), self.font, 0.7, (0, 255, 0), 2)
            
            # Add position info
            state = self.agent.get_state()
            pos = state.position
            pos_text = f"Pos: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]"
            cv2.putText(frame, pos_text, (10, 60), self.font, 0.5, (0, 255, 0), 1)
            
            # In semantic mode, show center pixel class
            if self.current_mode == "semantic" and "semantic_sensor" in observations:
                semantic = observations["semantic_sensor"]
                center_y, center_x = self.height // 2, self.width // 2
                semantic_id = semantic[center_y, center_x]
                
                if semantic_id in self.semantic_classes:
                    class_name = self.semantic_classes[semantic_id]
                else:
                    class_name = f"ID: {semantic_id}"
                
                cv2.putText(frame, f"Looking at: {class_name}", (10, 90), 
                           self.font, 0.5, (0, 255, 0), 1)
                
                # Draw crosshair
                cv2.drawMarker(frame, (center_x, center_y), (0, 255, 0), 
                              cv2.MARKER_CROSS, 20, 2)
        
        return frame
    
    def run(self):
        """Run the interactive viewer."""
        cv2.namedWindow("Replica Viewer", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Replica Viewer", self.width, self.height)
        
        # Mouse handling
        mouse_x, mouse_y = 0, 0
        mouse_pressed = False
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal mouse_x, mouse_y, mouse_pressed
            
            if event == cv2.EVENT_LBUTTONDOWN:
                mouse_pressed = True
                mouse_x, mouse_y = x, y
            elif event == cv2.EVENT_LBUTTONUP:
                mouse_pressed = False
            elif event == cv2.EVENT_MOUSEMOVE and mouse_pressed:
                dx = x - mouse_x
                dy = y - mouse_y
                
                # Rotate camera based on mouse movement
                rotation = self.agent.get_state().rotation
                
                # Horizontal rotation (around Y axis)
                yaw_delta = -dx * self.rotation_speed * 0.01
                yaw_rotation = np.quaternion(np.cos(yaw_delta/2), 0, np.sin(yaw_delta/2), 0)
                
                # Vertical rotation (around X axis)
                pitch_delta = -dy * self.rotation_speed * 0.01
                pitch_rotation = np.quaternion(np.cos(pitch_delta/2), np.sin(pitch_delta/2), 0, 0)
                
                # Apply rotations
                new_rotation = rotation * yaw_rotation * pitch_rotation
                
                state = self.agent.get_state()
                state.rotation = new_rotation
                self.agent.set_state(state)
                
                mouse_x, mouse_y = x, y
        
        cv2.setMouseCallback("Replica Viewer", mouse_callback)
        
        while True:
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord('w'):
                self.agent.act("move_forward")
            elif key == ord('s'):
                # Move backward
                state = self.agent.get_state()
                forward = state.rotation.inverse() * np.array([0, 0, -1])
                state.position -= forward * self.camera_speed
                self.agent.set_state(state)
            elif key == ord('a'):
                # Strafe left
                state = self.agent.get_state()
                right = state.rotation.inverse() * np.array([1, 0, 0])
                state.position -= right * self.camera_speed
                self.agent.set_state(state)
            elif key == ord('d'):
                # Strafe right
                state = self.agent.get_state()
                right = state.rotation.inverse() * np.array([1, 0, 0])
                state.position += right * self.camera_speed
                self.agent.set_state(state)
            elif key == ord('q'):
                # Move up
                state = self.agent.get_state()
                state.position[1] += self.camera_speed
                self.agent.set_state(state)
            elif key == ord('e'):
                # Move down
                state = self.agent.get_state()
                state.position[1] -= self.camera_speed
                self.agent.set_state(state)
            elif key == ord('t'):
                self.toggle_mode()
            elif key == ord('i'):
                self.show_info = not self.show_info
            elif key == ord('p'):
                self.take_screenshot()
            elif key == ord('r'):
                self._reset_camera()
            
            # Render and display frame
            frame = self.render_frame()
            cv2.imshow("Replica Viewer", frame)
        
        cv2.destroyAllWindows()
        self.sim.close()


def main():
    parser = argparse.ArgumentParser(description="Interactive Replica dataset viewer")
    parser.add_argument("scene_path", type=str, help="Path to Replica scene directory")
    parser.add_argument("--resolution", type=int, nargs=2, default=[640, 480],
                       help="Display resolution (width height)")
    
    args = parser.parse_args()
    
    try:
        viewer = InteractiveReplicaViewer(
            args.scene_path,
            tuple(args.resolution)
        )
        viewer.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()