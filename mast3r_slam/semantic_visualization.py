"""Semantic Visualization Module for MAST3R-SLAM"""

import torch
import numpy as np
import imgui
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import matplotlib.cm as cm
from mast3r_slam.visualization import Window, WindowMsg
from mast3r_slam.semantic_integration import SemanticSLAMBackend
from in3d.color import hex2rgba


@dataclass
class SemanticWindowMsg(WindowMsg):
    """Extended window message with semantic controls."""
    color_mode: str = "rgb"  # "rgb", "semantic", "confidence"
    show_labels: bool = False
    selected_instance: int = -1
    filter_instances: List[int] = None
    semantic_alpha: float = 0.5
    confidence_threshold: float = 0.5
    
    def __post_init__(self):
        if self.filter_instances is None:
            self.filter_instances = []


class SemanticVisualizationMixin:
    """Mixin class to add semantic visualization capabilities to the Window class."""
    
    def __init__(self, semantic_backend: Optional[SemanticSLAMBackend] = None):
        self.semantic_backend = semantic_backend
        self.semantic_state = SemanticWindowMsg()
        
        # Color mapping
        self.instance_colors = {}
        self.colormap = cm.get_cmap('tab20')
        self.confidence_colormap = cm.get_cmap('viridis')
        
        # Cache for performance
        self.semantic_point_colors = None
        self.last_semantic_update = -1
        
        # UI state
        self.show_semantic_panel = True
        self.instance_list = []
        self.label_to_instances = {}
        
    def update_semantic_colors(self, force_update: bool = False):
        """Update semantic color mapping for point cloud."""
        if self.semantic_backend is None:
            return
            
        # Check if update needed
        current_kf_count = len(self.keyframes)
        if not force_update and current_kf_count == self.last_semantic_update:
            return
            
        self.last_semantic_update = current_kf_count
        
        # Get all unique instances and their labels
        self.instance_list = []
        self.label_to_instances = {}
        
        for kf_idx in range(len(self.keyframes)):
            semantics = self.semantic_backend.keyframe_semantics.get(kf_idx)
            if semantics is None:
                continue
                
            labels, _ = semantics
            unique_labels = torch.unique(labels[labels > 0])
            
            for label in unique_labels:
                label_int = label.item()
                if label_int not in self.instance_list:
                    self.instance_list.append(label_int)
                    
                    # Get semantic class name
                    track_info = self.semantic_backend.track_manager.get_track_history(label_int)
                    if track_info and 'label' in track_info:
                        class_name = track_info['label']
                        if class_name not in self.label_to_instances:
                            self.label_to_instances[class_name] = []
                        self.label_to_instances[class_name].append(label_int)
        
        # Assign colors to instances
        for i, instance_id in enumerate(self.instance_list):
            if instance_id not in self.instance_colors:
                color = self.colormap(i % 20)[:3]
                self.instance_colors[instance_id] = color
    
    def get_semantic_point_colors(self, kf_idx: int, 
                                 original_colors: np.ndarray,
                                 point_indices: Optional[np.ndarray] = None) -> np.ndarray:
        """Get point colors based on semantic mode."""
        if self.semantic_backend is None or self.semantic_state.color_mode == "rgb":
            return original_colors
            
        semantics = self.semantic_backend.keyframe_semantics.get(kf_idx)
        if semantics is None:
            return original_colors
            
        labels, confidences = semantics
        
        # Handle point subset if indices provided
        if point_indices is not None:
            labels = labels[point_indices]
            confidences = confidences[point_indices]
            
        # Initialize output colors
        n_points = len(labels)
        colors = np.zeros((n_points, 3), dtype=np.float32)
        
        if self.semantic_state.color_mode == "semantic":
            # Color by instance
            for i in range(n_points):
                label = labels[i].item()
                if label == 0:  # Background
                    colors[i] = [0.5, 0.5, 0.5]
                elif label in self.instance_colors:
                    colors[i] = self.instance_colors[label]
                else:
                    colors[i] = [1.0, 0.0, 1.0]  # Magenta for unknown
                    
            # Apply filtering
            if self.semantic_state.filter_instances:
                mask = torch.zeros(n_points, dtype=torch.bool)
                for inst_id in self.semantic_state.filter_instances:
                    mask |= (labels == inst_id)
                colors[~mask.cpu().numpy()] = [0.2, 0.2, 0.2]  # Dim filtered points
                
        elif self.semantic_state.color_mode == "confidence":
            # Color by confidence
            conf_np = confidences.cpu().numpy()
            conf_normalized = np.clip(conf_np, 0, 1)
            
            for i in range(n_points):
                if labels[i] > 0:  # Only color labeled points
                    colors[i] = self.confidence_colormap(conf_normalized[i])[:3]
                else:
                    colors[i] = [0.3, 0.3, 0.3]
                    
            # Apply confidence threshold
            mask = conf_np < self.semantic_state.confidence_threshold
            colors[mask] = [0.2, 0.2, 0.2]
            
        # Blend with original colors if requested
        if self.semantic_state.semantic_alpha < 1.0:
            alpha = self.semantic_state.semantic_alpha
            colors = alpha * colors + (1 - alpha) * original_colors
            
        return colors
    
    def render_semantic_ui(self):
        """Render semantic visualization controls."""
        imgui.spacing()
        imgui.separator()
        imgui.text("Semantic Visualization")
        imgui.separator()
        
        # Color mode selection
        imgui.text("Color Mode:")
        color_modes = ["rgb", "semantic", "confidence"]
        for i, mode in enumerate(color_modes):
            if imgui.radio_button(f"{mode}##color", self.semantic_state.color_mode == mode):
                self.semantic_state.color_mode = mode
                self.update_semantic_colors(force_update=True)
                
        imgui.spacing()
        
        # Semantic-specific controls
        if self.semantic_state.color_mode != "rgb":
            _, self.semantic_state.semantic_alpha = imgui.slider_float(
                "Blend Alpha", self.semantic_state.semantic_alpha, 0.0, 1.0
            )
            
            if self.semantic_state.color_mode == "confidence":
                _, self.semantic_state.confidence_threshold = imgui.slider_float(
                    "Conf Threshold", self.semantic_state.confidence_threshold, 0.0, 1.0
                )
                
            imgui.spacing()
            
            # Instance filtering
            if imgui.collapsing_header("Instance Filter")[0]:
                imgui.text(f"Total instances: {len(self.instance_list)}")
                
                # Group by semantic class
                for class_name, instances in self.label_to_instances.items():
                    if imgui.tree_node(f"{class_name} ({len(instances)})##tree_{class_name}"):
                        for inst_id in instances:
                            is_filtered = inst_id in self.semantic_state.filter_instances
                            clicked, _ = imgui.checkbox(f"Instance {inst_id}##check_{inst_id}", is_filtered)
                            if clicked:
                                if is_filtered:
                                    self.semantic_state.filter_instances.remove(inst_id)
                                else:
                                    self.semantic_state.filter_instances.append(inst_id)
                        imgui.tree_pop()
                
                imgui.spacing()
                if imgui.button("Clear Filter"):
                    self.semantic_state.filter_instances = []
                imgui.same_line()
                if imgui.button("Select All"):
                    self.semantic_state.filter_instances = self.instance_list.copy()
                    
            # Statistics
            if imgui.collapsing_header("Semantic Statistics")[0]:
                if self.semantic_backend:
                    stats = self.get_semantic_statistics()
                    imgui.text(f"Labeled keyframes: {stats['labeled_keyframes']}/{stats['total_keyframes']}")
                    imgui.text(f"Total instances: {stats['total_instances']}")
                    imgui.text(f"Unique classes: {stats['unique_classes']}")
                    
                    if stats['class_distribution']:
                        imgui.spacing()
                        imgui.text("Class distribution:")
                        for class_name, count in stats['class_distribution'].items():
                            imgui.text(f"  {class_name}: {count}")
                            
        _, self.semantic_state.show_labels = imgui.checkbox("Show Labels", self.semantic_state.show_labels)
        
    def get_semantic_statistics(self) -> Dict:
        """Get semantic mapping statistics."""
        if self.semantic_backend is None:
            return {}
            
        stats = {
            'total_keyframes': len(self.keyframes),
            'labeled_keyframes': 0,
            'total_instances': len(self.instance_list),
            'unique_classes': len(self.label_to_instances),
            'class_distribution': {}
        }
        
        # Count labeled keyframes
        for i in range(len(self.keyframes)):
            if i in self.semantic_backend.keyframe_semantics:
                stats['labeled_keyframes'] += 1
                
        # Count instances per class
        for class_name, instances in self.label_to_instances.items():
            stats['class_distribution'][class_name] = len(instances)
            
        return stats


class SemanticWindow(Window, SemanticVisualizationMixin):
    """Extended Window class with semantic visualization capabilities."""
    
    def __init__(self, states, keyframes, main2viz, viz2main, 
                 semantic_backend: Optional[SemanticSLAMBackend] = None, **kwargs):
        Window.__init__(self, states, keyframes, main2viz, viz2main, **kwargs)
        SemanticVisualizationMixin.__init__(self, semantic_backend)
        
        # Override the color texture preparation
        self._original_prepare_textures = self.prepare_textures
        self.prepare_textures = self._semantic_prepare_textures
        
    def _semantic_prepare_textures(self, kf_idx, w, h):
        """Prepare textures with semantic coloring."""
        ptex, ctex, itex = self._original_prepare_textures(kf_idx, w, h)
        
        # Modify color texture based on semantic mode
        if self.semantic_state.color_mode != "rgb":
            # Read original colors
            original_colors = np.frombuffer(ctex.read(), dtype=np.uint8).reshape(-1, 3) / 255.0
            
            # Get semantic colors
            semantic_colors = self.get_semantic_point_colors(kf_idx, original_colors)
            
            # Write back to texture
            ctex.write((semantic_colors * 255).astype(np.uint8).tobytes())
            
        return ptex, ctex, itex
        
    def render_ui(self):
        """Extended UI rendering with semantic controls."""
        # Call parent render_ui
        super().render_ui()
        
        # Add semantic UI elements
        if self.semantic_backend is not None:
            self.render_semantic_ui()
            
        # Update semantic colors if needed
        self.update_semantic_colors()


def create_semantic_visualization(states, keyframes, main2viz, viz2main, 
                                semantic_backend: Optional[SemanticSLAMBackend] = None):
    """Factory function to create appropriate visualization window."""
    if semantic_backend is not None:
        return SemanticWindow(states, keyframes, main2viz, viz2main, semantic_backend)
    else:
        return Window(states, keyframes, main2viz, viz2main)