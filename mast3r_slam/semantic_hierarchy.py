"""
Hierarchical 3D Scene Graph for Semantic Understanding
Builds parent-child relationships between detected objects.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
import json
from collections import defaultdict

@dataclass
class SceneNode:
    """Node in the scene graph representing an object."""
    node_id: int
    class_name: str
    confidence: float
    mask: np.ndarray
    area: int
    children: List['SceneNode'] = field(default_factory=list)
    parent: Optional['SceneNode'] = None
    global_id: Optional[int] = None  # For 3D tracking
    
    def add_child(self, child: 'SceneNode'):
        """Add a child node."""
        self.children.append(child)
        child.parent = self
    
    def get_hierarchy_string(self, indent: int = 0) -> str:
        """Get string representation of hierarchy."""
        result = "  " * indent + f"- {self.class_name} (conf: {self.confidence:.1f}%, area: {self.area})"
        for child in self.children:
            result += "\n" + child.get_hierarchy_string(indent + 1)
        return result
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "area": int(self.area),
            "global_id": self.global_id,
            "children": [child.to_dict() for child in self.children]
        }

class SemanticSceneGraph:
    """Builds and manages hierarchical scene graph from semantic segmentation."""
    
    # Define semantic hierarchy rules
    SCENE_HIERARCHY = {
        # Room-level containers
        "room": ["desk", "table", "chair", "bookshelf", "wall", "floor", "ceiling"],
        
        # Furniture that contains objects
        "desk": ["laptop", "desktop computer", "monitor", "keyboard", "mouse", "lamp", "cup", "book", "paper", "phone"],
        "computer desk": ["laptop", "desktop computer", "monitor", "keyboard", "mouse", "lamp", "cup", "book", "paper"],
        "table": ["laptop", "cup", "book", "paper", "plate", "phone"],
        "bookshelf": ["book", "book spine", "decorative object", "container"],
        
        # Computer systems and their components
        "laptop": ["screen", "keyboard", "trackpad"],
        "laptop computer": ["screen", "monitor", "keyboard", "trackpad"],
        "desktop computer": ["monitor", "keyboard", "mouse", "computer case"],
        
        # Container objects
        "container": ["cup", "mug", "bottle"],
        "storage box": ["book", "paper", "small objects"],
    }
    
    # Spatial containment thresholds
    CONTAINMENT_THRESHOLD = 0.7  # 70% of child must be inside parent
    OVERLAP_THRESHOLD = 0.3      # 30% overlap to consider relationship
    
    def __init__(self):
        """Initialize the scene graph."""
        self.nodes: List[SceneNode] = []
        self.root_nodes: List[SceneNode] = []
        self.node_by_id: Dict[int, SceneNode] = {}
        
    def normalize_class_name(self, class_name: str) -> str:
        """Normalize class name for hierarchy matching."""
        # Remove common prefixes
        for prefix in ["a picture of a ", "a photo of a ", "a picture of ", "a photo of ", "a "]:
            if class_name.startswith(prefix):
                class_name = class_name[len(prefix):]
        return class_name.lower()
    
    def can_contain(self, parent_class: str, child_class: str) -> bool:
        """Check if parent class can contain child class."""
        parent_norm = self.normalize_class_name(parent_class)
        child_norm = self.normalize_class_name(child_class)
        
        # Check explicit hierarchy
        if parent_norm in self.SCENE_HIERARCHY:
            child_categories = self.SCENE_HIERARCHY[parent_norm]
            for category in child_categories:
                if category in child_norm or child_norm in category:
                    return True
        
        # Check general patterns
        # Desk/table can contain most smaller objects
        if any(furniture in parent_norm for furniture in ["desk", "table"]):
            if any(obj in child_norm for obj in ["keyboard", "mouse", "monitor", "laptop", "cup", "book", "lamp"]):
                return True
        
        # Computer systems contain their components
        if "laptop" in parent_norm or "computer" in parent_norm:
            if any(comp in child_norm for comp in ["keyboard", "screen", "monitor", "trackpad", "mouse"]):
                return True
                
        return False
    
    def compute_containment(self, child_mask: np.ndarray, parent_mask: np.ndarray) -> float:
        """Compute how much of child is contained in parent."""
        intersection = np.logical_and(child_mask, parent_mask).sum()
        child_area = child_mask.sum()
        return intersection / child_area if child_area > 0 else 0
    
    def compute_overlap(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute overlap ratio between two masks."""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0
    
    def build_graph(self, 
                   masks: List[np.ndarray],
                   class_names: List[str], 
                   confidences: List[float],
                   debug: bool = False) -> Dict[str, any]:
        """
        Build hierarchical scene graph from segmentation results.
        
        Returns:
            Dictionary containing the graph structure and debug info.
        """
        # Clear previous graph
        self.nodes.clear()
        self.root_nodes.clear()
        self.node_by_id.clear()
        
        # Create nodes
        for i, (mask, class_name, conf) in enumerate(zip(masks, class_names, confidences)):
            node = SceneNode(
                node_id=i,
                class_name=class_name,
                confidence=conf,
                mask=mask,
                area=mask.sum()
            )
            self.nodes.append(node)
            self.node_by_id[i] = node
        
        # Sort by area (largest first) for hierarchy building
        sorted_nodes = sorted(self.nodes, key=lambda n: n.area, reverse=True)
        
        # Build hierarchy
        assigned_children = set()
        debug_info = {"relationships": [], "decisions": []}
        
        for i, parent_node in enumerate(sorted_nodes):
            if parent_node.node_id in assigned_children:
                continue
                
            for j, child_node in enumerate(sorted_nodes[i+1:], i+1):
                if child_node.node_id in assigned_children:
                    continue
                    
                # Check spatial containment
                containment = self.compute_containment(child_node.mask, parent_node.mask)
                
                if containment >= self.CONTAINMENT_THRESHOLD:
                    # Check semantic compatibility
                    if self.can_contain(parent_node.class_name, child_node.class_name):
                        parent_node.add_child(child_node)
                        assigned_children.add(child_node.node_id)
                        
                        if debug:
                            debug_info["relationships"].append({
                                "parent": parent_node.node_id,
                                "child": child_node.node_id,
                                "parent_class": parent_node.class_name,
                                "child_class": child_node.class_name,
                                "containment": float(containment),
                                "reason": "spatial_and_semantic"
                            })
                    elif debug:
                        debug_info["decisions"].append({
                            "action": "skip_containment",
                            "parent": parent_node.node_id,
                            "child": child_node.node_id,
                            "reason": f"Semantic incompatible: {parent_node.class_name} cannot contain {child_node.class_name}",
                            "containment": float(containment)
                        })
                
                elif containment > 0.3 and debug:  # Some overlap but not enough
                    overlap = self.compute_overlap(child_node.mask, parent_node.mask)
                    debug_info["decisions"].append({
                        "action": "insufficient_containment",
                        "parent": parent_node.node_id,
                        "child": child_node.node_id,
                        "containment": float(containment),
                        "overlap": float(overlap),
                        "reason": f"Containment {containment:.2f} < {self.CONTAINMENT_THRESHOLD}"
                    })
        
        # Identify root nodes (nodes without parents)
        self.root_nodes = [node for node in self.nodes if node.parent is None]
        
        # Add scene-level analysis
        if debug:
            debug_info["stats"] = {
                "total_nodes": len(self.nodes),
                "root_nodes": len(self.root_nodes),
                "nodes_with_children": sum(1 for n in self.nodes if n.children),
                "leaf_nodes": sum(1 for n in self.nodes if not n.children),
                "max_depth": self._get_max_depth()
            }
        
        return {
            "graph": self.get_graph_dict(),
            "debug_info": debug_info if debug else None
        }
    
    def _get_max_depth(self, node: Optional[SceneNode] = None, current_depth: int = 0) -> int:
        """Get maximum depth of the tree."""
        if node is None:
            if not self.root_nodes:
                return 0
            return max(self._get_max_depth(root, 0) for root in self.root_nodes)
        
        if not node.children:
            return current_depth
        
        return max(self._get_max_depth(child, current_depth + 1) for child in node.children)
    
    def get_graph_dict(self) -> Dict[str, any]:
        """Get the scene graph as a dictionary."""
        return {
            "scene": {
                "root_objects": [node.to_dict() for node in self.root_nodes],
                "total_objects": len(self.nodes),
                "hierarchy_depth": self._get_max_depth()
            }
        }
    
    def get_flattened_labels(self, include_hierarchy: bool = True) -> Tuple[List[int], Dict[int, str]]:
        """
        Get flattened labels for point cloud, optionally with hierarchy info.
        
        Returns:
            keep_indices: Indices of masks to use
            label_map: Mapping of indices to hierarchical labels
        """
        keep_indices = []
        label_map = {}
        
        if include_hierarchy:
            # Include all nodes but with hierarchical labels
            for node in self.nodes:
                keep_indices.append(node.node_id)
                
                # Build hierarchical label
                hierarchy_path = []
                current = node
                while current is not None:
                    hierarchy_path.insert(0, self.normalize_class_name(current.class_name))
                    current = current.parent
                
                # Create label like "desk/laptop/keyboard"
                if len(hierarchy_path) > 1:
                    label = "/".join(hierarchy_path)
                else:
                    label = node.class_name
                    
                label_map[node.node_id] = label
        else:
            # Only keep root and high-confidence children
            confidence_threshold = 50.0
            
            def should_keep_node(node: SceneNode) -> bool:
                # Always keep root nodes
                if node.parent is None:
                    return True
                # Keep high-confidence children
                if node.confidence >= confidence_threshold:
                    return True
                # Keep if it's an important component
                important_components = ["keyboard", "monitor", "mouse", "screen"]
                if any(comp in self.normalize_class_name(node.class_name) for comp in important_components):
                    return True
                return False
            
            for node in self.nodes:
                if should_keep_node(node):
                    keep_indices.append(node.node_id)
                    label_map[node.node_id] = node.class_name
        
        return keep_indices, label_map
    
    def print_graph(self):
        """Print the scene graph hierarchy."""
        print("Scene Graph Hierarchy:")
        print("=" * 50)
        if not self.root_nodes:
            print("(No root nodes)")
        else:
            for root in self.root_nodes:
                print(root.get_hierarchy_string())
        print("=" * 50)
    
    def get_hierarchy_string(self) -> str:
        """Get the full hierarchy as a string."""
        if not self.root_nodes:
            return "(No root nodes)"
        
        result = []
        for root in self.root_nodes:
            result.append(root.get_hierarchy_string())
        return "\n".join(result)
    
    def get_scene_description(self) -> str:
        """Generate natural language description of the scene."""
        if not self.root_nodes:
            return "Empty scene"
        
        # Analyze root objects
        root_classes = [self.normalize_class_name(node.class_name) for node in self.root_nodes]
        
        # Determine scene type
        if any("desk" in cls for cls in root_classes):
            scene_type = "office workspace"
        elif any("table" in cls for cls in root_classes):
            scene_type = "table area"
        elif any("bookshelf" in cls for cls in root_classes):
            scene_type = "storage area"
        else:
            scene_type = "room"
        
        # Count important objects
        all_objects = []
        def collect_objects(node):
            all_objects.append(self.normalize_class_name(node.class_name))
            for child in node.children:
                collect_objects(child)
        
        for root in self.root_nodes:
            collect_objects(root)
        
        # Build description
        desc = f"This appears to be a {scene_type} containing "
        
        # List main objects
        main_objects = []
        for root in self.root_nodes[:3]:  # Top 3 root objects
            obj_desc = self.normalize_class_name(root.class_name)
            if root.children:
                child_count = len(root.children)
                obj_desc += f" (with {child_count} components)"
            main_objects.append(obj_desc)
        
        desc += ", ".join(main_objects)
        
        if len(self.root_nodes) > 3:
            desc += f" and {len(self.root_nodes) - 3} other objects"
        
        return desc