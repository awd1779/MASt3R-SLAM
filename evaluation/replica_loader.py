"""Load Replica dataset ground truth semantic data."""

import os
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import open3d as o3d


class ReplicaSemanticLoader:
    def __init__(self, scene_path: str):
        """Initialize the Replica semantic data loader.
        
        Args:
            scene_path: Path to Replica scene (e.g., /path/to/apartment_0)
        """
        self.scene_path = Path(scene_path)
        self.habitat_path = self.scene_path / "habitat"
        
        # Load semantic information
        self.semantic_info = self._load_semantic_info()
        self.class_names = self._extract_class_names()
        self.instance_to_class = self._create_instance_to_class_mapping()
        
    def _load_semantic_info(self) -> Dict:
        """Load semantic information from info_semantic.json."""
        info_path = self.habitat_path / "info_semantic.json"
        if not info_path.exists():
            raise FileNotFoundError(f"Semantic info not found: {info_path}")
            
        with open(info_path, 'r') as f:
            return json.load(f)
    
    def _extract_class_names(self) -> Dict[int, str]:
        """Extract class ID to name mapping from semantic info.
        
        Returns:
            Dictionary mapping class IDs to class names
        """
        class_names = {}
        
        # Replica uses "classes" field in info_semantic.json
        if "classes" in self.semantic_info:
            for class_entry in self.semantic_info["classes"]:
                class_id = class_entry["id"]
                class_name = class_entry["name"]
                class_names[class_id] = class_name
        
        return class_names
    
    def _create_instance_to_class_mapping(self) -> Dict[int, Tuple[int, str]]:
        """Create mapping from instance IDs to class IDs and names.
        
        Returns:
            Dictionary mapping instance IDs to (class_id, class_name) tuples
        """
        instance_to_class = {}
        
        # Parse objects in the semantic info
        if "objects" in self.semantic_info:
            for obj in self.semantic_info["objects"]:
                instance_id = obj["id"]
                class_id = obj["class_id"]
                class_name = self.class_names.get(class_id, "unknown")
                instance_to_class[instance_id] = (class_id, class_name)
        
        return instance_to_class
    
    def load_semantic_mesh(self) -> o3d.geometry.TriangleMesh:
        """Load the semantic mesh from mesh_semantic.ply.
        
        Returns:
            Open3D triangle mesh with semantic labels
        """
        mesh_path = self.habitat_path / "mesh_semantic.ply"
        if not mesh_path.exists():
            raise FileNotFoundError(f"Semantic mesh not found: {mesh_path}")
        
        # Load mesh
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        
        # The semantic mesh has vertex colors that encode semantic labels
        # We'll need to decode these later
        
        return mesh
    
    def load_semantic_mesh_with_labels(self) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray]:
        """Load semantic mesh and extract per-face semantic labels.
        
        Returns:
            Tuple of (mesh, face_labels) where face_labels contains semantic instance IDs
        """
        # First try to load the PLY file with custom properties
        import plyfile
        
        mesh_path = self.habitat_path / "mesh_semantic.ply"
        plydata = plyfile.PlyData.read(str(mesh_path))
        
        # Extract vertices
        vertices = np.vstack([
            plydata['vertex']['x'],
            plydata['vertex']['y'],
            plydata['vertex']['z']
        ]).T
        
        # Extract faces and their semantic labels
        faces = []
        face_labels = []
        
        for face in plydata['face']:
            vertex_indices = face['vertex_indices']
            faces.append(vertex_indices)
            
            # Check if object_id property exists
            if 'object_id' in face:
                face_labels.append(face['object_id'])
            else:
                # If no object_id, default to 0 (unknown)
                face_labels.append(0)
        
        faces = np.array(faces)
        face_labels = np.array(face_labels)
        
        # Create Open3D mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        
        # Also extract vertex colors if available
        if 'red' in plydata['vertex']:
            colors = np.vstack([
                plydata['vertex']['red'],
                plydata['vertex']['green'],
                plydata['vertex']['blue']
            ]).T / 255.0
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        
        return mesh, face_labels
    
    def get_vocabulary_for_zero_shot(self) -> List[str]:
        """Get the list of class names for zero-shot evaluation.
        
        Returns:
            List of unique class names in the scene
        """
        # Get unique class names from the scene
        unique_classes = set()
        
        for instance_id, (class_id, class_name) in self.instance_to_class.items():
            if class_name and class_name != "unknown":
                unique_classes.add(class_name)
        
        # Sort for consistency
        return sorted(list(unique_classes))
    
    def map_instance_to_class(self, instance_id: int) -> Tuple[int, str]:
        """Map an instance ID to its class ID and name.
        
        Args:
            instance_id: Instance ID from the semantic mesh
            
        Returns:
            Tuple of (class_id, class_name)
        """
        if instance_id in self.instance_to_class:
            return self.instance_to_class[instance_id]
        else:
            return (0, "unknown")
    
    def get_class_statistics(self) -> Dict[str, int]:
        """Get statistics about classes in the scene.
        
        Returns:
            Dictionary mapping class names to instance counts
        """
        class_counts = {}
        
        for instance_id, (class_id, class_name) in self.instance_to_class.items():
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += 1
        
        return class_counts
    
    def sample_points_from_mesh(self, mesh: o3d.geometry.TriangleMesh, 
                               face_labels: np.ndarray,
                               num_points: int = 100000) -> Tuple[np.ndarray, np.ndarray]:
        """Sample points from the mesh surface with their semantic labels.
        
        Args:
            mesh: Triangle mesh
            face_labels: Per-face semantic labels (instance IDs)
            num_points: Number of points to sample
            
        Returns:
            Tuple of (points, labels) where labels are class IDs
        """
        # Sample points from mesh surface
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)
        points = np.asarray(pcd.points)
        
        # For each sampled point, find the nearest face and get its label
        # This is a simplified approach - for better accuracy, you might want
        # to track which face each point was sampled from
        
        # Build a KDTree from face centers
        face_centers = []
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        for face in triangles:
            center = vertices[face].mean(axis=0)
            face_centers.append(center)
        
        face_centers = np.array(face_centers)
        
        # Find nearest face for each point
        from sklearn.neighbors import KDTree
        tree = KDTree(face_centers)
        _, indices = tree.query(points, k=1)
        indices = indices.flatten()
        
        # Get instance labels for points
        point_instance_labels = face_labels[indices]
        
        # Convert instance labels to class labels
        point_class_labels = []
        for instance_id in point_instance_labels:
            class_id, _ = self.map_instance_to_class(instance_id)
            point_class_labels.append(class_id)
        
        point_class_labels = np.array(point_class_labels)
        
        return points, point_class_labels


def test_loader():
    """Test the Replica loader functionality."""
    # Example usage
    scene_path = "/path/to/replica/apartment_0"
    
    try:
        loader = ReplicaSemanticLoader(scene_path)
        
        # Get vocabulary for zero-shot evaluation
        vocabulary = loader.get_vocabulary_for_zero_shot()
        print(f"Scene vocabulary ({len(vocabulary)} classes):")
        for i, class_name in enumerate(vocabulary):
            print(f"  {i}: {class_name}")
        
        # Get class statistics
        stats = loader.get_class_statistics()
        print("\nClass statistics:")
        for class_name, count in stats.items():
            print(f"  {class_name}: {count} instances")
        
        # Load semantic mesh
        mesh, face_labels = loader.load_semantic_mesh_with_labels()
        print(f"\nLoaded mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} faces")
        print(f"Unique instance IDs: {len(np.unique(face_labels))}")
        
        # Sample points
        points, labels = loader.sample_points_from_mesh(mesh, face_labels, num_points=10000)
        print(f"\nSampled {len(points)} points with {len(np.unique(labels))} unique class labels")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_loader()