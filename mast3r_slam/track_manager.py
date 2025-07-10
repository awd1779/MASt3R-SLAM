"""Global Track Manager for maintaining temporal consistency of instance IDs."""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Set
import networkx as nx
from collections import defaultdict


class UnionFind:
    """Efficient union-find data structure for track merging."""
    
    def __init__(self):
        self.parent = {}
        self.rank = defaultdict(int)
        
    def find(self, x: int) -> int:
        """Find with path compression."""
        if x not in self.parent:
            self.parent[x] = x
            return x
            
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
        
    def union(self, x: int, y: int) -> None:
        """Union by rank."""
        px, py = self.find(x), self.find(y)
        if px == py:
            return
            
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
            
    def get_canonical_id(self, x: int) -> int:
        """Get the canonical (root) ID for a track."""
        return self.find(x)


class GlobalTrackManager:
    """Maintains consistent instance IDs across frames and handles track merging."""
    
    def __init__(self, iou_threshold: float = 0.5, feature_threshold: float = 0.7):
        """
        Args:
            iou_threshold: Minimum IoU for matching tracks
            feature_threshold: Minimum feature similarity for matching (if features available)
        """
        self.iou_threshold = iou_threshold
        self.feature_threshold = feature_threshold
        
        # Track graph stores relationships between tracks
        self.track_graph = nx.Graph()
        
        # Union-find for efficient canonical ID queries
        self.union_find = UnionFind()
        
        # Track metadata
        self.track_info = {}  # track_id -> {label, frames, last_seen, confidence}
        
        # Frame-to-tracks mapping
        self.frame_tracks = defaultdict(set)  # frame_id -> set of track_ids
        
        # Global canonical ID counter
        self.next_canonical_id = 1
    
    def get_all_tracks(self) -> Dict[int, Dict]:
        """Get all track information."""
        return self.track_info
        
    def add_frame_tracks(self,
                        frame_id: int,
                        local_tracks: Dict[int, Dict],
                        masks: Optional[Dict[int, torch.Tensor]] = None) -> Dict[int, int]:
        """
        Add tracks from a new frame and return local-to-global mapping.
        
        Args:
            frame_id: Frame identifier
            local_tracks: Dict mapping local instance_id to track info
                         {instance_id: {'track_id': int, 'label': str, 'confidence': float}}
            masks: Optional dict of instance masks for better matching
            
        Returns:
            Dict mapping local instance_id to global canonical track_id
        """
        local_to_global = {}
        
        # Get existing tracks in recent frames for matching
        recent_frames = self._get_recent_frames(frame_id, window=10)
        candidate_tracks = self._get_tracks_in_frames(recent_frames)
        
        # Match each local track to existing global tracks
        for local_id, track_data in local_tracks.items():
            local_track_id = track_data['track_id']
            label = track_data['label']
            confidence = track_data.get('confidence', 1.0)
            
            # Try to match with existing tracks
            best_match = None
            best_score = 0.0
            
            if masks and local_id in masks:
                local_mask = masks[local_id]
                
                for global_track_id in candidate_tracks:
                    if self.track_info[global_track_id]['label'] != label:
                        continue
                        
                    # Compute matching score
                    score = self._compute_track_similarity(
                        local_mask, global_track_id, recent_frames
                    )
                    
                    if score > best_score and score > self.iou_threshold:
                        best_score = score
                        best_match = global_track_id
            
            if best_match is not None:
                # Merge with existing track
                canonical_id = self.union_find.get_canonical_id(best_match)
                self._update_track_info(canonical_id, frame_id, confidence)
            else:
                # Create new global track
                canonical_id = self._create_new_track(local_track_id, label, frame_id, confidence)
                
            local_to_global[local_id] = canonical_id
            self.frame_tracks[frame_id].add(canonical_id)
            
        return local_to_global
    
    def merge_tracks_on_loop_closure(self,
                                   frame1_id: int,
                                   frame2_id: int,
                                   semantic_labels1: torch.Tensor,
                                   semantic_labels2: torch.Tensor) -> None:
        """
        Merge tracks when a loop closure is detected.
        
        Args:
            frame1_id, frame2_id: Frame IDs involved in loop closure
            semantic_labels1, semantic_labels2: Semantic labels for 3D points
        """
        tracks1 = self.frame_tracks[frame1_id]
        tracks2 = self.frame_tracks[frame2_id]
        
        # Count label occurrences
        label_counts1 = self._count_labels(semantic_labels1)
        label_counts2 = self._count_labels(semantic_labels2)
        
        # Find tracks to merge based on significant overlap
        for track1 in tracks1:
            if track1 not in label_counts1:
                continue
                
            info1 = self.track_info[track1]
            
            for track2 in tracks2:
                if track2 not in label_counts2:
                    continue
                    
                info2 = self.track_info[track2]
                
                # Only merge tracks with same semantic label
                if info1['label'] != info2['label']:
                    continue
                    
                # Check if there's significant overlap
                overlap = min(label_counts1[track1], label_counts2[track2])
                total = label_counts1[track1] + label_counts2[track2]
                
                if overlap / total > 0.3:  # 30% overlap threshold
                    # Merge tracks
                    self.union_find.union(track1, track2)
                    self.track_graph.add_edge(track1, track2, weight=overlap/total)
                    
    def get_canonical_mapping(self, track_ids: List[int]) -> Dict[int, int]:
        """Get canonical IDs for a list of tracks."""
        return {tid: self.union_find.get_canonical_id(tid) for tid in track_ids}
    
    def get_track_history(self, canonical_id: int) -> Dict:
        """Get full history of a track including all merged tracks."""
        # Find all tracks that map to this canonical ID
        related_tracks = []
        for track_id in self.track_info:
            if self.union_find.get_canonical_id(track_id) == canonical_id:
                related_tracks.append(track_id)
                
        # Aggregate information
        all_frames = set()
        total_confidence = 0.0
        count = 0
        label = None
        
        for track_id in related_tracks:
            info = self.track_info[track_id]
            all_frames.update(info['frames'])
            total_confidence += info['confidence'] * len(info['frames'])
            count += len(info['frames'])
            if label is None:
                label = info['label']
                
        return {
            'canonical_id': canonical_id,
            'related_tracks': related_tracks,
            'frames': sorted(all_frames),
            'label': label,
            'avg_confidence': total_confidence / count if count > 0 else 0.0,
            'num_observations': count
        }
    
    def _get_recent_frames(self, current_frame: int, window: int = 10) -> List[int]:
        """Get list of recent frame IDs."""
        return [f for f in range(max(0, current_frame - window), current_frame)]
    
    def _get_tracks_in_frames(self, frame_ids: List[int]) -> Set[int]:
        """Get all tracks that appear in given frames."""
        tracks = set()
        for fid in frame_ids:
            tracks.update(self.frame_tracks[fid])
        return tracks
    
    def _compute_track_similarity(self,
                                 mask: torch.Tensor,
                                 track_id: int,
                                 recent_frames: List[int]) -> float:
        """
        Compute similarity between a mask and an existing track.
        
        For now, returns a simple heuristic. In practice, this could:
        - Use IoU with projected masks
        - Use feature similarity
        - Use motion prediction
        """
        # Simple heuristic: tracks are likely to continue if seen recently
        track_info = self.track_info[track_id]
        last_seen = track_info['last_seen']
        
        if last_seen in recent_frames[-3:]:  # Seen in last 3 frames
            return 0.8
        elif last_seen in recent_frames:
            return 0.6
        else:
            return 0.3
            
    def _create_new_track(self, 
                         track_id: int,
                         label: str,
                         frame_id: int,
                         confidence: float) -> int:
        """Create a new global track."""
        canonical_id = self.next_canonical_id
        self.next_canonical_id += 1
        
        self.track_info[canonical_id] = {
            'label': label,
            'frames': [frame_id],
            'last_seen': frame_id,
            'confidence': confidence,
            'created': frame_id
        }
        
        self.track_graph.add_node(canonical_id)
        self.union_find.parent[canonical_id] = canonical_id
        
        return canonical_id
        
    def _update_track_info(self,
                          track_id: int,
                          frame_id: int,
                          confidence: float) -> None:
        """Update track information with new observation."""
        info = self.track_info[track_id]
        info['frames'].append(frame_id)
        info['last_seen'] = frame_id
        # Update confidence with running average
        n = len(info['frames'])
        info['confidence'] = ((n-1) * info['confidence'] + confidence) / n
        
    def _count_labels(self, semantic_labels: torch.Tensor) -> Dict[int, int]:
        """Count occurrences of each label."""
        unique, counts = torch.unique(semantic_labels[semantic_labels > 0], return_counts=True)
        return {int(u): int(c) for u, c in zip(unique, counts)}
        
    def visualize_track_graph(self, save_path: Optional[str] = None):
        """Visualize the track relationship graph."""
        import matplotlib.pyplot as plt
        
        # Create subgraph of connected components
        components = list(nx.connected_components(self.track_graph))
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.track_graph)
        
        # Color nodes by canonical ID
        node_colors = []
        for node in self.track_graph.nodes():
            canonical = self.union_find.get_canonical_id(node)
            node_colors.append(canonical)
            
        nx.draw(self.track_graph, pos, 
                node_color=node_colors, 
                with_labels=True,
                cmap='tab20',
                node_size=500)
                
        plt.title(f"Track Relationship Graph ({len(components)} components)")
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
            
        plt.close()