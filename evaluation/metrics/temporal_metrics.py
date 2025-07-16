"""Temporal consistency metrics for evaluating object tracking across frames."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import networkx as nx
from scipy.optimize import linear_sum_assignment

from .base_metrics import BaseMetrics


class TemporalConsistencyMetrics(BaseMetrics):
    """Evaluate temporal consistency of semantic and instance segmentation."""
    
    def __init__(self,
                 iou_threshold: float = 0.5,
                 distance_threshold: float = 0.05,
                 min_track_length: int = 3,
                 output_dir: Optional[str] = None):
        """Initialize temporal consistency metrics.
        
        Args:
            iou_threshold: IoU threshold for matching instances across frames
            distance_threshold: Distance threshold for 3D point matching
            min_track_length: Minimum track length to consider
            output_dir: Directory to save outputs
        """
        super().__init__("TemporalConsistency", output_dir)
        
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        self.min_track_length = min_track_length
        
        self.reset()
    
    def reset(self):
        """Reset internal state."""
        self.results = {}
        self.frame_data = []
        self.predicted_tracks = defaultdict(list)  # track_id -> list of (frame_idx, instance_data)
        self.gt_tracks = defaultdict(list)
        self.track_graph = nx.Graph()
    
    def compute_instance_overlap(self,
                                instance1: Dict,
                                instance2: Dict,
                                points1: np.ndarray,
                                points2: np.ndarray) -> float:
        """Compute overlap between instances in consecutive frames.
        
        Args:
            instance1: Instance data from frame 1
            instance2: Instance data from frame 2
            points1: Point cloud from frame 1
            points2: Point cloud from frame 2
            
        Returns:
            Overlap score (IoU-like metric)
        """
        # Extract instance points
        mask1 = instance1['mask']
        mask2 = instance2['mask']
        
        if 'points_3d' in instance1 and 'points_3d' in instance2:
            # Use 3D points if available
            pts1 = instance1['points_3d']
            pts2 = instance2['points_3d']
        else:
            # Use 2D projections
            pts1 = points1[mask1]
            pts2 = points2[mask2]
        
        if len(pts1) == 0 or len(pts2) == 0:
            return 0.0
        
        # Compute overlap using nearest neighbors
        from sklearn.neighbors import KDTree
        tree = KDTree(pts2)
        distances, _ = tree.query(pts1, k=1)
        
        overlap_count = np.sum(distances.flatten() < self.distance_threshold)
        union_size = len(pts1) + len(pts2) - overlap_count
        
        return overlap_count / union_size if union_size > 0 else 0.0
    
    def add_frame(self,
                 frame_idx: int,
                 pred_instances: List[Dict],
                 gt_instances: List[Dict],
                 pred_points: Optional[np.ndarray] = None,
                 gt_points: Optional[np.ndarray] = None):
        """Add frame data for temporal analysis.
        
        Args:
            frame_idx: Frame index
            pred_instances: List of predicted instances with 'track_id', 'mask', 'label'
            gt_instances: List of ground truth instances
            pred_points: Predicted point cloud
            gt_points: Ground truth point cloud
        """
        frame_data = {
            'frame_idx': frame_idx,
            'pred_instances': pred_instances,
            'gt_instances': gt_instances,
            'pred_points': pred_points,
            'gt_points': gt_points
        }
        
        self.frame_data.append(frame_data)
        
        # Update tracks
        for inst in pred_instances:
            if 'track_id' in inst:
                self.predicted_tracks[inst['track_id']].append((frame_idx, inst))
        
        for inst in gt_instances:
            if 'track_id' in inst:
                self.gt_tracks[inst['track_id']].append((frame_idx, inst))
    
    def compute_track_consistency(self, track: List[Tuple[int, Dict]]) -> Dict[str, float]:
        """Compute consistency metrics for a single track.
        
        Args:
            track: List of (frame_idx, instance_data) tuples
            
        Returns:
            Dictionary of track consistency metrics
        """
        if len(track) < 2:
            return {'consistency_score': 1.0, 'label_switches': 0}
        
        # Sort by frame index
        track = sorted(track, key=lambda x: x[0])
        
        # Compute label consistency
        labels = [inst['label'] for _, inst in track if 'label' in inst]
        label_switches = sum(1 for i in range(1, len(labels)) if labels[i] != labels[i-1])
        
        # Compute spatial consistency (overlap between consecutive frames)
        overlaps = []
        for i in range(1, len(track)):
            frame_idx1, inst1 = track[i-1]
            frame_idx2, inst2 = track[i]
            
            # Only compute overlap for consecutive frames
            if frame_idx2 - frame_idx1 == 1:
                # Find frame data
                frame1_data = next((f for f in self.frame_data if f['frame_idx'] == frame_idx1), None)
                frame2_data = next((f for f in self.frame_data if f['frame_idx'] == frame_idx2), None)
                
                if frame1_data and frame2_data:
                    overlap = self.compute_instance_overlap(
                        inst1, inst2,
                        frame1_data.get('pred_points', np.array([])),
                        frame2_data.get('pred_points', np.array([]))
                    )
                    overlaps.append(overlap)
        
        # Compute consistency score
        consistency_score = np.mean(overlaps) if overlaps else 0.0
        
        # Compute track fragmentation
        frame_indices = [frame_idx for frame_idx, _ in track]
        gaps = sum(1 for i in range(1, len(frame_indices)) 
                  if frame_indices[i] - frame_indices[i-1] > 1)
        
        return {
            'consistency_score': consistency_score,
            'label_switches': label_switches,
            'track_length': len(track),
            'track_gaps': gaps,
            'mean_overlap': consistency_score
        }
    
    def compute_tracking_metrics(self) -> Dict[str, float]:
        """Compute CLEAR MOT metrics for multi-object tracking."""
        # Initialize counters
        total_gt = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_idsw = 0  # ID switches
        
        # Previous frame matches for ID switch detection
        prev_matches = {}
        
        # Process each frame
        for frame_data in sorted(self.frame_data, key=lambda x: x['frame_idx']):
            pred_instances = frame_data['pred_instances']
            gt_instances = frame_data['gt_instances']
            
            # Get current frame track IDs
            pred_tracks = {inst['track_id']: inst for inst in pred_instances if 'track_id' in inst}
            gt_tracks = {inst['track_id']: inst for inst in gt_instances if 'track_id' in inst}
            
            # Match instances based on overlap
            matches = []
            if pred_tracks and gt_tracks:
                cost_matrix = np.zeros((len(pred_tracks), len(gt_tracks)))
                pred_ids = list(pred_tracks.keys())
                gt_ids = list(gt_tracks.keys())
                
                for i, pred_id in enumerate(pred_ids):
                    for j, gt_id in enumerate(gt_ids):
                        overlap = self.compute_instance_overlap(
                            pred_tracks[pred_id], gt_tracks[gt_id],
                            frame_data.get('pred_points', np.array([])),
                            frame_data.get('gt_points', np.array([]))
                        )
                        cost_matrix[i, j] = -overlap  # Negative for minimization
                
                # Hungarian matching
                pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
                
                for pred_idx, gt_idx in zip(pred_indices, gt_indices):
                    if -cost_matrix[pred_idx, gt_idx] >= self.iou_threshold:
                        pred_id = pred_ids[pred_idx]
                        gt_id = gt_ids[gt_idx]
                        matches.append((pred_id, gt_id))
                        
                        # Check for ID switches
                        if gt_id in prev_matches and prev_matches[gt_id] != pred_id:
                            total_idsw += 1
            
            # Update counters
            total_gt += len(gt_tracks)
            total_tp += len(matches)
            total_fp += len(pred_tracks) - len(matches)
            total_fn += len(gt_tracks) - len(matches)
            
            # Update previous matches
            prev_matches = {gt_id: pred_id for pred_id, gt_id in matches}
        
        # Compute CLEAR MOT metrics
        mota = 1 - (total_fp + total_fn + total_idsw) / total_gt if total_gt > 0 else 0
        motp = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        
        # Track fragmentation metric
        track_fragmentations = 0
        for track_id, track in self.predicted_tracks.items():
            track_metrics = self.compute_track_consistency(track)
            if track_metrics['track_gaps'] > 0:
                track_fragmentations += track_metrics['track_gaps']
        
        return {
            'MOTA': mota,  # Multi-Object Tracking Accuracy
            'MOTP': motp,  # Multi-Object Tracking Precision
            'ID_switches': total_idsw,
            'track_fragmentations': track_fragmentations,
            'true_positives': total_tp,
            'false_positives': total_fp,
            'false_negatives': total_fn
        }
    
    def compute(self,
               prediction: Dict,
               ground_truth: Dict,
               **kwargs) -> Dict[str, float]:
        """Compute temporal consistency metrics.
        
        Note: This method assumes add_frame() has been called for all frames.
        
        Args:
            prediction: Not used (data added via add_frame)
            ground_truth: Not used (data added via add_frame)
            
        Returns:
            Dictionary of temporal metrics
        """
        if not self.frame_data:
            raise ValueError("No frame data added. Call add_frame() first.")
        
        metrics = {}
        
        # Compute tracking metrics
        tracking_metrics = self.compute_tracking_metrics()
        metrics.update(tracking_metrics)
        
        # Compute per-track consistency
        consistency_scores = []
        label_switch_counts = []
        track_lengths = []
        
        for track_id, track in self.predicted_tracks.items():
            if len(track) >= self.min_track_length:
                track_metrics = self.compute_track_consistency(track)
                consistency_scores.append(track_metrics['consistency_score'])
                label_switch_counts.append(track_metrics['label_switches'])
                track_lengths.append(track_metrics['track_length'])
        
        # Aggregate track metrics
        if consistency_scores:
            metrics['mean_consistency_score'] = np.mean(consistency_scores)
            metrics['std_consistency_score'] = np.std(consistency_scores)
            metrics['total_label_switches'] = sum(label_switch_counts)
            metrics['mean_track_length'] = np.mean(track_lengths)
            metrics['num_tracks'] = len(consistency_scores)
        else:
            metrics['mean_consistency_score'] = 0.0
            metrics['std_consistency_score'] = 0.0
            metrics['total_label_switches'] = 0
            metrics['mean_track_length'] = 0.0
            metrics['num_tracks'] = 0
        
        # Compute track purity (percentage of tracks with no label switches)
        pure_tracks = sum(1 for count in label_switch_counts if count == 0)
        metrics['track_purity'] = pure_tracks / len(label_switch_counts) if label_switch_counts else 0.0
        
        self.results.update(metrics)
        return metrics
    
    def visualize_tracks(self, output_path: Optional[str] = None):
        """Visualize tracks over time (creates a track timeline plot).
        
        Args:
            output_path: Path to save visualization
        """
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot predicted tracks
        for track_id, track in self.predicted_tracks.items():
            if len(track) >= self.min_track_length:
                frames = [frame_idx for frame_idx, _ in track]
                labels = [inst.get('label', 0) for _, inst in track]
                ax1.scatter(frames, [track_id] * len(frames), c=labels, cmap='tab20', s=50)
        
        ax1.set_xlabel('Frame Index')
        ax1.set_ylabel('Track ID')
        ax1.set_title('Predicted Tracks')
        ax1.grid(True, alpha=0.3)
        
        # Plot ground truth tracks
        for track_id, track in self.gt_tracks.items():
            frames = [frame_idx for frame_idx, _ in track]
            labels = [inst.get('label', 0) for _, inst in track]
            ax2.scatter(frames, [track_id] * len(frames), c=labels, cmap='tab20', s=50)
        
        ax2.set_xlabel('Frame Index')
        ax2.set_ylabel('Track ID')
        ax2.set_title('Ground Truth Tracks')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150)
        else:
            plt.show()
        
        plt.close()