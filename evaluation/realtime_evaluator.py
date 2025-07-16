"""Real-time evaluation module for online semantic SLAM metrics."""

import numpy as np
from typing import Dict, List, Optional, Any, Callable
import threading
import queue
import time
from pathlib import Path
import json
from collections import deque, defaultdict
import matplotlib.pyplot as plt
from datetime import datetime

from evaluation.metrics import (
    InstanceSegmentationMetrics,
    TemporalConsistencyMetrics,
    BoundaryAccuracyMetrics,
    EfficiencyMetrics
)
from evaluation.semantic_3d_metrics import Semantic3DMetrics


class RealtimeEvaluator:
    """Real-time evaluation system for semantic SLAM."""
    
    def __init__(self,
                 output_dir: str,
                 semantic_classes: List[str],
                 eval_interval: int = 30,
                 plot_interval: int = 60,
                 enable_plotting: bool = True):
        """Initialize real-time evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
            semantic_classes: List of semantic class names
            eval_interval: Frames between evaluations
            plot_interval: Frames between plot updates
            enable_plotting: Whether to show live plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.semantic_classes = semantic_classes
        self.num_classes = len(semantic_classes)
        self.eval_interval = eval_interval
        self.plot_interval = plot_interval
        self.enable_plotting = enable_plotting
        
        # Initialize metrics
        self._init_metrics()
        
        # Frame buffer for temporal analysis
        self.frame_buffer = deque(maxlen=eval_interval * 2)
        
        # Metric history
        self.metric_history = defaultdict(list)
        self.timestamp_history = []
        
        # Threading for background evaluation
        self.eval_queue = queue.Queue()
        self.running = False
        self.eval_thread = None
        
        # Live plotting
        if self.enable_plotting:
            self._init_plots()
    
    def _init_metrics(self):
        """Initialize all metric calculators."""
        # 3D semantic metrics
        self.semantic_metrics = Semantic3DMetrics(
            self.num_classes,
            class_names={i: name for i, name in enumerate(self.semantic_classes)}
        )
        
        # Instance segmentation metrics
        self.instance_metrics = InstanceSegmentationMetrics(
            output_dir=str(self.output_dir / "instance_metrics")
        )
        
        # Temporal consistency metrics
        self.temporal_metrics = TemporalConsistencyMetrics(
            output_dir=str(self.output_dir / "temporal_metrics")
        )
        
        # Boundary accuracy metrics
        self.boundary_metrics = BoundaryAccuracyMetrics(
            output_dir=str(self.output_dir / "boundary_metrics")
        )
        
        # Efficiency metrics
        self.efficiency_metrics = EfficiencyMetrics(
            track_gpu=True,
            output_dir=str(self.output_dir / "efficiency_metrics")
        )
        
        # Start efficiency monitoring
        self.efficiency_metrics.start_monitoring()
    
    def _init_plots(self):
        """Initialize live plotting."""
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('Real-time Semantic SLAM Evaluation')
        
        # Configure subplots
        self.axes[0, 0].set_title('Semantic mIoU')
        self.axes[0, 0].set_xlabel('Frame')
        self.axes[0, 0].set_ylabel('mIoU')
        
        self.axes[0, 1].set_title('Instance mAP')
        self.axes[0, 1].set_xlabel('Frame')
        self.axes[0, 1].set_ylabel('mAP')
        
        self.axes[1, 0].set_title('Temporal Consistency')
        self.axes[1, 0].set_xlabel('Frame')
        self.axes[1, 0].set_ylabel('MOTA')
        
        self.axes[1, 1].set_title('Processing Speed')
        self.axes[1, 1].set_xlabel('Frame')
        self.axes[1, 1].set_ylabel('FPS')
        
        plt.tight_layout()
    
    def start(self):
        """Start background evaluation thread."""
        if self.running:
            return
        
        self.running = True
        self.eval_thread = threading.Thread(target=self._evaluation_loop)
        self.eval_thread.daemon = True
        self.eval_thread.start()
    
    def stop(self):
        """Stop evaluation and save results."""
        self.running = False
        if self.eval_thread:
            self.eval_thread.join()
        
        # Stop efficiency monitoring
        self.efficiency_metrics.stop_monitoring()
        
        # Save final results
        self._save_results()
    
    def _evaluation_loop(self):
        """Background thread for processing evaluation tasks."""
        while self.running:
            try:
                # Get evaluation task with timeout
                task = self.eval_queue.get(timeout=1.0)
                
                if task['type'] == 'frame':
                    self._process_frame_evaluation(task['data'])
                elif task['type'] == 'checkpoint':
                    self._process_checkpoint_evaluation(task['data'])
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Evaluation error: {e}")
    
    def add_frame(self,
                 frame_id: int,
                 rgb: np.ndarray,
                 depth: Optional[np.ndarray],
                 pred_semantic: Optional[np.ndarray],
                 pred_instance: Optional[np.ndarray],
                 gt_semantic: Optional[np.ndarray],
                 gt_instance: Optional[np.ndarray],
                 pose: np.ndarray,
                 processing_time: float,
                 **kwargs):
        """Add a frame for evaluation.
        
        Args:
            frame_id: Frame identifier
            rgb: RGB image
            depth: Depth map in meters
            pred_semantic: Predicted semantic labels
            pred_instance: Predicted instance labels
            gt_semantic: Ground truth semantic labels
            gt_instance: Ground truth instance labels
            pose: Camera pose (4x4)
            processing_time: Time taken to process frame
            **kwargs: Additional data
        """
        # Record efficiency metrics
        self.efficiency_metrics.record_frame_time(processing_time)
        
        # Store frame data
        frame_data = {
            'frame_id': frame_id,
            'rgb': rgb,
            'depth': depth,
            'pred_semantic': pred_semantic,
            'pred_instance': pred_instance,
            'gt_semantic': gt_semantic,
            'gt_instance': gt_instance,
            'pose': pose,
            'timestamp': time.time(),
            **kwargs
        }
        
        self.frame_buffer.append(frame_data)
        
        # Add to temporal metrics
        if pred_instance is not None and gt_instance is not None:
            self._update_temporal_metrics(frame_data)
        
        # Check if we should evaluate
        if frame_id % self.eval_interval == 0:
            self.eval_queue.put({
                'type': 'frame',
                'data': frame_data
            })
        
        # Update plots
        if self.enable_plotting and frame_id % self.plot_interval == 0:
            self._update_plots()
    
    def _process_frame_evaluation(self, frame_data: Dict[str, Any]):
        """Process evaluation for a single frame."""
        frame_id = frame_data['frame_id']
        
        # 2D metrics (if semantic masks available)
        if frame_data['pred_semantic'] is not None and frame_data['gt_semantic'] is not None:
            # Compute 2D IoU
            pred_flat = frame_data['pred_semantic'].flatten()
            gt_flat = frame_data['gt_semantic'].flatten()
            
            # Update confusion matrix
            for pred, gt in zip(pred_flat, gt_flat):
                if 0 <= pred < self.num_classes and 0 <= gt < self.num_classes:
                    self.semantic_metrics.confusion_matrix[gt, pred] += 1
            
            # Get current metrics
            metrics_2d = self.semantic_metrics.compute_metrics()
            
            # Store in history
            self.metric_history['semantic_miou'].append(metrics_2d['mean_iou'])
            self.metric_history['semantic_accuracy'].append(metrics_2d['overall_accuracy'])
        
        # Instance metrics
        if frame_data['pred_instance'] is not None and frame_data['gt_instance'] is not None:
            # Prepare instance data
            instance_pred = self._prepare_instance_data(
                frame_data['pred_instance'],
                frame_data['pred_semantic']
            )
            instance_gt = self._prepare_instance_data(
                frame_data['gt_instance'],
                frame_data['gt_semantic']
            )
            
            # Compute instance metrics
            if instance_pred and instance_gt:
                instance_results = self.instance_metrics.compute(
                    instance_pred, instance_gt
                )
                
                self.metric_history['instance_map'].append(instance_results.get('mAP', 0))
                self.metric_history['instance_map50'].append(instance_results.get('mAP_50', 0))
        
        # Efficiency metrics
        fps_metrics = self.efficiency_metrics.compute_fps_metrics()
        self.metric_history['fps'].append(fps_metrics.get('fps_mean', 0))
        
        # Timestamp
        self.timestamp_history.append(frame_id)
        
        # Save checkpoint
        if frame_id % (self.eval_interval * 10) == 0:
            self._save_checkpoint(frame_id)
    
    def _update_temporal_metrics(self, frame_data: Dict[str, Any]):
        """Update temporal consistency metrics."""
        # Extract instance data
        pred_instances = self._extract_instances(
            frame_data['pred_instance'],
            frame_data['pred_semantic']
        )
        gt_instances = self._extract_instances(
            frame_data['gt_instance'],
            frame_data['gt_semantic']
        )
        
        # Add to temporal metrics
        self.temporal_metrics.add_frame(
            frame_data['frame_id'],
            pred_instances,
            gt_instances
        )
        
        # Compute metrics periodically
        if frame_data['frame_id'] % (self.eval_interval * 5) == 0:
            temporal_results = self.temporal_metrics.compute({}, {})
            self.metric_history['mota'].append(temporal_results.get('MOTA', 0))
            self.metric_history['motp'].append(temporal_results.get('MOTP', 0))
    
    def _prepare_instance_data(self,
                              instance_mask: np.ndarray,
                              semantic_mask: Optional[np.ndarray]) -> Dict[str, Any]:
        """Prepare instance data for metrics computation."""
        unique_instances = np.unique(instance_mask)
        unique_instances = unique_instances[unique_instances > 0]  # Remove background
        
        if len(unique_instances) == 0:
            return {}
        
        # Create mock 3D points for 2D evaluation
        h, w = instance_mask.shape
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        points = np.stack([x_coords, y_coords, np.ones_like(x_coords)], axis=-1)
        points = points.reshape(-1, 3)
        
        instance_labels = instance_mask.flatten()
        
        data = {
            'points': points,
            'instance_labels': instance_labels,
            'confidence_scores': {int(inst_id): 1.0 for inst_id in unique_instances}
        }
        
        if semantic_mask is not None:
            data['semantic_labels'] = semantic_mask.flatten()
        
        return data
    
    def _extract_instances(self,
                          instance_mask: np.ndarray,
                          semantic_mask: Optional[np.ndarray]) -> List[Dict]:
        """Extract individual instances from masks."""
        instances = []
        unique_ids = np.unique(instance_mask)
        
        for inst_id in unique_ids:
            if inst_id == 0:  # Skip background
                continue
            
            mask = instance_mask == inst_id
            
            instance = {
                'track_id': int(inst_id),
                'mask': mask,
                'label': int(semantic_mask[mask][0]) if semantic_mask is not None else 0
            }
            
            instances.append(instance)
        
        return instances
    
    def _update_plots(self):
        """Update live plots."""
        if not self.enable_plotting or not self.timestamp_history:
            return
        
        # Clear axes
        for ax in self.axes.flat:
            ax.clear()
        
        frames = self.timestamp_history
        
        # Plot semantic mIoU
        if 'semantic_miou' in self.metric_history:
            self.axes[0, 0].plot(frames, self.metric_history['semantic_miou'], 'b-')
            self.axes[0, 0].set_title(f'Semantic mIoU (Current: {self.metric_history["semantic_miou"][-1]:.3f})')
            self.axes[0, 0].set_xlabel('Frame')
            self.axes[0, 0].set_ylabel('mIoU')
            self.axes[0, 0].grid(True, alpha=0.3)
        
        # Plot instance mAP
        if 'instance_map' in self.metric_history:
            self.axes[0, 1].plot(frames, self.metric_history['instance_map'], 'g-', label='mAP')
            if 'instance_map50' in self.metric_history:
                self.axes[0, 1].plot(frames, self.metric_history['instance_map50'], 'g--', label='mAP@50')
            self.axes[0, 1].set_title(f'Instance mAP (Current: {self.metric_history["instance_map"][-1]:.3f})')
            self.axes[0, 1].set_xlabel('Frame')
            self.axes[0, 1].set_ylabel('mAP')
            self.axes[0, 1].legend()
            self.axes[0, 1].grid(True, alpha=0.3)
        
        # Plot temporal consistency
        if 'mota' in self.metric_history:
            frames_temporal = frames[::5]  # Temporal metrics computed less frequently
            mota_values = self.metric_history['mota'][-len(frames_temporal):]
            self.axes[1, 0].plot(frames_temporal, mota_values, 'r-')
            self.axes[1, 0].set_title(f'MOTA (Current: {mota_values[-1] if mota_values else 0:.3f})')
            self.axes[1, 0].set_xlabel('Frame')
            self.axes[1, 0].set_ylabel('MOTA')
            self.axes[1, 0].grid(True, alpha=0.3)
        
        # Plot FPS
        if 'fps' in self.metric_history:
            self.axes[1, 1].plot(frames, self.metric_history['fps'], 'm-')
            self.axes[1, 1].axhline(y=30, color='r', linestyle='--', label='Real-time (30 FPS)')
            self.axes[1, 1].set_title(f'Processing Speed (Current: {self.metric_history["fps"][-1]:.1f} FPS)')
            self.axes[1, 1].set_xlabel('Frame')
            self.axes[1, 1].set_ylabel('FPS')
            self.axes[1, 1].legend()
            self.axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)
    
    def _save_checkpoint(self, frame_id: int):
        """Save evaluation checkpoint."""
        checkpoint = {
            'frame_id': frame_id,
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'semantic': self.semantic_metrics.compute_metrics(),
                'efficiency': self.efficiency_metrics.compute({}, {}),
                'temporal': self.temporal_metrics.compute({}, {}) if len(self.temporal_metrics.frame_data) > 0 else {}
            },
            'metric_history': dict(self.metric_history)
        }
        
        checkpoint_path = self.output_dir / f'checkpoint_frame_{frame_id:06d}.json'
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def _save_results(self):
        """Save final evaluation results."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'num_frames': len(self.timestamp_history),
            'semantic_classes': self.semantic_classes,
            'final_metrics': {
                'semantic': self.semantic_metrics.compute_metrics(),
                'instance': self.instance_metrics.get_results(),
                'temporal': self.temporal_metrics.compute({}, {}) if len(self.temporal_metrics.frame_data) > 0 else {},
                'boundary': self.boundary_metrics.get_results(),
                'efficiency': self.efficiency_metrics.compute({}, {})
            },
            'metric_history': dict(self.metric_history),
            'timestamp_history': self.timestamp_history
        }
        
        # Save JSON results
        results_path = self.output_dir / 'final_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save metric plots
        if self.metric_history:
            self._save_metric_plots()
        
        # Save individual metric reports
        self.semantic_metrics.save_results()
        self.instance_metrics.save_results()
        self.temporal_metrics.save_results()
        self.boundary_metrics.save_results()
        self.efficiency_metrics.save_results()
        
        # Generate summary report
        self._generate_summary_report()
    
    def _save_metric_plots(self):
        """Save metric history plots."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        frames = self.timestamp_history
        
        # Semantic mIoU
        if 'semantic_miou' in self.metric_history:
            axes[0, 0].plot(frames, self.metric_history['semantic_miou'])
            axes[0, 0].set_title('Semantic mIoU over Time')
            axes[0, 0].set_xlabel('Frame')
            axes[0, 0].set_ylabel('mIoU')
            axes[0, 0].grid(True)
        
        # Instance mAP
        if 'instance_map' in self.metric_history:
            axes[0, 1].plot(frames, self.metric_history['instance_map'])
            axes[0, 1].set_title('Instance mAP over Time')
            axes[0, 1].set_xlabel('Frame')
            axes[0, 1].set_ylabel('mAP')
            axes[0, 1].grid(True)
        
        # Temporal MOTA
        if 'mota' in self.metric_history:
            frames_temporal = frames[::5]
            mota_values = self.metric_history['mota'][-len(frames_temporal):]
            axes[1, 0].plot(frames_temporal, mota_values)
            axes[1, 0].set_title('MOTA over Time')
            axes[1, 0].set_xlabel('Frame')
            axes[1, 0].set_ylabel('MOTA')
            axes[1, 0].grid(True)
        
        # FPS
        if 'fps' in self.metric_history:
            axes[1, 1].plot(frames, self.metric_history['fps'])
            axes[1, 1].axhline(y=30, color='r', linestyle='--', label='Real-time')
            axes[1, 1].set_title('Processing Speed over Time')
            axes[1, 1].set_xlabel('Frame')
            axes[1, 1].set_ylabel('FPS')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metric_history.png', dpi=150)
        plt.close()
    
    def _generate_summary_report(self):
        """Generate a summary report."""
        report_path = self.output_dir / 'evaluation_summary.txt'
        
        with open(report_path, 'w') as f:
            f.write("=== Semantic SLAM Evaluation Summary ===\n\n")
            f.write(f"Total frames evaluated: {len(self.timestamp_history)}\n")
            f.write(f"Evaluation interval: {self.eval_interval} frames\n\n")
            
            # Semantic metrics
            if self.semantic_metrics.results:
                f.write("Semantic Segmentation Metrics:\n")
                f.write(f"  Mean IoU: {self.semantic_metrics.results.get('mean_iou', 0):.3f}\n")
                f.write(f"  Overall Accuracy: {self.semantic_metrics.results.get('overall_accuracy', 0):.3f}\n\n")
            
            # Instance metrics
            if self.instance_metrics.results:
                f.write("Instance Segmentation Metrics:\n")
                f.write(f"  mAP: {self.instance_metrics.results.get('mAP', 0):.3f}\n")
                f.write(f"  mAP@50: {self.instance_metrics.results.get('mAP_50', 0):.3f}\n\n")
            
            # Temporal metrics
            if self.temporal_metrics.results:
                f.write("Temporal Consistency Metrics:\n")
                f.write(f"  MOTA: {self.temporal_metrics.results.get('MOTA', 0):.3f}\n")
                f.write(f"  MOTP: {self.temporal_metrics.results.get('MOTP', 0):.3f}\n")
                f.write(f"  ID Switches: {self.temporal_metrics.results.get('ID_switches', 0)}\n\n")
            
            # Efficiency metrics
            if self.efficiency_metrics.results:
                f.write("Efficiency Metrics:\n")
                f.write(f"  Mean FPS: {self.efficiency_metrics.results.get('fps_mean', 0):.1f}\n")
                f.write(f"  CPU Memory: {self.efficiency_metrics.results.get('cpu_memory_mean_mb', 0):.0f} MB\n")
                if 'gpu_memory_mean_mb' in self.efficiency_metrics.results:
                    f.write(f"  GPU Memory: {self.efficiency_metrics.results.get('gpu_memory_mean_mb', 0):.0f} MB\n")
        
        print(f"Evaluation complete. Results saved to: {self.output_dir}")