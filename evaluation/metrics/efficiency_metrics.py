"""Computational efficiency metrics for semantic SLAM evaluation."""

import time
import psutil
import numpy as np
from typing import Dict, List, Optional, Callable, Any
import threading
from collections import defaultdict, deque
import GPUtil

from .base_metrics import BaseMetrics


class EfficiencyMetrics(BaseMetrics):
    """Track computational efficiency metrics including timing, memory, and throughput."""
    
    def __init__(self,
                 track_gpu: bool = True,
                 memory_sample_interval: float = 0.1,
                 window_size: int = 100,
                 output_dir: Optional[str] = None):
        """Initialize efficiency metrics tracker.
        
        Args:
            track_gpu: Whether to track GPU metrics
            memory_sample_interval: Interval for memory sampling (seconds)
            window_size: Window size for moving averages
            output_dir: Directory to save outputs
        """
        super().__init__("Efficiency", output_dir)
        
        self.track_gpu = track_gpu and self._check_gpu_available()
        self.memory_sample_interval = memory_sample_interval
        self.window_size = window_size
        
        self.reset()
    
    def reset(self):
        """Reset internal state."""
        self.results = {}
        self.timing_data = defaultdict(list)
        self.memory_samples = []
        self.gpu_memory_samples = []
        self.frame_times = deque(maxlen=self.window_size)
        self.component_times = defaultdict(lambda: deque(maxlen=self.window_size))
        
        # Memory monitoring thread
        self._monitoring = False
        self._monitor_thread = None
        
        # Process handle for memory tracking
        self.process = psutil.Process()
    
    def _check_gpu_available(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            GPUtil.getGPUs()
            return True
        except:
            print("GPU monitoring not available. Install gputil for GPU metrics.")
            return False
    
    def start_monitoring(self):
        """Start background memory monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_memory)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop background memory monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
    
    def _monitor_memory(self):
        """Background thread for memory monitoring."""
        while self._monitoring:
            # CPU memory
            mem_info = self.process.memory_info()
            self.memory_samples.append({
                'timestamp': time.time(),
                'rss': mem_info.rss / 1024**2,  # MB
                'vms': mem_info.vms / 1024**2,  # MB
                'percent': self.process.memory_percent()
            })
            
            # GPU memory
            if self.track_gpu:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # Use first GPU
                        self.gpu_memory_samples.append({
                            'timestamp': time.time(),
                            'used': gpu.memoryUsed,  # MB
                            'total': gpu.memoryTotal,  # MB
                            'percent': gpu.memoryUtil * 100
                        })
                except:
                    pass
            
            time.sleep(self.memory_sample_interval)
    
    def time_operation(self, operation_name: str):
        """Context manager for timing operations.
        
        Usage:
            with metrics.time_operation('feature_extraction'):
                # Your code here
                pass
        """
        return TimingContext(self, operation_name)
    
    def record_frame_time(self, frame_time: float):
        """Record processing time for a frame.
        
        Args:
            frame_time: Time in seconds to process one frame
        """
        self.frame_times.append(frame_time)
    
    def record_component_time(self, component: str, time_taken: float):
        """Record processing time for a specific component.
        
        Args:
            component: Name of the component (e.g., 'segmentation', 'tracking')
            time_taken: Time in seconds
        """
        self.component_times[component].append(time_taken)
        self.timing_data[component].append(time_taken)
    
    def compute_fps_metrics(self) -> Dict[str, float]:
        """Compute frames per second metrics."""
        if not self.frame_times:
            return {'fps': 0.0, 'fps_std': 0.0}
        
        frame_times = np.array(self.frame_times)
        fps_values = 1.0 / (frame_times + 1e-10)
        
        return {
            'fps_mean': np.mean(fps_values),
            'fps_std': np.std(fps_values),
            'fps_min': np.min(fps_values),
            'fps_max': np.max(fps_values),
            'frame_time_mean': np.mean(frame_times),
            'frame_time_std': np.std(frame_times)
        }
    
    def compute_memory_metrics(self) -> Dict[str, float]:
        """Compute memory usage metrics."""
        metrics = {}
        
        # CPU memory metrics
        if self.memory_samples:
            rss_values = [s['rss'] for s in self.memory_samples]
            metrics.update({
                'cpu_memory_mean_mb': np.mean(rss_values),
                'cpu_memory_max_mb': np.max(rss_values),
                'cpu_memory_std_mb': np.std(rss_values)
            })
        
        # GPU memory metrics
        if self.gpu_memory_samples:
            used_values = [s['used'] for s in self.gpu_memory_samples]
            percent_values = [s['percent'] for s in self.gpu_memory_samples]
            metrics.update({
                'gpu_memory_mean_mb': np.mean(used_values),
                'gpu_memory_max_mb': np.max(used_values),
                'gpu_memory_mean_percent': np.mean(percent_values),
                'gpu_memory_max_percent': np.max(percent_values)
            })
        
        return metrics
    
    def compute_latency_metrics(self) -> Dict[str, float]:
        """Compute latency metrics for different components."""
        metrics = {}
        
        for component, times in self.component_times.items():
            if times:
                times_array = np.array(times) * 1000  # Convert to milliseconds
                metrics.update({
                    f'{component}_latency_mean_ms': np.mean(times_array),
                    f'{component}_latency_std_ms': np.std(times_array),
                    f'{component}_latency_p50_ms': np.percentile(times_array, 50),
                    f'{component}_latency_p95_ms': np.percentile(times_array, 95),
                    f'{component}_latency_p99_ms': np.percentile(times_array, 99)
                })
        
        return metrics
    
    def compute_throughput_metrics(self,
                                  num_points: Optional[List[int]] = None,
                                  num_objects: Optional[List[int]] = None) -> Dict[str, float]:
        """Compute throughput metrics.
        
        Args:
            num_points: List of point counts per frame
            num_objects: List of object counts per frame
            
        Returns:
            Throughput metrics
        """
        metrics = {}
        
        if num_points and self.frame_times:
            points_per_second = [n / t for n, t in zip(num_points[-len(self.frame_times):], self.frame_times)]
            metrics['points_per_second_mean'] = np.mean(points_per_second)
            metrics['points_per_second_std'] = np.std(points_per_second)
        
        if num_objects and self.frame_times:
            objects_per_second = [n / t for n, t in zip(num_objects[-len(self.frame_times):], self.frame_times)]
            metrics['objects_per_second_mean'] = np.mean(objects_per_second)
        
        return metrics
    
    def compute_scalability_metrics(self,
                                   scene_sizes: Optional[List[int]] = None) -> Dict[str, float]:
        """Compute scalability metrics based on scene complexity.
        
        Args:
            scene_sizes: List of scene sizes (e.g., total points in map)
            
        Returns:
            Scalability metrics
        """
        if not scene_sizes or not self.frame_times:
            return {}
        
        # Analyze how performance scales with scene size
        scene_sizes = np.array(scene_sizes[-len(self.frame_times):])
        frame_times = np.array(list(self.frame_times))
        
        # Compute correlation between scene size and frame time
        if len(scene_sizes) > 10:
            correlation = np.corrcoef(scene_sizes, frame_times)[0, 1]
            
            # Fit linear model to estimate scalability
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(scene_sizes.reshape(-1, 1), frame_times)
            
            # Predict time for different scene sizes
            test_sizes = [10000, 50000, 100000, 500000]
            predictions = model.predict(np.array(test_sizes).reshape(-1, 1))
            
            metrics = {
                'scalability_correlation': correlation,
                'time_per_1k_points_ms': model.coef_[0] * 1000 * 1000,  # ms per 1k points
            }
            
            for size, pred_time in zip(test_sizes, predictions):
                metrics[f'predicted_time_{size//1000}k_points_ms'] = pred_time * 1000
            
            return metrics
        
        return {}
    
    def compute(self,
               prediction: Dict[str, Any],
               ground_truth: Dict[str, Any],
               **kwargs) -> Dict[str, float]:
        """Compute all efficiency metrics.
        
        Args:
            prediction: Not used directly
            ground_truth: Not used directly
            **kwargs: Additional data like num_points, num_objects, scene_sizes
            
        Returns:
            Dictionary of efficiency metrics
        """
        metrics = {}
        
        # FPS metrics
        metrics.update(self.compute_fps_metrics())
        
        # Memory metrics
        metrics.update(self.compute_memory_metrics())
        
        # Latency metrics
        metrics.update(self.compute_latency_metrics())
        
        # Throughput metrics
        if 'num_points' in kwargs or 'num_objects' in kwargs:
            metrics.update(self.compute_throughput_metrics(
                kwargs.get('num_points'),
                kwargs.get('num_objects')
            ))
        
        # Scalability metrics
        if 'scene_sizes' in kwargs:
            metrics.update(self.compute_scalability_metrics(kwargs['scene_sizes']))
        
        # System utilization
        metrics['cpu_percent'] = psutil.cpu_percent(interval=0.1)
        metrics['num_threads'] = threading.active_count()
        
        self.results.update(metrics)
        return metrics
    
    def generate_efficiency_report(self) -> str:
        """Generate a formatted efficiency report."""
        report = []
        report.append("=== Efficiency Report ===\n")
        
        # Real-time performance
        fps_metrics = self.compute_fps_metrics()
        report.append("Real-time Performance:")
        report.append(f"  Average FPS: {fps_metrics.get('fps_mean', 0):.1f} ± {fps_metrics.get('fps_std', 0):.1f}")
        report.append(f"  Frame time: {fps_metrics.get('frame_time_mean', 0)*1000:.1f} ± {fps_metrics.get('frame_time_std', 0)*1000:.1f} ms")
        
        # Component breakdown
        report.append("\nComponent Latencies:")
        latency_metrics = self.compute_latency_metrics()
        components = set(key.split('_')[0] for key in latency_metrics.keys())
        for component in sorted(components):
            mean_key = f'{component}_latency_mean_ms'
            p95_key = f'{component}_latency_p95_ms'
            if mean_key in latency_metrics:
                report.append(f"  {component}: {latency_metrics[mean_key]:.1f} ms (p95: {latency_metrics.get(p95_key, 0):.1f} ms)")
        
        # Memory usage
        mem_metrics = self.compute_memory_metrics()
        report.append("\nMemory Usage:")
        if 'cpu_memory_mean_mb' in mem_metrics:
            report.append(f"  CPU: {mem_metrics['cpu_memory_mean_mb']:.0f} MB (peak: {mem_metrics.get('cpu_memory_max_mb', 0):.0f} MB)")
        if 'gpu_memory_mean_mb' in mem_metrics:
            report.append(f"  GPU: {mem_metrics['gpu_memory_mean_mb']:.0f} MB (peak: {mem_metrics.get('gpu_memory_max_mb', 0):.0f} MB)")
        
        return "\n".join(report)


class TimingContext:
    """Context manager for timing code blocks."""
    
    def __init__(self, metrics: EfficiencyMetrics, operation_name: str):
        self.metrics = metrics
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        self.metrics.record_component_time(self.operation_name, elapsed)