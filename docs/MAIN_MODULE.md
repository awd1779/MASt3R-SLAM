# main.py - Entry Point and Pipeline Orchestration

## Overview
`main.py` is the entry point for MASt3R-SLAM. It orchestrates the entire SLAM pipeline by managing three parallel processes: frontend tracking, backend optimization, and real-time visualization.

## Command-Line Interface

### Basic Usage
```bash
python main.py --scene <dataset_name> [options]
```

### Arguments

#### Required Arguments
- `--scene`: Dataset name or path to process

#### Dataset Selection
- `--dataset`: Dataset type (default: "demo")
  - Options: `tum`, `euorc`, `7scenes`, `eth3d`, `demo`

#### Input Sources
- `--root`: Path to dataset root directory
- `--video`: Path to video file (alternative to dataset)
- `--cam`: Camera device ID for live input (default: 0)
- `--img_dir`: Directory containing image sequence

#### Camera Configuration  
- `--K`: Camera intrinsics matrix (fx, fy, cx, cy)
- `--width`, `--height`: Image dimensions for live camera
- `--no-use_calibration`: Disable camera calibration usage

#### Output Options
- `--output_dir`: Directory for saving results (default: "./output")
- `--save_ply`: Save final point cloud
- `--save_refined`: Save refined point cloud (calibrated mode only)
- `--save_kf`: Save keyframe images

#### Performance Settings
- `--processes`: Number of processes (default: 3)
  - 3: Full system with visualization
  - 2: No visualization 
  - 1: Single-threaded mode
- `--img_stride`: Frame subsampling factor (default: 1)

#### Configuration
- `--config`: Path to YAML configuration file

## Pipeline Architecture

### System Modes
The SLAM system operates in four modes:

```python
class Mode(Enum):
    INIT = 0        # Initial frame processing
    TRACKING = 1    # Normal tracking mode
    RELOC = 2       # Relocalization after tracking failure
    TERMINATED = 3  # System shutdown
```

### Process Structure

#### 1. Main Process (Frontend)
Handles frame-by-frame tracking:
```python
def run_frontend():
    while mode != Mode.TERMINATED:
        frame = get_next_frame()
        
        if mode == Mode.INIT:
            # Process first frame
            init_system(frame)
            mode = Mode.TRACKING
            
        elif mode == Mode.TRACKING:
            # Track current frame
            success = track_frame(frame)
            if not success:
                mode = Mode.RELOC
                
        elif mode == Mode.RELOC:
            # Attempt relocalization
            success = relocalize(frame)
            if success:
                mode = Mode.TRACKING
```

#### 2. Backend Process
Runs global optimization:
```python
def run_backend():
    factor_graph = FactorGraph()
    
    while True:
        task = task_queue.get()
        
        if task == "optimize":
            # Add new factors
            factor_graph.add_keyframe_factors()
            # Run optimization
            factor_graph.optimize()
            # Update poses
            update_shared_keyframes()
```

#### 3. Visualization Process
Real-time 3D rendering:
```python
def run_visualization():
    viewer = Viewer3D()
    
    while running:
        # Get updated keyframes
        keyframes = get_dirty_keyframes()
        # Update visualization
        viewer.update(keyframes)
        # Handle user input
        viewer.process_events()
```

## Key Functions

### Initialization
```python
def init(args, dataset, model, tracker, shared_states, shared_keyframes):
    """Initialize SLAM system with first frame"""
    # Process first frame
    frame = dataset[0]
    # MASt3R mono inference
    pred = mast3r_inference_mono(model, frame.img)
    # Create first keyframe
    keyframe = Frame(...)
    # Initialize shared states
    shared_states.set_latest_frame(keyframe)
```

### Frame Tracking
```python
def track_frame(frame_idx, dataset, model, tracker, shared_states):
    """Track current frame against latest keyframe"""
    # Get current frame
    curr_frame = dataset[frame_idx]
    # Get latest keyframe
    ref_frame = shared_states.get_latest_frame()
    
    # Perform tracking
    track_result = tracker.track(model, ref_frame, curr_frame)
    
    if track_result.is_keyframe:
        # Add to keyframe buffer
        shared_keyframes.add(curr_frame)
        # Queue backend optimization
        task_queue.put("optimize")
```

### Relocalization
```python
def relocalize(frame, model, retrieval_db, shared_keyframes):
    """Attempt to relocalize after tracking failure"""
    # Query similar keyframes
    candidates = retrieval_db.query(frame)
    
    for candidate in candidates:
        # Try tracking against candidate
        success = try_track(frame, candidate)
        if success:
            return True
    return False
```

## Data Structures

### SharedStates
Manages system-wide state across processes:
- Current frame data
- System mode (INIT, TRACKING, etc.)
- Statistics (frame count, keyframe count)

### Task Queue
Communication between frontend and backend:
- `"optimize"`: Trigger global optimization
- `"add_keyframe"`: New keyframe available
- `"loop_closure"`: Loop detected

## Configuration Integration

The system loads configuration from YAML files:
```python
# Load configuration
if args.config:
    with open(args.config) as f:
        config = yaml.safe_load(f)
        
# Apply to components
tracker = Tracker(**config['tracking'])
backend = Backend(**config['optimization'])
```

## Error Handling

### Tracking Failures
- Automatically switches to relocalization mode
- Attempts to find similar keyframes
- Falls back to reinitialization if needed

### Resource Management
- Graceful shutdown on errors
- Cleanup of shared memory
- Process termination handling

## Performance Considerations

### Multi-Process Benefits
- Frontend runs at camera rate
- Backend optimization doesn't block tracking
- Visualization updates asynchronously

### Memory Efficiency
- Shared memory for zero-copy data access
- Pre-allocated keyframe buffer
- Lazy loading of images

### GPU Utilization
- MASt3R model on GPU
- CUDA kernels for optimization
- Batch processing where possible

## Example Usage

### Running on TUM Dataset
```bash
python main.py --scene freiburg1_desk --dataset tum --root ./datasets/TUM
```

### Live Camera with Calibration
```bash
python main.py --scene live --cam 0 --K 520 520 320 240 --width 640 --height 480
```

### Video File Processing
```bash
python main.py --scene my_video --video ./videos/test.mp4 --save_ply
```

## Integration Points

The main module integrates with:
- **Dataloader**: For dataset/input handling
- **Tracker**: For frame-to-frame tracking
- **Global Optimizer**: For backend optimization
- **Visualization**: For real-time display
- **Retrieval Database**: For loop closure

This modular design allows easy extension and modification of individual components while maintaining the overall pipeline structure.