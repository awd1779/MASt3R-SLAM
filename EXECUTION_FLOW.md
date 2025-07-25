# Execution Flow of main_semantic_tracked_3d.py

## Main Entry Point
- **main_semantic_tracked_3d.py** - Main script that orchestrates everything

## Core SLAM Components (from mast3r_slam/)

### 1. Configuration & Data Loading
- **config.py** - Loads and manages configuration
- **dataloader.py** - Loads dataset (images, poses, etc.)

### 2. Core SLAM Processing
- **frame.py** - Frame management (SharedKeyframes, SharedStates, create_frame)
- **mast3r_utils.py** - MASt3R model loading and inference
- **tracker.py** - FrameTracker for visual tracking
- **global_opt.py** - FactorGraph for global optimization
- **multiprocess_utils.py** - Queue management for multiprocessing
- **lietorch_utils.py** - Lie group operations (imported by other modules)

### 3. Semantic Processing Pipeline
- **semantic_frame.py** - SharedSemanticKeyframes, create_semantic_frame
- **grounded_sam2_real_3d.py** - Main semantic processor with 3D tracking
  - Starts the semantic processing thread
  - Processes images through Grounded-SAM2
- **grounded_sam2_config.py** - Configuration for Grounded-SAM2
- **grounded_sam2_real.py** - Core Grounded-SAM2 implementation (imported by grounded_sam2_real_3d.py)

### 4. Object Tracking
- **geometric_3d_tracker.py** - 3D geometric tracking for objects
- **tracking_utils.py** - Utility functions for tracking (imported by geometric tracker)

### 5. Semantic Reconstruction
- **dense_semantic_reconstruction_tracked_v2.py** - Creates semantic point cloud with tracking
- **dense_semantic_reconstruction_v2.py** - Core semantic reconstruction (imported by tracked version)
- **semantic_overlay.py** - Creates overlay visualization

### 6. Visualization
- **visualization.py** - Main visualization window (run_visualization)
- **visualization_utils.py** - Utility functions for visualization

### 7. Evaluation & Saving
- **evaluate.py** - Evaluation metrics
- **save_semantic_keyframes.py** - Saves semantic statistics and metadata

### 8. Additional Utilities (imported internally)
- **geometry.py** - Geometric operations
- **matching.py** - Feature matching
- **image.py** - Image processing utilities
- **retrieval_database.py** - Database for keyframe retrieval
- **nonlinear_optimizer.py** - Optimization routines
- **replica_vocabulary_loader.py** - Loads vocabulary for Replica dataset

## External Dependencies
- **thirdparty/mast3r/** - MASt3R model implementation
- **Grounded-SAM2** models (loaded at runtime)
- **BERT** model for text encoding

## Execution Flow Summary
1. Load configuration and dataset
2. Initialize MASt3R model and semantic processors
3. Start visualization process (if enabled)
4. Start semantic processing thread
5. Process frames:
   - Visual SLAM tracking
   - Send keyframes to semantic processor
   - Semantic segmentation and 3D tracking
6. Global optimization
7. Dense semantic reconstruction
8. Save results

## Key Processing Threads
1. **Main thread** - SLAM processing
2. **Visualization process** - Real-time visualization
3. **Semantic processing thread** - Grounded-SAM2 processing
4. **Backend optimization** - Runs periodically