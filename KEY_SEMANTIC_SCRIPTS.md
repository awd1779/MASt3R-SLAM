# Key Scripts for Semantic SLAM Pipeline

## Core Semantic Processing Chain

### 1. **main_semantic_tracked_3d.py**
- Entry point
- Orchestrates the entire pipeline
- Manages threads and processes

### 2. **grounded_sam2_real_3d.py**
- Processes images through Grounded-SAM2
- Detects and segments objects
- Manages semantic processing thread
- Calls grounded_sam2_real.py internally

### 3. **grounded_sam2_real.py**
- Core Grounded-SAM2 implementation
- Performs actual detection and segmentation
- Handles vocabulary and filtering

### 4. **geometric_3d_tracker.py**
- Tracks objects across frames using 3D geometry
- Assigns consistent track IDs
- Manages global sliding window tracking

### 5. **dense_semantic_reconstruction_v2.py**
- **THIS IS WHERE THE CHAIR/SOFA ISSUE OCCURS**
- Applies semantic masks to 3D points
- Filters points based on labels
- Creates semantic point cloud

### 6. **dense_semantic_reconstruction_tracked_v2.py**
- Wrapper around dense_semantic_reconstruction_v2.py
- Adds tracking information to reconstruction
- Manages track-to-label mapping

### 7. **semantic_overlay.py**
- Creates visualization by overlaying semantic colors
- Uses nearest neighbor matching
- This is why chair/sofa appear colored even with 0 points

## The Problem Location
The issue with chair/sofa having 0 points occurs in:
- **dense_semantic_reconstruction_v2.py** at the mask application stage
- Lines where masks are sorted and applied
- The filtering condition `has_label = semantic_mask_flat > 0`

## Key Configuration Files
- **config/replica_semantic_auto_tracked_3d_global_improved_filtered.yaml**
- Contains all parameters for detection, tracking, and reconstruction