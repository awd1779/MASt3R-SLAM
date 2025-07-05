# Semantic Processing Debug Visualization

This document explains how to use the debug visualization features added to the MASt3R-SLAM semantic processor for inspecting SAM segmentation and CLIP classification outputs.

## Overview

The debug visualization system helps identify issues in the semantic mapping pipeline by saving intermediate outputs:

1. **SAM Mask Visualization**: Shows instance segmentation results from SAM
2. **CLIP Crops Visualization**: Shows the cropped segments fed to CLIP for classification
3. **Classification Results**: Shows CLIP predictions with confidence scores and statistics

## Quick Start

### 1. Enable Debug Mode

Debug visualization is controlled by global flags in `semantic_processor.py`:

```python
from mast3r_slam.semantic_processor import enable_debug_visualization, set_debug_options

# Enable debug visualization
enable_debug_visualization(True, output_dir="debug_semantic_output")

# Configure what to save (optional)
set_debug_options(
    save_sam=True,           # Save SAM mask overlays
    save_clips=True,         # Save CLIP input crops  
    save_classifications=True, # Save classification results
    max_masks=20             # Limit masks to visualize (performance)
)
```

### 2. Run the Test Script

Test the debug functionality with a synthetic image:

```bash
cd /home/ubuntu/MASt3R-SLAM
python test_semantic_debug.py
```

This will create visualizations in the `test_debug_output/` directory.

### 3. Run SLAM with Debug Visualization

The debug visualization is automatically enabled in the current setup. When you run MASt3R-SLAM, debug outputs will be saved to `debug_semantic_output/` for:

- The initial frame (processed in `main.py`)
- New keyframes (processed in `tracker.py`)

```bash
python main.py --dataset datasets/tum/rgbd_dataset_freiburg1_room/ --config config/calib.yaml
```

## Output Structure

Debug outputs are organized as follows:

```
debug_semantic_output/
├── frame_000000_20231204_143022/     # Frame ID + timestamp
│   ├── sam_masks/                     # SAM segmentation results
│   │   ├── sam_overview_frame_000000.png        # All masks overlay
│   │   ├── mask_00_area_15420_frame_000000.png  # Individual masks
│   │   └── mask_01_area_8934_frame_000000.png
│   ├── clip_crops/                    # CLIP input crops
│   │   ├── clip_crops_grid_frame_000000.png     # All crops in grid
│   │   ├── crop_00_frame_000000.png             # Individual crops
│   │   └── crop_01_frame_000000.png
│   └── classifications/               # Classification results
│       ├── classification_summary_frame_000000.png  # Statistics & charts
│       └── classification_data_frame_000000.json    # Detailed data
```

## Understanding the Visualizations

### SAM Mask Outputs

1. **sam_overview_frame_XXXXXX.png**:
   - Original image
   - All masks overlaid with different colors
   - Largest individual mask
   - Mask area distribution histogram

2. **mask_XX_area_YYYY_frame_XXXXXX.png**:
   - Individual mask highlighted on original image
   - Binary mask visualization
   - Mask boundary overlay

### CLIP Crop Outputs

1. **clip_crops_grid_frame_XXXXXX.png**:
   - Grid showing all crops fed to CLIP
   - Useful for seeing what CLIP "sees" for each segment

2. **crop_XX_frame_XXXXXX.png**:
   - Individual crops as 224x224 images
   - These are the exact inputs to CLIP classification

### Classification Results

1. **classification_summary_frame_XXXXXX.png**:
   - Original image
   - Confidence score distribution 
   - Predicted class distribution
   - Confidence vs class scatter plot
   - Top/bottom confident predictions

2. **classification_data_frame_XXXXXX.json**:
   - Detailed metadata for each segment
   - Predicted classes and confidence scores
   - Segment areas and statistics

## Debugging Common Issues

### Issue 1: Most areas painted with wrong colors in PLY

**Check these visualizations:**

1. **SAM masks**: Are the segments reasonable? Look for:
   - Over-segmentation (too many small masks)
   - Under-segmentation (large areas not separated)
   - Missing important objects

2. **CLIP crops**: Do the crops contain clear, identifiable objects?
   - Crops should be meaningful object parts
   - Check if crops are too small/blurry
   - Verify crops aren't mostly background

3. **Classifications**: Are the confidence scores reasonable?
   - Low confidence scores (< 20) indicate uncertainty
   - Check if similar objects get different labels
   - Look for systematic misclassifications

### Issue 2: Poor segmentation quality

**Adjust SAM parameters** in `semantic_processor.py`:

```python
DEFAULT_SAM_POINTS_PER_SIDE = 32      # Increase for more segments
DEFAULT_SAM_MIN_MASK_REGION_AREA = 50 # Increase to filter small segments
DEFAULT_SAM_PRED_IOU_THRESH = 0.88    # Increase for higher quality
DEFAULT_SAM_STABILITY_SCORE_THRESH = 0.95  # Increase for more stable masks
```

### Issue 3: Poor classification accuracy

**Check/modify text prompts** in `semantic_processor.py`:

```python
TEXT_PROMPTS = [
    "background", "wall", "floor", "ceiling", 
    "chair", "desk", "table", "monitor", "cup", "book",
    # Add more specific prompts for your environment
    "wooden table", "office chair", "computer monitor"
]
```

## Configuration Options

### Global Debug Flags

Located in `semantic_processor.py`:

```python
DEBUG_VISUALIZATION = True              # Master enable/disable
DEBUG_OUTPUT_DIR = "debug_semantic_output"  # Output directory
DEBUG_SAVE_SAM_MASKS = True            # Save SAM visualizations
DEBUG_SAVE_CLIP_CROPS = True           # Save CLIP crop visualizations  
DEBUG_SAVE_CLASSIFICATION_RESULTS = True  # Save classification results
DEBUG_MAX_MASKS_TO_VISUALIZE = 20      # Performance limit
```

### Runtime Control

```python
# Disable debug for performance during normal runs
enable_debug_visualization(False)

# Enable only for specific frames
local_mask, local_map = process_frame_for_semantics(
    image_tensor_chw_0_1_rgb=frame_image,
    text_prompts_for_clip=TEXT_PROMPTS,
    enable_debug_viz=True,  # Override global setting
    frame_id=frame.frame_id
)
```

## Performance Considerations

- Debug visualization adds ~2-3 seconds per frame
- Disable for production runs: `enable_debug_visualization(False)`
- Limit masks to visualize: `set_debug_options(max_masks=10)`
- Debug outputs can use significant disk space

## Next Steps

1. Run the test script to verify functionality
2. Process a few frames of your dataset with debug enabled
3. Analyze the visualizations to identify issues
4. Adjust SAM parameters or CLIP prompts based on findings
5. Disable debug mode for full dataset processing

## Troubleshooting

- **Import errors**: Ensure matplotlib is installed (`pip install matplotlib`)
- **No debug output**: Check that `DEBUG_VISUALIZATION = True`
- **Permission errors**: Ensure write access to output directory
- **Memory issues**: Reduce `DEBUG_MAX_MASKS_TO_VISUALIZE`