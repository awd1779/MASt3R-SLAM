# Grounded-SAM2 Installation Guide

This guide covers the installation of real Grounded-SAM2 dependencies for the MAST3R-SLAM semantic integration.

## Prerequisites

- CUDA-capable GPU (10GB+ VRAM recommended)
- Python 3.8+
- PyTorch with CUDA support
- Git

## Installation Steps

### 1. Install SAM2 (Segment Anything 2)

```bash
# Clone SAM2 repository
cd /path/to/libs  # Choose your library directory
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2

# Install SAM2
pip install -e .

# Download model checkpoints
mkdir checkpoints
cd checkpoints

# Download hiera_b+ model (recommended for paper)
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt

# Optional: Download other models
# wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt
# wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt
# wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
```

### 2. Install Grounding DINO

```bash
# Clone Grounding DINO repository
cd /path/to/libs
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO

# Install dependencies
pip install -e .

# Download model checkpoints
mkdir weights
cd weights

# Download Swin-B model (recommended for paper)
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth

# Optional: Download Swin-T model (faster, less accurate)
# wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

### 3. Additional Dependencies

```bash
# Install additional required packages
pip install supervision transformers

# For CUDA kernels (optional but recommended)
pip install ninja
```

## Configuration

Update the model paths in your code or configuration:

```python
# In mast3r_slam/grounded_sam2_config.py or similar
SAM2_CHECKPOINTS = {
    "hiera_b+": "/path/to/libs/segment-anything-2/checkpoints/sam2_hiera_base_plus.pt",
    # Add other models as needed
}

GROUNDING_DINO_CHECKPOINTS = {
    "grounding_dino_swin-b": "/path/to/libs/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth",
    # Add other models as needed
}
```

## Verify Installation

Test the installation with this script:

```python
# test_grounded_sam2_install.py
import torch

# Test SAM2
try:
    from sam2.build_sam import build_sam2_video_predictor
    print("✓ SAM2 import successful")
except ImportError as e:
    print(f"✗ SAM2 import failed: {e}")

# Test Grounding DINO
try:
    from groundingdino.util.inference import load_model
    from groundingdino.util.slconfig import SLConfig
    print("✓ Grounding DINO import successful")
except ImportError as e:
    print(f"✗ Grounding DINO import failed: {e}")

# Check CUDA
if torch.cuda.is_available():
    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("✗ CUDA not available")
```

## Common Issues

### 1. CUDA Out of Memory
- Use smaller models (hiera_tiny instead of hiera_b+)
- Reduce batch size in configuration
- Use gradient checkpointing

### 2. Import Errors
- Ensure all repositories are properly installed with `pip install -e .`
- Check Python path includes the library directories
- Verify compatible PyTorch version

### 3. Model Download Issues
- Use alternative download methods (browser, curl)
- Check file integrity with checksums
- Ensure sufficient disk space

## Running with Real Models

Once installed, update your config:

```yaml
# config/semantic_slam.yaml
semantic_segmentation:
  use_mock: false  # Enable real models
  grounded_sam2:
    model_type: "hiera_b+"
    grounding_model: "grounding_dino_swin-b"
    device: "cuda:1"  # Adjust based on your setup
```

Then run:
```bash
python main_semantic.py --config config/semantic_slam.yaml --dataset <path>
```

## Performance Tips

1. **GPU Selection**: Use a dedicated GPU for semantic processing (cuda:1)
2. **Model Selection**: Start with smaller models for testing
3. **Batch Processing**: Process multiple frames together when possible
4. **Memory Management**: Clear cache between sequences

## Expected Performance

With recommended models (hiera_b+ + swin-b):
- Initialization: ~10-15 seconds
- Per-frame processing: ~50ms
- Video tracking: ~30ms propagation
- Total overhead: <100ms per frame