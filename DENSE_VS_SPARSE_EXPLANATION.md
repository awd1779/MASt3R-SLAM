# Dense vs Sparse Semantic Reconstruction

## Sparse Reconstruction
- **What**: Only projects semantic labels onto SLAM feature points
- **Points**: ~2,000-10,000 points (depends on scene complexity)
- **Coverage**: Only keypoints tracked by SLAM for pose estimation
- **Advantages**: 
  - Fast computation
  - Guaranteed accurate 3D positions
  - Good for real-time visualization
- **Disadvantages**:
  - Very limited coverage
  - Misses most of the segmented regions
  - Not suitable for dense scene understanding

## Dense Reconstruction
- **What**: Projects ALL segmented pixels into 3D space
- **Points**: ~300,000-500,000 points per dataset
- **Coverage**: Every pixel that has:
  1. Valid depth estimation
  2. Semantic segmentation label
- **Advantages**:
  - Complete 3D semantic representation
  - Full object boundaries preserved
  - Suitable for applications like:
    - 3D scene understanding
    - Robotic navigation
    - AR/VR applications
- **Disadvantages**:
  - Higher computational cost
  - Larger file sizes
  - May include more noise in depth estimates

## Example from Your Results:
- **Sparse**: ~63,000 points (only SLAM features)
- **Dense**: 381,574 points (6x more coverage)

### Label Distribution (Dense):
- Chair: 189,164 points (49.6%)
- Table: 158,631 points (41.6%)
- Person: 20,334 points (5.3%)
- Bottle: 13,445 points (3.5%)

The dense reconstruction gives you the full semantic 3D map of the environment!