# dataloader.py - Dataset Handling and I/O

## Overview
The `dataloader.py` module provides a unified interface for loading various datasets and input sources. It handles camera calibration, image preprocessing, and ground truth pose loading for evaluation.

## Class Hierarchy

```
MonocularDataset (Abstract Base)
├── TUMDataset
├── SevenScenesDataset  
├── EuorcDataset
├── ETH3DDataset
├── VideoDataset
├── ImageFolderDataset
├── CameraDataset (Live input)
└── RealsenseDataset
```

## Core Classes

### MonocularDataset (Base Class)
Abstract base class defining the common interface.

```python
class MonocularDataset(ABC):
    def __init__(self, 
                 root: str,
                 scene: str,
                 start_frame: int = 0,
                 end_frame: int = -1,
                 img_skip: int = 1,
                 use_calibration: bool = True,
                 center_principal_point: bool = True,
                 target_img_size: int = 512):
```

#### Key Parameters
- **root**: Dataset root directory
- **scene**: Scene/sequence name
- **img_skip**: Frame subsampling factor
- **use_calibration**: Whether to use camera calibration
- **center_principal_point**: Center the principal point
- **target_img_size**: Resize images to this size

#### Abstract Methods
```python
@abstractmethod
def get_image(self, idx: int) -> np.ndarray:
    """Load image at index"""
    
@abstractmethod
def get_intrinsics(self) -> Intrinsics:
    """Get camera intrinsics"""
    
@abstractmethod
def get_pose(self, idx: int) -> Optional[np.ndarray]:
    """Get ground truth pose if available"""
```

### Intrinsics Class
Manages camera calibration parameters.

```python
class Intrinsics:
    def __init__(self, K, width, height, 
                 distortion_model=None, dist_coef=None):
        self.fx = K[0, 0]
        self.fy = K[1, 1]  
        self.cx = K[0, 2]
        self.cy = K[1, 2]
        self.width = width
        self.height = height
```

#### Key Methods

##### scale()
```python
def scale(self, scale_factor):
    """Scale intrinsics for image resizing"""
    self.fx *= scale_factor
    self.fy *= scale_factor
    self.cx *= scale_factor
    self.cy *= scale_factor
```

##### undistort_image()
```python
def undistort_image(self, img):
    """Remove lens distortion"""
    if self.dist_coef is not None:
        return cv2.undistort(img, self.K, self.dist_coef)
    return img
```

##### center_principal_point()
```python
def center_principal_point(self):
    """Adjust principal point to image center"""
    self.cx = self.width / 2
    self.cy = self.height / 2
```

## Dataset Implementations

### 1. TUMDataset
For TUM RGB-D benchmark sequences.

```python
class TUMDataset(MonocularDataset):
    """TU Munich RGB-D dataset loader"""
```

#### Features
- Loads from `rgb.txt` association file
- Hardcoded intrinsics for Freiburg 1/2/3 cameras
- Supports radial and tangential distortion
- Ground truth poses from `groundtruth.txt`

#### File Structure
```
scene/
├── rgb.txt               # Image list with timestamps
├── groundtruth.txt      # Ground truth trajectory  
└── rgb/                 # RGB images
    ├── 1305031102.175304.png
    └── ...
```

#### Intrinsics
```python
# Freiburg 1
K = [[517.3, 0, 318.6],
     [0, 516.5, 255.3],
     [0, 0, 1]]

# Freiburg 2  
K = [[520.9, 0, 325.1],
     [0, 521.0, 249.7],
     [0, 0, 1]]
```

### 2. SevenScenesDataset
Microsoft 7-Scenes indoor dataset.

```python
class SevenScenesDataset(MonocularDataset):
    """Microsoft 7-Scenes dataset loader"""
```

#### Features
- Natural sorting of color-*.png images
- Fixed intrinsics for all scenes
- Pose files as pose-*.txt (4x4 matrices)

#### File Structure
```
scene/seq-01/
├── frame-000000.color.png
├── frame-000000.pose.txt
└── ...
```

### 3. EuorcDataset
EuRoC MAV visual-inertial dataset.

```python
class EuorcDataset(MonocularDataset):
    """EuRoC MAV dataset loader"""
```

#### Features
- Always undistorts due to high distortion
- Loads calibration from sensor.yaml
- Bilinear interpolation for pose timestamps
- Converts from body to camera frame

#### Calibration Loading
```python
def load_calibration(self):
    """Load from sensor.yaml"""
    with open(calib_path) as f:
        calib = yaml.safe_load(f)
    
    # Extract camera matrix
    K = np.array(calib['cam0']['intrinsics'])
    
    # Distortion coefficients
    dist = np.array(calib['cam0']['distortion_coefficients'])
```

### 4. ETH3DDataset
ETH3D SLAM benchmark dataset.

```python
class ETH3DDataset(MonocularDataset):
    """ETH3D SLAM dataset loader"""
```

#### Features
- Similar format to TUM
- Separate calibration.txt file
- High-quality ground truth

#### Calibration Format
```
# calibration.txt
fx fy cx cy
```

### 5. VideoDataset
Load from video files (MP4, AVI, etc.).

```python
class VideoDataset(MonocularDataset):
    """Video file dataset loader"""
```

#### Features
- Uses torchcodec for fast decoding (optional)
- Falls back to OpenCV if unavailable
- Estimates timestamps from FPS
- No ground truth poses

#### Implementation
```python
def __init__(self, video_path, intrinsics=None):
    # Try torchcodec first
    try:
        import torchcodec
        self.use_torchcodec = True
        self.decoder = torchcodec.VideoDecoder(video_path)
    except:
        # Fall back to OpenCV
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
```

### 6. ImageFolderDataset
Load from directory of images.

```python
class ImageFolderDataset(MonocularDataset):
    """Image folder dataset loader"""
```

#### Features
- Supports common formats (jpg, png, etc.)
- Natural sorting of filenames
- Optional intrinsics from file

### 7. Live Camera Datasets

#### CameraDataset
Generic webcam/USB camera input.

```python
class CameraDataset(MonocularDataset):
    def __init__(self, cam_id=0, width=640, height=480, fps=30):
        self.cap = cv2.VideoCapture(cam_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
```

#### RealsenseDataset
Intel RealSense depth camera (RGB only).

```python
class RealsenseDataset(MonocularDataset):
    """RealSense camera input"""
    
    def __init__(self):
        import pyrealsense2 as rs
        self.pipeline = rs.pipeline()
        
        # Configure streams
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, 
                           rs.format.bgr8, 30)
```

## Common Operations

### Image Preprocessing
All datasets apply consistent preprocessing:

```python
def preprocess_image(self, img):
    """Standard preprocessing pipeline"""
    
    # 1. Undistort if calibrated
    if self.use_calibration and self.has_distortion:
        img = self.intrinsics.undistort_image(img)
    
    # 2. Resize to target size
    if img.shape[0] != self.target_img_size:
        scale = self.target_img_size / img.shape[0]
        img = cv2.resize(img, None, fx=scale, fy=scale)
        self.intrinsics.scale(scale)
    
    # 3. Center principal point if requested
    if self.center_principal_point:
        self.intrinsics.center_principal_point()
    
    return img
```

### Pose Handling
Ground truth poses are handled consistently:

```python
def load_poses(self):
    """Load ground truth trajectory"""
    
    # Load pose file
    poses = self.load_pose_file()
    
    # Convert to common format (4x4 matrices)
    poses = self.convert_pose_format(poses)
    
    # Align timestamps
    self.aligned_poses = self.align_timestamps(
        self.image_timestamps,
        poses
    )
```

## Dataset Creation

### Factory Function
```python
def create_dataset(args):
    """Create dataset from command-line args"""
    
    if args.video:
        return VideoDataset(args.video, args.K)
    
    elif args.cam is not None:
        if args.cam == "realsense":
            return RealsenseDataset()
        else:
            return CameraDataset(args.cam, args.width, args.height)
    
    elif args.img_dir:
        return ImageFolderDataset(args.img_dir, args.K)
    
    else:
        # Standard datasets
        dataset_map = {
            'tum': TUMDataset,
            '7scenes': SevenScenesDataset,
            'euroc': EuorcDataset,
            'eth3d': ETH3DDataset
        }
        
        dataset_class = dataset_map[args.dataset]
        return dataset_class(
            args.root,
            args.scene,
            use_calibration=args.use_calibration
        )
```

## Evaluation Support

### Trajectory Saving
```python
def save_trajectory(dataset, poses, filename):
    """Save trajectory in TUM format"""
    
    with open(filename, 'w') as f:
        for i, pose in enumerate(poses):
            timestamp = dataset.get_timestamp(i)
            
            # Convert to position + quaternion
            position = pose[:3, 3]
            quaternion = matrix_to_quaternion(pose[:3, :3])
            
            f.write(f"{timestamp} {' '.join(map(str, position))} "
                   f"{' '.join(map(str, quaternion))}\n")
```

### Metric Computation
Integration with evaluation tools:

```python
# Save for evaluation
save_trajectory(dataset, estimated_poses, "estimated.txt")
save_trajectory(dataset, ground_truth_poses, "groundtruth.txt")

# Run evaluation (called from bash)
os.system("evo_ape tum groundtruth.txt estimated.txt -va --plot")
```

## Configuration Examples

### Running on TUM Dataset
```python
dataset = TUMDataset(
    root="/path/to/TUM",
    scene="rgbd_dataset_freiburg1_desk",
    use_calibration=True,
    img_skip=1
)
```

### Live Camera with Custom Intrinsics
```python
K = np.array([[520, 0, 320],
              [0, 520, 240],
              [0, 0, 1]])

dataset = CameraDataset(
    cam_id=0,
    width=640,
    height=480
)
dataset.intrinsics = Intrinsics(K, 640, 480)
```

### Video with Unknown Calibration
```python
dataset = VideoDataset(
    video_path="my_video.mp4",
    use_calibration=False  # Will estimate from FOV
)
```

## Performance Considerations

### Efficient Loading
- **Lazy Loading**: Images loaded on demand
- **Caching**: Optional caching for small datasets
- **Parallel Loading**: Can use DataLoader for batching

### Memory Management
- **Resolution Control**: Downsample for speed
- **Format Conversion**: Consistent RGB format
- **Buffer Reuse**: For live camera sources

This module provides flexible dataset handling for various SLAM benchmarks and real-world input sources, with consistent preprocessing and evaluation support.