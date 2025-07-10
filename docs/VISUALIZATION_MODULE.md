# visualization.py - Real-time 3D Visualization

## Overview
The `visualization.py` module provides an interactive 3D viewer for real-time visualization of the SLAM reconstruction. It uses ModernGL for efficient GPU rendering and imgui for the user interface.

## Core Components

### Viewer3D Class
Main visualization class managing the 3D rendering pipeline.

```python
class Viewer3D:
    def __init__(self,
                 camera_z=-3,
                 camera_y=2, 
                 up=(0, -1, 0),
                 fov=50,
                 pointcloud_size=0.02,
                 keyframes=None,
                 states=None,
                 processes=3,
                 filter_depth=5,
                 confidence_thresh=1.5,
                 target_img_size=3,
                 visualize=True,
                 offscreen=False,
                 use_mask=False,
                 mode="surfel"):
```

#### Key Parameters
- **camera_z/y**: Initial camera position
- **fov**: Field of view in degrees
- **pointcloud_size**: Point/surfel size
- **confidence_thresh**: Minimum confidence to display points
- **mode**: Rendering mode ("points", "surfel", "triangle")
- **offscreen**: Headless rendering for recording

### Camera Control

#### Camera Class
Manages viewer camera with smooth motion.

```python
class Camera:
    def __init__(self, position, target, up, fov):
        self.position = np.array(position)
        self.target = np.array(target)
        self.up = np.array(up)
        self.fov = fov
```

#### Interactive Controls
- **Mouse Drag**: Rotate camera around target
- **Scroll**: Zoom in/out
- **WASD**: Move camera position
- **Arrow Keys**: Rotate view
- **Space**: Reset view
- **F**: Follow latest keyframe

### Rendering Pipeline

#### 1. Point Cloud Rendering
Renders 3D points with confidence-based coloring.

```python
def create_pointcloud_vao(self, keyframe):
    """Create vertex array object for point cloud"""
    
    # Get 3D points and colors
    points = keyframe.get_pointmap()
    colors = keyframe.img.reshape(-1, 3)
    confidence = keyframe.C.flatten()
    
    # Filter by confidence
    mask = confidence > self.confidence_thresh
    points = points[mask]
    colors = colors[mask]
    
    # Create vertex buffer
    vbo = self.ctx.buffer(np.hstack([points, colors]).astype('f4'))
    
    # Create VAO
    vao = self.ctx.vertex_array(
        self.point_program,
        [(vbo, '3f 3f', 'in_position', 'in_color')]
    )
    
    return vao
```

#### 2. Surfel Rendering
Oriented discs for better surface representation.

```python
def render_surfels(self, keyframe):
    """Render points as oriented discs"""
    
    # Compute normals from depth gradients
    normals = compute_normals(keyframe.X_canon)
    
    # Pack data for shader
    surfel_data = np.hstack([
        points,           # 3D position
        normals,          # Surface normal
        colors,           # RGB color
        radii            # Disc radius
    ])
    
    # Render with geometry shader
    self.surfel_program['u_mvp'] = mvp_matrix
    self.surfel_vao.render(moderngl.POINTS)
```

#### 3. Camera Frustum Display
Shows keyframe poses as wireframe pyramids.

```python
def create_frustum_mesh(self, pose, intrinsics, scale=0.1):
    """Create frustum mesh for camera pose"""
    
    # Image plane corners in camera space
    corners = np.array([
        [0, 0, 1],
        [intrinsics.width, 0, 1],
        [intrinsics.width, intrinsics.height, 1],
        [0, intrinsics.height, 1]
    ])
    
    # Unproject to 3D
    rays = intrinsics.unproject(corners) * scale
    
    # Transform to world space
    world_corners = pose @ np.hstack([rays, np.ones((4, 1))]).T
    
    # Create wireframe
    vertices = [pose.translation, *world_corners[:3]]
    indices = [(0,1), (0,2), (0,3), (0,4), 
               (1,2), (2,3), (3,4), (4,1)]
    
    return vertices, indices
```

### GUI Components

#### Control Panel
Interactive parameter adjustment using imgui.

```python
def render_gui(self):
    """Render GUI controls"""
    
    imgui.begin("SLAM Controls")
    
    # System controls
    if imgui.button("Pause" if not self.paused else "Resume"):
        self.toggle_pause()
    
    # Visualization settings
    imgui.text("Visualization")
    
    # Confidence threshold slider
    changed, self.confidence_thresh = imgui.slider_float(
        "Confidence", 
        self.confidence_thresh,
        0.0, 10.0
    )
    
    # Point size
    changed, self.pointcloud_size = imgui.slider_float(
        "Point Size",
        self.pointcloud_size,
        0.001, 0.1
    )
    
    # Render mode
    if imgui.combo("Mode", self.mode, ["points", "surfel", "triangle"]):
        self.update_render_mode()
    
    imgui.end()
```

#### Information Display
Shows system statistics.

```python
def render_stats(self):
    """Display SLAM statistics"""
    
    imgui.begin("Statistics")
    
    # Frame counts
    imgui.text(f"Frames: {self.states.n_frames}")
    imgui.text(f"Keyframes: {self.states.n_keyframes}")
    imgui.text(f"Failed Tracks: {self.states.n_failed_tracks}")
    
    # Performance
    imgui.text(f"FPS: {self.fps:.1f}")
    imgui.text(f"Points: {self.total_points:,}")
    
    # Current mode
    mode_str = ["INIT", "TRACKING", "RELOC", "TERMINATED"]
    imgui.text(f"Mode: {mode_str[self.states.mode]}")
    
    imgui.end()
```

### Shader Programs

#### Point Shader
Basic point rendering with per-vertex color.

```glsl
// Vertex shader
#version 330
in vec3 in_position;
in vec3 in_color;
out vec3 v_color;
uniform mat4 u_mvp;

void main() {
    gl_Position = u_mvp * vec4(in_position, 1.0);
    gl_PointSize = u_point_size / gl_Position.w;
    v_color = in_color;
}

// Fragment shader  
#version 330
in vec3 v_color;
out vec4 f_color;

void main() {
    f_color = vec4(v_color, 1.0);
}
```

#### Surfel Shader
Oriented disc rendering with shading.

```glsl
// Geometry shader for surfel expansion
#version 330
layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

uniform mat4 u_mvp;
uniform mat4 u_mv;
uniform float u_radius;

void main() {
    vec4 center = gl_in[0].gl_Position;
    vec3 normal = v_normal[0];
    
    // Create oriented quad
    vec3 right = cross(normal, vec3(0,1,0));
    vec3 up = cross(right, normal);
    
    // Emit vertices
    for(int i = 0; i < 4; i++) {
        vec2 offset = quad_offsets[i] * u_radius;
        vec3 pos = center.xyz + right * offset.x + up * offset.y;
        gl_Position = u_mvp * vec4(pos, 1.0);
        emit_vertex();
    }
}
```

### Multi-threaded Updates

#### Dirty Keyframe System
Efficient updates for modified keyframes.

```python
def update_keyframes(self):
    """Update visualization for dirty keyframes"""
    
    # Get modified keyframes
    dirty_kfs = self.keyframes.get_dirty_keyframes()
    
    if len(dirty_kfs) > 0:
        for kf in dirty_kfs:
            idx = kf.frame_idx
            
            # Update point cloud
            if idx in self.point_vaos:
                self.point_vaos[idx].release()
            
            self.point_vaos[idx] = self.create_pointcloud_vao(kf)
            
            # Update frustum
            self.update_frustum(idx, kf.pose)
```

### Rendering Optimizations

#### 1. Frustum Culling
Only render visible keyframes.

```python
def cull_keyframes(self, view_frustum):
    """Frustum culling for keyframes"""
    
    visible_kfs = []
    for kf_idx in range(self.keyframes.n_keyframes):
        # Bounding sphere test
        sphere = self.compute_bounding_sphere(kf_idx)
        if view_frustum.intersects_sphere(sphere):
            visible_kfs.append(kf_idx)
    
    return visible_kfs
```

#### 2. Level of Detail
Adjust rendering quality by distance.

```python
def get_lod_level(self, distance):
    """Determine level of detail"""
    
    if distance < 5:
        return 0  # Full quality
    elif distance < 20:
        return 1  # Medium quality
    else:
        return 2  # Low quality
```

#### 3. Point Cloud Decimation
Reduce points for distant keyframes.

```python
def decimate_pointcloud(self, points, confidence, level):
    """Downsample point cloud by level"""
    
    if level == 0:
        return points, confidence
    
    # Grid-based decimation
    voxel_size = 0.01 * (2 ** level)
    indices = voxelize(points, voxel_size)
    
    return points[indices], confidence[indices]
```

### Recording and Export

#### Video Recording
Save visualization as video.

```python
def start_recording(self, filename, fps=30):
    """Start recording frames"""
    
    self.recording = True
    self.video_writer = cv2.VideoWriter(
        filename,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (self.width, self.height)
    )
```

#### Screenshot Capture
```python
def capture_screenshot(self):
    """Capture current frame"""
    
    # Read framebuffer
    pixels = self.ctx.screen.read()
    img = Image.frombytes('RGB', (self.width, self.height), pixels)
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    
    # Save with timestamp
    filename = f"screenshot_{time.time():.0f}.png"
    img.save(filename)
```

## Usage Examples

### Basic Visualization
```python
# Create viewer
viewer = Viewer3D(
    keyframes=shared_keyframes,
    states=shared_states,
    confidence_thresh=1.5
)

# Run visualization loop
while viewer.running:
    viewer.update()
    viewer.render()
```

### Custom Rendering
```python
# Add custom geometry
viewer.add_mesh(
    vertices=my_vertices,
    indices=my_indices,
    colors=my_colors
)

# Custom shader
viewer.load_shader(
    vertex_src=my_vertex_shader,
    fragment_src=my_fragment_shader
)
```

### Headless Rendering
```python
# Create offscreen viewer
viewer = Viewer3D(
    keyframes=shared_keyframes,
    offscreen=True,
    width=1920,
    height=1080
)

# Render and save
viewer.render()
viewer.save_image("output.png")
```

## Configuration

### Performance Settings
```yaml
visualization:
  confidence_thresh: 1.5    # Filter low-confidence points
  filter_depth: 5          # Maximum depth to render
  pointcloud_size: 0.02    # Point display size
  render_every: 3          # Skip frames for performance
```

### Quality Settings
```yaml
visualization:
  mode: "surfel"           # Better surface representation
  antialiasing: 4          # MSAA samples
  shadow_mapping: true     # Enable shadows
  ambient_occlusion: true  # Screen-space AO
```

## Integration with SLAM

### Communication
- Reads keyframes from shared memory
- No direct modification of SLAM data
- Dirty flag system for updates

### Synchronization
- Runs in separate process
- Non-blocking visualization
- Graceful handling of data updates

This module provides an essential tool for understanding and debugging the SLAM system's behavior in real-time, with interactive controls for detailed inspection of the reconstruction.