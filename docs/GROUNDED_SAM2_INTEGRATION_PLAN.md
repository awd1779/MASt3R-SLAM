# Grounded-SAM2 + MAST3R-SLAM Integration Plan v2

### **Overview**

This document outlines a comprehensive plan for integrating Grounded-SAM2 semantic segmentation into the MAST3R-SLAM framework. The goal is to create a fully segmented, globally consistent 3D dense map while maintaining real-time performance. This version includes an experimental plan for validating key architectural decisions.

### **Architecture Overview**

The architecture uses a **decoupled, parallel processing** model. Grounded-SAM2 runs as a separate process, consuming in-memory image data and producing semantic outputs without blocking the main SLAM thread. This ensures MAST3R-SLAM's core performance remains unaffected, with data fused asynchronously.

### **High-Level Phase Summary**

* **Phase 1: Core Integration & Data Pipeline.** Establish the parallel process architecture and the communication pipeline using in-memory data transfer, making the system agnostic to source file formats (PNG, JPG, etc.).
* **Phase 2: 3D Semantic Fusion.** Develop a highly performant module to project 2D segmentation masks into the 3D map, focusing on vectorized, GPU-accelerated fusion and robust temporal track management.
* **Phase 3: Backend Integration & Consistency.** Enhance the SLAM backend to enforce semantic consistency, starting with semantic verification for loop closures and progressing to direct integration into the optimization graph.
* **Phase 4: Visualization, API & User Interaction.** Build the necessary tools for user interaction, including a semantic map viewer and a high-level API for dynamic, language-based querying.

### **Technical Implementation Details**

#### **Phase 1: Core Integration**

* **1.1 Extended Data Structures:**
    * Extend `Frame` to `SemanticFrame` to hold RLE-compressed masks, instance IDs, track IDs, and confidence scores.
    * Implement `SharedSemanticKeyframes` using `SharedMemory` for efficient bulk data transfer.
* **1.2 Semantic Process & Data Handling:**
    * Create a `SemanticProcessor` class running in a separate process.
    * This process initializes `GroundedSAM2VideoPredictor` and consumes image data from a queue.
    * **Data Format Handling**: The `SemanticProcessor` will receive raw image data (e.g., NumPy arrays) directly, not file paths. This makes the pipeline independent of the original dataset's file format (`.png`, `.jpg`, etc.).
    ```python
    # In SemanticProcessor.run()
    def run(self):
        while True:
            frame_data = self.frame_queue.get()
            if frame_data is None: break
    
            # Directly use the in-memory NumPy array
            image_numpy_array = frame_data['img']
    
            # The predictor operates on the array, not a file
            masks, tracks = self.process_frame(image_numpy_array)
            # ...
    ```
* **1.3 Main Pipeline Integration & Synchronization:**
    * The main SLAM process loads or captures a frame, resulting in an in-memory image array.
    * This array is put onto the `frame_queue` for the `SemanticProcessor`.
    ```python
    # In main.py loop
    # frame.img is an in-memory NumPy array [H, W, 3]
    frame_queue.put({'img': frame.img.copy(), ...})
    ```
    * **Asynchronicity Handling**: The main process will handle delayed semantic data. Keyframes may exist briefly without semantics; downstream modules must be robust to this.

#### **Phase 2: 3D Semantic Fusion**

* **2.1 Performant Semantic Pointmap Fusion:**
    * **Critical Task**: Implement a highly optimized, vectorized function for semantic fusion. **Avoid Python `for` loops over pixels.**
    * **Method**: Use **vectorized NumPy** operations for 3D reprojection. For maximum performance, this step can be offloaded to a **custom CUDA kernel**.
* **2.2 Temporal Track Management:**
    * Implement `GlobalTrackManager` using a `networkx` graph and a union-find data structure to merge and maintain canonical instance IDs across loop closures.

#### **Phase 3: Backend Integration**

* **3.1 Loop Closure Semantic Verification (Priority 1):**
    * Use semantics as a robust check for loop closure candidates. Implement `verify_semantic_consistency` to compare the semantic histogram and instance ID overlap between two keyframes to reject false positives.
* **3.2 Semantic Consistency Factors (Priority 2):**
    * **Advanced Task**: Implement a `SemanticFactor` class for the pose graph optimization. This is an experimental feature that should be disabled by default.

#### **Phase 4: Visualization & API**

* **4.1 Semantic Visualization:**
    * Extend the `Viewer3D` to color the point cloud by semantic or instance labels, with UI controls for filtering and selection.
* **4.2 Dynamic Query API:**
    * Implement `SemanticSLAMAPI` for high-level map interaction.
    * `find_objects(text_query)`: Passes the query directly to Grounded-SAM's grounding model.
    * `add_new_prompt(text_prompt)`: Dynamically adds new object types for the `SemanticProcessor` to find.

### **Experimental Validation Plan**

To justify the choice of `Grounded-SAM` over alternative front-ends, a comparative experiment will be conducted.

* **Baseline (Segment-then-Classify):** Implement a front-end that uses SAM to generate class-agnostic masks, followed by a CLIP encoder to classify each mask. This mirrors the approach in related works.
* **Proposed (Prompt-and-Segment):** The `Grounded-SAM` architecture as defined in this plan.

#### **Hypotheses**

1.  **Efficiency:** The `Grounded-SAM` approach will have lower computational overhead (higher FPS, lower GPU load).
2.  **Accuracy:** The `Grounded-SAM` approach will yield a higher quality 3D semantic map (higher mIoU) by reducing initial perception errors.

#### **Metrics**

* **Quantitative:** End-to-end system FPS, GPU memory usage, 3D semantic segmentation mIoU, and object-level detection/miss/false-positive rates.
* **Qualitative:** Side-by-side visualizations of challenging scenes to demonstrate the difference in segmentation quality.

### **Configuration File**

```yaml
# config/semantic_slam.yaml
inherit: "base.yaml"

semantic_segmentation:
  enabled: true
  
  # Perception front-end choice for experiments
  # Options: "grounded_sam" (proposed), "sam_clip" (baseline)
  frontend_type: "grounded_sam"

  # Grounded-SAM2 settings
  grounded_sam2:
    model_type: "hiera_b+"
    grounding_model: "grounding_dino_swin-b"
    device: "cuda:1"
    confidence_threshold: 0.35
  
  # Initial Vocabulary (can be updated dynamically via API)
  initial_vocabulary: ["person", "chair", "table", "car", "bottle"]
     
  # Backend settings
  backend:
    enable_semantic_verification: true
    enable_semantic_factor: false
    semantic_weight: 0.05
```

### **Key Design Decisions**

1.  **Decoupled Process Architecture**: Keeps SLAM performance independent of semantic processing load.
2.  **In-Memory Data Pipeline**: Makes the system agnostic to source file formats (PNG, JPG) and avoids I/O bottlenecks.
3.  **Vectorized 3D Fusion**: Critical for achieving real-time performance in the data fusion step.
4.  **Semantics for Robustness**: Using semantics first to verify loop closures before using it as a weaker constraint in the main optimization.
5.  **Dynamic Vocabulary**: The API allows for interactive, open-vocabulary mapping that is not limited to a predefined list.

