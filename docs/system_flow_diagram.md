# Semantic SLAM System Flow Diagram

## High-Level Architecture

```mermaid
graph TB
    subgraph "Input"
        IMG[RGB Image Stream]
    end
    
    subgraph "Main Process"
        FP[Frame Processor]
        TRACK[Frame Tracker]
        KF_DEC{Keyframe<br/>Decision}
    end
    
    subgraph "MASt3R Model"
        FEAT[Dense Feature<br/>Extraction]
        DEPTH[Monocular<br/>Depth Estimation]
        MATCH[Feature<br/>Matching]
    end
    
    subgraph "Semantic Process"
        GDINO[GroundingDINO<br/>Object Detection]
        SAM2[SAM2<br/>Segmentation]
        INST[Instance<br/>Tracking]
    end
    
    subgraph "Backend Process"
        FG[Factor Graph<br/>Optimization]
        LOOP[Loop Closure<br/>Detection]
        SEM_VER[Semantic<br/>Verification]
    end
    
    subgraph "Shared Memory"
        SK[Shared<br/>Keyframes]
        SSK[Shared Semantic<br/>Keyframes]
        STATE[Shared<br/>States]
    end
    
    subgraph "Outputs"
        POSE[Camera<br/>Poses]
        PC[3D Point<br/>Cloud]
        SEM_PC[Semantic<br/>Point Cloud]
        VIZ[Real-time<br/>Visualization]
    end
    
    %% Main flow
    IMG --> FP
    FP --> FEAT
    FEAT --> TRACK
    TRACK --> KF_DEC
    
    %% Keyframe branch
    KF_DEC -->|Yes| SK
    KF_DEC -->|Yes| GDINO
    KF_DEC -->|No| FP
    
    %% MASt3R outputs
    FEAT --> DEPTH
    DEPTH --> SK
    MATCH --> POSE
    
    %% Semantic flow
    GDINO --> SAM2
    SAM2 --> INST
    INST --> SSK
    
    %% Backend flow
    SK --> FG
    SSK --> SEM_VER
    FG --> LOOP
    LOOP --> SEM_VER
    SEM_VER --> FG
    
    %% State management
    FG --> STATE
    TRACK --> STATE
    STATE --> POSE
    
    %% Final outputs
    SK --> PC
    SSK --> SEM_PC
    PC --> VIZ
    SEM_PC --> VIZ
    
    %% Styling
    classDef input fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef process fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef model fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef semantic fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
    classDef memory fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    classDef output fill:#e0f2f1,stroke:#004d40,stroke-width:2px
    
    class IMG input
    class FP,TRACK,KF_DEC process
    class FEAT,DEPTH,MATCH model
    class GDINO,SAM2,INST semantic
    class SK,SSK,STATE memory
    class POSE,PC,SEM_PC,VIZ output
    class FG,LOOP,SEM_VER process
```

## Detailed Data Flow

### 1. Frame Processing Pipeline
```
RGB Image → Resize & Normalize → Feature Extraction → Dense Descriptors
                                                    ↓
                                              Depth Prediction
                                                    ↓
                                              3D Point Map (X,C)
```

### 2. Tracking and Keyframe Selection
```
Current Frame → Match with Recent Keyframes → Estimate Relative Pose
                                            ↓
                                     Tracking Quality Check
                                            ↓
                            [Good Tracking]     [Poor Tracking]
                                  ↓                    ↓
                            Update Pose          Relocalization
                                  ↓
                          Keyframe Criteria
                         (motion, quality)
                                  ↓
                    [New KF]            [Continue]
                        ↓                    ↓
                 Add to Keyframes      Next Frame
```

### 3. Semantic Processing Pipeline
```
Keyframe Image → GroundingDINO → Bounding Boxes + Labels
                                        ↓
                                   SAM2 Segmentation
                                        ↓
                                   Binary Masks (RLE)
                                        ↓
                                 Instance Tracking
                                        ↓
                            Semantic Keyframe Data:
                            - Masks (RLE encoded)
                            - Labels (object classes)
                            - Confidences
                            - Instance IDs
```

### 4. Backend Optimization
```
Keyframes → Build Factor Graph → Add Visual Factors
                              ↓
                    Loop Closure Detection
                              ↓
                   Semantic Verification
                              ↓
              [Consistent]        [Inconsistent]
                    ↓                   ↓
              Add Loop Factor      Reject Loop
                    ↓
            Global Optimization
                    ↓
          Updated Keyframe Poses
```

### 5. Dense Reconstruction
```
All Keyframes → Project Semantic Masks to 3D → Semantic Point Clouds
                                             ↓
                                    Merge All Keyframes
                                             ↓
                                     Remove Duplicates
                                             ↓
                              Dense Semantic Point Cloud
```

## Key Data Structures

### Frame Data
```
Frame {
    id: frame number
    image: RGB tensor
    T_WC: SE3 pose
    pointmap: (X, C) 3D points + features
}
```

### Semantic Data
```
SemanticData {
    frame_id: original frame number
    keyframe_idx: keyframe index
    masks: RLE-encoded binary masks
    labels: object class names
    confidences: detection scores
    boxes: bounding boxes
}
```

### Output Point Cloud
```
Point {
    position: (x, y, z)
    color: RGB
    semantic_label: object class
    instance_id: unique object ID
    confidence: semantic confidence
}
```

## Process Communication

1. **Main → Backend**: Keyframe queue for optimization
2. **Main → Semantic**: Image queue for segmentation  
3. **Semantic → Main**: Result queue with masks/labels
4. **Backend → Main**: Updated poses via shared memory
5. **All → Visualization**: Real-time updates

This architecture enables:
- Real-time SLAM tracking (30+ FPS)
- Asynchronous semantic processing
- Global consistency through optimization
- Dense semantic reconstruction