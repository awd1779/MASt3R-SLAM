# retrieval_database.py - Loop Closure Detection

## Overview
The `retrieval_database.py` module implements a sophisticated image retrieval system for loop closure detection. It uses HOW (Hierarchical One-vs-the-rest Walking) descriptors with ASMK (Aggregated Selective Match Kernels) for efficient similarity search.

## Architecture

### RetrievalDatabase Class
Main class for managing the retrieval system.

```python
class RetrievalDatabase:
    def __init__(self, mast3r_model, 
                 retrieval_config=None,
                 verbose=False):
        self.mast3r_model = mast3r_model
        self.db = None
        self.query_cnt = 0
        self.verbose = verbose
```

### Core Components

#### 1. Feature Extraction
Extracts global and local features from MASt3R backbone.

```python
def extract_features(self, keyframe):
    """Extract HOW features from keyframe"""
    
    # Get MASt3R backbone features
    backbone_features = keyframe.mast3r_features
    
    # Project through HOW layers
    local_features = self.how_net.proj(backbone_features)
    local_features = self.how_net.attn(local_features)
    
    # L2 normalize
    local_features = F.normalize(local_features, dim=-1)
    
    return local_features
```

#### 2. Visual Vocabulary
Quantizes features to visual words for efficient indexing.

```python
class VisualVocabulary:
    def __init__(self, n_clusters=65536):
        self.n_clusters = n_clusters
        self.centroids = None
        self.inverted_index = defaultdict(list)
    
    def quantize(self, features):
        """Assign features to visual words"""
        # Find nearest centroids
        distances = torch.cdist(features, self.centroids)
        assignments = distances.argmin(dim=1)
        
        return assignments
```

## HOW Descriptor System

### Hierarchical Aggregation
HOW uses multiple levels of spatial aggregation:

```python
def compute_how_descriptor(self, local_features, positions):
    """Compute hierarchical descriptor"""
    
    descriptors = []
    
    # Level 0: Global aggregation
    global_desc = self.aggregate_features(
        local_features, 
        weight_fn=self.global_weight
    )
    descriptors.append(global_desc)
    
    # Level 1: Spatial regions (2x2)
    for region in self.get_spatial_regions(2, 2):
        mask = self.point_in_region(positions, region)
        region_desc = self.aggregate_features(
            local_features[mask],
            weight_fn=self.region_weight
        )
        descriptors.append(region_desc)
    
    # Level 2: Finer regions (4x4)
    for region in self.get_spatial_regions(4, 4):
        mask = self.point_in_region(positions, region)
        if mask.sum() > 0:
            region_desc = self.aggregate_features(
                local_features[mask],
                weight_fn=self.fine_weight
            )
            descriptors.append(region_desc)
    
    # Concatenate all levels
    return torch.cat(descriptors)
```

### ASMK Aggregation
Aggregated Selective Match Kernels for robust matching:

```python
def asmk_aggregate(self, features, visual_words, alpha=3.0):
    """ASMK aggregation with selectivity"""
    
    # Compute residuals
    residuals = features - self.centroids[visual_words]
    
    # Selective function (power normalization)
    selectivity = torch.sign(residuals) * torch.abs(residuals).pow(alpha)
    
    # Aggregate per visual word
    aggregated = {}
    for word_id in torch.unique(visual_words):
        mask = visual_words == word_id
        aggregated[word_id] = selectivity[mask].sum(dim=0)
    
    return aggregated
```

## Database Operations

### 1. Adding Keyframes
Incremental database updates without full rebuild:

```python
def add(self, keyframe):
    """Add keyframe to database"""
    
    # Extract features
    features = self.extract_features(keyframe)
    
    # Quantize to visual words
    visual_words = self.vocabulary.quantize(features)
    
    # Update inverted index
    for i, word in enumerate(visual_words):
        self.inverted_index[word.item()].append({
            'keyframe_id': keyframe.frame_idx,
            'feature_id': i,
            'residual': features[i] - self.centroids[word]
        })
    
    # Store global descriptor
    global_desc = self.compute_how_descriptor(features)
    self.global_descriptors[keyframe.frame_idx] = global_desc
    
    self.n_keyframes += 1
```

### 2. Querying
Efficient similarity search with early termination:

```python
def query(self, query_keyframe, k=5, threshold=0.5):
    """Find k most similar keyframes"""
    
    # Extract query features
    query_features = self.extract_features(query_keyframe)
    query_words = self.vocabulary.quantize(query_features)
    
    # Fast candidate selection using inverted index
    candidates = self.get_candidates(query_words, top_n=k*10)
    
    # Detailed scoring of candidates
    scores = {}
    for candidate_id in candidates:
        score = self.compute_similarity(
            query_features, 
            query_words,
            candidate_id
        )
        
        if score > threshold:
            scores[candidate_id] = score
    
    # Return top-k
    top_k = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    
    return [kf_id for kf_id, score in top_k]
```

### 3. Similarity Computation
Detailed matching between query and database images:

```python
def compute_similarity(self, query_features, query_words, db_keyframe_id):
    """Compute ASMK similarity score"""
    
    similarity = 0.0
    
    # Get database features for this keyframe
    db_entries = self.get_keyframe_entries(db_keyframe_id)
    
    # Match visual words
    for q_idx, q_word in enumerate(query_words):
        # Find matching database features
        db_matches = [e for e in db_entries if e['word'] == q_word]
        
        for db_match in db_matches:
            # Compute match kernel
            kernel_value = self.match_kernel(
                query_features[q_idx],
                db_match['residual'],
                q_word
            )
            
            similarity += kernel_value
    
    # Normalize by geometric mean
    n_query = len(query_words)
    n_db = len(db_entries)
    similarity /= np.sqrt(n_query * n_db)
    
    return similarity
```

## GPU Acceleration

### Custom CUDA Kernels

#### Fast Quantization
```cuda
__global__ void quantize_features_kernel(
    const float* features,      // [N, D]
    const float* centroids,     // [K, D]
    int* assignments,           // [N]
    float* distances,           // [N]
    int N, int K, int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    float min_dist = INFINITY;
    int best_cluster = 0;
    
    // Find nearest centroid
    for (int k = 0; k < K; k++) {
        float dist = 0;
        for (int d = 0; d < D; d++) {
            float diff = features[idx * D + d] - centroids[k * D + d];
            dist += diff * diff;
        }
        
        if (dist < min_dist) {
            min_dist = dist;
            best_cluster = k;
        }
    }
    
    assignments[idx] = best_cluster;
    distances[idx] = min_dist;
}
```

#### Parallel ASMK Scoring
```cuda
__global__ void asmk_scoring_kernel(
    const float* query_residuals,
    const float* db_residuals,
    const int* query_words,
    const int* db_words,
    float* scores,
    int n_query, int n_db, int dim
) {
    int q_idx = blockIdx.x;
    int db_idx = threadIdx.x;
    
    if (q_idx >= n_query || db_idx >= n_db) return;
    
    // Check if visual words match
    if (query_words[q_idx] != db_words[db_idx]) {
        scores[q_idx * n_db + db_idx] = 0;
        return;
    }
    
    // Compute match kernel
    float kernel = 0;
    for (int d = 0; d < dim; d++) {
        float q_r = query_residuals[q_idx * dim + d];
        float db_r = db_residuals[db_idx * dim + d];
        kernel += q_r * db_r;  // Dot product
    }
    
    // Apply selectivity function
    kernel = fmax(0, kernel);  // ReLU
    kernel = powf(kernel, 3.0);  // Power normalization
    
    scores[q_idx * n_db + db_idx] = kernel;
}
```

## Advanced Features

### 1. Geometric Verification
Validates retrieval results with spatial consistency:

```python
def geometric_verification(self, query_kf, retrieved_kf):
    """Verify retrieval with geometric constraints"""
    
    # Match local features
    matches_q, matches_r = self.match_local_features(
        query_kf, retrieved_kf
    )
    
    if len(matches_q) < 8:
        return False, None
    
    # Estimate fundamental matrix
    F, inliers = cv2.findFundamentalMat(
        matches_q, matches_r,
        cv2.RANSAC, 3.0
    )
    
    inlier_ratio = np.sum(inliers) / len(matches_q)
    
    return inlier_ratio > 0.3, F
```

### 2. Temporal Filtering
Prevents matching recent keyframes:

```python
def filter_temporal_neighbors(self, query_id, candidates, min_distance=50):
    """Remove temporally close candidates"""
    
    filtered = []
    for candidate_id in candidates:
        temporal_dist = abs(candidate_id - query_id)
        
        if temporal_dist >= min_distance:
            filtered.append(candidate_id)
    
    return filtered
```

### 3. Multi-Scale Features
Extracts features at multiple scales:

```python
def extract_multiscale_features(self, keyframe):
    """Extract features at multiple scales"""
    
    scales = [0.5, 1.0, 1.5]
    all_features = []
    
    for scale in scales:
        # Resize image
        scaled_img = resize_image(keyframe.img, scale)
        
        # Extract features
        features = self.extract_features_from_image(scaled_img)
        
        # Rescale positions
        features = rescale_feature_positions(features, 1.0/scale)
        
        all_features.append(features)
    
    # Combine features
    return torch.cat(all_features)
```

## Integration with SLAM

### Loop Closure Pipeline
```python
# In global_opt.py
def detect_loop_closures(self, current_kf_idx):
    """Main loop closure detection"""
    
    # Get current keyframe
    current_kf = self.shared_keyframes.get(current_kf_idx)
    
    # Query database (before adding current)
    candidates = self.retrieval_db.query(
        current_kf, 
        k=self.retrieval_k,
        threshold=self.retrieval_thresh
    )
    
    # Add current keyframe to database
    self.retrieval_db.add(current_kf)
    
    # Process candidates
    loop_closures = []
    for candidate_idx in candidates:
        # Geometric verification
        candidate_kf = self.shared_keyframes.get(candidate_idx)
        
        valid, relative_pose = self.verify_loop_closure(
            current_kf, candidate_kf
        )
        
        if valid:
            loop_closures.append({
                'from': current_kf_idx,
                'to': candidate_idx,
                'pose': relative_pose
            })
    
    return loop_closures
```

## Memory Management

### Efficient Storage
```python
class CompactInvertedIndex:
    """Memory-efficient inverted index"""
    
    def __init__(self):
        # Use array instead of list for each posting
        self.postings = {}
        self.posting_sizes = {}
        self.buffer_size = 1000
    
    def add_posting(self, word_id, keyframe_id, feature_id):
        if word_id not in self.postings:
            # Pre-allocate buffer
            self.postings[word_id] = np.zeros(
                (self.buffer_size, 2), dtype=np.int32
            )
            self.posting_sizes[word_id] = 0
        
        # Add to buffer
        idx = self.posting_sizes[word_id]
        self.postings[word_id][idx] = [keyframe_id, feature_id]
        self.posting_sizes[word_id] += 1
        
        # Resize if needed
        if idx >= self.buffer_size - 1:
            self.resize_buffer(word_id)
```

## Configuration

### Retrieval Settings
```yaml
retrieval:
  vocab_size: 65536          # Visual vocabulary size
  how_layers: 3              # HOW hierarchy levels
  asmk_alpha: 3.0           # ASMK selectivity
  min_temporal_dist: 50     # Frames between matches
  geometric_verification: true
```

### Performance Tuning
```yaml
retrieval:
  max_features_per_image: 1000   # Limit features
  quantization_batch_size: 100   # GPU batch size
  candidate_list_factor: 10      # Candidates per query
  early_termination_thresh: 0.8  # Stop if good match found
```

## Debugging Tools

### Database Statistics
```python
def get_database_stats(self):
    """Compute database statistics"""
    
    stats = {
        'n_keyframes': self.n_keyframes,
        'n_visual_words': len(self.inverted_index),
        'avg_features_per_kf': np.mean([
            len(self.get_keyframe_entries(i)) 
            for i in range(self.n_keyframes)
        ]),
        'index_memory_mb': self.compute_memory_usage() / 1024**2,
        'vocab_usage': self.compute_vocabulary_usage()
    }
    
    return stats
```

### Retrieval Visualization
```python
def visualize_retrieval_results(query_img, retrieved_imgs, scores):
    """Visualize retrieval results"""
    
    fig, axes = plt.subplots(1, len(retrieved_imgs) + 1)
    
    # Query image
    axes[0].imshow(query_img)
    axes[0].set_title('Query')
    
    # Retrieved images
    for i, (img, score) in enumerate(zip(retrieved_imgs, scores)):
        axes[i+1].imshow(img)
        axes[i+1].set_title(f'Score: {score:.3f}')
    
    plt.tight_layout()
    return fig
```

This module enables robust loop closure detection through efficient image retrieval, helping maintain global consistency in long-term SLAM operation.