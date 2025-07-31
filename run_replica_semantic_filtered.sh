#!/bin/bash
# Run script for testing label filtering fix

# Set the dataset
DATASET="datasets/room_0"

# Run with the new filtered configuration
echo "Running semantic SLAM with label filtering..."
python main_semantic_tracked_3d.py \
    --dataset $DATASET \
    --config config/replica_semantic_auto_tracked_3d_global_improved_filtered.yaml \
    --save-as logs/tracked_3d_global_improved_filtered/room_0 \
    --verbose \
    --no-viz

# Check if completed successfully
if [ $? -eq 0 ]; then
    echo "Semantic SLAM completed successfully!"
    
    # Run the test script to analyze results
    echo ""
    echo "Analyzing results..."
    # TODO: Create test_label_filtering.py script
    # python test_label_filtering.py logs/tracked_3d_global_improved_filtered
    
    # Run bbox visualization for filtered results
    echo ""
    echo "Creating filtered bbox visualization..."
    # TODO: Create visualize_3d_bboxes_filtered.py script
    # python visualize_3d_bboxes_filtered.py \
    #     --ply logs/tracked_3d_global_improved_filtered/room_0/room_0_semantic_dense_tracked_3d.ply \
    #     --tracking-json logs/tracked_3d_global_improved_filtered/room_0/room_0_semantic_dense_tracked_3d.tracking.json \
    #     --output logs/tracked_3d_global_improved_filtered/room_0/bboxes_furniture_filtered.ply \
    #     --filter chair sofa blanket table \
    #     --min-points 100
    
    echo ""
    echo "Done! Check the results in logs/tracked_3d_global_improved_filtered/"
else
    echo "Semantic SLAM failed!"
    exit 1
fi
