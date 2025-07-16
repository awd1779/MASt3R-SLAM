#!/bin/bash

# Fix OpenGL driver path
export LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri

# Fix GLIBC version mismatch by using system libstdc++
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# Alternative: Force software rendering if hardware acceleration fails
# export LIBGL_ALWAYS_SOFTWARE=1

# Run semantic SLAM with visualization
echo "Starting semantic SLAM with visualization..."
python main_semantic.py --dataset datasets/lab_videos/1fps --config config/semantic_slam.yaml --verbose "$@"
