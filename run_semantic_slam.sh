#!/bin/bash

python main_semantic.py --dataset /home/ubuntu/restart_from_scratch/datasets/tum/rgbd_dataset_freiburg1_desk/ --config config/semantic_slam.yaml --verbose "$@"
