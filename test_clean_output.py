#!/usr/bin/env python3
"""Test script to demonstrate the cleaner output with logging control."""

import subprocess
import sys

print("Testing MASt3R-SLAM with different verbosity levels...")
print("="*60)

# Test dataset
dataset = "datasets/my"
config = "config/semantic_slam.yaml"

# Test 1: Normal mode (quiet)
print("\n1. Running in QUIET mode (default):")
print("-"*40)
cmd = [sys.executable, "main_semantic.py", "--dataset", dataset, "--config", config, "--no-viz"]
print(f"Command: {' '.join(cmd)}")
print("\nOutput:")
subprocess.run(cmd)

print("\n" + "="*60)

# Test 2: Verbose mode
print("\n2. Running in VERBOSE mode:")
print("-"*40)
cmd = [sys.executable, "main_semantic.py", "--dataset", dataset, "--config", config, "--no-viz", "--verbose"]
print(f"Command: {' '.join(cmd)}")
print("\nOutput:")
subprocess.run(cmd)