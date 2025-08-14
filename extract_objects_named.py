#!/usr/bin/env python3
"""
Extract individual objects from semantic PLY files with proper names.
"""

import sys
import struct
import numpy as np
import json
from pathlib import Path
import re

def clean_filename(name):
    """Clean object name for use as filename."""
    # Remove 'a ' prefix and clean up characters
    name = re.sub(r'^a\s+', '', name)
    name = re.sub(r'[^\w\-_.]', '_', name)
    return name

def extract_objects_with_names(input_ply, tracking_json, output_dir, min_points=10):
    """Extract objects from semantic PLY file with proper names."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading: {input_ply}")
    print(f"Using tracking info: {tracking_json}")
    
    # Load label mapping
    with open(tracking_json, 'r') as f:
        tracking_data = json.load(f)
    
    label_to_name = tracking_data['label_to_name']
    print(f"Found {len(label_to_name)} label mappings")
    
    # Read the PLY file
    with open(input_ply, 'rb') as f:
        # Read header
        header = []
        vertex_count = 0
        while True:
            line = f.readline().decode('ascii')
            header.append(line)
            if 'element vertex' in line:
                vertex_count = int(line.split()[-1])
            if 'end_header' in line:
                break
        
        print(f"Found {vertex_count:,} vertices")
        
        # Read binary data (float x, float y, float z, uchar r, uchar g, uchar b, int label)
        format_string = '<fffBBBi'  # little endian
        vertex_size = struct.calcsize(format_string)
        
        vertices = []
        for i in range(vertex_count):
            data = f.read(vertex_size)
            if len(data) < vertex_size:
                break
            vertex = struct.unpack(format_string, data)
            vertices.append(vertex)
    
    vertices = np.array(vertices)
    print(f"Loaded {len(vertices):,} vertices")
    
    # Extract label column (last column, index 6)
    labels = vertices[:, 6].astype(int)
    unique_labels = np.unique(labels)
    
    print(f"Found {len(unique_labels)} unique labels")
    
    # Extract each object
    extracted = 0
    summary = []
    
    for label_id in unique_labels:
        if label_id == 0:  # Skip background
            continue
        
        # Get vertices for this label
        mask = labels == label_id
        obj_vertices = vertices[mask]
        
        if len(obj_vertices) < min_points:
            continue
        
        # Get object name
        label_str = str(label_id)
        if label_str in label_to_name:
            object_name = label_to_name[label_str]
            clean_name = clean_filename(object_name)
            output_file = output_path / f"{clean_name}_{label_id:04d}.ply"
        else:
            object_name = f"unknown_object_{label_id}"
            output_file = output_path / f"unknown_{label_id:04d}.ply"
        
        # Write object PLY
        with open(output_file, 'wb') as f:
            # Write header
            header_str = f"""ply
format binary_little_endian 1.0
comment {object_name}
element vertex {len(obj_vertices)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property int label
end_header
"""
            f.write(header_str.encode('ascii'))
            
            # Write binary data
            for v in obj_vertices:
                x, y, z = v[0], v[1], v[2]
                r, g, b = int(v[3]), int(v[4]), int(v[5])
                label = int(v[6])
                packed = struct.pack('<fffBBBi', x, y, z, r, g, b, label)
                f.write(packed)
        
        extracted += 1
        summary.append({
            'label_id': label_id,
            'name': object_name,
            'filename': output_file.name,
            'points': len(obj_vertices)
        })
        
        print(f"Extracted {object_name}: {len(obj_vertices):,} points -> {output_file.name}")
    
    # Write summary
    summary_file = output_path / "object_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Extracted Objects Summary\n")
        f.write("========================\n\n")
        for obj in sorted(summary, key=lambda x: x['points'], reverse=True):
            f.write(f"{obj['filename']:<50} | {obj['name']:<30} | {obj['points']:>8,} points\n")
    
    print(f"\nExtracted {extracted} objects to {output_path}/")
    print(f"Summary saved to: {summary_file}")
    
    # Show top objects by size
    print(f"\nTop 10 largest objects:")
    for obj in sorted(summary, key=lambda x: x['points'], reverse=True)[:10]:
        print(f"  {obj['name']}: {obj['points']:,} points ({obj['filename']})")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python extract_objects_named.py input.ply tracking.json output_dir/ [min_points]")
        print("Example: python extract_objects_named.py semantic.ply tracking.json objects/ 100")
        sys.exit(1)
    
    input_ply = sys.argv[1]
    tracking_json = sys.argv[2]
    output_dir = sys.argv[3]
    min_points = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    
    extract_objects_with_names(input_ply, tracking_json, output_dir, min_points)