#!/usr/bin/env python3
"""
Extract individual objects from semantic PLY files.
Reads a semantic PLY file and creates separate PLY files for each object/label.
"""

import argparse
import numpy as np
from pathlib import Path
import re
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def read_ply_header(file_path):
    """Read PLY header to get format and property information."""
    with open(file_path, 'rb') as f:
        header_lines = []
        while True:
            line = f.readline().decode('ascii').strip()
            header_lines.append(line)
            if line == 'end_header':
                break
        
        # Parse header information
        vertex_count = 0
        properties = []
        binary_format = False
        
        for line in header_lines:
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            elif line.startswith('format'):
                if 'binary' in line:
                    binary_format = True
            elif line.startswith('property'):
                properties.append(line)
        
        return header_lines, vertex_count, properties, binary_format

def read_ply_data(file_path):
    """Read PLY file and return points with all properties."""
    header_lines, vertex_count, properties, binary_format = read_ply_header(file_path)
    
    if binary_format:
        raise NotImplementedError("Binary PLY format not supported yet. Please use ASCII format.")
    
    # Parse property names and types
    prop_info = []
    for prop in properties:
        parts = prop.split()
        if len(parts) >= 3:
            prop_type = parts[1]
            prop_name = parts[2]
            prop_info.append((prop_name, prop_type))
    
    logger.info(f"Properties found: {[name for name, _ in prop_info]}")
    
    # Read vertex data
    with open(file_path, 'r') as f:
        # Skip header
        for line in f:
            if line.strip() == 'end_header':
                break
        
        # Read vertex data
        vertices = []
        for i in range(vertex_count):
            line = f.readline().strip()
            if not line:
                break
            values = line.split()
            vertices.append(values)
    
    return np.array(vertices), prop_info, header_lines

def extract_label_from_comment(comment_line):
    """Extract label name from PLY comment line."""
    # Look for patterns like "comment label_id_to_name: {1: 'table_kf0_inst1', ...}"
    if 'label_id_to_name' in comment_line:
        # Extract the dictionary part
        dict_match = re.search(r'\{([^}]+)\}', comment_line)
        if dict_match:
            dict_str = dict_match.group(1)
            label_mapping = {}
            
            # Parse key-value pairs
            pairs = dict_str.split(',')
            for pair in pairs:
                if ':' in pair:
                    key_str, value_str = pair.split(':', 1)
                    key = int(key_str.strip())
                    value = value_str.strip().strip("'\"")
                    label_mapping[key] = value
            
            return label_mapping
    return {}

def write_ply_file(file_path, vertices, prop_info, header_template, label_name=None):
    """Write vertices to PLY file with proper header."""
    vertex_count = len(vertices)
    
    with open(file_path, 'w') as f:
        # Write header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        if label_name:
            f.write(f"comment Extracted object: {label_name}\n")
        f.write(f"element vertex {vertex_count}\n")
        
        # Write property definitions
        for prop_name, prop_type in prop_info:
            f.write(f"property {prop_type} {prop_name}\n")
        
        f.write("end_header\n")
        
        # Write vertex data
        for vertex in vertices:
            f.write(" ".join(str(v) for v in vertex) + "\n")

def extract_objects_from_ply(input_file, output_dir, min_points=10):
    """Extract individual objects from semantic PLY file."""
    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Reading PLY file: {input_path}")
    
    # Read PLY data
    vertices, prop_info, header_lines = read_ply_data(input_path)
    
    if len(vertices) == 0:
        logger.error("No vertices found in PLY file")
        return
    
    logger.info(f"Loaded {len(vertices):,} vertices")
    
    # Find label column
    label_col_idx = None
    for i, (prop_name, _) in enumerate(prop_info):
        if prop_name in ['label', 'label_id', 'semantic_label', 'object_id']:
            label_col_idx = i
            logger.info(f"Found label column: {prop_name} at index {i}")
            break
    
    if label_col_idx is None:
        logger.error("No label column found. Expected column names: label, label_id, semantic_label, object_id")
        return
    
    # Extract label mapping from comments if available
    label_mapping = {}
    for line in header_lines:
        if line.startswith('comment'):
            mapping = extract_label_from_comment(line)
            label_mapping.update(mapping)
    
    # Get unique labels
    label_column = vertices[:, label_col_idx].astype(int)
    unique_labels = np.unique(label_column)
    
    logger.info(f"Found {len(unique_labels)} unique labels: {unique_labels}")
    
    if label_mapping:
        logger.info("Label mapping found:")
        for label_id, label_name in sorted(label_mapping.items()):
            count = np.sum(label_column == label_id)
            logger.info(f"  {label_id}: {label_name} ({count:,} points)")
    
    # Extract each object
    extracted_count = 0
    total_points_extracted = 0
    
    for label_id in unique_labels:
        # Skip background/unlabeled (usually 0)
        if label_id == 0:
            continue
            
        # Get points for this label
        mask = label_column == label_id
        object_vertices = vertices[mask]
        
        if len(object_vertices) < min_points:
            logger.info(f"Skipping label {label_id}: only {len(object_vertices)} points (< {min_points})")
            continue
        
        # Determine output filename
        if label_id in label_mapping:
            label_name = label_mapping[label_id]
            # Clean label name for filename
            safe_name = re.sub(r'[^\w\-_.]', '_', label_name)
            output_file = output_path / f"{safe_name}_{label_id:04d}.ply"
        else:
            label_name = f"object_{label_id}"
            output_file = output_path / f"object_{label_id:04d}.ply"
        
        # Write object PLY file
        write_ply_file(output_file, object_vertices, prop_info, header_lines, label_name)
        
        extracted_count += 1
        total_points_extracted += len(object_vertices)
        
        logger.info(f"Extracted {label_name}: {len(object_vertices):,} points -> {output_file}")
    
    # Summary
    logger.info(f"\nExtraction complete:")
    logger.info(f"  Input file: {input_path}")
    logger.info(f"  Output directory: {output_path}")
    logger.info(f"  Objects extracted: {extracted_count}")
    logger.info(f"  Total points extracted: {total_points_extracted:,}")
    logger.info(f"  Minimum points threshold: {min_points}")
    
    # Create summary file
    summary_file = output_path / "extraction_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Semantic Object Extraction Summary\n")
        f.write(f"==================================\n\n")
        f.write(f"Input file: {input_path}\n")
        f.write(f"Output directory: {output_path}\n")
        f.write(f"Total input points: {len(vertices):,}\n")
        f.write(f"Objects extracted: {extracted_count}\n")
        f.write(f"Total points extracted: {total_points_extracted:,}\n")
        f.write(f"Minimum points threshold: {min_points}\n\n")
        
        f.write("Objects:\n")
        for label_id in unique_labels:
            if label_id == 0:
                continue
            mask = label_column == label_id
            count = np.sum(mask)
            if count >= min_points:
                if label_id in label_mapping:
                    label_name = label_mapping[label_id]
                    safe_name = re.sub(r'[^\w\-_.]', '_', label_name)
                    filename = f"{safe_name}_{label_id:04d}.ply"
                else:
                    label_name = f"object_{label_id}"
                    filename = f"object_{label_id:04d}.ply"
                f.write(f"  {filename}: {label_name} ({count:,} points)\n")
    
    logger.info(f"Summary saved to: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="Extract individual objects from semantic PLY files")
    parser.add_argument("input_ply", help="Input semantic PLY file")
    parser.add_argument("output_dir", help="Output directory for individual object PLY files")
    parser.add_argument("--min-points", type=int, default=10, 
                       help="Minimum number of points required to extract an object (default: 10)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Validate input file
    if not Path(args.input_ply).exists():
        logger.error(f"Input file does not exist: {args.input_ply}")
        return 1
    
    try:
        extract_objects_from_ply(args.input_ply, args.output_dir, args.min_points)
        return 0
    except Exception as e:
        logger.error(f"Error extracting objects: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())