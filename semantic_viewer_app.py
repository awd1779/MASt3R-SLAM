#!/usr/bin/env python3
"""
Gradio Web App for MASt3R-SLAM Semantic Visualization
Access via web browser to visualize and query semantic point clouds
"""

import gradio as gr
import numpy as np
from pathlib import Path
from plyfile import PlyData
import plotly.graph_objects as go
import json
import pandas as pd
import colorsys


class SemanticViewerApp:
    def __init__(self):
        self.ply_data = None
        self.points = None
        self.colors = None
        self.semantic_ids = None
        self.semantic_map = {}
        self.current_ply_path = None
        
    def load_ply(self, ply_path):
        """Load PLY file and extract data."""
        if not ply_path or not Path(ply_path).exists():
            return "Please provide a valid PLY file path", None, None
        
        try:
            self.current_ply_path = ply_path
            self.ply_data = PlyData.read(ply_path)
            vertices = self.ply_data['vertex']
            
            # Extract points
            self.points = np.vstack([
                vertices['x'],
                vertices['y'],
                vertices['z']
            ]).T
            
            # Extract colors
            self.colors = np.vstack([
                vertices['red'],
                vertices['green'],
                vertices['blue']
            ]).T
            
            # Extract semantic IDs
            self.semantic_ids = None
            vertex_dtype = vertices.data.dtype
            for prop in ['semantic_id', 'quality']:
                if prop in vertex_dtype.names:
                    self.semantic_ids = vertices.data[prop]
                    break
            
            if self.semantic_ids is None:
                self.semantic_ids = np.zeros(len(self.points), dtype=np.int32)
            
            # Load semantic map from comments
            self.semantic_map = {0: "background"}
            if hasattr(self.ply_data, 'comments'):
                in_map = False
                for comment in self.ply_data.comments:
                    if comment == 'label_map_start':
                        in_map = True
                    elif comment == 'label_map_end':
                        in_map = False
                    elif in_map and comment.startswith('label_id'):
                        parts = comment.split(':', 1)
                        if len(parts) == 2:
                            label_id = int(parts[0].split()[-1])
                            self.semantic_map[label_id] = parts[1].strip()
            
            stats = f"Loaded {len(self.points):,} points with {len(np.unique(self.semantic_ids))} unique classes"
            return stats, self.get_statistics_df(), self.visualize_3d()
            
        except Exception as e:
            return f"Error loading PLY: {str(e)}", None, None
    
    def get_statistics_df(self):
        """Get statistics as a DataFrame."""
        if self.semantic_ids is None:
            return None
        
        unique_ids, counts = np.unique(self.semantic_ids, return_counts=True)
        
        data = []
        for label_id, count in zip(unique_ids, counts):
            name = self.semantic_map.get(label_id, f"unknown_{label_id}")
            percentage = count / len(self.points) * 100
            data.append({
                'ID': label_id,
                'Class Name': name,
                'Point Count': f"{count:,}",
                'Percentage': f"{percentage:.1f}%"
            })
        
        df = pd.DataFrame(data)
        return df.sort_values('Point Count', ascending=False)
    
    def generate_semantic_colors(self):
        """Generate distinct colors for semantic classes."""
        unique_ids = np.unique(self.semantic_ids)
        colors = np.zeros((len(self.points), 3))
        
        for i, label_id in enumerate(unique_ids):
            if label_id == 0:
                color = [50, 50, 50]  # Dark gray for background
            else:
                hue = (i - 1) * 360.0 / max(1, len(unique_ids) - 1)
                rgb = colorsys.hsv_to_rgb(hue/360.0, 0.8, 0.9)
                color = [int(c * 255) for c in rgb]
            
            mask = self.semantic_ids == label_id
            colors[mask] = color
        
        return colors.astype(np.uint8)
    
    def visualize_3d(self, class_filter=None, color_mode='rgb', max_points=50000):
        """Create 3D visualization with Plotly."""
        if self.points is None:
            return None
        
        # Apply class filter if specified
        if class_filter and class_filter != "All":
            mask = None
            for label_id, name in self.semantic_map.items():
                if class_filter.lower() in name.lower():
                    if mask is None:
                        mask = self.semantic_ids == label_id
                    else:
                        mask |= self.semantic_ids == label_id
            
            if mask is None or not mask.any():
                return None
            
            points = self.points[mask]
            colors = self.colors[mask] if color_mode == 'rgb' else self.generate_semantic_colors()[mask]
            ids = self.semantic_ids[mask]
        else:
            points = self.points
            colors = self.colors if color_mode == 'rgb' else self.generate_semantic_colors()
            ids = self.semantic_ids
        
        # Subsample if too many points
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]
            colors = colors[indices]
            ids = ids[indices]
        
        # Create hover text
        hover_text = []
        for i in range(len(points)):
            label_id = ids[i]
            name = self.semantic_map.get(label_id, f"unknown_{label_id}")
            hover_text.append(f"Class: {name}<br>ID: {label_id}<br>Pos: ({points[i,0]:.2f}, {points[i,1]:.2f}, {points[i,2]:.2f})")
        
        # Create Plotly figure
        fig = go.Figure(data=[go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=['rgb({},{},{})'.format(r, g, b) for r, g, b in colors],
            ),
            text=hover_text,
            hoverinfo='text'
        )])
        
        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            title=f"Point Cloud Visualization ({len(points):,} points shown)",
            width=800,
            height=600
        )
        
        return fig
    
    def query_objects(self, query_text):
        """Query objects by name."""
        if self.semantic_ids is None:
            return "No PLY file loaded", None
        
        results = []
        total_points = 0
        
        for label_id, name in self.semantic_map.items():
            if query_text.lower() in name.lower():
                mask = self.semantic_ids == label_id
                count = mask.sum()
                if count > 0:
                    points = self.points[mask]
                    bbox_min = points.min(axis=0)
                    bbox_max = points.max(axis=0)
                    center = (bbox_min + bbox_max) / 2
                    size = bbox_max - bbox_min
                    
                    results.append({
                        'ID': label_id,
                        'Name': name,
                        'Points': f"{count:,}",
                        'Center': f"({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})",
                        'Size': f"({size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f})"
                    })
                    total_points += count
        
        if results:
            df = pd.DataFrame(results)
            summary = f"Found {len(results)} objects matching '{query_text}' with {total_points:,} total points"
            return summary, df
        else:
            return f"No objects found matching '{query_text}'", None
    
    def export_filtered(self, class_filter):
        """Export filtered points to new PLY file."""
        if self.points is None:
            return "No PLY file loaded"
        
        # Find matching classes
        mask = None
        for label_id, name in self.semantic_map.items():
            if class_filter.lower() in name.lower():
                if mask is None:
                    mask = self.semantic_ids == label_id
                else:
                    mask |= self.semantic_ids == label_id
        
        if mask is None or not mask.any():
            return f"No objects found matching '{class_filter}'"
        
        # Create output path
        output_path = Path(self.current_ply_path).parent / f"filtered_{class_filter.replace(' ', '_')}.ply"
        
        # Filter data
        filtered_points = self.points[mask]
        filtered_colors = self.colors[mask]
        filtered_ids = self.semantic_ids[mask]
        
        # Create PLY structure
        from plyfile import PlyElement
        vertex = np.zeros(len(filtered_points), dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ('semantic_id', 'u1' if filtered_ids.max() <= 255 else 'u2')
        ])
        
        vertex['x'] = filtered_points[:, 0]
        vertex['y'] = filtered_points[:, 1]
        vertex['z'] = filtered_points[:, 2]
        vertex['red'] = filtered_colors[:, 0]
        vertex['green'] = filtered_colors[:, 1]
        vertex['blue'] = filtered_colors[:, 2]
        vertex['semantic_id'] = filtered_ids
        
        # Write PLY
        el = PlyElement.describe(vertex, 'vertex')
        PlyData([el], comments=self.ply_data.comments if hasattr(self.ply_data, 'comments') else []).write(str(output_path))
        
        return f"Exported {len(filtered_points):,} points to {output_path}"


# Create Gradio interface
def create_app():
    viewer = SemanticViewerApp()
    
    with gr.Blocks(title="MASt3R-SLAM Semantic Viewer") as app:
        gr.Markdown("# MASt3R-SLAM Semantic Point Cloud Viewer")
        gr.Markdown("Web-based visualization and query tool for semantic point clouds")
        
        with gr.Tab("Load & Visualize"):
            with gr.Row():
                ply_input = gr.Textbox(
                    label="PLY File Path",
                    placeholder="e.g., logs/hierarchical_test/my_room/my_seg_color.ply",
                    value="logs/hierarchical_test/my_room/my_seg_color.ply"
                )
                load_btn = gr.Button("Load PLY", variant="primary")
            
            load_status = gr.Textbox(label="Status")
            
            with gr.Row():
                with gr.Column(scale=1):
                    stats_table = gr.Dataframe(label="Object Statistics")
                    color_mode = gr.Radio(
                        choices=["rgb", "semantic"],
                        value="rgb",
                        label="Color Mode"
                    )
                    class_filter = gr.Dropdown(
                        choices=["All"],
                        value="All",
                        label="Filter by Class"
                    )
                    update_viz_btn = gr.Button("Update Visualization")
                
                with gr.Column(scale=2):
                    plot_3d = gr.Plot(label="3D Visualization")
        
        with gr.Tab("Query & Export"):
            with gr.Row():
                query_input = gr.Textbox(
                    label="Query Objects",
                    placeholder="e.g., chair, table, computer"
                )
                query_btn = gr.Button("Search")
            
            query_result = gr.Textbox(label="Query Result")
            query_table = gr.Dataframe(label="Found Objects")
            
            with gr.Row():
                export_filter = gr.Textbox(
                    label="Export Filter",
                    placeholder="e.g., chair"
                )
                export_btn = gr.Button("Export Filtered PLY")
            
            export_status = gr.Textbox(label="Export Status")
        
        # Event handlers
        def on_load(ply_path):
            status, stats, plot = viewer.load_ply(ply_path)
            
            # Update class filter choices
            if viewer.semantic_map:
                classes = ["All"] + sorted(set(viewer.semantic_map.values()))
                return status, stats, plot, gr.Dropdown(choices=classes, value="All")
            
            return status, stats, plot, gr.Dropdown(choices=["All"], value="All")
        
        def on_update_viz(class_filter, color_mode):
            plot = viewer.visualize_3d(class_filter, color_mode)
            return plot
        
        def on_query(query_text):
            return viewer.query_objects(query_text)
        
        def on_export(class_filter):
            return viewer.export_filtered(class_filter)
        
        # Connect events
        load_btn.click(
            on_load,
            inputs=[ply_input],
            outputs=[load_status, stats_table, plot_3d, class_filter]
        )
        
        update_viz_btn.click(
            on_update_viz,
            inputs=[class_filter, color_mode],
            outputs=[plot_3d]
        )
        
        query_btn.click(
            on_query,
            inputs=[query_input],
            outputs=[query_result, query_table]
        )
        
        export_btn.click(
            on_export,
            inputs=[export_filter],
            outputs=[export_status]
        )
    
    return app


if __name__ == "__main__":
    # Check if required packages are installed
    try:
        import gradio
        import plotly
        import pandas
    except ImportError as e:
        print("Missing required packages. Please install:")
        print("  pip install gradio plotly pandas")
        exit(1)
    
    app = create_app()
    
    # Launch with public link for AWS access
    app.launch(
        server_name="0.0.0.0",  # Listen on all interfaces
        server_port=7860,        # Default Gradio port
        share=True               # Create public URL
    )