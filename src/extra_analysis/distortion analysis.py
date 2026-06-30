from hylite import io
from visualization import *
import numpy as np
import cv2

# Aruco detection
import importlib.util
spec = importlib.util.spec_from_file_location(
    "aruco_detection",
    "/Users/nova98/Documents/Nova/Marker-detection/src/aruco_detection.py"
)
aruco_detection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aruco_detection)


def calculate_edge_lengths(marker_dict):
    """
    Calculate the length of each edge for each marker.
    Edges are ordered clockwise starting from corner 0.
    
    Args:
        marker_dict: Dictionary with marker_id as key and corners array as value
        
    Returns:
        Dictionary with marker_id as key and list of edge lengths as value
    """
    edge_lengths = {}
    
    for marker_id, corners in marker_dict.items():
        # corners is typically shape (1, 4, 2) - extract the 4 corner points
        corner_points = corners[0]  # Shape: (4, 2)
        
        # Calculate distances for all 4 edges (clockwise)
        edges = []
        for i in range(4):
            pt1 = corner_points[i]
            pt2 = corner_points[(i + 1) % 4]  # Next corner (wraps to 0 at end)
            distance = np.linalg.norm(pt2 - pt1)
            edges.append(distance)
        
        edge_lengths[marker_id] = edges
    
    return edge_lengths


def visualize_marker_edges(img_bgr, marker_dict, edge_lengths, save_path=None):
    """
    Visualize the edges of markers with different colors for each edge.
    
    Args:
        img_bgr: Input image (BGR)
        marker_dict: Dictionary with marker_id as key and corners array as value
        edge_lengths: Dictionary with marker_id as key and list of edge lengths as value
        save_path: Optional path to save the visualization image
    """
    # Create a copy to draw on
    img_visualization = img_bgr.copy()
    
    # Define colors for edges: red, orange, magenta, yellow
    colors = [
        (0, 0, 255),      # Red (BGR)
        (0, 165, 255),    # Orange
        (255, 0, 255),    # Magenta
        (0, 255, 255)     # Yellow
    ]
    
    # Draw edges for each marker
    for marker_id, corners in marker_dict.items():
        corner_points = corners[0].astype(int)  # Convert to int for drawing
        
        # Draw each edge with a different color
        for i in range(4):
            pt1 = corner_points[i]
            pt2 = corner_points[(i + 1) % 4]
            
            # Get color for this edge
            color = colors[i % len(colors)]
            
            # Get edge length for annotation
            edge_len = edge_lengths[marker_id][i]
            
            # Draw line
            cv2.line(img_visualization, tuple(pt1), tuple(pt2), color, 1)
            
            # Calculate midpoint for text
            midpoint = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
            
            # Put text with edge length
            cv2.putText(
                img_visualization,
                f'E{i}: {edge_len:.1f}',
                midpoint,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                color,
                1
            )
        
        # Mark corners
        for i, pt in enumerate(corner_points):
            cv2.circle(img_visualization, tuple(pt), 5, (255, 255, 255), -1)
            cv2.putText(
                img_visualization,
                f'C{i}',
                tuple(pt + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 255, 255),
                1
            )
    
    # Display
    cv2.imshow('Marker Edges Visualization', img_visualization)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save image if path provided
    if save_path:
        cv2.imwrite(save_path, img_visualization, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(f"✓ Image saved to: {save_path}")
    
    return img_visualization


def visualize_marker_ids(img_bgr, marker_dict, save_path=None):
    """
    Visualize only marker IDs at the center of each marker.
    
    Args:
        img_bgr: Input image (BGR)
        marker_dict: Dictionary with marker_id as key and corners array as value
        save_path: Optional path to save the visualization image
        
    Returns:
        Visualization image
    """
    # Create a copy to draw on
    img_visualization = img_bgr.copy()
    
    # Draw marker IDs
    for marker_id, corners in marker_dict.items():
        corner_points = corners[0].astype(int)  # Convert to int for drawing
        
        # Calculate marker center
        center = corner_points.mean(axis=0).astype(int)
        
        # Draw marker ID at center
        cv2.putText(
            img_visualization,
            f'M{marker_id}',
            tuple(center),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )
        
        # Draw a circle around the center for visibility
        cv2.circle(img_visualization, tuple(center), 15, (255, 255, 255), 1)
    
    # Display
    cv2.imshow('Marker IDs', img_visualization)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save image if path provided
    if save_path:
        cv2.imwrite(save_path, img_visualization, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(f"✓ Marker IDs image saved to: {save_path}")
    
    return img_visualization


def visualize_single_edge(img_bgr, marker_dict, edge_lengths, edge_index, save_path=None):
    """
    Visualize a single edge for all markers (without marker IDs).
    
    Args:
        img_bgr: Input image (BGR)
        marker_dict: Dictionary with marker_id as key and corners array as value
        edge_lengths: Dictionary with marker_id as key and list of edge lengths as value
        edge_index: Edge index (0-3) to visualize
        save_path: Optional path to save the visualization image
        
    Returns:
        Visualization image
    """
    # Create a copy to draw on
    img_visualization = img_bgr.copy()
    
    # Define colors for edges: red, orange, magenta, yellow
    colors = [
        (0, 0, 255),      # Red (BGR)
        (0, 165, 255),    # Orange
        (255, 0, 255),    # Magenta
        (0, 255, 255)     # Yellow
    ]
    
    edge_names = ['Edge 0 (Red)', 'Edge 1 (Orange)', 'Edge 2 (Magenta)', 'Edge 3 (Yellow)']
    color = colors[edge_index]
    edge_name = edge_names[edge_index]
    
    # Draw only the specified edge for each marker
    for marker_id, corners in marker_dict.items():
        corner_points = corners[0].astype(int)  # Convert to int for drawing
        
        # Get the two corner points for this edge
        pt1 = corner_points[edge_index]
        pt2 = corner_points[(edge_index + 1) % 4]
        
        # Get edge length for annotation
        edge_len = edge_lengths[marker_id][edge_index]
        
        # Draw line
        cv2.line(img_visualization, tuple(pt1), tuple(pt2), color, 1)
        
        # Calculate midpoint for text
        midpoint = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        
        # Put text with edge length only (no marker ID)
        cv2.putText(
            img_visualization,
            f'{edge_len:.1f}px',
            midpoint,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1
        )
        
        # # Mark both corners
        # for i, pt in enumerate([pt1, pt2]):
        #     cv2.circle(img_visualization, tuple(pt), 6, color, -1)
        #     cv2.circle(img_visualization, tuple(pt), 6, (255, 255, 255), 1)
    
    # Add title with edge information
    cv2.putText(
        img_visualization,
        edge_name,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        1
    )
    
    # Display
    cv2.imshow(f'Marker {edge_name}', img_visualization)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save image if path provided
    if save_path:
        cv2.imwrite(save_path, img_visualization, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(f"✓ {edge_name} saved to: {save_path}")
    
    return img_visualization


def print_edge_statistics(edge_lengths):
    """
    Print edge length statistics for validation.
    
    Args:
        edge_lengths: Dictionary with marker_id as key and list of edge lengths as value
    """
    print("\n" + "="*60)
    print("MARKER EDGE LENGTH ANALYSIS")
    print("="*60)
    
    for marker_id, edges in sorted(edge_lengths.items()):
        print(f"\nMarker ID: {marker_id}")
        print(f"  Edge 0 (Red):     {edges[0]:.2f} pixels")
        print(f"  Edge 1 (Orange):  {edges[1]:.2f} pixels")
        print(f"  Edge 2 (Magenta): {edges[2]:.2f} pixels")
        print(f"  Edge 3 (Yellow):  {edges[3]:.2f} pixels")
        print(f"  Average length:  {np.mean(edges):.2f} pixels")
        print(f"  Std deviation:   {np.std(edges):.2f} pixels")
        print(f"  Min length:      {np.min(edges):.2f} pixels")
        print(f"  Max length:      {np.max(edges):.2f} pixels")


if __name__ == "__main__":
    print("Starting processing1...")
    # image load
    path = '/Users/nova98/Documents/Nova/Helios+/FX10/20260527/FX10_Charucoreallyflat_2026-05-27_07-22-27/capture/FX10_Charucoreallyflat_2026-05-27_07-22-27.hdr'
    image = io.load(path)
    img_bgr = plot_hyimage(image)
    # aruco marker detction
    marker_dict = aruco_detection.getAruco(img_bgr)
    
    # Calculate edge lengths
    edge_lengths = calculate_edge_lengths(marker_dict)
    
    # Print statistics
    print_edge_statistics(edge_lengths)
    
    # Save 4 separate edge images
    import os
    from datetime import datetime
    
    output_dir = '/Users/nova98/Documents/Nova/3d_localization'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    edge_names = ['red', 'orange', 'magenta', 'yellow']
    
    # Save 4 separate edge images
    print("\n" + "="*60)
    print("SAVING INDIVIDUAL EDGE VISUALIZATIONS")
    print("="*60)
    
    for edge_idx in range(4):
        output_path = os.path.join(output_dir, f'marker_edge_{edge_names[edge_idx]}_{timestamp}.png')
        visualize_single_edge(img_bgr, marker_dict, edge_lengths, edge_idx, save_path=output_path)