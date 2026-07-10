'''
Data fusion of ROI from point cloud and Gray scale HSI image
'''

import numpy as np
import open3d as o3d
import cv2
import sys
import importlib
from pathlib import Path
from matplotlib import pyplot as plt

CLOUD_ANALYSIS_SRC = Path("/Users/nova98/Documents/Nova/cloud_analysis/src")
if str(CLOUD_ANALYSIS_SRC) not in sys.path:
    sys.path.insert(0, str(CLOUD_ANALYSIS_SRC))

# Point projection to 2D space
cloud_analysis = importlib.import_module("analysis")
viz = importlib.import_module("viz")


def dataloader(cloud_path, img_path, visualisation=True):
    # Load point cloud
    pcd = o3d.io.read_point_cloud(cloud_path)
    cloud_points = np.asarray(pcd.points)

    # Load grayscale image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Temporary fix for img
    # Rotate the image by 90 degrees clockwise + flip vertically
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = cv2.flip(img, 1)  # Flip vertically

    if visualisation:
        # Visualize point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud_points)
        o3d.visualization.draw_geometries([pcd])

        # Visualize image
        cv2.imshow("Grayscale Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return cloud_points, img


def overlay_heightmap_on_image(gray_img, height_map, alpha=0.45):
    if gray_img is None or height_map is None:
        raise ValueError("gray_img and height_map must be valid arrays")

    if gray_img.ndim != 2:
        raise ValueError("gray_img must be a grayscale image")

    if height_map.shape[:2] != gray_img.shape[:2]:
        height_map = cv2.resize(
            height_map,
            (gray_img.shape[1], gray_img.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    valid_mask = ~np.isnan(height_map)
    if not np.any(valid_mask):
        raise ValueError("height_map has no valid values (all NaN)")

    valid_values = height_map[valid_mask]
    z_min = valid_values.min()
    z_max = valid_values.max()

    if np.isclose(z_max, z_min):
        normalized = np.zeros_like(height_map, dtype=np.uint8)
    else:
        normalized_float = (height_map - z_min) / (z_max - z_min)
        normalized_float = np.where(valid_mask, normalized_float, 0.0)
        normalized = np.clip(normalized_float * 255.0, 0, 255).astype(np.uint8)

    normalized[~valid_mask] = 0

    heatmap_bgr = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    base_bgr = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(base_bgr, 1.0 - alpha, heatmap_bgr, alpha, 0)

    overlay = base_bgr.copy()
    overlay[valid_mask] = blended[valid_mask]
    return overlay, z_min, z_max


def display_heatmap_with_colorbar(fused_heatmap, z_min, z_max):
    """Display heatmap with colorbar showing height values with JET colormap (blue to red)."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Display the RGB image
    ax.imshow(cv2.cvtColor(fused_heatmap, cv2.COLOR_BGR2RGB))
    ax.set_title("Height Map Overlay on Image (Blue=Low, Red=High)", fontsize=14, fontweight="bold")
    ax.axis("off")
    
    # Create ScalarMappable with JET colormap to match the heatmap colors
    sm = ScalarMappable(cmap=plt.cm.jet, norm=Normalize(vmin=z_min, vmax=z_max))
    sm.set_array([])
    
    # Create colorbar with the JET colormap
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", pad=0.02, fraction=0.046)
    cbar.set_label(f"Height (units)\n[{z_min:.2f} to {z_max:.2f}]", fontsize=12, fontweight="bold")
    
    # Set colorbar tick labels to represent actual height values
    cbar_ticks = np.linspace(z_min, z_max, 5)  # 5 ticks with actual height values
    cbar.set_ticks(cbar_ticks)
    cbar.ax.set_yticklabels([f"{h:.2f}" for h in cbar_ticks])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    
    CLOUD_PATH = "/Users/nova98/Documents/Nova/cloud_data/20260629/test2_ROI/roi.ply"
    IMG_PATH = "/Users/nova98/Documents/Nova/cloud_data/20260629/test2_ROI/cropped_ROI.png"

    # Step 1: Load data
    cloud_points, hsi_img = dataloader(CLOUD_PATH, IMG_PATH)

    # Step 2: Project point cloud to 2D space with height map
    proj_img, pixel_to_point_indices, height_map = cloud_analysis.project_to_2D(
        cloud_points,
        height=hsi_img.shape[0],
        width=hsi_img.shape[1],
        visualisation=True,
        return_height_map=True,
    )

    # Step 3: Overlay height map (blue=low, red=high) onto the grayscale image
    fused_heatmap, z_min, z_max = overlay_heightmap_on_image(hsi_img, height_map, alpha=0.6)

    # Step 4: Display with colorbar showing height values
    display_heatmap_with_colorbar(fused_heatmap, z_min, z_max)