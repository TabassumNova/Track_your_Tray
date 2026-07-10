'''
Data fusion of ROI from point cloud and Gray scale HSI image
'''

import numpy as np
import open3d as o3d
import cv2
import sys
import importlib
from pathlib import Path

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

if __name__ == "__main__":
    
    CLOUD_PATH = "/Users/nova98/Documents/Nova/cloud_data/20260629/test2_ROI/roi.ply"
    IMG_PATH = "/Users/nova98/Documents/Nova/cloud_data/20260629/test2_ROI/cropped_ROI.png"

    # Step 1: Load data
    cloud_points, hsi_img = dataloader(CLOUD_PATH, IMG_PATH)

    # Step 2: Project point cloud to 2D space
    proj_img, pixel_to_point_indices = cloud_analysis.project_to_2D(cloud_points, height=1000, width=1000, visualisation=True)
    pass