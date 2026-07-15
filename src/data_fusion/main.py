'''
Data fusion of ROI from point cloud and Gray scale HSI image
'''

import sys
import importlib
from pathlib import Path
from matplotlib import pyplot as plt

# Ensure project root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloader import *
from projection import *
from visualisation import *
from src.segmentation.sam2_segmentation import *

CLOUD_ANALYSIS_SRC = Path("/Users/nova98/Documents/Nova/cloud_analysis/src")
if str(CLOUD_ANALYSIS_SRC) not in sys.path:
    sys.path.insert(0, str(CLOUD_ANALYSIS_SRC))

# Point projection to 2D space
cloud_analysis = importlib.import_module("analysis")
viz = importlib.import_module("viz")


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

    # Step 5: Contours detection on HSI image
    # # ── SAM2 segmentation ────────────────────────────────────────────────────
    SAM2_CHECKPOINT = "/Users/nova98/Documents/Nova/3d_localization/sam_checkpoints/sam2.1_hiera_tiny.pt"
    SAM2_MODEL_TYPE = "tiny"  # 'tiny', 'small', 'base_plus', or 'large'
    DEVICE = "cpu"
    sam2_countours = run_SAM2(SAM2_CHECKPOINT, SAM2_MODEL_TYPE, DEVICE, hsi_img)