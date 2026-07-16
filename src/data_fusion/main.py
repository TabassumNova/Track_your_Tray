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
from src.segmentation.contour_detection import *

CLOUD_ANALYSIS_SRC = Path("/Users/nova98/Documents/Nova/cloud_analysis/src")
if str(CLOUD_ANALYSIS_SRC) not in sys.path:
    sys.path.insert(0, str(CLOUD_ANALYSIS_SRC))

# Point projection to 2D space
cloud_analysis = importlib.import_module("analysis")
viz = importlib.import_module("viz")


def crop_bounding_box(hsi_img, heatmap, boxes, max_boxes=None):
    """Crop each bounding box from HSI and heatmap and visualize side by side."""
    if hsi_img is None or heatmap is None:
        raise ValueError("hsi_img and heatmap must be valid arrays")

    if boxes is None or len(boxes) == 0:
        print("No boxes to crop for Step 8.")
        return

    h, w = hsi_img.shape[:2]
    total = len(boxes) if max_boxes is None else min(len(boxes), int(max_boxes))

    for i, (x, y, bw, bh) in enumerate(boxes[:total], start=1):
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(w, int(x + bw))
        y1 = min(h, int(y + bh))

        if x1 <= x0 or y1 <= y0:
            continue

        hsi_crop = hsi_img[y0:y1, x0:x1]
        heatmap_crop = heatmap[y0:y1, x0:x1]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(hsi_crop, cmap="gray")
        axes[0].set_title(f"HSI Crop #{i}")
        axes[0].axis("off")

        height_vis = np.ma.masked_invalid(heatmap_crop)
        axes[1].imshow(height_vis, cmap="jet")
        axes[1].set_title(f"Fused Heatmap Crop #{i}")
        axes[1].axis("off")

        fig.suptitle(f"Box #{i}: x={x0}, y={y0}, w={x1 - x0}, h={y1 - y0}")
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

    # Step 5: Contours detection on HSI image
    # # ── SAM2 segmentation ────────────────────────────────────────────────────
    SAM2_CHECKPOINT = "/Users/nova98/Documents/Nova/3d_localization/sam_checkpoints/sam2.1_hiera_tiny.pt"
    SAM2_MODEL_TYPE = "tiny"  # 'tiny', 'small', 'base_plus', or 'large'
    DEVICE = "cpu"
    sam2_countours = run_SAM2(SAM2_CHECKPOINT, SAM2_MODEL_TYPE, DEVICE, hsi_img)

    # Step 6: Remove the contours that are very near to the border
    filtered_contours = filter_border_contours(
        contours=sam2_countours,
        img_shape=hsi_img.shape,
        border_margin=20,
    )
    print(f"Contours after border filtering: {len(filtered_contours)}/{len(sam2_countours)}")
    # Visualize the filtered contours on the original HSI image
    show_filtered_contours_on_hsi(hsi_img, filtered_contours)

    # Step 7: Create bounding boxes around filtered contours
    img_with_boxes, boxes = generate_bounding_boxes(hsi_img, filtered_contours, visualisation=True)

    # Step 8: Visualize the cropped regions from HSI and fused heatmap side by side
    crop_bounding_box(hsi_img, height_map, boxes)
