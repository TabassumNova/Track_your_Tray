

import hylite
from hylite import io
import csv
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.roi_detection import *
from src.segmentation.sam_segmentation import *
from src.segmentation.sam2_segmentation import *
from src.visualization import *
from src.segmentation.contour_detection import *
from src.mapping import *


from src.sensor_axis_distortion_analysis.dataloader import *
from src.sensor_axis_distortion_analysis.pose_analysis_pipeline import *

import time

'''
Aruco detection module
1. Clone the repository:
    - git clone https://github.com/TabassumNova/Marker-detection.git
2. Ensure the path to aruco_detection.py is correct in the import statement below.
'''
# Aruco detection
import importlib.util
spec = importlib.util.spec_from_file_location(
    "aruco_detection",
    "/Users/nova98/Documents/Nova/Marker-detection/src/aruco_detection.py" # Update this path to the actual location of aruco_detection.py
)
aruco_detection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aruco_detection)


def order_corners_top_left_clockwise(corners_xy):
    """Return 4 corners ordered as [top-left, top-right, bottom-right, bottom-left]."""
    pts = np.asarray(corners_xy, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(d)]
    bottom_left = pts[np.argmax(d)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)



def aruco_detection_and_mapping(path, aruco_ids):
    print("Starting processing1...")
    # image load
    image = io.load(path)
    img_bgr = plot_hyimage(image)
    # aruco marker detction
    marker_dict = aruco_detection.getAruco(img_bgr)
    # # # roi detection
    CONSIDERED_MARKER = [34, 38, 39, 37, 35, 46, 45, 42, 49, 53, 43, 32, 74] # <-- Big black tray
    # CONSIDERED_MARKER = [65, 59, 60, 61, 58, 62, 57, 56, 70, 71, 72] # <-- Small black tray
    # CONSIDERED_MARKER = [68, 63, 64, 67]
    roi_pts = find_ROI(img_bgr, marker_dict, considered_markers=CONSIDERED_MARKER)
    
    # Crop ROI
    roi_cropped, img_warped, warped_roi_pts, warped_marker_dict = crop_roi_from_image(img_bgr, roi_pts, marker_dict, roi_size_px=1000, visualize=True)
    img_bgr1 = roi_cropped  # For subsequent processing, focus on the cropped ROI

    # Map the 4 corners of the middle Aruco markers in millimeter scale.
    selected_pixels = []
    selected_pixels_for_mapping = []
    marker_ids_found = []
    missing_ids = []
    for marker_id in aruco_ids:
        corners = warped_marker_dict.get(marker_id)
        if corners is None:
            missing_ids.append(marker_id)
            continue

        ordered_corners = order_corners_top_left_clockwise(corners)
        corners_xy = []
        corners_for_map = []
        for x, y in ordered_corners:
            px = int(round(x))
            py = int(round(y))
            corners_xy.append((px, py))
            corners_for_map.append((px, py, 0))

        selected_pixels.extend(corners_xy)
        selected_pixels_for_mapping.append(corners_for_map)
        marker_ids_found.append(marker_id)

    if missing_ids:
        print(f"Warning: missing middle Aruco IDs in warped ROI: {missing_ids}")

    if not selected_pixels_for_mapping:
        raise ValueError("No middle Aruco corners found in warped_marker_dict.")

    pixels_mm_nested = map_pixels_to_mm(
        warped_roi_pts,
        selected_pixels_for_mapping,
        roi_size_mm=318.0,
        visualize=True,
        img=img_warped,
    )
    pixels_mm = [pt_mm for marker_mm in pixels_mm_nested for pt_mm in marker_mm]

    csv_path = os.path.join(os.path.dirname(__file__), "middle_aruco_corners_mm.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header_ids = [str(marker_id) for marker_id in aruco_ids]
        writer.writerow(["corner", *header_ids])

        marker_mm_by_id = {
            marker_id: marker_mm
            for marker_id, marker_mm in zip(marker_ids_found, pixels_mm_nested)
        }

        for corner_idx in range(4):
            row = [f"corner{corner_idx + 1}"]
            for marker_id in aruco_ids:
                marker_mm = marker_mm_by_id.get(marker_id)
                if marker_mm is None:
                    row.append("")
                else:
                    x_mm, y_mm = marker_mm[corner_idx]
                    x_mm = int(round(float(x_mm)))
                    y_mm = int(round(float(y_mm)))
                    row.append(f"({x_mm}, {y_mm})")
            writer.writerow(row)

    print(f"Mapped {len(pixels_mm)} middle Aruco corner points to mm scale.")
    print(f"Saved marker corner CSV: {csv_path}")
    



if __name__ == "__main__":
    dataset_path = '/Users/nova98/Documents/Nova/Helios+/FX10/20260616/10cm_Cube'
    TRAY_ARUCOS = [34, 38, 39, 37, 35, 46, 45, 42, 49, 53, 43, 32, 74] # <-- Big black tray
    MEASUREMENT_ARUCOS = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111] 
    TRAY_SIZE = 325 # Actual tray size in mm (for the big black tray)
    CUBE_ARUCO = 3
    CUBE_SQUARE_SIZE = 50 # in mm
    CUBE_MARKER_SIZE = 40 # in mm
    TRAY_SQUARE_SIZE = 25 # in mm  # Applies to both TRAY_ARUCOS and MEASUREMENT_ARUCOS
    TRAY_MARKER_SIZE = 18 # in mm  # Applies to both TRAY_ARUCOS and MEASUREMENT_ARUCOS
    
    
    
    pipeline = PoseAnalysisPipeline(
        dataset_path=dataset_path,
        tray_size=TRAY_SIZE,
        tray_aruco_ids=TRAY_ARUCOS,
        measurement_aruco_ids=MEASUREMENT_ARUCOS,
        cube_aruco_id=CUBE_ARUCO,
        cube_square_size=CUBE_SQUARE_SIZE,
        cube_marker_size=CUBE_MARKER_SIZE,
        tray_square_size=TRAY_SQUARE_SIZE,
        tray_marker_size=TRAY_MARKER_SIZE
    )

    pipeline.start_analysis()

    # path = '/Users/nova98/Documents/Nova/Helios+/FX10/20260616/FX10_ArucoCube_10cm_pose11_2026-06-16_11-43-03/capture/FX10_ArucoCube_10cm_pose11_2026-06-16_11-43-03.hdr'
    # aruco_ids = [11]
    # aruco_detection_and_mapping(path, MEASUREMENT_ARUCOS)
    
