import hylite
from hylite import io
import csv
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from roi_detection import *
from segmentation.sam_segmentation import *
from segmentation.sam2_segmentation import *
from visualization import *
from segmentation.contour_detection import *
from main import *

import time

# Aruco detection
import importlib.util
spec = importlib.util.spec_from_file_location(
    "aruco_detection",
    "/Users/nova98/Documents/Nova/Marker-detection/src/aruco_detection.py"
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


def create_optimal_camera_matrix(img_height, img_width):
    """Create an optimal camera matrix based on image dimensions (no actual calibration).
    
    Args:
        img_height (int): Image height in pixels.
        img_width (int): Image width in pixels.
    
    Returns:
        K (np.ndarray): 3x3 camera intrinsic matrix.
        dist_coeffs (np.ndarray): Distortion coefficients (all zeros).
    """
    # Estimate focal length as the average of width and height
    focal_length = (img_width + img_height) / 2.0
    cx = img_width / 2.0
    cy = img_height / 2.0
    
    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    dist_coeffs = np.zeros(5, dtype=np.float32)  # No distortion
    
    return K, dist_coeffs


def estimate_aruco_poses(img_bgr, warped_marker_dict, marker_ids, marker_size_mm=60.0):
    """Estimate 6-DOF poses for detected ArUco markers.
    
    Args:
        img_bgr (np.ndarray): Input image in BGR format.
        warped_marker_dict (dict): Marker ID -> corners from ArUco detection.
        marker_ids (list): List of marker IDs to estimate pose for.
        marker_size_mm (float): Physical size of the marker in mm.
    
    Returns:
        poses (dict): {marker_id: {'rvec': rvec, 'tvec': tvec}}.
        img_with_axes (np.ndarray): Image with 3D axes drawn for each marker.
    """
    img_height, img_width = img_bgr.shape[:2]
    K, dist_coeffs = create_optimal_camera_matrix(img_height, img_width)
    
    # Define 3D marker corners in marker coordinate frame (top-left clockwise)
    # Marker is centered at origin, lies in z=0 plane
    marker_size = marker_size_mm / 2.0  # Half-size from center
    object_points = np.array([
        [-marker_size,  marker_size, 0],  # top-left
        [ marker_size,  marker_size, 0],  # top-right
        [ marker_size, -marker_size, 0],  # bottom-right
        [-marker_size, -marker_size, 0],  # bottom-left
    ], dtype=np.float32)
    
    poses = {}
    img_with_axes = img_bgr.copy()
    
    for marker_id in marker_ids:
        if marker_id not in warped_marker_dict:
            continue
        
        # Get ordered corners
        corners = warped_marker_dict[marker_id]
        ordered_corners = order_corners_top_left_clockwise(corners)
        image_points = ordered_corners.astype(np.float32)
        
        # Estimate pose
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            K,
            dist_coeffs,
            useExtrinsicGuess=False,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success:
            poses[marker_id] = {'rvec': rvec, 'tvec': tvec, 'success': True}
            
            # Draw 3D axes at marker origin
            axis_length = marker_size_mm  # Length of axis in mm (50mm)
            img_with_axes = cv2.drawFrameAxes(
                img_with_axes,
                K,
                dist_coeffs,
                rvec,
                tvec,
                axis_length,
                thickness=2
            )
            
            # Draw marker ID near the center
            center = np.mean(image_points, axis=0).astype(int)
            cv2.putText(
                img_with_axes,
                f"ID:{marker_id}",
                tuple(center),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
        else:
            poses[marker_id] = {'success': False}
    
    return poses, img_with_axes


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
    aruco_ids = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 11, 3] 
    # aruco_ids = [101, 104] 
    path = '/Users/nova98/Documents/Nova/Helios+/FX10/20260616/FX10_ArucoCube_10cm_pose11_2026-06-16_11-43-03/capture/FX10_ArucoCube_10cm_pose11_2026-06-16_11-43-03.hdr'
    # aruco_ids = [11]
    aruco_detection_and_mapping(path, aruco_ids)
    
