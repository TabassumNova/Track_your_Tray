
import hylite
from hylite import io

import cv2
import numpy as np
import matplotlib.pyplot as plt
from roi_detection import *

# Aruco detection
import importlib.util
spec = importlib.util.spec_from_file_location(
    "aruco_detection",
    "/Users/nova98/Documents/Nova/Marker-detection/src/aruco_detection.py"
)
aruco_detection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aruco_detection)

def plot_hyimage(image):
    
    # Find the band index closest to 770 nm (FX10) and 1322 nm (FX17)
    wavelengths = image.get_wavelengths()
    band_idx = np.argmin(np.abs(wavelengths - 770.0))

    # Extract band and normalize to uint8 (0-255), handling NaN
    band_data = image.data[:, :, band_idx].astype(np.float32)
    band_data = np.nan_to_num(band_data, nan=0.0)
    band_norm = cv2.normalize(band_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Stack into BGR (grayscale equivalent) for cv2 processing
    img_bgr = cv2.merge([band_norm, band_norm, band_norm])

    # Mirror along the x-axis (vertical flip)
    img_bgr = cv2.flip(img_bgr, 0)

    cv2.imshow('Band at ~770 nm', img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img_bgr

def edge_detection(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Display the edges
    cv2.imshow('Edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to detect contours in an image using OpenCV
def detect_contours(img_bgr, visualize=True):
    """
    Detect contours in a BGR image using OpenCV.
    Args:
        img_bgr (np.ndarray): Input image in BGR format.
        visualize (bool): If True, display the contours on the image.
    Returns:
        contours (list): Detected contours.
        hierarchy (np.ndarray): Contour hierarchy.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)
    # Find contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if visualize:
        img_contours = img_bgr.copy()
        cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
        cv2.imshow('Contours', img_contours)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return contours, hierarchy

def filter_contours(img_bgr, contours, roi_pts, marker_dict, visualize=True):
    """
    Filter contours to only those inside the ROI polygon, excluding contours
    that belong to Aruco markers.

    Args:
        img_bgr     (np.ndarray): Input image in BGR format.
        contours    (list):       Contours from cv2.findContours.
        roi_pts     (list):       Four corner points defining the ROI polygon.
        marker_dict (dict):       {marker_id: corners} from Aruco detection,
                                  where corners has shape (1, 4, 2).
        visualize   (bool):       If True, display the filtered contours.

    Returns:
        filtered (list): Contours inside the ROI and not on any Aruco marker.
    """
    roi_poly = np.int32(roi_pts)

    # Build a list of Aruco marker polygons for quick lookup
    marker_polys = []
    for corners in marker_dict.values():
        pts = np.int32(corners[0])   # shape (4, 2)
        marker_polys.append(pts)

    filtered = []
    for cnt in contours:
        # Compute centroid via moments
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        pt = (cx, cy)

        # Must be inside the ROI polygon
        if cv2.pointPolygonTest(roi_poly, pt, False) < 0:
            continue

        # Must NOT be inside any Aruco marker polygon
        inside_marker = any(
            cv2.pointPolygonTest(mpoly, pt, False) >= 0
            for mpoly in marker_polys
        )
        if inside_marker:
            continue

        filtered.append(cnt)

    if visualize:
        vis = img_bgr.copy()
        # Draw the ROI boundary
        # cv2.polylines(vis, [roi_poly], isClosed=True, color=(0, 255, 255), thickness=2)
        # Draw the filtered contours
        cv2.drawContours(vis, filtered, -1, (0, 0, 255), 2)
        cv2.imshow('Filtered Contours (inside ROI, no markers)', vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return filtered

def draw_bounding_boxes(img_bgr, contours, visualize=True):
    """
    Draw bounding boxes around the given contours and visualize them.
    Args:
        img_bgr (np.ndarray): Input image in BGR format.
        contours (list): List of contours to draw bounding boxes around.
        visualize (bool): If True, display the image with bounding boxes.
    Returns:
        img_with_boxes (np.ndarray): Image with bounding boxes drawn.
        boxes (list): List of bounding box coordinates (x, y, w, h).
    """
    img_with_boxes = img_bgr.copy()
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, w, h))
        cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (255, 0, 0), 2)
    if visualize:
        cv2.imshow('Bounding Boxes', img_with_boxes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img_with_boxes, boxes


if __name__ == "__main__":
    # image load
    path = '/Users/nova98/Documents/Nova/Helios+/FX10/20260323/FX10_Aruco_random_2026-03-23_12-34-11/capture/FX10_Aruco_random_2026-03-23_12-34-11.hdr'
    image = io.load(path)
    img_bgr = plot_hyimage(image)
    # aruco marker detction
    marker_dict = aruco_detection.getAruco(img_bgr)
    # # # roi detection
    roi_pts = find_ROI(img_bgr, marker_dict)
    
    # grid line detection
    # find_grid(img_bgr, marker_dict)

    pass
    # # Edge detection
    # edge_detection(img_bgr)
    
    # Contour detection
    contours, hierarchy = detect_contours(img_bgr, visualize=True)

    # Filter contours: inside ROI only, Aruco markers excluded
    filtered = filter_contours(img_bgr, contours, roi_pts, marker_dict, visualize=True)
    print(f"Contours after filtering: {len(filtered)}")

    # Draw bounding boxes around filtered contours
    img_with_boxes, boxes = draw_bounding_boxes(img_bgr, filtered, visualize=True)
