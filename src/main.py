
import hylite
from hylite import io

import cv2
import numpy as np
import matplotlib.pyplot as plt
from roi_detection import *
from sam_segmentation import *


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
    edges = cv2.Canny(gray, 100, 150, apertureSize=3)
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
        # Check if every point of the contour is inside the ROI polygon
        all_inside = True
        for pt in cnt.reshape(-1, 2):
            pt_clean = (int(pt[0]), int(pt[1]))
            if cv2.pointPolygonTest(roi_poly, pt_clean, False) < 0:
                all_inside = False
                break
        if not all_inside:
            continue

        # Must NOT be inside any Aruco marker polygon (centroid test)
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        centroid = (cx, cy)
        inside_marker = any(
            cv2.pointPolygonTest(mpoly, centroid, False) >= 0
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

def select_bright_pixels(img_bgr, contours, num_pixels=10, visualize=False):
    """
    For each contour, select the top N brightest pixels inside the contour (by grayscale value).
    Optionally plot them in yellow with larger size.
    Args:
        img_bgr (np.ndarray): Input image in BGR format.
        contours (list): List of contours (as from cv2.findContours).
        num_pixels (int): Number of pixels to select per contour.
        visualize (bool): If True, plot the selected pixels on the image.
    Returns:
        List of lists: Each sublist contains (x, y, value) tuples for the selected pixels in a contour.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    selected_pixels = []
    img_pixels = img_bgr.copy() if visualize else None
    for cnt in contours:
        # Create a mask for the contour
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        # Get coordinates of all pixels inside the contour
        ys, xs = np.where(mask == 255)
        values = gray[ys, xs]
        # Compute centroid of the contour
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            selected_pixels.append([])
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        # Compute distance from centroid for each pixel
        dists = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        # Sort pixels by distance to centroid (ascending)
        center_sorted_idx = np.argsort(dists)
        # Take a subset of pixels closest to the centroid (e.g., 30 pixels or all if fewer)
        center_n = min(30, len(center_sorted_idx))
        center_pixels_idx = center_sorted_idx[:center_n]
        # Among these, select the brightest num_pixels
        center_values = values[center_pixels_idx]
        center_xs = xs[center_pixels_idx]
        center_ys = ys[center_pixels_idx]
        # Sort by grayscale value (descending)
        bright_idx = np.argsort(center_values)[::-1]
        chosen = []
        for idx in bright_idx[:num_pixels]:
            x, y, val = center_xs[idx], center_ys[idx], center_values[idx]
            chosen.append((x, y, val))
            if visualize:
                cv2.circle(img_pixels, (int(x), int(y)), radius=5, color=(0, 255, 255), thickness=-1)
        selected_pixels.append(chosen)
    if visualize:
        cv2.imshow('Brightest Pixels (Yellow)', img_pixels)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return selected_pixels


if __name__ == "__main__":
    # image load
    path = '/Users/nova98/Documents/Nova/Helios+/FX10/20260323/FX10_Aruco_random_2026-03-23_08-45-11/capture/FX10_Aruco_random_2026-03-23_07-45-43.hdr'
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
    # contours, hierarchy = detect_contours(img_bgr, visualize=True)
    # Alternative way using SAM
    CHECKPOINT = "/Users/nova98/Documents/Nova/3d_localization/sam_checkpoints/sam_vit_h_4b8939.pth"  # <-- replace with checkpoint path
    MODEL_TYPE = "vit_h"   # 'vit_h', 'vit_l', or 'vit_b'
    DEVICE = "cpu"         # 'cuda' if GPU available
    mask_generator = load_sam_model(CHECKPOINT, model_type=MODEL_TYPE, device=DEVICE)
    masks = run_sam_everything(img_bgr, mask_generator)
    result_bgr = visualize_sam_masks(img_bgr, masks)
    contours = masks_to_contours(masks)

    # Filter contours: inside ROI only, Aruco markers excluded
    filtered = filter_contours(img_bgr, contours, roi_pts, marker_dict, visualize=True)
    print(f"Contours after filtering: {len(filtered)}")

    # Select bright pixels within each filtered contour
    bright_pixels = select_bright_pixels(img_bgr, filtered, num_pixels=10)
    for i, pixels in enumerate(bright_pixels):
        print(f"Contour {i}: {pixels}")


    # Draw bounding boxes around filtered contours
    img_with_boxes, boxes = draw_bounding_boxes(img_bgr, filtered, visualize=True)

    # Select and plot the 10 brightest pixels inside each filtered contour
    selected_pixels = select_bright_pixels(img_bgr, filtered, num_pixels=10, visualize=True)
