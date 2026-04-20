import hylite
from hylite import io

import cv2
import numpy as np
import matplotlib.pyplot as plt
from roi_detection import *
from segmentation.sam_segmentation import *
from segmentation.sam2_segmentation import *
from visualization import *
from segmentation.contour_detection import *


# Aruco detection
import importlib.util
spec = importlib.util.spec_from_file_location(
    "aruco_detection",
    "/Users/nova98/Documents/Nova/Marker-detection/src/aruco_detection.py"
)
aruco_detection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aruco_detection)



def edge_detection(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Display the edges
    cv2.imshow('Edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



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


def map_pixels_to_mm(roi_pts, selected_pixels, roi_size_mm=325.0, visualize=True, img_bgr=None):
    """
    Map pixel coordinates to millimeter scale using the ROI corners via perspective transform.

    The ROI is treated as a square of side roi_size_mm.  Corner ordering must
    match find_ROI: [top-left, top-right, bottom-right, bottom-left].

    Args:
        roi_pts        (list):            Four corner points of the ROI in pixel space.
        selected_pixels (list of lists):  Output from select_bright_pixels —
                                          each sublist contains (x, y, value) tuples.
        roi_size_mm    (float):           Physical side length of the ROI in mm (default 325).
        visualize      (bool):            If True, display the image with mm labels.
        img_bgr        (np.ndarray):      Image to annotate (required when visualize=True).

    Returns:
        pixels_mm (list of lists): Each sublist contains (x_mm, y_mm) tuples,
                                   one per pixel in the corresponding contour.
    """
    # Perspective transform: pixel space → mm space
    src = np.float32(roi_pts)  # [TL, TR, BR, BL]
    dst = np.float32([
        [0,           0          ],
        [roi_size_mm, 0          ],
        [roi_size_mm, roi_size_mm],
        [0,           roi_size_mm],
    ])
    M = cv2.getPerspectiveTransform(src, dst)

    pixels_mm = []
    for contour_pixels in selected_pixels:
        contour_mm = []
        for (x, y, _val) in contour_pixels:
            pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
            pt_mm = cv2.perspectiveTransform(pt, M)[0][0]
            contour_mm.append((float(pt_mm[0]), float(pt_mm[1])))
        pixels_mm.append(contour_mm)

    if visualize and img_bgr is not None:
        vis = img_bgr.copy()
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        thickness  = 2
        arrow_len  = 30   # px from dot tip to label anchor

        for contour_pixels, contour_mm in zip(selected_pixels, pixels_mm):
            for (x, y, _val), (x_mm, y_mm) in zip(contour_pixels, contour_mm):
                px, py = int(x), int(y)

                # Arrow tip offset: place label to the right and slightly above
                tip_x = px + arrow_len
                tip_y = py - arrow_len

                label = f"({x_mm:.1f}, {y_mm:.1f}) mm"
                (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                # Draw a filled dark rectangle behind the text for readability
                pad = 4
                cv2.rectangle(
                    vis,
                    (tip_x - pad, tip_y - th - pad),
                    (tip_x + tw + pad, tip_y + baseline + pad),
                    (30, 30, 30),
                    cv2.FILLED,
                )

                # Arrow from dot to label box
                cv2.arrowedLine(
                    vis,
                    (px, py),
                    (tip_x, tip_y),
                    (0, 255, 255),
                    thickness=2,
                    tipLength=0.25,
                )

                # Dot at the pixel location
                cv2.circle(vis, (px, py), 5, (0, 255, 255), -1)

                # Label text
                cv2.putText(
                    vis, label,
                    (tip_x, tip_y),
                    font, font_scale, (255, 255, 0), thickness, cv2.LINE_AA,
                )

        cv2.imshow("Pixel Positions in mm", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return pixels_mm



def warp_contours_to_original(contours, roi_pts, roi_size_px=400, img_bgr=None, visualize=True):
    """
    Warp contours from the cropped ROI image back to the original image using inverse perspective transform.

    Args:
        contours (list): Contours in the cropped ROI image (list of np.ndarray).
        roi_pts (list): Four corner points of the ROI in the original image (TL, TR, BR, BL).
        roi_size_px (int): Size of the cropped ROI image (width/height in px).
        img_bgr (np.ndarray): Original image to visualize on (optional).
        visualize (bool): If True, display the contours on the original image.

    Returns:
        contours_orig (list): Contours mapped to the original image coordinates.
    """
    # Perspective transform: cropped ROI -> original image
    dst = np.float32([
        [0, 0],
        [roi_size_px - 1, 0],
        [roi_size_px - 1, roi_size_px - 1],
        [0, roi_size_px - 1],
    ])
    src = np.float32(roi_pts)
    Minv = cv2.getPerspectiveTransform(dst, src)

    contours_orig = []
    for cnt in contours:
        cnt = cnt.astype(np.float32)
        cnt_warped = cv2.perspectiveTransform(cnt, Minv)
        contours_orig.append(cnt_warped.astype(np.int32))

    if visualize and img_bgr is not None:
        vis = img_bgr.copy()
        cv2.drawContours(vis, contours_orig, -1, (0, 0, 255), 2)
        cv2.imshow("Contours posed in original image", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return contours_orig





if __name__ == "__main__":
    # image load
    path = '/Users/nova98/Documents/Nova/Helios+/FX10/20260323/FX10_Aruco_random_2026-03-23_13-13-47_nicd_with_objects/capture/FX10_Aruco_random_2026-03-23_13-13-47.hdr'
    image = io.load(path)
    img_bgr = plot_hyimage(image)
    # aruco marker detction
    marker_dict = aruco_detection.getAruco(img_bgr)
    # # # roi detection
    CONSIDERED_MARKER = [34, 38, 39, 37, 35, 46, 45, 42, 49, 53, 43, 32] # <-- specify which markers are on the tray
    roi_pts = find_ROI(img_bgr, marker_dict, considered_markers=CONSIDERED_MARKER)
    

    # Crop ROI
    roi_cropped = crop_roi_from_image(img_bgr, roi_pts, roi_size_px=400, visualize=True)
    img_bgr1 = roi_cropped  # For subsequent processing, focus on the cropped ROI

    # # Test: Apply blob detection
    # keypoints = detect_blobs(img_bgr, visualize=True, min_area=50, max_area=5000, threshold=10)

    # # Test: Apply haris corner detection
    # corners = detect_harris_corners(img_bgr, visualize=True, block_size=2, ksize=3, k=0.04, threshold_rel=0.01)
    
    # Contour detection
    contours, hierarchy = detect_contours(img_bgr1, visualize=True)

    import time
    # ── SAM1 segmentation ────────────────────────────────────────────────────
    SAM1_CHECKPOINT_PATH = "/Users/nova98/Documents/Nova/3d_localization/sam_checkpoints"
    SAM1_MODEL_TYPE = "vit_b"   # 'vit_h', 'vit_l', or 'vit_b'
    DEVICE = "cpu"              # 'cuda' if GPU available
    start_time = time.time()
    sam1_contours = run_SAM1(img_bgr1, SAM1_CHECKPOINT_PATH, SAM1_MODEL_TYPE, DEVICE)
    print(f"SAM1 segmentation took {time.time() - start_time:.2f} seconds")

    # ── SAM2 segmentation ────────────────────────────────────────────────────
    SAM2_CHECKPOINT = "/Users/nova98/Documents/Nova/3d_localization/sam_checkpoints/sam2.1_hiera_tiny.pt"
    SAM2_MODEL_TYPE = "tiny"  # 'tiny', 'small', 'base_plus', or 'large'
    start_time = time.time()
    DEVICE = "cpu"
    sam2_mask_generator = load_sam2_model(SAM2_CHECKPOINT, model_type=SAM2_MODEL_TYPE, device=DEVICE)
    sam2_masks = run_sam2_everything(img_bgr1, sam2_mask_generator)
    result_bgr_sam2 = visualize_sam2_masks(img_bgr1, sam2_masks)
    sam2_contours = sam2_masks_to_contours(sam2_masks)
    print(f"SAM2 segmentation took {time.time() - start_time:.2f} seconds")

    # Choose which contours to use downstream (swap between sam1_contours / sam2_contours)
    contours = sam2_contours

    # Pose the contours in the original img_bgr (not cropped)
    contours_orig = warp_contours_to_original(contours, roi_pts, roi_size_px=400, img_bgr=img_bgr, visualize=True)

    # Filter contours: inside ROI only, Aruco markers excluded
    filtered = filter_contours(img_bgr, contours_orig, roi_pts, marker_dict, visualize=True)
    print(f"Contours after filtering: {len(filtered)}")

    # Select bright pixels within each filtered contour
    bright_pixels = select_bright_pixels(img_bgr, filtered, num_pixels=10)
    # for i, pixels in enumerate(bright_pixels):
    #     print(f"Contour {i}: {pixels}")


    # Draw bounding boxes around filtered contours
    img_with_boxes, boxes = draw_bounding_boxes(img_bgr, filtered, visualize=True)

    # Select and plot the 10 brightest pixels inside each filtered contour
    selected_pixels = select_bright_pixels(img_bgr, filtered, num_pixels=10, visualize=True)

    # Map the pixels in millimeter scale.
    pixels_mm = map_pixels_to_mm(roi_pts, selected_pixels, roi_size_mm=325.0, visualize=True, img_bgr=img_bgr)

