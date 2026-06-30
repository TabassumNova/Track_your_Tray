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

import time

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


def map_pixels_to_mm(roi_pts, selected_pixels, roi_size_mm=325.0, visualize=True, img=None):
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
        img            (np.ndarray):      Image to annotate (required when visualize=True).

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

    if visualize and img is not None:
        vis = img.copy()
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


def select_pixels_by_click(img_bgr, window_name='Select Pixels (click to add, q to finish)'):
    """
    Open an interactive window and let the user click pixel locations.
    Each click is recorded as a separate single-pixel entry, compatible
    with the output format of select_bright_pixels / map_pixels_to_mm.

    Controls:
        Left-click        — add a pixel at the clicked position.
        Right-click       — remove the last added pixel.
        Scroll wheel up   — zoom in  (centred on cursor).
        Scroll wheel down — zoom out (centred on cursor).
        + / =             — zoom in  (centred on image centre, keyboard fallback).
        - / _             — zoom out (centred on image centre, keyboard fallback).
        r                 — reset zoom and pan.
        Middle-click drag — pan the view.
        q / Enter         — confirm selection and close the window.

    Args:
        img_bgr     (np.ndarray): Input image in BGR format.
        window_name (str):        Title of the OpenCV window.

    Returns:
        selected_pixels (list of lists): Each sublist contains one
            (x, y, grayscale_value) tuple, one sublist per click.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = img_bgr.shape[:2]

    # Display size: cap at 1200 px on the larger side so the window fits on screen
    DISP_MAX = 1200
    scale_init = min(1.0, DISP_MAX / max(W, H))
    DW = int(W * scale_init)   # display-window width  (pixels on screen)
    DH = int(H * scale_init)   # display-window height (pixels on screen)

    clicked = []    # list of (x, y, val) in full-resolution image coordinates

    # ── View state ────────────────────────────────────────────────────────────
    # zoom  : magnification factor relative to the display window
    #         zoom=1 → the whole image fills the display window
    #         zoom=2 → a quarter of the image fills the display window
    # off_x/off_y : top-left corner of the visible region in *display* px coords
    state = {
        'zoom':   1.0,
        'off_x':  0.0,
        'off_y':  0.0,
        'pan':    False,
        'pan_sx': 0,
        'pan_sy': 0,
        'pan_ox': 0.0,
        'pan_oy': 0.0,
        'cursor_sx': DW // 2,   # last known cursor position in display coords
        'cursor_sy': DH // 2,
    }
    ZOOM_STEP = 1.20
    MIN_ZOOM  = 1.0
    MAX_ZOOM  = 30.0

    def _disp_to_img(sx, sy):
        """Convert display-window coords → full-resolution image coords."""
        img_x = (sx + state['off_x']) / state['zoom'] / scale_init
        img_y = (sy + state['off_y']) / state['zoom'] / scale_init
        return img_x, img_y

    def _img_to_disp(ix, iy):
        """Convert full-resolution image coords → display-window coords."""
        sx = ix * scale_init * state['zoom'] - state['off_x']
        sy = iy * scale_init * state['zoom'] - state['off_y']
        return sx, sy

    def _clamp_offset():
        max_off_x = max(0.0, DW * state['zoom'] - DW)
        max_off_y = max(0.0, DH * state['zoom'] - DH)
        state['off_x'] = max(0.0, min(state['off_x'], max_off_x))
        state['off_y'] = max(0.0, min(state['off_y'], max_off_y))

    def _apply_zoom(new_zoom, anchor_sx, anchor_sy):
        """Change zoom, keeping the image point under (anchor_sx, anchor_sy) fixed."""
        new_zoom = max(MIN_ZOOM, min(new_zoom, MAX_ZOOM))
        # image-coord of the anchor before zoom change
        img_ax = (anchor_sx + state['off_x']) / state['zoom'] / scale_init
        img_ay = (anchor_sy + state['off_y']) / state['zoom'] / scale_init
        state['zoom'] = new_zoom
        # re-compute offset so the same image point sits under the anchor
        state['off_x'] = img_ax * scale_init * new_zoom - anchor_sx
        state['off_y'] = img_ay * scale_init * new_zoom - anchor_sy
        _clamp_offset()

    def _render():
        z = state['zoom']
        ox, oy = state['off_x'], state['off_y']

        # Each display pixel corresponds to 1/z of a prescaled (DW×DH) pixel
        # Visit region in the DW×DH "scaled image" coordinates
        src_x0 = int(ox / z)
        src_y0 = int(oy / z)
        src_w  = int(DW / z) + 1
        src_h  = int(DH / z) + 1

        # Map to full-resolution image coords
        full_x0 = int(src_x0 / scale_init)
        full_y0 = int(src_y0 / scale_init)
        full_x1 = min(int((src_x0 + src_w) / scale_init) + 1, W)
        full_y1 = min(int((src_y0 + src_h) / scale_init) + 1, H)
        full_x0 = max(0, full_x0)
        full_y0 = max(0, full_y0)

        crop = img_bgr[full_y0:full_y1, full_x0:full_x1]
        frame = cv2.resize(crop, (DW, DH), interpolation=cv2.INTER_LINEAR)

        # Apply pan/zoom via warpAffine on the DW×DH frame
        M_zoom = np.float32([
            [z,  0, -ox],
            [0,  z, -oy],
        ])
        frame = cv2.warpAffine(
            cv2.resize(img_bgr, (DW, DH), interpolation=cv2.INTER_LINEAR),
            M_zoom, (DW, DH),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # Draw clicked points in display coords
        for i, (ix, iy, _) in enumerate(clicked):
            sx, sy = _img_to_disp(ix, iy)
            sx, sy = int(sx), int(sy)
            if -20 <= sx < DW + 20 and -20 <= sy < DH + 20:
                cv2.circle(frame, (sx, sy), 6, (0, 255, 255), -1)
                cv2.circle(frame, (sx, sy), 6, (0, 180, 180), 1)
                label = f"{i + 1}: ({int(ix)}, {int(iy)})"
                cv2.putText(frame, label, (sx + 9, sy - 9),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(frame,
                    f"Zoom: {z:.1f}x | clicks: {len(clicked)} | +/- zoom | r reset | q quit",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(window_name, frame)

    def _on_mouse(event, sx, sy, flags, param):
        state['cursor_sx'] = sx
        state['cursor_sy'] = sy
        ix, iy = _disp_to_img(sx, sy)
        ix = max(0.0, min(ix, W - 1.0))
        iy = max(0.0, min(iy, H - 1.0))

        if event == cv2.EVENT_LBUTTONDOWN:
            val = int(gray[int(round(iy)), int(round(ix))])
            clicked.append((ix, iy, val))
            _render()

        elif event == cv2.EVENT_RBUTTONDOWN and clicked:
            clicked.pop()
            _render()

        elif event == cv2.EVENT_MBUTTONDOWN:
            state['pan']    = True
            state['pan_sx'] = sx
            state['pan_sy'] = sy
            state['pan_ox'] = state['off_x']
            state['pan_oy'] = state['off_y']

        elif event == cv2.EVENT_MBUTTONUP:
            state['pan'] = False

        elif event == cv2.EVENT_MOUSEMOVE and state['pan']:
            state['off_x'] = state['pan_ox'] - (sx - state['pan_sx'])
            state['off_y'] = state['pan_oy'] - (sy - state['pan_sy'])
            _clamp_offset()
            _render()

        elif event == cv2.EVENT_MOUSEWHEEL:
            # flags is a raw C int passed as Python int; on some platforms
            # a negative scroll delta arrives as a large positive value —
            # reinterpret as signed 32-bit integer to get the correct sign.
            delta = np.int32(flags)
            new_zoom = (state['zoom'] * ZOOM_STEP
                        if delta > 0
                        else state['zoom'] / ZOOM_STEP)
            _apply_zoom(new_zoom, sx, sy)
            _render()

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    _render()
    cv2.setMouseCallback(window_name, _on_mouse)
    print("[select_pixels_by_click] Scroll/+/- to zoom, middle-drag to pan, "
          "left-click to add, right-click to undo, r to reset, q/Enter to confirm.")

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord('q'), 13):            # quit / confirm
            break
        elif key in (ord('+'), ord('=')):    # zoom in (keyboard)
            _apply_zoom(state['zoom'] * ZOOM_STEP,
                        state['cursor_sx'], state['cursor_sy'])
            _render()
        elif key in (ord('-'), ord('_')):    # zoom out (keyboard)
            _apply_zoom(state['zoom'] / ZOOM_STEP,
                        state['cursor_sx'], state['cursor_sy'])
            _render()
        elif key == ord('r'):                # reset view
            state['zoom'] = 1.0
            state['off_x'] = 0.0
            state['off_y'] = 0.0
            _render()

    cv2.destroyWindow(window_name)
    print(f"[select_pixels_by_click] {len(clicked)} pixel(s) selected.")

    # Return format: list of single-element sublists — same as select_bright_pixels
    return [[(int(round(x)), int(round(y)), val)] for x, y, val in clicked]



if __name__ == "__main__":
    print("Starting processing1...")
    # image load
    path = '/Users/nova98/Documents/Nova/Helios+/FX10/20260323/FX10_Aruco_random_2026-03-23_13-13-47_nicd_with_objects/capture/FX10_Aruco_random_2026-03-23_13-13-47.hdr'
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

    # # # Select pixels from mouse click by user (for testing purposes)
    # selected_pixels = select_pixels_by_click(img_warped)

    
    # # Test: Apply blob detection
    # keypoints = detect_blobs(img_bgr, visualize=True, min_area=50, max_area=5000, threshold=10)

    # # Test: Apply haris corner detection
    # corners = detect_harris_corners(img_bgr, visualize=True, block_size=2, ksize=3, k=0.04, threshold_rel=0.01)
    
    # Contour detection
    contours, hierarchy = detect_contours(img_bgr1, visualize=True)

    
    # # ── SAM1 segmentation ────────────────────────────────────────────────────
    # SAM1_CHECKPOINT_PATH = "/Users/nova98/Documents/Nova/3d_localization/sam_checkpoints"
    # SAM1_MODEL_TYPE = "vit_b"   # 'vit_h', 'vit_l', or 'vit_b'
    # DEVICE = "cpu"              # 'cuda' if GPU available
    # start_time = time.time()
    # sam1_contours = run_SAM1(img_bgr1, SAM1_CHECKPOINT_PATH, SAM1_MODEL_TYPE, DEVICE)
    # print(f"SAM1 segmentation took {time.time() - start_time:.2f} seconds")
    
    # # ── SAM2 segmentation ────────────────────────────────────────────────────
    # SAM2_CHECKPOINT = "/Users/nova98/Documents/Nova/3d_localization/sam_checkpoints/sam2.1_hiera_tiny.pt"
    # SAM2_MODEL_TYPE = "tiny"  # 'tiny', 'small', 'base_plus', or 'large'
    # start_time = time.time()
    # DEVICE = "cpu"
    # sam2_mask_generator = load_sam2_model(SAM2_CHECKPOINT, model_type=SAM2_MODEL_TYPE, device=DEVICE)
    # sam2_masks = run_sam2_everything(img_bgr1, sam2_mask_generator)
    # result_bgr_sam2 = visualize_sam2_masks(img_bgr1, sam2_masks)
    # sam2_contours = sam2_masks_to_contours(sam2_masks)
    # print(f"SAM2 segmentation took {time.time() - start_time:.2f} seconds")

    # Choose which contours to use downstream (swap between sam1_contours / sam2_contours)
    # contours = sam2_contours

    # Pose the contours in the original img_bgr (not cropped)
    contours_orig = warp_contours_to_original(contours, warped_roi_pts, roi_size_px=1000, img_bgr=img_warped, visualize=True)

    # Filter contours: inside ROI only, Aruco markers excluded
    filtered = filter_contours(img_warped, contours_orig, warped_roi_pts, warped_marker_dict, visualize=True)
    print(f"Contours after filtering: {len(filtered)}")

    # Select bright pixels within each filtered contour
    # bright_pixels = select_bright_pixels(img_warped, filtered, num_pixels=10)
    # for i, pixels in enumerate(bright_pixels):
    #     print(f"Contour {i}: {pixels}")


    # Draw bounding boxes around filtered contours
    img_with_boxes, boxes = draw_bounding_boxes(img_warped, filtered, visualize=True)

    # Select and plot the 10 brightest pixels inside each filtered contour
    selected_pixels = select_bright_pixels(img_warped, filtered, num_pixels=10, visualize=True)
    

    # Map the pixels in millimeter scale.
    pixels_mm = map_pixels_to_mm(warped_roi_pts, selected_pixels, roi_size_mm=311.0, visualize=True, img=img_warped)

