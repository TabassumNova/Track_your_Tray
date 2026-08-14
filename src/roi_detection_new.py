'''
This script is for detecting ROI with more flexibility on choosing the edges


'''
from shutil import which

import numpy as np
import cv2


def extract_marker_side_corners(marker_dict):
    """
    Build side-wise corner groups per marker from image perspective.

    Returns a dict with keys: top, bottom, left, right.
    Each value is a list of {'id': marker_id, 'pt': np.array([x, y])}.
    For each marker, two corners are added per side.
    """
    side_corners = {}

    for marker_id, corners in marker_dict.items():
        pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] < 2:
            continue

        sorted_by_y = np.argsort(pts[:, 1])
        sorted_by_x = np.argsort(pts[:, 0])

        top_indices = sorted_by_y[:2]
        bottom_indices = sorted_by_y[-2:]
        left_indices = sorted_by_x[:2]
        right_indices = sorted_by_x[-2:]

        side_corners[marker_id] = {'top': [pts[i] for i in top_indices],
                                   'bottom': [pts[i] for i in bottom_indices],
                                   'left': [pts[i] for i in left_indices],
                                   'right': [pts[i] for i in right_indices]}

        # for idx in top_indices:
        #     side_corners['top'].append({'id': marker_id, 'pt': pts[idx]})
        # for idx in bottom_indices:
        #     side_corners['bottom'].append({'id': marker_id, 'pt': pts[idx]})
        # for idx in left_indices:
        #     side_corners['left'].append({'id': marker_id, 'pt': pts[idx]})
        # for idx in right_indices:
        #     side_corners['right'].append({'id': marker_id, 'pt': pts[idx]})

    return side_corners


def select_roi_edge_markers(marker_side_corners, markers_per_side=2):
    """
    Group markers into ROI sides (top, bottom, left, right) and flatten points.

    The selected markers per side are chosen from image perspective:
    - top: markers whose top side has the smallest y
    - bottom: markers whose bottom side has the largest y
    - left: markers whose left side has the smallest x
    - right: markers whose right side has the largest x

    Returns a dict with keys top/bottom/left/right where each value is a list
    of {'id': marker_id, 'pt': np.array([x, y])}.
    """
    edge_markers = {}
    if not marker_side_corners:
        return edge_markers

    marker_items = list(marker_side_corners.items())

    top_markers = sorted(
        marker_items,
        key=lambda item: float(np.mean([pt[1] for pt in item[1]['top']]))
    )[:markers_per_side]
    edge_markers['top'] = top_markers
    bottom_markers = sorted(
        marker_items,
        key=lambda item: float(np.mean([pt[1] for pt in item[1]['bottom']])),
        reverse=True
    )[:markers_per_side]
    edge_markers['bottom'] = bottom_markers
    left_markers = sorted(
        marker_items,
        key=lambda item: float(np.mean([pt[0] for pt in item[1]['left']]))
    )[:markers_per_side]
    edge_markers['left'] = left_markers
    right_markers = sorted(
        marker_items,
        key=lambda item: float(np.mean([pt[0] for pt in item[1]['right']])),
        reverse=True
    )[:markers_per_side]
    edge_markers['right'] = right_markers


    return edge_markers


def get_extreme_line_endpoints(all_corners, mode, side, threshold=15):
    """
    all_corners: list of dicts with 'pt' key
    mode: 'horizontal' or 'vertical'
    side: 'top', 'bottom', 'left', 'right'
    threshold: pixel threshold for close points
    Returns two endpoints (np.array) for the fitted line.
    """
    if mode == 'horizontal':
        # Use y value for sorting and thresholding
        sorted_pts = sorted(all_corners, key=lambda x: x['pt'][1])
        if side == 'top':
            ref_y = sorted_pts[0]['pt'][1]
            close_pts = [p for p in sorted_pts if abs(p['pt'][1] - ref_y) <= threshold]
        else:  # 'bottom'
            ref_y = sorted_pts[-1]['pt'][1]
            close_pts = [p for p in reversed(sorted_pts) if abs(p['pt'][1] - ref_y) <= threshold]
        if len(close_pts) >= 2:
            xs = np.array([p['pt'][0] for p in close_pts])
            ys = np.array([p['pt'][1] for p in close_pts])
            m, b = np.polyfit(xs, ys, 1)
            x_min, x_max = xs.min(), xs.max()
            pt1 = np.array([x_min, m * x_min + b])
            pt2 = np.array([x_max, m * x_max + b])
            return pt1, pt2
        else:
            if side == 'top':
                return sorted_pts[0]['pt'], sorted_pts[1]['pt']
            else:
                return sorted_pts[-2]['pt'], sorted_pts[-1]['pt']
    elif mode == 'vertical':
        # Use x value for sorting and thresholding
        sorted_pts = sorted(all_corners, key=lambda x: x['pt'][0])
        if side == 'left':
            ref_x = sorted_pts[0]['pt'][0]
            close_pts = [p for p in sorted_pts if abs(p['pt'][0] - ref_x) <= threshold]
        else:  # 'right'
            ref_x = sorted_pts[-1]['pt'][0]
            close_pts = [p for p in reversed(sorted_pts) if abs(p['pt'][0] - ref_x) <= threshold]
        if len(close_pts) >= 2:
            xs = np.array([p['pt'][0] for p in close_pts])
            ys = np.array([p['pt'][1] for p in close_pts])
            m, b = np.polyfit(ys, xs, 1)
            y_min, y_max = ys.min(), ys.max()
            pt1 = np.array([m * y_min + b, y_min])
            pt2 = np.array([m * y_max + b, y_max])
            return pt1, pt2
        else:
            if side == 'left':
                return sorted_pts[0]['pt'], sorted_pts[1]['pt']
            else:
                return sorted_pts[-2]['pt'], sorted_pts[-1]['pt']
    else:
        raise ValueError("mode must be 'horizontal' or 'vertical'")

    
def line_from_points(p1, p2):
    # Returns (A, B, C) for line Ax + By = C
    A = p2[1] - p1[1]
    B = p1[0] - p2[0]
    C = A * p1[0] + B * p1[1]
    return A, B, C

def intersection(L1, L2):
    # L1, L2: (A, B, C)
    D = L1[0]*L2[1] - L2[0]*L1[1]
    if D == 0:
        return None  # Parallel
    Dx = L1[2]*L2[1] - L2[2]*L1[1]
    Dy = L1[0]*L2[2] - L2[0]*L1[2]
    x = Dx / D
    y = Dy / D
    return np.array([x, y])

def find_ROI(image, marker_dict, considered_markers, visualisation=True):

    # # Flatten all corners for only considered markers
    # all_corners = []
    # for marker_id, corners in marker_dict.items():
    #     if marker_id in considered_markers and corners is not None:
    #         for i, pt in enumerate(corners):
    #             all_corners.append({'id': marker_id, 'pt': pt})

    # Remove markers from marker dict that are not in considered_markers
    marker_dict = {k: v for k, v in marker_dict.items() if k in considered_markers and v is not None}

    # For each marker, collect side-specific corners from image perspective.
    marker_side_points = extract_marker_side_corners(marker_dict)

    # Now group the markers by side (top, bottom, left, right) for edges
    # e.g. For selection of 'Top' marker, the 'top' side of 2 markers have the smallest y value 
    roi_edge_points = select_roi_edge_markers(marker_side_points, markers_per_side=2)

    
    # 
    
    roi_pts = []


    # Draw ROI polygon (optional)
    if visualisation:
        img_vis = image.copy()
        for pt in roi_pts:
            if pt is not None:
                cv2.circle(img_vis, tuple(np.int32(pt)), 10, (0,0,255), 2)
        # Draw filled rectangle (polygon) with the 4 ROI points
        roi_pts_int = np.int32(roi_pts)
        cv2.polylines(img_vis, [roi_pts_int], isClosed=True, color=(0,255,0), thickness=3)

        # Show the result
        cv2.imshow('ROI Detection', img_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return np.array(roi_pts)

def crop_roi_from_image(img_bgr, roi_pts, marker_dict, roi_size_px=400, visualisation=True):
    """
    Crop the ROI area from the input image using a perspective transform.

    Args:
        img_bgr (np.ndarray): Input image (BGR).
        roi_pts (list): Four corner points of the ROI in pixel space (TL, TR, BR, BL).
        roi_size_px (int): Output size (width and height in pixels) for the cropped ROI.
        visualisation (bool): If True, display the cropped ROI.

    Returns:
        roi_cropped (np.ndarray): Cropped, perspective-corrected ROI image.
    """
    src = np.float32(roi_pts)
    dst = np.float32([
        [0, 0],
        [roi_size_px - 1, 0],
        [roi_size_px - 1, roi_size_px - 1],
        [0, roi_size_px - 1],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    roi_cropped = cv2.warpPerspective(img_bgr, M, (roi_size_px, roi_size_px))

    # Warp the whole image without cropping:
    # Find where the original image corners land after the transform,
    # then shift M so the full warped image fits within a positive canvas.
    h, w = img_bgr.shape[:2]
    img_corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(img_corners, M)
    x_min = warped_corners[:, 0, 0].min()
    y_min = warped_corners[:, 0, 1].min()
    x_max = warped_corners[:, 0, 0].max()
    y_max = warped_corners[:, 0, 1].max()
    out_w = int(np.ceil(x_max - x_min))
    out_h = int(np.ceil(y_max - y_min))
    # Translation matrix to shift the result so no pixels are cut off
    T = np.array([[1, 0, -x_min],
                  [0, 1, -y_min],
                  [0, 0, 1]], dtype=np.float64)
    M_full = T @ M
    img_warped = cv2.warpPerspective(img_bgr, M_full, (out_w, out_h))

    # Warp the ROI points using M_full
    roi_pts_arr = np.float32(roi_pts).reshape(-1, 1, 2)
    warped_roi_pts = cv2.perspectiveTransform(roi_pts_arr, M_full).reshape(-1, 2)

    # Warp all marker corners
    warped_marker_dict = {}
    for marker_id, corners in marker_dict.items():
        # corners_arr = np.float32(corners)  # shape (1, 4, 2)
        corners_arr = np.float32(corners).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners_arr, M_full)
        warped_marker_dict[marker_id] = warped_corners

    if visualisation:
        # Visualize warped ROI points and markers on img_warped
        img_warped_vis = img_warped.copy()
        cv2.polylines(img_warped_vis, [np.int32(warped_roi_pts)], isClosed=True, color=(0, 255, 0), thickness=2)
        for pt in warped_roi_pts:
            cv2.circle(img_warped_vis, tuple(np.int32(pt)), 5, (0, 0, 255), -1)
        # Draw all warped markers
        for corners in warped_marker_dict.values():
            cv2.polylines(img_warped_vis, [np.int32(corners)], isClosed=True, color=(255, 0, 0), thickness=2)
            for pt in corners[0]:
                cv2.circle(img_warped_vis, tuple(np.int32(pt)), 4, (255, 0, 0), -1)

        cv2.imwrite("cropped_ROI.png", roi_cropped)
        cv2.imshow("Cropped ROI", roi_cropped)
        cv2.imshow("Warped Full Image", img_warped_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return roi_cropped, img_warped, warped_roi_pts, warped_marker_dict
