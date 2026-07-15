
from shutil import which

import numpy as np
import cv2


def get_extreme_line_endpoints(all_corners, mode, side, threshold=10):
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

    # Flatten all corners for only considered markers
    all_corners = []
    for marker_id, corners in marker_dict.items():
        if marker_id in considered_markers and corners is not None:
            for i, pt in enumerate(corners):
                all_corners.append({'id': marker_id, 'pt': pt})

    
    # 1. For each side, get 2 extreme points (for y, use horizontal corner; for x, use vertical corner)
    top_pts = get_extreme_line_endpoints(all_corners, mode='horizontal', side='top')
    bottom_pts = get_extreme_line_endpoints(all_corners, mode='horizontal', side='bottom')
    left_pts = get_extreme_line_endpoints(all_corners, mode='vertical', side='left')
    right_pts = get_extreme_line_endpoints(all_corners, mode='vertical', side='right')


    # 2. Fit lines from the selected extreme points
    top_line = line_from_points(top_pts[0], top_pts[1])
    bottom_line = line_from_points(bottom_pts[0], bottom_pts[1])
    left_line = line_from_points(left_pts[0], left_pts[1])
    right_line = line_from_points(right_pts[0], right_pts[1])

    # 3. Intersections: (top & left), (top & right), (bottom & left), (bottom & right)
    roi_pts = [
        intersection(top_line, left_line),
        intersection(top_line, right_line),
        intersection(bottom_line, right_line),
        intersection(bottom_line, left_line),
    ]


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
