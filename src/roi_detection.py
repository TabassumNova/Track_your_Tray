
from shutil import which

import numpy as np
import cv2

CONSIDERED_MARKER = [34, 38, 39, 37, 35, 46, 45, 42, 49, 53, 43, 32]

def get_extreme_line_endpoints(all_corners, mode, side, threshold=3):
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
            if which == 'left':
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

def find_ROI(image, marker_dict):

    # Flatten all corners for only considered markers
    all_corners = []
    for marker_id, corners in marker_dict.items():
        if marker_id in CONSIDERED_MARKER:
            for i, pt in enumerate(corners[0]):
                all_corners.append({'id': marker_id, 'pt': pt})

    # 1. Find markers with extreme values among considered markers only
    # top = min(all_corners, key=lambda x: x['pt'][1])
    # bottom = max(all_corners, key=lambda x: x['pt'][1])
    # left = min(all_corners, key=lambda x: x['pt'][0])
    # right = max(all_corners, key=lambda x: x['pt'][0])

    
    # 2. For each, get the corresponding point (for y, use horizontal corner; for x, use vertical corner)
    # For top/bottom, use the leftmost or rightmost corner of that marker (horizontal)
    top_pts = get_extreme_line_endpoints(all_corners, mode='horizontal', side='top')
    bottom_pts = get_extreme_line_endpoints(all_corners, mode='horizontal', side='bottom')
    left_pts = get_extreme_line_endpoints(all_corners, mode='vertical', side='left')
    right_pts = get_extreme_line_endpoints(all_corners, mode='vertical', side='right')

    # 3. Draw lines between each pair
    # Top-Bottom (vertical), Left-Right (horizontal)
    # Actually, we want 4 lines: top, bottom, left, right
    # So, for each side, find the two points that define the line
    # We'll use the top and bottom points for the vertical lines, left and right for horizontal
    # But for intersection, we need the equations of the lines

    # For visualization (optional):
    img_vis = image.copy()
    # cv2.circle(img_vis, tuple(np.int32(top_pts[0])), 4, (255,0,0), -1)
    # cv2.circle(img_vis, tuple(np.int32(top_pts[1])), 4, (255,0,0), -1)
    # cv2.circle(img_vis, tuple(np.int32(bottom_pts[0])), 4, (0,255,0), -1)
    # cv2.circle(img_vis, tuple(np.int32(bottom_pts[1])), 4, (0,255,0), -1)
    # cv2.circle(img_vis, tuple(np.int32(left_pts[0])), 4, (0,0,255), -1)
    # cv2.circle(img_vis, tuple(np.int32(left_pts[1])), 4, (0,0,255), -1)
    # cv2.circle(img_vis, tuple(np.int32(right_pts[0])), 8, (0,255,255), -1)
    # cv2.circle(img_vis, tuple(np.int32(right_pts[1])), 8, (0,255,255), -1)

    # 4. Find intersection points of the 4 lines
    # Define lines: top-bottom (vertical), left-right (horizontal)
    # We'll define lines as (pt1, pt2)
    # For a rectangle, the ROI corners are the intersections of these lines:
    # (top, left), (top, right), (bottom, left), (bottom, right)

    # Define lines
    # Top and bottom: horizontal lines
    # Left and right: vertical lines
    # For each, we need two points. We'll use the topmost and bottommost for horizontal, leftmost and rightmost for vertical
    # For simplicity, use the same point for both ends (since we only have one per extreme)
    # But for intersection, that's enough
    top_line = line_from_points(top_pts[0], top_pts[1])
    bottom_line = line_from_points(bottom_pts[0], bottom_pts[1])
    left_line = line_from_points(left_pts[0], left_pts[1])
    right_line = line_from_points(right_pts[0], right_pts[1])

    # Intersections: (top & left), (top & right), (bottom & left), (bottom & right)
    roi_pts = [
        intersection(top_line, left_line),
        intersection(top_line, right_line),
        intersection(bottom_line, right_line),
        intersection(bottom_line, left_line),
    ]



    # Draw ROI polygon (optional)
    for pt in roi_pts:
        if pt is not None:
            cv2.circle(img_vis, tuple(np.int32(pt)), 10, (0,0,255), 2)
    # Draw filled rectangle (polygon) with the 4 ROI points
    roi_pts_int = np.int32(roi_pts)
    cv2.polylines(img_vis, [roi_pts_int], isClosed=True, color=(0,255,0), thickness=3)
    # cv2.fillPoly(img_vis, [roi_pts_int], color=(0,128,255))  # semi-transparent orange fill

    # Show the result
    cv2.imshow('ROI Detection', img_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return np.array(roi_pts)

def find_grid(image, marker_dict):
    img_vis = image.copy()
    
    h, w = img_vis.shape[:2]
    for marker_id, corners in marker_dict.items():
        if marker_id in CONSIDERED_MARKER:
            top_pts = get_horizontal_corners(marker_dict[marker_id], 'top')
            bottom_pts = get_horizontal_corners(marker_dict[marker_id], 'bottom')
            left_pts = get_vertical_corners(marker_dict[marker_id], 'left')
            right_pts = get_vertical_corners(marker_dict[marker_id], 'right')

            # Fit lines and extend to image borders
            # Top line (horizontal): y = m*x + b
            x1, y1 = top_pts[0]
            x2, y2 = top_pts[1]
            if x2 != x1:
                m_top = (y2 - y1) / (x2 - x1)
                b_top = y1 - m_top * x1
                pt1_top = (0, int(b_top))
                pt2_top = (w-1, int(m_top * (w-1) + b_top))
            else:
                pt1_top = (int(x1), 0)
                pt2_top = (int(x1), h-1)
            cv2.line(img_vis, pt1_top, pt2_top, (255, 0, 255), 2)

            # Bottom line (horizontal): y = m*x + b
            x1, y1 = bottom_pts[0]
            x2, y2 = bottom_pts[1]
            if x2 != x1:
                m_bot = (y2 - y1) / (x2 - x1)
                b_bot = y1 - m_bot * x1
                pt1_bot = (0, int(b_bot))
                pt2_bot = (w-1, int(m_bot * (w-1) + b_bot))
            else:
                pt1_bot = (int(x1), 0)
                pt2_bot = (int(x1), h-1)
            cv2.line(img_vis, pt1_bot, pt2_bot, (0, 255, 255), 2)

            # Left line (vertical): x = c
            x1, y1 = left_pts[0]
            x2, y2 = left_pts[1]
            if y2 != y1:
                m_left = (x2 - x1) / (y2 - y1)
                b_left = x1 - m_left * y1
                pt1_left = (int(m_left * 0 + b_left), 0)
                pt2_left = (int(m_left * (h-1) + b_left), h-1)
            else:
                pt1_left = (0, int(y1))
                pt2_left = (w-1, int(y1))
            cv2.line(img_vis, pt1_left, pt2_left, (0, 255, 0), 2)

            # Right line (vertical): x = c
            x1, y1 = right_pts[0]
            x2, y2 = right_pts[1]
            if y2 != y1:
                m_right = (x2 - x1) / (y2 - y1)
                b_right = x1 - m_right * y1
                pt1_right = (int(m_right * 0 + b_right), 0)
                pt2_right = (int(m_right * (h-1) + b_right), h-1)
            else:
                pt1_right = (0, int(y1))
                pt2_right = (w-1, int(y1))
            cv2.line(img_vis, pt1_right, pt2_right, (0, 128, 255), 2)

    # Show the result
    cv2.imshow('Grid Lines', img_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()