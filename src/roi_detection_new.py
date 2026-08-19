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


def select_roi_edge_markers(marker_side_corners, markers_per_side=2, position_tolerance_px=50):
    """
    Group markers into ROI sides (top, bottom, left, right) and flatten points.

    The selected markers per side are chosen from image perspective:
    - top: markers whose top side has the smallest y
    - bottom: markers whose bottom side has the largest y
    - left: markers whose left side has the smallest x
    - right: markers whose right side has the largest x

    Returns a dict with keys top/bottom/left/right containing selected markers.

    Selection logic per side:
    1) Sort markers by extreme-side position.
    2) Keep first marker as base.
    3) Keep additional markers only if their side position is within
       position_tolerance_px of the base marker on the same axis.
    """
    edge_markers = {}
    if not marker_side_corners:
        return edge_markers

    marker_items = list(marker_side_corners.items())

    def _select_with_tolerance(sorted_markers, side_key, axis):
        if not sorted_markers:
            return []

        selected = [sorted_markers[0]]
        base_marker = sorted_markers[0]
        base_pos = float(np.mean([pt[axis] for pt in base_marker[1][side_key]]))

        for candidate in sorted_markers[1:]:
            if len(selected) >= markers_per_side:
                break
            candidate_pos = float(np.mean([pt[axis] for pt in candidate[1][side_key]]))
            if abs(candidate_pos - base_pos) <= float(position_tolerance_px):
                selected.append(candidate)

        return selected

    top_markers = sorted(
        marker_items,
        key=lambda item: float(np.mean([pt[1] for pt in item[1]['top']]))
    )
    edge_markers['top'] = _select_with_tolerance(top_markers, side_key='top', axis=1)

    bottom_markers = sorted(
        marker_items,
        key=lambda item: float(np.mean([pt[1] for pt in item[1]['bottom']])),
        reverse=True
    )
    edge_markers['bottom'] = _select_with_tolerance(bottom_markers, side_key='bottom', axis=1)

    left_markers = sorted(
        marker_items,
        key=lambda item: float(np.mean([pt[0] for pt in item[1]['left']]))
    )
    edge_markers['left'] = _select_with_tolerance(left_markers, side_key='left', axis=0)

    right_markers = sorted(
        marker_items,
        key=lambda item: float(np.mean([pt[0] for pt in item[1]['right']])),
        reverse=True
    )
    edge_markers['right'] = _select_with_tolerance(right_markers, side_key='right', axis=0)


    return edge_markers


def create_edge_point_groups(edge_markers):
    """
    Create the 8 requested edge point groups from selected edge markers.

    Output groups:
    1. top_markers_top_points
    2. top_markers_bottom_points
    3. bottom_markers_top_points
    4. bottom_markers_bottom_points
    5. left_markers_left_points
    6. left_markers_right_points
    7. right_markers_left_points
    8. right_markers_right_points
    """
    edge_point_groups = {
        'top_markers_top_points': [],
        'top_markers_bottom_points': [],
        'bottom_markers_top_points': [],
        'bottom_markers_bottom_points': [],
        'left_markers_left_points': [],
        'left_markers_right_points': [],
        'right_markers_left_points': [],
        'right_markers_right_points': [],
    }

    def _append_points(selected_markers, marker_side, target_group):
        for marker_id, marker_sides in selected_markers:
            for pt in marker_sides[marker_side]:
                edge_point_groups[target_group].append((float(pt[0]), float(pt[1])))

    top_markers = edge_markers.get('top', [])
    bottom_markers = edge_markers.get('bottom', [])
    left_markers = edge_markers.get('left', [])
    right_markers = edge_markers.get('right', [])

    _append_points(top_markers, 'top', 'top_markers_top_points')
    _append_points(top_markers, 'bottom', 'top_markers_bottom_points')
    _append_points(bottom_markers, 'top', 'bottom_markers_top_points')
    _append_points(bottom_markers, 'bottom', 'bottom_markers_bottom_points')
    _append_points(left_markers, 'left', 'left_markers_left_points')
    _append_points(left_markers, 'right', 'left_markers_right_points')
    _append_points(right_markers, 'left', 'right_markers_left_points')
    _append_points(right_markers, 'right', 'right_markers_right_points')

    return edge_point_groups


def get_extreme_line_endpoints(all_corners, mode, side, threshold=15):
    """
    all_corners: list of (x, y) tuples
    mode: 'horizontal' or 'vertical'
    side: 'top', 'bottom', 'left', 'right'
    threshold: pixel threshold for close points
    Returns two endpoints (np.array) for the fitted line.
    """
    points = list(all_corners)
    if len(points) < 2:
        raise ValueError("At least two valid points are required to fit an edge line")

    if mode == 'horizontal':
        # Use y value for sorting and thresholding
        # sorted_pts = sorted(points, key=lambda pt: pt[1])
        # if side == 'top':
        #     ref_y = sorted_pts[0][1]
        #     close_pts = [pt for pt in sorted_pts if abs(pt[1] - ref_y) <= threshold]
        # else:  # 'bottom'
        #     ref_y = sorted_pts[-1][1]
        #     close_pts = [pt for pt in reversed(sorted_pts) if abs(pt[1] - ref_y) <= threshold]
        if len(points) >= 2:
            xs = np.array([pt[0] for pt in points])
            ys = np.array([pt[1] for pt in points])
            m, b = np.polyfit(xs, ys, 1)
            x_min, x_max = xs.min(), xs.max()
            pt1 = np.array([x_min, m * x_min + b])
            pt2 = np.array([x_max, m * x_max + b])
            return pt1, pt2
        else:
            if side == 'top':
                return points[0], points[1]
            else:
                return points[-2], points[-1]
    elif mode == 'vertical':
        # # Use x value for sorting and thresholding
        # sorted_pts = sorted(points, key=lambda pt: pt[0])
        # if side == 'left':
        #     ref_x = sorted_pts[0][0]
        #     close_pts = [pt for pt in sorted_pts if abs(pt[0] - ref_x) <= threshold]
        # else:  # 'right'
        #     ref_x = sorted_pts[-1][0]
        #     close_pts = [pt for pt in reversed(sorted_pts) if abs(pt[0] - ref_x) <= threshold]
        if len(points) >= 2:
            xs = np.array([pt[0] for pt in points])
            ys = np.array([pt[1] for pt in points])
            m, b = np.polyfit(ys, xs, 1)
            y_min, y_max = ys.min(), ys.max()
            pt1 = np.array([m * y_min + b, y_min])
            pt2 = np.array([m * y_max + b, y_max])
            return pt1, pt2
        else:
            if side == 'left':
                return points[0], points[1]
            else:
                return points[-2], points[-1]
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
    marker_side_corners = extract_marker_side_corners(marker_dict)

    # Now group the markers by side (top, bottom, left, right) for edges
    # e.g. For selection of 'Top' marker, the 'top' side of 2 markers have the smallest y value 
    roi_edge_markers = select_roi_edge_markers(marker_side_corners, markers_per_side=2)
    edge_point_groups = create_edge_point_groups(roi_edge_markers)
    
    top_line1_pts = get_extreme_line_endpoints(edge_point_groups['top_markers_top_points'],
                                            mode='horizontal', 
                                            side='top')
    top_line2_pts = get_extreme_line_endpoints(edge_point_groups['top_markers_bottom_points'],
                                            mode='horizontal', 
                                            side='top')
    bottom_line1_pts = get_extreme_line_endpoints(edge_point_groups['bottom_markers_top_points'],
                                            mode='horizontal',          
                                            side='top')
    bottom_line2_pts = get_extreme_line_endpoints(edge_point_groups['bottom_markers_bottom_points'],
                                            mode='horizontal', 
                                            side='top')
    left_line1_pts = get_extreme_line_endpoints(edge_point_groups['left_markers_left_points'],
                                            mode='vertical', 
                                            side='left')
    left_line2_pts = get_extreme_line_endpoints(edge_point_groups['left_markers_right_points'],
                                            mode='vertical',    
                                            side='left')
    right_line1_pts = get_extreme_line_endpoints(edge_point_groups['right_markers_left_points'],
                                            mode='vertical',    
                                            side='left')
    right_line2_pts = get_extreme_line_endpoints(edge_point_groups['right_markers_right_points'],
                                            mode='vertical',    
                                            side='left')



    # 2. Fit lines from the selected extreme points
    top_line1 = line_from_points(top_line1_pts[0], top_line1_pts[1])
    top_line2 = line_from_points(top_line2_pts[0], top_line2_pts[1])
    bottom_line1 = line_from_points(bottom_line1_pts[0], bottom_line1_pts[1])
    bottom_line2 = line_from_points(bottom_line2_pts[0], bottom_line2_pts[1])
    left_line1 = line_from_points(left_line1_pts[0], left_line1_pts[1])
    left_line2 = line_from_points(left_line2_pts[0], left_line2_pts[1])
    right_line1 = line_from_points(right_line1_pts[0], right_line1_pts[1])
    right_line2 = line_from_points(right_line2_pts[0], right_line2_pts[1])


    roi_pts = []


    # Draw edge lines and ROI polygon (optional)
    if visualisation:
        img_vis = image.copy()

        # (line number, endpoints, color in BGR)
        line_specs = [
            (1, top_line1_pts, (0, 0, 255)),
            (2, top_line2_pts, (0, 165, 255)),
            (3, bottom_line1_pts, (0, 255, 255)),
            (4, bottom_line2_pts, (0, 255, 0)),
            (5, left_line1_pts, (255, 0, 0)),
            (6, left_line2_pts, (255, 0, 255)),
            (7, right_line1_pts, (255, 255, 0)),
            (8, right_line2_pts, (128, 128, 255)),
        ]

        h, w = img_vis.shape[:2]

        for line_number, line_pts, color in line_specs:
            p1 = np.asarray(line_pts[0], dtype=np.float32)
            p2 = np.asarray(line_pts[1], dtype=np.float32)

            direction = p2 - p1
            norm = float(np.linalg.norm(direction))
            if norm < 1e-6:
                continue

            unit_dir = direction / norm
            scale = float(max(h, w) * 4)

            start = (
                int(round(float(p1[0] - unit_dir[0] * scale))),
                int(round(float(p1[1] - unit_dir[1] * scale))),
            )
            end = (
                int(round(float(p2[0] + unit_dir[0] * scale))),
                int(round(float(p2[1] + unit_dir[1] * scale))),
            )

            ok, p1_clip, p2_clip = cv2.clipLine((0, 0, w, h), start, end)
            if not ok:
                continue

            cv2.line(img_vis, p1_clip, p2_clip, color, 2)

            mid_x = int((p1_clip[0] + p2_clip[0]) / 2)
            mid_y = int((p1_clip[1] + p2_clip[1]) / 2)
            cv2.putText(
                img_vis,
                str(line_number),
                (mid_x + 8, mid_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )

        if len(roi_pts) >= 3:
            roi_pts_int = np.int32(roi_pts)
            cv2.polylines(img_vis, [roi_pts_int], isClosed=True, color=(0, 255, 0), thickness=3)

        # Show the result
        cv2.imshow('ROI Detection', img_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return np.array(roi_pts)

def crop_roi_from_image(img_bgr, roi_pts, output_path, roi_size_px=400):
    """
    Crop the ROI area from the input image using a perspective transform.

    Args:
        img_bgr (np.ndarray): Input image (BGR).
        roi_pts (list): Four corner points of the ROI in pixel space (TL, TR, BR, BL).
        output_path (str): Path where the cropped ROI will be saved.
        roi_size_px (int): Output size (width and height in pixels) for the cropped ROI.

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

    # cv2.imwrite(output_path, roi_cropped)
    return roi_cropped

    # # Warp the whole image without cropping:
    # # Find where the original image corners land after the transform,
    # # then shift M so the full warped image fits within a positive canvas.
    # h, w = img_bgr.shape[:2]
    # img_corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
    # warped_corners = cv2.perspectiveTransform(img_corners, M)
    # x_min = warped_corners[:, 0, 0].min()
    # y_min = warped_corners[:, 0, 1].min()
    # x_max = warped_corners[:, 0, 0].max()
    # y_max = warped_corners[:, 0, 1].max()
    # out_w = int(np.ceil(x_max - x_min))
    # out_h = int(np.ceil(y_max - y_min))
    # # Translation matrix to shift the result so no pixels are cut off
    # T = np.array([[1, 0, -x_min],
    #               [0, 1, -y_min],
    #               [0, 0, 1]], dtype=np.float64)
    # M_full = T @ M
    # img_warped = cv2.warpPerspective(img_bgr, M_full, (out_w, out_h))

    # # Warp the ROI points using M_full
    # roi_pts_arr = np.float32(roi_pts).reshape(-1, 1, 2)
    # warped_roi_pts = cv2.perspectiveTransform(roi_pts_arr, M_full).reshape(-1, 2)

    # # Warp all marker corners
    # warped_marker_dict = {}
    # for marker_id, corners in marker_dict.items():
    #     # corners_arr = np.float32(corners)  # shape (1, 4, 2)
    #     corners_arr = np.float32(corners).reshape(-1, 1, 2)
    #     warped_corners = cv2.perspectiveTransform(corners_arr, M_full)
    #     warped_marker_dict[marker_id] = warped_corners

    # if visualisation:
    #     # Visualize warped ROI points and markers on img_warped
    #     img_warped_vis = img_warped.copy()
    #     cv2.polylines(img_warped_vis, [np.int32(warped_roi_pts)], isClosed=True, color=(0, 255, 0), thickness=2)
    #     for pt in warped_roi_pts:
    #         cv2.circle(img_warped_vis, tuple(np.int32(pt)), 5, (0, 0, 255), -1)
    #     # Draw all warped markers
    #     for corners in warped_marker_dict.values():
    #         cv2.polylines(img_warped_vis, [np.int32(corners)], isClosed=True, color=(255, 0, 0), thickness=2)
    #         for pt in corners[0]:
    #             cv2.circle(img_warped_vis, tuple(np.int32(pt)), 4, (255, 0, 0), -1)

    #     cv2.imwrite("cropped_ROI.png", roi_cropped)
    #     cv2.imshow("Cropped ROI", roi_cropped)
    #     cv2.imshow("Warped Full Image", img_warped_vis)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    # return roi_cropped, img_warped, warped_roi_pts, warped_marker_dict
