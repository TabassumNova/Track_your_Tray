import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from track_your_tray.roi_detection.roi_detection import (
    create_edge_point_groups,
    crop_roi_from_image,
    extract_marker_side_corners,
    get_extreme_line_endpoints,
    intersection,
    line_from_points,
    select_roi_edge_markers,
)
from track_your_tray.roi_detection.visualization import plot_hyimage


def parse_marker_ids(marker_ids_text: str) -> List[int]:
    ids: List[int] = []
    for token in marker_ids_text.replace(";", ",").split(","):
        value = token.strip()
        if not value:
            continue
        ids.append(int(value))
    return ids


def load_image_to_bgr(image_path: str) -> np.ndarray:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image path not found: {image_path}")

    if path.suffix.lower() == ".hdr":
        import hylite
        from hylite import io

        hy_image = io.load(str(path))
        return plot_hyimage(hy_image, visualisation=False)

    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    return img_bgr


def load_aruco_module(aruco_script_path: str):
    script_path = Path(aruco_script_path)
    if not script_path.exists():
        raise FileNotFoundError(f"Aruco script not found: {aruco_script_path}")

    spec = importlib.util.spec_from_file_location("aruco_detection", str(script_path))
    if spec is None or spec.loader is None:
        raise ImportError("Could not create import spec for aruco_detection script")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def detect_marker_corners(img_bgr: np.ndarray, aruco_script_path: str, corner_key: str = "outer_corners") -> Dict[int, np.ndarray]:
    aruco_detection = load_aruco_module(aruco_script_path)
    marker_dict_raw = aruco_detection.getAruco(
        img_bgr,
        aruco_dict_id=cv2.aruco.DICT_4X4_1000,
        visualisation=False,
    )

    marker_dict = {k: v for k, v in marker_dict_raw.items() if corner_key in v}
    marker_dict = {k: v[corner_key] for k, v in marker_dict.items()}
    marker_dict = {k: v for k, v in marker_dict.items() if v is not None}
    return marker_dict


def compute_edge_candidates(marker_dict: Dict[int, np.ndarray], considered_markers: List[int], markers_per_side: int = 2):
    selected_markers = {k: v for k, v in marker_dict.items() if k in considered_markers and v is not None}
    marker_side_corners = extract_marker_side_corners(selected_markers)
    roi_edge_markers = select_roi_edge_markers(marker_side_corners, markers_per_side=markers_per_side)
    edge_point_groups = create_edge_point_groups(roi_edge_markers)

    endpoints = {
        "top_1": get_extreme_line_endpoints(edge_point_groups["top_markers_top_points"], mode="horizontal", side="top"),
        "top_2": get_extreme_line_endpoints(edge_point_groups["top_markers_bottom_points"], mode="horizontal", side="top"),
        "bottom_1": get_extreme_line_endpoints(edge_point_groups["bottom_markers_top_points"], mode="horizontal", side="top"),
        "bottom_2": get_extreme_line_endpoints(edge_point_groups["bottom_markers_bottom_points"], mode="horizontal", side="top"),
        "left_1": get_extreme_line_endpoints(edge_point_groups["left_markers_left_points"], mode="vertical", side="left"),
        "left_2": get_extreme_line_endpoints(edge_point_groups["left_markers_right_points"], mode="vertical", side="left"),
        "right_1": get_extreme_line_endpoints(edge_point_groups["right_markers_left_points"], mode="vertical", side="left"),
        "right_2": get_extreme_line_endpoints(edge_point_groups["right_markers_right_points"], mode="vertical", side="left"),
    }

    lines_abc = {name: line_from_points(pts[0], pts[1]) for name, pts in endpoints.items()}

    return selected_markers, edge_point_groups, endpoints, lines_abc


def compute_roi_from_selected_lines(lines_abc, top_choice: str, bottom_choice: str, left_choice: str, right_choice: str):
    top_line = lines_abc[top_choice]
    bottom_line = lines_abc[bottom_choice]
    left_line = lines_abc[left_choice]
    right_line = lines_abc[right_choice]

    roi_pts = [
        intersection(top_line, left_line),
        intersection(top_line, right_line),
        intersection(bottom_line, right_line),
        intersection(bottom_line, left_line),
    ]

    if any(pt is None for pt in roi_pts):
        raise ValueError("Selected lines produced parallel intersections; choose a different line combination")

    return np.array(roi_pts, dtype=np.float32)


def save_cropped_roi(img_bgr: np.ndarray, roi_pts: np.ndarray, input_image_path: str, roi_size_px: int = 400) -> Tuple[str, np.ndarray]:
    """
    Crop and save the ROI to the same parent directory as the input image.

    Args:
        img_bgr (np.ndarray): Input image (BGR).
        roi_pts (np.ndarray): Four corner points of the ROI (TL, TR, BR, BL).
        input_image_path (str): Path to the input image file.
        roi_size_px (int): Output size for the cropped ROI.

    Returns:
        Tuple[str, np.ndarray]: Output file path and cropped image.
    """
    input_path = Path(input_image_path)
    output_dir = input_path.parent
    output_filename = f"{input_path.stem}_cropped_roi.png"
    output_path = str(output_dir / output_filename)

    roi_cropped = crop_roi_from_image(img_bgr, roi_pts, output_path, roi_size_px=roi_size_px)
    return output_path, roi_cropped


def draw_candidates_and_roi(
    img_bgr: np.ndarray,
    endpoints: Dict[str, Tuple[np.ndarray, np.ndarray]],
    selected_line_names: Dict[str, str],
    marker_corners: Dict[int, np.ndarray] | None = None,
    roi_pts: np.ndarray | None = None,
) -> np.ndarray:
    img_vis = img_bgr.copy()

    color_map = {
        "top_1": (0, 0, 255),
        "top_2": (0, 165, 255),
        "bottom_1": (0, 255, 255),
        "bottom_2": (0, 255, 0),
        "left_1": (255, 0, 0),
        "left_2": (255, 0, 255),
        "right_1": (255, 255, 0),
        "right_2": (128, 128, 255),
    }

    ordered_names = ["top_1", "top_2", "bottom_1", "bottom_2", "left_1", "left_2", "right_1", "right_2"]
    h, w = img_vis.shape[:2]

    for idx, line_name in enumerate(ordered_names, start=1):
        line_pts = endpoints[line_name]
        color = color_map[line_name]
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
            str(idx),
            (mid_x + 8, mid_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    if marker_corners:
        for marker_id, corners in marker_corners.items():
            corners_arr = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
            if corners_arr.shape[0] == 0:
                continue

            cv2.polylines(img_vis, [np.int32(corners_arr)], isClosed=True, color=(255, 255, 255), thickness=2)
            center = np.mean(corners_arr, axis=0)
            label_pos = (int(center[0]) + 6, int(center[1]) - 6)
            cv2.putText(
                img_vis,
                f"ID {marker_id}",
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    if roi_pts is not None and len(roi_pts) == 4:
        cv2.polylines(img_vis, [np.int32(roi_pts)], isClosed=True, color=(255, 255, 255), thickness=3)
        for i, pt in enumerate(roi_pts):
            cv2.circle(img_vis, tuple(np.int32(pt)), 6, (255, 255, 255), -1)
            cv2.putText(
                img_vis,
                str(i + 1),
                (int(pt[0]) + 8, int(pt[1]) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    for side_name, line_name in selected_line_names.items():
        cv2.putText(
            img_vis,
            f"{side_name}: {line_name}",
            (12, 28 + 24 * list(selected_line_names.keys()).index(side_name)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return img_vis
