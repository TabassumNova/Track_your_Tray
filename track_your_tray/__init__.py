"""Track Your Tray - ROI Detection and Tracking System."""

__version__ = "0.1.0"
__author__ = "Your Name"

# Make key functions importable
from track_your_tray.roi_detection import (
    crop_roi_from_image,
    extract_marker_side_corners,
    intersection,
    line_from_points,
    select_roi_edge_markers,
)
from track_your_tray.pipeline import (
    parse_marker_ids,
    load_image_to_bgr,
    detect_marker_corners,
    compute_roi_from_selected_lines,
    save_cropped_roi,
)

__all__ = [
    "crop_roi_from_image",
    "extract_marker_side_corners",
    "intersection",
    "line_from_points",
    "select_roi_edge_markers",
    "parse_marker_ids",
    "load_image_to_bgr",
    "detect_marker_corners",
    "compute_roi_from_selected_lines",
    "save_cropped_roi",
]
