from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from track_your_tray.pipeline.pipeline import (
    compute_edge_candidates,
    compute_roi_from_selected_lines,
    detect_marker_corners,
    draw_candidates_and_roi,
    load_image_to_bgr,
    parse_marker_ids,
    save_cropped_roi,
)


st.set_page_config(page_title="Tray ROI Assistant", layout="wide")
st.title("Tray ROI Assistant")
st.caption("Load an image, detect marker-based edge candidates, and select final ROI edges.")

with st.sidebar:
    st.header("Inputs")
    image_path = st.text_input(
        "Image path",
        value="",
        placeholder="/path/to/image.png or /path/to/capture.hdr",
    )
    marker_ids_text = st.text_input(
        "Considered marker IDs",
        value="9, 12, 20, 21",
        help="Comma-separated marker IDs",
    )
    aruco_script_path = st.text_input(
        "Aruco detection script path",
        value="/Users/nova98/Documents/Nova/Marker-detection/src/aruco_detection.py",
    )
    corner_key = st.selectbox("Corner set", ["outer_corners", "inner_corners"], index=0)
    markers_per_side = st.slider("Markers per side", min_value=1, max_value=4, value=2, step=1)
    run_btn = st.button("Detect Candidate Edges", type="primary")

if "endpoints" not in st.session_state:
    st.session_state.endpoints = None
if "img_bgr" not in st.session_state:
    st.session_state.img_bgr = None
if "selected_markers" not in st.session_state:
    st.session_state.selected_markers = None
if "marker_dict" not in st.session_state:
    st.session_state.marker_dict = None
if "image_path_input" not in st.session_state:
    st.session_state.image_path_input = None

if run_btn:
    try:
        if not image_path.strip():
            raise ValueError("Please provide an image path")

        considered_markers = parse_marker_ids(marker_ids_text)
        if not considered_markers:
            raise ValueError("Please provide at least one marker ID")

        img_bgr = load_image_to_bgr(image_path)
        marker_dict = detect_marker_corners(img_bgr, aruco_script_path, corner_key=corner_key)

        selected_markers, edge_point_groups, endpoints, lines_abc = compute_edge_candidates(
            marker_dict,
            considered_markers,
            markers_per_side=markers_per_side,
        )

        st.session_state.img_bgr = img_bgr
        st.session_state.endpoints = endpoints
        st.session_state.lines_abc = lines_abc
        st.session_state.selected_markers = selected_markers
        st.session_state.marker_dict = marker_dict
        st.session_state.edge_point_groups = edge_point_groups
        st.session_state.image_path_input = image_path
        st.success(f"Detected markers: {len(marker_dict)} | Considered markers found: {len(selected_markers)}")
    except Exception as exc:
        st.error(str(exc))

if st.session_state.endpoints is not None and st.session_state.img_bgr is not None:
    detected_ids = sorted(st.session_state.marker_dict.keys()) if st.session_state.marker_dict else []

    st.subheader("Detected ArUco IDs")
    st.markdown("All detected IDs:")
    st.code(str(detected_ids), language="text")

    st.subheader("Select ROI Edge Lines")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        top_choice = st.selectbox("Top side", ["1", "2"], index=0)
    with c2:
        bottom_choice = st.selectbox("Bottom side", ["3", "4"], index=1)
    with c3:
        left_choice = st.selectbox("Left side", ["5", "6"], index=0)
    with c4:
        right_choice = st.selectbox("Right side", ["7", "8"], index=1)

    selected_line_names = {
        "top": top_choice,
        "bottom": bottom_choice,
        "left": left_choice,
        "right": right_choice,
    }

    roi_pts = None
    roi_error = None
    try:
        # Map user selections (1-8) to actual line keys (top_1, top_2, etc.)
        line_mapping = {
            "1": "top_1", "2": "top_2",
            "3": "bottom_1", "4": "bottom_2",
            "5": "left_1", "6": "left_2",
            "7": "right_1", "8": "right_2",
        }
        roi_pts = compute_roi_from_selected_lines(
            st.session_state.lines_abc,
            top_choice=line_mapping[top_choice],
            bottom_choice=line_mapping[bottom_choice],
            left_choice=line_mapping[left_choice],
            right_choice=line_mapping[right_choice],
        )
    except Exception as exc:
        roi_error = str(exc)

    overlay_bgr = draw_candidates_and_roi(
        st.session_state.img_bgr,
        st.session_state.endpoints,
        selected_line_names,
        marker_corners=st.session_state.selected_markers,
        roi_pts=roi_pts,
    )

    st.image(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB), caption="Candidate lines and selected ROI", use_container_width=True)

    if roi_error:
        st.warning(roi_error)
    elif roi_pts is not None:
        st.markdown("Selected ROI points (TL, TR, BR, BL):")
        st.code(np.array2string(roi_pts, precision=2), language="text")

        # Add save button
        col1, col2 = st.columns([1, 3])
        with col1:
            save_btn = st.button("Save Cropped ROI", type="primary")
        with col2:
            st.empty()

        if save_btn:
            try:
                # Show where we're trying to save
                st.info(f"Saving to: {st.session_state.image_path_input}")
                
                output_path, roi_cropped = save_cropped_roi(
                    st.session_state.img_bgr,
                    roi_pts,
                    st.session_state.image_path_input,
                    roi_size_px=400,
                )
                st.success(f"✅ ROI saved successfully to:\n`{output_path}`")
                st.image(cv2.cvtColor(roi_cropped, cv2.COLOR_BGR2RGB), caption="Saved Cropped ROI", use_container_width=True)
            except Exception as exc:
                st.error(f"❌ Error saving ROI:\n{str(exc)}\n\nTraceback:\n{type(exc).__name__}")

    with st.expander("Inspect edge point groups"):
        st.json(st.session_state.edge_point_groups)

else:
    st.info("Provide inputs in the sidebar and click 'Detect Candidate Edges'.")
