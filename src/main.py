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
from mapping import *
from pixel_selection import *

import time

'''
Aruco detection module
1. Clone the repository:
    - git clone https://github.com/TabassumNova/Marker-detection.git
2. Ensure the path to aruco_detection.py is correct in the import statement below.
'''
# Aruco detection
import importlib.util
spec = importlib.util.spec_from_file_location(
    "aruco_detection",
    "/Users/nova98/Documents/Nova/Marker-detection/src/aruco_detection.py" # Update this path to the actual location of aruco_detection.py
)
aruco_detection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aruco_detection)




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
    pixels_mm = map_pixels_to_mm(warped_roi_pts, selected_pixels, roi_size_mm=318.0, visualize=True, img=img_warped)

