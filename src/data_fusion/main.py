'''
Data fusion of ROI from point cloud and Gray scale HSI image
'''

import sys
import importlib
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

# Ensure project root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloader import *
from projection import *
from visualisation import *
from src.segmentation.sam_segmentation import *
from src.segmentation.sam2_segmentation import *
from src.segmentation.contour_detection import *

CLOUD_ANALYSIS_SRC = Path("/Users/nova98/Documents/Nova/cloud_analysis/src")
if str(CLOUD_ANALYSIS_SRC) not in sys.path:
    sys.path.insert(0, str(CLOUD_ANALYSIS_SRC))

# Point projection to 2D space
# Remove the cached local 'projection' module so importlib resolves to
# the external one in CLOUD_ANALYSIS_SRC (inserted at the front of sys.path).
sys.modules.pop("projection", None)
cloud_analysis = importlib.import_module("projection")
viz = importlib.import_module("viz")

def SIFT_feature_matching(img1, img2, method='flann', ratio_threshold=0.75,
                          min_matches=4, visualize=True, title=""):
    """
    Perform SIFT feature matching between a grayscale HSI crop and a height-map crop.

    Args:
        img1 (np.ndarray): Grayscale HSI crop (uint8 or float).
        img2 (np.ndarray): Height-map crop (float, may contain NaN).
        method (str): 'bf' for Brute-Force or 'flann' for FLANN (default: 'flann').
        ratio_threshold (float): Lowe's ratio test threshold (default: 0.75).
        min_matches (int): Minimum good matches required (default: 4).
        visualize (bool): Whether to display results (default: True).
        title (str): Optional title prefix for figures.

    Returns:
        dict with keys: status, matches, kp1, kp2, homography (or None).
    """

    def _to_uint8(img):
        """Normalise any 2-D array (including NaN floats) to uint8."""
        img = img.astype(np.float32)
        valid = ~np.isnan(img)
        if not np.any(valid):
            return np.zeros(img.shape, dtype=np.uint8)
        vmin, vmax = img[valid].min(), img[valid].max()
        out = np.zeros_like(img)
        if not np.isclose(vmin, vmax):
            out[valid] = (img[valid] - vmin) / (vmax - vmin) * 255.0
        return np.clip(out, 0, 255).astype(np.uint8)

    # Ensure both images are uint8 grayscale
    gray1 = _to_uint8(img1) if img1.dtype != np.uint8 else img1
    gray2 = _to_uint8(img2)  # heatmap always needs normalisation

    print(f"[SIFT] HSI crop: {gray1.shape}, Heatmap crop: {gray2.shape}")

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    print(f"[SIFT] Keypoints — HSI: {len(kp1)}, Heatmap: {len(kp2)}")

    empty_result = {
        'status': 'insufficient_matches',
        'matches': [],
        'kp1': kp1,
        'kp2': kp2,
        'homography': None,
    }

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        print("[SIFT] Not enough keypoints — skipping.")
        return empty_result

    # ── Matching ──────────────────────────────────────────────────────────────
    good_matches = []
    if method.lower() == 'bf':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        raw = bf.knnMatch(des1, des2, k=2)
        for pair in raw:
            if len(pair) == 2:
                m, n = pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append([m])
    else:  # FLANN (default)
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE = 1
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        raw = flann.knnMatch(des1, des2, k=2)
        for m, n in raw:
            if m.distance < ratio_threshold * n.distance:
                good_matches.append([m])

    print(f"[SIFT] Good matches after ratio test: {len(good_matches)}")

    if len(good_matches) < min_matches:
        print(f"[SIFT] Insufficient matches ({len(good_matches)} < {min_matches}).")
        return empty_result

    # ── Homography ────────────────────────────────────────────────────────────
    src_pts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # ── Visualization ─────────────────────────────────────────────────────────
    if visualize:
        prefix = f"{title} — " if title else ""

        # Panel 1: matched keypoints side-by-side
        img_matches = cv2.drawMatchesKnn(
            gray1, kp1, gray2, kp2, good_matches,
            None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,
        )
        plt.figure(figsize=(15, 5))
        plt.imshow(img_matches, cmap='gray')
        plt.title(f"{prefix}SIFT Matches ({len(good_matches)} good matches)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        plt.close()

        # Panel 2: homography outline on heatmap
        if homography is not None:
            h, w = gray1.shape
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            dst_corners = cv2.perspectiveTransform(corners, homography)

            heatmap_bgr = cv2.cvtColor(gray2, cv2.COLOR_GRAY2BGR)
            heatmap_bgr = cv2.polylines(heatmap_bgr, [np.int32(dst_corners)], True, (0, 255, 0), 2)

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].imshow(gray1, cmap='gray')
            axes[0].set_title(f"{prefix}HSI Crop")
            axes[0].axis("off")

            axes[1].imshow(cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB))
            axes[1].set_title(f"{prefix}HSI Region in Heatmap")
            axes[1].axis("off")

            plt.tight_layout()
            plt.show()
            plt.close()

    return {
        'status': 'success',
        'matches': good_matches,
        'kp1': kp1,
        'kp2': kp2,
        'homography': homography,
    }

def crop_bounding_box(hsi_img, heatmap, boxes, pad = 30, max_boxes=None, visualisation= False):
    """Crop each bounding box from HSI and heatmap and visualize side by side."""
    if hsi_img is None or heatmap is None:
        raise ValueError("hsi_img and heatmap must be valid arrays")

    if boxes is None or len(boxes) == 0:
        print("No boxes to crop for Step 8.")
        return

    h, w = hsi_img.shape[:2]
    total = len(boxes) if max_boxes is None else min(len(boxes), int(max_boxes))

    for i, (x, y, bw, bh) in enumerate(boxes[:total], start=1):
        x0 = max(0, int(x - pad))
        y0 = max(0, int(y - pad))
        x1 = min(w, int(x + bw + pad))
        y1 = min(h, int(y + bh + pad))

        if x1 <= x0 or y1 <= y0:
            continue

        hsi_crop = hsi_img[y0:y1, x0:x1]
        heatmap_crop = heatmap[y0:y1, x0:x1]

        # # SIFT feature matching between HSI crop and heatmap crop
        # SIFT_feature_matching(
        #     hsi_crop,
        #     heatmap_crop,
        #     method='flann',
        #     visualize=True,
        #     title=f"Box #{i}",
        # )

        if visualisation:
            # Box coordinates relative to the cropped image.
            box_x = int(x - x0)
            box_y = int(y - y0)
            box_w = int(bw)
            box_h = int(bh)

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].imshow(hsi_crop, cmap="gray")
            axes[0].add_patch(Rectangle((box_x, box_y), box_w, box_h, fill=False, edgecolor="lime", linewidth=2))
            axes[0].add_patch(Rectangle((0, 0), 1, 1, transform=axes[0].transAxes, fill=False, edgecolor="red", linewidth=2, clip_on=False))
            axes[0].set_title(f"HSI Crop #{i}")
            axes[0].axis("off")

            height_vis = np.ma.masked_invalid(heatmap_crop)
            axes[1].imshow(height_vis, cmap="jet")
            axes[1].add_patch(Rectangle((box_x, box_y), box_w, box_h, fill=False, edgecolor="lime", linewidth=2))
            axes[1].add_patch(Rectangle((0, 0), 1, 1, transform=axes[1].transAxes, fill=False, edgecolor="red", linewidth=2, clip_on=False))
            axes[1].set_title(f"Fused Heatmap Crop #{i}")
            axes[1].axis("off")

            fig.suptitle(f"Box #{i}: x={x0}, y={y0}, w={x1 - x0}, h={y1 - y0}")
            plt.tight_layout()
            plt.show()
            plt.close()

if __name__ == "__main__":
    
    CLOUD_PATH = "/Users/nova98/Documents/Nova/cloud_data/20260629/test2_ROI/roi.ply"
    IMG_PATH = "/Users/nova98/Documents/Nova/cloud_data/20260629/test2_ROI/cropped_ROI.png"

    # Step 1: Load data
    cloud_points, hsi_img = dataloader(CLOUD_PATH, IMG_PATH)

    # Step 2: Project point cloud to 2D space with height map
    proj_img, pixel_to_point_indices, height_map = cloud_analysis.project_to_2D(
        cloud_points,
        height=hsi_img.shape[0],
        width=hsi_img.shape[1],
        visualisation=True,
        return_height_map=True,
    )

    # Step 3: Overlay height map (blue=low, red=high) onto the grayscale image
    fused_heatmap, z_min, z_max = overlay_heightmap_on_image(hsi_img, height_map, alpha=0.6)

    # Step 4: Display with colorbar showing height values
    display_heatmap_with_colorbar(fused_heatmap, z_min, z_max)

    # Step 5: Contours detection on HSI image
    # # # ── SAM2 segmentation ────────────────────────────────────────────────────
    # SAM2_CHECKPOINT = "/Users/nova98/Documents/Nova/3d_localization/sam_checkpoints/sam2.1_hiera_tiny.pt"
    # SAM2_MODEL_TYPE = "tiny"  # 'tiny', 'small', 'base_plus', or 'large'
    # DEVICE = "cpu"
    # countours = run_SAM2(SAM2_CHECKPOINT, SAM2_MODEL_TYPE, DEVICE, hsi_img)

    # # # ── SAM1 segmentation ────────────────────────────────────────────────────
    SAM1_CHECKPOINT_PATH = "/Users/nova98/Documents/Nova/3d_localization/sam_checkpoints"
    SAM1_MODEL_TYPE = "vit_b"   # 'vit_h', 'vit_l', or 'vit_b'
    DEVICE = "cpu"              # 'cuda' if GPU available
    contours = run_SAM1(hsi_img, SAM1_CHECKPOINT_PATH, SAM1_MODEL_TYPE, DEVICE)

    # Step 6: Remove the contours that are very near to the border
    filtered_contours = filter_border_contours(
        contours=contours,
        img_shape=hsi_img.shape,
        border_margin=20,
    )
    print(f"Contours after border filtering: {len(filtered_contours)}/{len(contours)}")
    # Visualize the filtered contours on the original HSI image
    show_filtered_contours_on_hsi(hsi_img, filtered_contours)

    # Step 7: Create bounding boxes around filtered contours
    img_with_boxes, boxes = generate_bounding_boxes(hsi_img, filtered_contours, visualisation=True)

    # Step 8: Visualize the cropped regions from HSI and fused heatmap side by side
    crop_bounding_box(hsi_img, height_map, boxes, pad=30, max_boxes=None, visualisation=True)
