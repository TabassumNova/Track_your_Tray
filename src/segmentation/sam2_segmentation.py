"""
SAM2 (Segment Anything Model 2) - Automatic Everything Mode
Segments all objects in an image automatically without any prompts.

Requires: pip install sam2
Checkpoints and model configs: https://github.com/facebookresearch/segment-anything-2
"""

import cv2
import numpy as np
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


# Map of human-readable model type names to their Hydra config names.
# SAM2 configs live under  <sam2_package>/configs/sam2/
# so the name passed to build_sam2 must include the "sam2/" subdirectory prefix.
SAM2_MODEL_CONFIGS = {
    "tiny":       "configs/sam2.1/sam2.1_hiera_t.yaml",
    "small":      "configs/sam2.1/sam2.1_hiera_s.yaml",
    "base_plus":  "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "large":      "configs/sam2.1/sam2.1_hiera_l.yaml",
}


def load_sam2_model(checkpoint_path, model_type="base_plus", device="cpu"):
    """
    Load the SAM2 model from a checkpoint.

    Args:
        checkpoint_path (str): Path to the SAM2 model checkpoint (.pt file).
        model_type (str): SAM2 model type: 'tiny', 'small', 'base_plus', or 'large'.
        device (str): 'cuda' for GPU, 'cpu' for CPU.

    Returns:
        SAM2AutomaticMaskGenerator: Ready-to-use mask generator.
    """
    if model_type not in SAM2_MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Choose from: {list(SAM2_MODEL_CONFIGS.keys())}"
        )
    model_cfg = SAM2_MODEL_CONFIGS[model_type]
    sam2 = build_sam2(model_cfg, checkpoint_path, device=device)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        min_mask_region_area=100,
    )
    return mask_generator


def run_sam2_everything(img_bgr, mask_generator):
    """
    Run SAM2 in automatic everything mode and return all masks.

    Args:
        img_bgr (np.ndarray): Input image in BGR format (from OpenCV).
        mask_generator (SAM2AutomaticMaskGenerator): Loaded SAM2 mask generator.

    Returns:
        masks (list[dict]): List of mask dicts sorted by area (largest first).
            Each dict contains:
                'segmentation': np.ndarray (bool mask, H x W)
                'area': int
                'bbox': [x, y, w, h]
                'predicted_iou': float
                'stability_score': float
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(img_rgb)
    masks = sorted(masks, key=lambda m: m["area"], reverse=True)
    print(f"SAM2 found {len(masks)} segments.")
    return masks


def visualize_sam2_masks(img_bgr, masks, alpha=0.4):
    """
    Overlay all SAM2 masks on the image with random colours.

    Args:
        img_bgr (np.ndarray): Original image in BGR format.
        masks (list[dict]): Masks from run_sam2_everything.
        alpha (float): Transparency of the mask overlay.

    Returns:
        result_bgr (np.ndarray): BGR image with coloured overlays.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    overlay = img_rgb.copy().astype(np.float32)

    np.random.seed(42)
    for mask in masks:
        color = np.random.randint(0, 255, 3, dtype=np.uint8).tolist()
        seg = mask["segmentation"]
        overlay[seg] = (1 - alpha) * overlay[seg] + alpha * np.array(color, dtype=np.float32)

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    result_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

    cv2.imshow(f"SAM2 - All Segments ({len(masks)} masks)", result_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return result_bgr


def sam2_masks_to_contours(masks):
    """
    Convert SAM2 binary masks to OpenCV contours.

    Args:
        masks (list[dict]): Masks from run_sam2_everything.

    Returns:
        all_contours (list): List of contour arrays (largest contour per mask).
    """
    all_contours = []
    for mask in masks:
        seg = mask["segmentation"].astype(np.uint8) * 255
        cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            largest = max(cnts, key=cv2.contourArea)
            all_contours.append(largest)
    return all_contours

def run_SAM2(checkpoint_path, model_type, device, img_bgr1):
    start_time = time.time()
    sam2_mask_generator = load_sam2_model(checkpoint_path, model_type=model_type, device=device)
    sam2_masks = run_sam2_everything(img_bgr1, sam2_mask_generator)
    result_bgr_sam2 = visualize_sam2_masks(img_bgr1, sam2_masks)
    sam2_contours = sam2_masks_to_contours(sam2_masks)
    print(f"SAM2 segmentation took {time.time() - start_time:.2f} seconds")

    return sam2_contours


if __name__ == "__main__":
    import os

    # ── Configuration ─────────────────────────────────────────────────────────
    IMAGE_PATH  = "/path/to/your/image.png"            # <-- replace with your image path
    CHECKPOINT  = "/path/to/sam2_checkpoint.pt"        # <-- replace with checkpoint path
    MODEL_TYPE  = "base_plus"                          # 'tiny', 'small', 'base_plus', 'large'
    DEVICE      = "cpu"                                # 'cuda' if GPU available
    # ──────────────────────────────────────────────────────────────────────────

    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")
    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(f"SAM2 checkpoint not found: {CHECKPOINT}")

    img_bgr = cv2.imread(IMAGE_PATH)

    print(f"Loading SAM2 model ({MODEL_TYPE}) ...")
    import time
    t0 = time.time()
    mask_generator = load_sam2_model(CHECKPOINT, model_type=MODEL_TYPE, device=DEVICE)
    masks = run_sam2_everything(img_bgr, mask_generator)
    print(f"SAM2 segmentation took {time.time() - t0:.2f} seconds")

    visualize_sam2_masks(img_bgr, masks)
    contours = sam2_masks_to_contours(masks)
    print(f"Extracted {len(contours)} contours from SAM2 masks.")
