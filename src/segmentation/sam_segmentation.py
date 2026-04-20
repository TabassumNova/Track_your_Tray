"""
SAM (Segment Anything Model) - Automatic Everything Mode
Segments all objects in an image automatically without any prompts.
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


def load_sam_model(checkpoint_path, model_type="vit_h", device="cpu"):
    """
    Load the SAM model from a checkpoint.
    Args:
        checkpoint_path (str): Path to the SAM model checkpoint (.pth file).
        model_type (str): SAM model type: 'vit_h', 'vit_l', or 'vit_b'.
        device (str): 'cuda' for GPU, 'cpu' for CPU.
    Returns:
        SamAutomaticMaskGenerator: Ready-to-use mask generator.
    """
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,           # Grid density for automatic prompts
        pred_iou_thresh=0.88,         # Filter low-quality masks
        stability_score_thresh=0.95,  # Filter unstable masks
        min_mask_region_area=100,     # Ignore very small regions
    )
    return mask_generator


def run_sam_everything(img_bgr, mask_generator):
    """
    Run SAM in automatic everything mode and return all masks.
    Args:
        img_bgr (np.ndarray): Input image in BGR format (from OpenCV).
        mask_generator (SamAutomaticMaskGenerator): Loaded SAM mask generator.
    Returns:
        masks (list[dict]): List of mask dicts sorted by area (largest first).
            Each dict contains:
                'segmentation': np.ndarray (bool mask, H x W)
                'area': int
                'bbox': [x, y, w, h]
                'predicted_iou': float
                'stability_score': float
                'point_coords': [[x, y]]
                'crop_box': [x, y, w, h]
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(img_rgb)
    # Sort by area descending
    masks = sorted(masks, key=lambda m: m["area"], reverse=True)
    print(f"SAM found {len(masks)} segments.")
    return masks


def visualize_sam_masks(img_bgr, masks, alpha=0.4):
    """
    Overlay all SAM masks on the image with random colours.
    Args:
        img_bgr (np.ndarray): Original image in BGR format.
        masks (list[dict]): Masks from run_sam_everything.
        alpha (float): Transparency of the mask overlay.
    Returns:
        overlay (np.ndarray): BGR image with coloured overlays.
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

    cv2.imshow(f"SAM - All Segments ({len(masks)} masks)", result_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def masks_to_contours(masks):
    """
    Convert SAM binary masks to OpenCV contours.
    Args:
        masks (list[dict]): Masks from run_sam_everything.
    Returns:
        all_contours (list): List of contour arrays (one per mask, largest contour per mask).
    """
    all_contours = []
    for mask in masks:
        seg = mask["segmentation"].astype(np.uint8) * 255
        cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            # Keep the largest contour for this mask
            largest = max(cnts, key=cv2.contourArea)
            all_contours.append(largest)
    return all_contours

def run_SAM1(img, checkpoint_folder, model_type, device):
    '''
    Args:
        img (np.ndarray): Input image in BGR format.
        checkpoint_folder (str): Directory containing the SAM model checkpoint.
        model_type (str): SAM model type: 'vit_h', 'vit_l', or 'vit_b'.
        device (str): 'cuda' for GPU, 'cpu' for CPU.
    Returns:
        sam1_contours (list): List of contours detected by SAM1.
    '''
    
    base_dir = checkpoint_folder
    checkpoint_file = None
    model_type_lower = model_type.lower()
    for fname in os.listdir(base_dir):
        if model_type_lower in fname.lower() and (fname.endswith('.pth') or fname.endswith('.pt')):
            checkpoint_file = os.path.join(base_dir, fname)
            break

    sam1_mask_generator = load_sam_model(checkpoint_file, model_type=model_type, device=device)
    sam1_masks = run_sam_everything(img, sam1_mask_generator)
    visualize_sam_masks(img, sam1_masks)
    sam1_contours = masks_to_contours(sam1_masks)
    return sam1_contours

