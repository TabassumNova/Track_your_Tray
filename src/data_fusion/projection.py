import numpy as np
import cv2


def overlay_heightmap_on_image(gray_img, height_map, alpha=0.45):
    '''
    Overlay a height map onto a grayscale image using a colormap.
    
    Args:
        gray_img (np.ndarray): Grayscale image. Shape: (H, W).
        height_map (np.ndarray): Height map with the same dimensions as gray_img. Shape: (H, W).
        alpha (float): Blending factor between 0 and 1.
    
    Returns:
        overlay (np.ndarray): Image with height map overlay.
        z_min (float): Minimum height value.
        z_max (float): Maximum height value.
    '''
    if gray_img is None or height_map is None:
        raise ValueError("gray_img and height_map must be valid arrays")

    if gray_img.ndim != 2:
        raise ValueError("gray_img must be a grayscale image")

    if height_map.shape[:2] != gray_img.shape[:2]:
        height_map = cv2.resize(
            height_map,
            (gray_img.shape[1], gray_img.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    valid_mask = ~np.isnan(height_map)
    if not np.any(valid_mask):
        raise ValueError("height_map has no valid values (all NaN)")

    valid_values = height_map[valid_mask]
    z_min = valid_values.min()
    z_max = valid_values.max()

    if np.isclose(z_max, z_min):
        normalized = np.zeros_like(height_map, dtype=np.uint8)
    else:
        normalized_float = (height_map - z_min) / (z_max - z_min)
        normalized_float = np.where(valid_mask, normalized_float, 0.0)
        normalized = np.clip(normalized_float * 255.0, 0, 255).astype(np.uint8)

    normalized[~valid_mask] = 0

    heatmap_bgr = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    base_bgr = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    blended = cv2.addWeighted(base_bgr, 1.0 - alpha, heatmap_bgr, alpha, 0)

    overlay = base_bgr.copy()
    overlay[valid_mask] = blended[valid_mask]
    return overlay, z_min, z_max