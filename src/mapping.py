import cv2
import numpy as np

def map_pixels_to_mm(roi_pts, selected_pixels, roi_size_mm=325.0, visualisation=True, img=None):
    """
    Map pixel coordinates to millimeter scale using the ROI corners via perspective transform.

    The ROI is treated as a square of side roi_size_mm.  Corner ordering must
    match find_ROI: [top-left, top-right, bottom-right, bottom-left].

    Args:
        roi_pts        (list):            Four corner points of the ROI in pixel space.
        selected_pixels (list of lists):  Output from select_bright_pixels —
                                          each sublist contains (x, y, value) tuples.
        roi_size_mm    (float):           Physical side length of the ROI in mm (default 325).
        visualisation (bool):            If True, display the image with mm labels.
        img            (np.ndarray):      Image to annotate (required when visualisation=True).

    Returns:
        pixels_mm (list of lists): Each sublist contains (x_mm, y_mm) tuples,
                                   one per pixel in the corresponding contour.
    """
    # Perspective transform: pixel space → mm space
    src = np.float32(roi_pts)  # [TL, TR, BR, BL]
    dst = np.float32([
        [0,           0          ],
        [roi_size_mm, 0          ],
        [roi_size_mm, roi_size_mm],
        [0,           roi_size_mm],
    ])
    M = cv2.getPerspectiveTransform(src, dst)

    pixels_mm = []
    for contour_pixels in selected_pixels:
        contour_mm = []
        for (x, y, _val) in contour_pixels:
            pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
            pt_mm = cv2.perspectiveTransform(pt, M)[0][0]
            contour_mm.append((float(pt_mm[0]), float(pt_mm[1])))
        pixels_mm.append(contour_mm)

    if visualisation and img is not None:
        vis = img.copy()
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        thickness  = 2
        arrow_len  = 30   # px from dot tip to label anchor

        for contour_pixels, contour_mm in zip(selected_pixels, pixels_mm):
            for (x, y, _val), (x_mm, y_mm) in zip(contour_pixels, contour_mm):
                px, py = int(x), int(y)

                # Arrow tip offset: place label to the right and slightly above
                tip_x = px + arrow_len
                tip_y = py - arrow_len

                label = f"({x_mm:.1f}, {y_mm:.1f}) mm"
                (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                # Draw a filled dark rectangle behind the text for readability
                pad = 4
                cv2.rectangle(
                    vis,
                    (tip_x - pad, tip_y - th - pad),
                    (tip_x + tw + pad, tip_y + baseline + pad),
                    (30, 30, 30),
                    cv2.FILLED,
                )

                # Arrow from dot to label box
                cv2.arrowedLine(
                    vis,
                    (px, py),
                    (tip_x, tip_y),
                    (0, 255, 255),
                    thickness=2,
                    tipLength=0.25,
                )

                # Dot at the pixel location
                cv2.circle(vis, (px, py), 5, (0, 255, 255), -1)

                # Label text
                cv2.putText(
                    vis, label,
                    (tip_x, tip_y),
                    font, font_scale, (255, 255, 0), thickness, cv2.LINE_AA,
                )

        cv2.imshow("Pixel Positions in mm", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return pixels_mm


def warp_contours_to_original(contours, roi_pts, roi_size_px=400, img_bgr=None, visualize=True):
    """
    Warp contours from the cropped ROI image back to the original image using inverse perspective transform.

    Args:
        contours (list): Contours in the cropped ROI image (list of np.ndarray).
        roi_pts (list): Four corner points of the ROI in the original image (TL, TR, BR, BL).
        roi_size_px (int): Size of the cropped ROI image (width/height in px).
        img_bgr (np.ndarray): Original image to visualize on (optional).
        visualize (bool): If True, display the contours on the original image.

    Returns:
        contours_orig (list): Contours mapped to the original image coordinates.
    """
    # Perspective transform: cropped ROI -> original image
    dst = np.float32([
        [0, 0],
        [roi_size_px - 1, 0],
        [roi_size_px - 1, roi_size_px - 1],
        [0, roi_size_px - 1],
    ])
    src = np.float32(roi_pts)
    Minv = cv2.getPerspectiveTransform(dst, src)

    contours_orig = []
    for cnt in contours:
        cnt = cnt.astype(np.float32)
        cnt_warped = cv2.perspectiveTransform(cnt, Minv)
        contours_orig.append(cnt_warped.astype(np.int32))

    if visualize and img_bgr is not None:
        vis = img_bgr.copy()
        cv2.drawContours(vis, contours_orig, -1, (0, 0, 255), 2)
        cv2.imshow("Contours posed in original image", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return contours_orig