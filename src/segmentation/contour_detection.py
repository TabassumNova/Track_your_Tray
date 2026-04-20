import cv2
import numpy as np

# Function to detect contours in an image using OpenCV
def detect_contours(img_bgr, visualize=True):
    """
    Detect contours in a BGR image using OpenCV.
    Args:
        img_bgr (np.ndarray): Input image in BGR format.
        visualize (bool): If True, display the contours on the image.
    Returns:
        contours (list): Detected contours.
        hierarchy (np.ndarray): Contour hierarchy.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 150, apertureSize=3)
    # Find contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if visualize:
        img_contours = img_bgr.copy()
        cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
        cv2.imshow('Contours', img_contours)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return contours, hierarchy

def detect_blobs(img_bgr, visualize=True, min_area=50, max_area=5000, threshold=10):
    """
    Apply blob detection to the input image using OpenCV's SimpleBlobDetector.

    Args:
        img_bgr (np.ndarray): Input image (BGR or grayscale).
        visualize (bool): If True, display the image with detected blobs.
        min_area (float): Minimum area of blobs to detect.
        max_area (float): Maximum area of blobs to detect.
        threshold (float): Minimum threshold for blob detection.

    Returns:
        keypoints (list): List of detected keypoints (cv2.KeyPoint objects).
    """
    # Convert to grayscale if needed
    if len(img_bgr.shape) == 3:
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_bgr

    # Set up the detector with parameters
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = min_area
    params.maxArea = max_area
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    params.minThreshold = threshold
    params.maxThreshold = 255
    params.thresholdStep = 10

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img_gray)

    if visualize:
        im_with_keypoints = cv2.drawKeypoints(
            img_bgr, keypoints, np.array([]), (0, 0, 255),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        cv2.imshow("Blobs", im_with_keypoints)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return keypoints


def detect_harris_corners(img_bgr, visualize=True, block_size=2, ksize=3, k=0.04, threshold_rel=0.01):
    """
    Detect Harris corners in the input image and visualize them.

    Args:
        img_bgr (np.ndarray): Input image (BGR or grayscale).
        visualize (bool): If True, display the image with detected corners.
        block_size (int): Neighborhood size for corner detection.
        ksize (int): Aperture parameter for the Sobel operator.
        k (float): Harris detector free parameter.
        threshold_rel (float): Relative threshold (fraction of max response) for corner selection.

    Returns:
        corners (np.ndarray): Array of (x, y) coordinates of detected corners.
    """
    # Convert to grayscale if needed
    if len(img_bgr.shape) == 3:
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_bgr

    img_gray = np.float32(img_gray)
    dst = cv2.cornerHarris(img_gray, block_size, ksize, k)
    dst = cv2.dilate(dst, None)
    thresh = threshold_rel * dst.max()
    corners = np.argwhere(dst > thresh)
    corners = np.flip(corners, axis=1)  # (row, col) -> (x, y)

    if visualize:
        vis = img_bgr.copy()
        for x, y in corners:
            cv2.circle(vis, (x, y), 5, (0, 0, 255), 2)
        cv2.imshow("Harris Corners", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return corners