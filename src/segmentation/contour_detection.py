import cv2
import numpy as np

# Function to detect contours in an image using OpenCV
def detect_contours(img_bgr, visualisation=True):
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
    if visualisation:
        img_contours = img_bgr.copy()
        cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
        cv2.imshow('Contours', img_contours)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return contours, hierarchy

def detect_blobs(img_bgr, visualisation=True, min_area=50, max_area=5000, threshold=10):
    """
    Apply blob detection to the input image using OpenCV's SimpleBlobDetector.

    Args:
        img_bgr (np.ndarray): Input image (BGR or grayscale).
        visualisation (bool): If True, display the image with detected blobs.
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

    if visualisation:
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

def edge_detection(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Display the edges
    cv2.imshow('Edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def filter_border_contours(contours, img_shape, border_margin=20):
    '''
    Filter out the countours that touch or very near to the border
    Args:
        contours (list): List of contours to filter.
        img_shape (tuple): Shape of the image (height, width).
        border_margin (int): Margin from the border to consider for filtering.
    Returns:
        list: Filtered contours that do not touch the border.
    
    '''
    
    if contours is None or len(contours) == 0:
        return []

    h, w = img_shape[:2]
    filtered = []

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)

        # Reject contours touching or very close to the image border.
        touches_left = x <= border_margin
        touches_top = y <= border_margin
        touches_right = (x + cw) >= (w - border_margin)
        touches_bottom = (y + ch) >= (h - border_margin)

        if touches_left or touches_top or touches_right or touches_bottom:
            continue

        filtered.append(cnt)

    return filtered




def filter_contours(img_bgr, contours, roi_pts, marker_dict, visualisation=True):
    """
    Filter contours to only those inside the ROI polygon, excluding contours
    that belong to Aruco markers.

    Args:
        img_bgr     (np.ndarray): Input image in BGR format.
        contours    (list):       Contours from cv2.findContours.
        roi_pts     (list):       Four corner points defining the ROI polygon.
        marker_dict (dict):       {marker_id: corners} from Aruco detection,
                                  where corners has shape (1, 4, 2).
        visualisation (bool):       If True, display the filtered contours.

    Returns:
        filtered (list): Contours inside the ROI and not on any Aruco marker.
    """
    roi_poly = np.int32(roi_pts)

    # Build a list of Aruco marker polygons for quick lookup
    marker_polys = []
    for corners in marker_dict.values():
        pts = np.int32(corners[0])   # shape (4, 2)
        marker_polys.append(pts)

    filtered = []
    for cnt in contours:
        # Check if every point of the contour is inside the ROI polygon
        all_inside = True
        for pt in cnt.reshape(-1, 2):
            pt_clean = (int(pt[0]), int(pt[1]))
            if cv2.pointPolygonTest(roi_poly, pt_clean, False) < 0:
                all_inside = False
                break
        if not all_inside:
            continue

        # Must NOT be inside any Aruco marker polygon (centroid test)
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        centroid = (cx, cy)
        inside_marker = any(
            cv2.pointPolygonTest(mpoly, centroid, False) >= 0
            for mpoly in marker_polys
        )
        if inside_marker:
            continue

        filtered.append(cnt)

    if visualisation:
        vis = img_bgr.copy()
        # Draw the ROI boundary
        # cv2.polylines(vis, [roi_poly], isClosed=True, color=(0, 255, 255), thickness=2)
        # Draw the filtered contours
        cv2.drawContours(vis, filtered, -1, (0, 0, 255), 2)
        cv2.imshow('Filtered Contours (inside ROI, no markers)', vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return filtered


def draw_bounding_boxes(img_bgr, contours, visualisation=True):
    """
    Draw bounding boxes around the given contours and visualize them.
    Args:
        img_bgr (np.ndarray): Input image in BGR format.
        contours (list): List of contours to draw bounding boxes around.
        visualisation (bool): If True, display the image with bounding boxes.
    Returns:
        img_with_boxes (np.ndarray): Image with bounding boxes drawn.
        boxes (list): List of bounding box coordinates (x, y, w, h).
    """
    img_with_boxes = img_bgr.copy()
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, w, h))
        cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (255, 0, 0), 2)
    if visualisation:
        cv2.imshow('Bounding Boxes', img_with_boxes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img_with_boxes, boxes