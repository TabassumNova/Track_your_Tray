import cv2
import numpy as np

def select_bright_pixels(img_bgr, contours, num_pixels=10, visualize=False):
    """
    For each contour, select the top N brightest pixels inside the contour (by grayscale value).
    Optionally plot them in yellow with larger size.
    Args:
        img_bgr (np.ndarray): Input image in BGR format.
        contours (list): List of contours (as from cv2.findContours).
        num_pixels (int): Number of pixels to select per contour.
        visualize (bool): If True, plot the selected pixels on the image.
    Returns:
        List of lists: Each sublist contains (x, y, value) tuples for the selected pixels in a contour.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    selected_pixels = []
    img_pixels = img_bgr.copy() if visualize else None
    for cnt in contours:
        # Create a mask for the contour
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        # Get coordinates of all pixels inside the contour
        ys, xs = np.where(mask == 255)
        values = gray[ys, xs]
        # Compute centroid of the contour
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            selected_pixels.append([])
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        # Compute distance from centroid for each pixel
        dists = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        # Sort pixels by distance to centroid (ascending)
        center_sorted_idx = np.argsort(dists)
        # Take a subset of pixels closest to the centroid (e.g., 30 pixels or all if fewer)
        center_n = min(30, len(center_sorted_idx))
        center_pixels_idx = center_sorted_idx[:center_n]
        # Among these, select the brightest num_pixels
        center_values = values[center_pixels_idx]
        center_xs = xs[center_pixels_idx]
        center_ys = ys[center_pixels_idx]
        # Sort by grayscale value (descending)
        bright_idx = np.argsort(center_values)[::-1]
        chosen = []
        for idx in bright_idx[:num_pixels]:
            x, y, val = center_xs[idx], center_ys[idx], center_values[idx]
            chosen.append((x, y, val))
            if visualize:
                cv2.circle(img_pixels, (int(x), int(y)), radius=5, color=(0, 255, 255), thickness=-1)
        selected_pixels.append(chosen)
    if visualize:
        cv2.imshow('Brightest Pixels (Yellow)', img_pixels)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return selected_pixels




def select_pixels_by_click(img_bgr, window_name='Select Pixels (click to add, q to finish)'):
    """
    Open an interactive window and let the user click pixel locations.
    Each click is recorded as a separate single-pixel entry, compatible
    with the output format of select_bright_pixels / map_pixels_to_mm.

    Controls:
        Left-click        — add a pixel at the clicked position.
        Right-click       — remove the last added pixel.
        Scroll wheel up   — zoom in  (centred on cursor).
        Scroll wheel down — zoom out (centred on cursor).
        + / =             — zoom in  (centred on image centre, keyboard fallback).
        - / _             — zoom out (centred on image centre, keyboard fallback).
        r                 — reset zoom and pan.
        Middle-click drag — pan the view.
        q / Enter         — confirm selection and close the window.

    Args:
        img_bgr     (np.ndarray): Input image in BGR format.
        window_name (str):        Title of the OpenCV window.

    Returns:
        selected_pixels (list of lists): Each sublist contains one
            (x, y, grayscale_value) tuple, one sublist per click.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = img_bgr.shape[:2]

    # Display size: cap at 1200 px on the larger side so the window fits on screen
    DISP_MAX = 1200
    scale_init = min(1.0, DISP_MAX / max(W, H))
    DW = int(W * scale_init)   # display-window width  (pixels on screen)
    DH = int(H * scale_init)   # display-window height (pixels on screen)

    clicked = []    # list of (x, y, val) in full-resolution image coordinates

    # ── View state ────────────────────────────────────────────────────────────
    # zoom  : magnification factor relative to the display window
    #         zoom=1 → the whole image fills the display window
    #         zoom=2 → a quarter of the image fills the display window
    # off_x/off_y : top-left corner of the visible region in *display* px coords
    state = {
        'zoom':   1.0,
        'off_x':  0.0,
        'off_y':  0.0,
        'pan':    False,
        'pan_sx': 0,
        'pan_sy': 0,
        'pan_ox': 0.0,
        'pan_oy': 0.0,
        'cursor_sx': DW // 2,   # last known cursor position in display coords
        'cursor_sy': DH // 2,
    }
    ZOOM_STEP = 1.20
    MIN_ZOOM  = 1.0
    MAX_ZOOM  = 30.0

    def _disp_to_img(sx, sy):
        """Convert display-window coords → full-resolution image coords."""
        img_x = (sx + state['off_x']) / state['zoom'] / scale_init
        img_y = (sy + state['off_y']) / state['zoom'] / scale_init
        return img_x, img_y

    def _img_to_disp(ix, iy):
        """Convert full-resolution image coords → display-window coords."""
        sx = ix * scale_init * state['zoom'] - state['off_x']
        sy = iy * scale_init * state['zoom'] - state['off_y']
        return sx, sy

    def _clamp_offset():
        max_off_x = max(0.0, DW * state['zoom'] - DW)
        max_off_y = max(0.0, DH * state['zoom'] - DH)
        state['off_x'] = max(0.0, min(state['off_x'], max_off_x))
        state['off_y'] = max(0.0, min(state['off_y'], max_off_y))

    def _apply_zoom(new_zoom, anchor_sx, anchor_sy):
        """Change zoom, keeping the image point under (anchor_sx, anchor_sy) fixed."""
        new_zoom = max(MIN_ZOOM, min(new_zoom, MAX_ZOOM))
        # image-coord of the anchor before zoom change
        img_ax = (anchor_sx + state['off_x']) / state['zoom'] / scale_init
        img_ay = (anchor_sy + state['off_y']) / state['zoom'] / scale_init
        state['zoom'] = new_zoom
        # re-compute offset so the same image point sits under the anchor
        state['off_x'] = img_ax * scale_init * new_zoom - anchor_sx
        state['off_y'] = img_ay * scale_init * new_zoom - anchor_sy
        _clamp_offset()

    def _render():
        z = state['zoom']
        ox, oy = state['off_x'], state['off_y']

        # Each display pixel corresponds to 1/z of a prescaled (DW×DH) pixel
        # Visit region in the DW×DH "scaled image" coordinates
        src_x0 = int(ox / z)
        src_y0 = int(oy / z)
        src_w  = int(DW / z) + 1
        src_h  = int(DH / z) + 1

        # Map to full-resolution image coords
        full_x0 = int(src_x0 / scale_init)
        full_y0 = int(src_y0 / scale_init)
        full_x1 = min(int((src_x0 + src_w) / scale_init) + 1, W)
        full_y1 = min(int((src_y0 + src_h) / scale_init) + 1, H)
        full_x0 = max(0, full_x0)
        full_y0 = max(0, full_y0)

        crop = img_bgr[full_y0:full_y1, full_x0:full_x1]
        frame = cv2.resize(crop, (DW, DH), interpolation=cv2.INTER_LINEAR)

        # Apply pan/zoom via warpAffine on the DW×DH frame
        M_zoom = np.float32([
            [z,  0, -ox],
            [0,  z, -oy],
        ])
        frame = cv2.warpAffine(
            cv2.resize(img_bgr, (DW, DH), interpolation=cv2.INTER_LINEAR),
            M_zoom, (DW, DH),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # Draw clicked points in display coords
        for i, (ix, iy, _) in enumerate(clicked):
            sx, sy = _img_to_disp(ix, iy)
            sx, sy = int(sx), int(sy)
            if -20 <= sx < DW + 20 and -20 <= sy < DH + 20:
                cv2.circle(frame, (sx, sy), 6, (0, 255, 255), -1)
                cv2.circle(frame, (sx, sy), 6, (0, 180, 180), 1)
                label = f"{i + 1}: ({int(ix)}, {int(iy)})"
                cv2.putText(frame, label, (sx + 9, sy - 9),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(frame,
                    f"Zoom: {z:.1f}x | clicks: {len(clicked)} | +/- zoom | r reset | q quit",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(window_name, frame)

    def _on_mouse(event, sx, sy, flags, param):
        state['cursor_sx'] = sx
        state['cursor_sy'] = sy
        ix, iy = _disp_to_img(sx, sy)
        ix = max(0.0, min(ix, W - 1.0))
        iy = max(0.0, min(iy, H - 1.0))

        if event == cv2.EVENT_LBUTTONDOWN:
            val = int(gray[int(round(iy)), int(round(ix))])
            clicked.append((ix, iy, val))
            _render()

        elif event == cv2.EVENT_RBUTTONDOWN and clicked:
            clicked.pop()
            _render()

        elif event == cv2.EVENT_MBUTTONDOWN:
            state['pan']    = True
            state['pan_sx'] = sx
            state['pan_sy'] = sy
            state['pan_ox'] = state['off_x']
            state['pan_oy'] = state['off_y']

        elif event == cv2.EVENT_MBUTTONUP:
            state['pan'] = False

        elif event == cv2.EVENT_MOUSEMOVE and state['pan']:
            state['off_x'] = state['pan_ox'] - (sx - state['pan_sx'])
            state['off_y'] = state['pan_oy'] - (sy - state['pan_sy'])
            _clamp_offset()
            _render()

        elif event == cv2.EVENT_MOUSEWHEEL:
            # flags is a raw C int passed as Python int; on some platforms
            # a negative scroll delta arrives as a large positive value —
            # reinterpret as signed 32-bit integer to get the correct sign.
            delta = np.int32(flags)
            new_zoom = (state['zoom'] * ZOOM_STEP
                        if delta > 0
                        else state['zoom'] / ZOOM_STEP)
            _apply_zoom(new_zoom, sx, sy)
            _render()

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    _render()
    cv2.setMouseCallback(window_name, _on_mouse)
    print("[select_pixels_by_click] Scroll/+/- to zoom, middle-drag to pan, "
          "left-click to add, right-click to undo, r to reset, q/Enter to confirm.")

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord('q'), 13):            # quit / confirm
            break
        elif key in (ord('+'), ord('=')):    # zoom in (keyboard)
            _apply_zoom(state['zoom'] * ZOOM_STEP,
                        state['cursor_sx'], state['cursor_sy'])
            _render()
        elif key in (ord('-'), ord('_')):    # zoom out (keyboard)
            _apply_zoom(state['zoom'] / ZOOM_STEP,
                        state['cursor_sx'], state['cursor_sy'])
            _render()
        elif key == ord('r'):                # reset view
            state['zoom'] = 1.0
            state['off_x'] = 0.0
            state['off_y'] = 0.0
            _render()

    cv2.destroyWindow(window_name)
    print(f"[select_pixels_by_click] {len(clicked)} pixel(s) selected.")

    # Return format: list of single-element sublists — same as select_bright_pixels
    return [[(int(round(x)), int(round(y)), val)] for x, y, val in clicked]