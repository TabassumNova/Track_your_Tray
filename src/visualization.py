import numpy as np
import cv2

def plot_hyimage(image):
    '''
    Plot a hyperspectral image by selecting a specific band.
    Args:
    - image: HyImage object
    '''
    
    # Find the band index closest to 770 nm (FX10) and 1322 nm (FX17)
    wavelengths = image.get_wavelengths()
    # select selected_band automatically on wavelengths array
    selected_band = 770.0 if wavelengths[-1] < 1005.0 else 1322.0
    band_idx = np.argmin(np.abs(wavelengths - selected_band))

    # Extract band and normalize to uint8 (0-255), handling NaN
    band_data = image.data[:, :, band_idx].astype(np.float32)
    band_data = np.nan_to_num(band_data, nan=0.0)
    band_norm = cv2.normalize(band_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Stack into BGR (grayscale equivalent) for cv2 processing
    img_bgr = cv2.merge([band_norm, band_norm, band_norm])

    # Mirror along the x-axis (vertical flip)
    img_bgr = cv2.flip(img_bgr, 0)

    cv2.imshow(f'Band at ~{selected_band} nm', img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img_bgr