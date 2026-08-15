
# 3D_localization

<!-- ![Overview](images/Overview.png) -->

<!-- <table align="center" border="0" cellpadding="0" cellspacing="0"> -->
<tr><td align="center" style="padding:4px 0;"><a href="src/main.py"><img src="https://img.shields.io/badge/1. HSI_Input-C5E3F0?style=for-the-badge" alt="HSI Input"></a></td></tr>

<tr><td align="center" style="padding:0; line-height:1.2; font-size:16px;">→</td></tr>

<tr><td align="center" style="padding:4px 0;"><a href="src/visualization.py"><img src="https://img.shields.io/badge/2. Pre_Processing-CDEAF3?style=for-the-badge" alt="Pre Processing"></a></td></tr>

<tr><td align="center" style="padding:0; line-height:1.2; font-size:16px;">→</td></tr>

<tr><td align="center" style="padding:4px 0;"><a href="https://github.com/TabassumNova/Marker-detection"><img src="https://img.shields.io/badge/3. Aruco_Marker_Detection-D4EEF5?style=for-the-badge" alt="Aruco Marker Detection"></a></td></tr>

<tr><td align="center" style="padding:0; line-height:1.2; font-size:16px;">→</td></tr>

<tr><td align="center" style="padding:4px 0;"><a href="src/roi_detection.py"><img src="https://img.shields.io/badge/4. Tray_ROI_Detection-C8EBE8?style=for-the-badge" alt="Tray ROI Detection"></a></td><tr>

<tr><td align="center" style="padding:0; line-height:1.2; font-size:16px;">→</td></tr>

<tr><td align="center" style="padding:4px 0;"><a href="src/roi_detection.py"><img src="https://img.shields.io/badge/5. Perspective_Warp_To_Square_ROI-D0EDEA?style=for-the-badge" alt="Perspective Warp To Square ROI"></a></td></tr>

<tr><td align="center" style="padding:0; line-height:1.2; font-size:16px;">→</td></tr>

<tr><td align="center" style="padding:4px 0;"><a href="src/segmentation"><img src="https://img.shields.io/badge/6. Contour_Detection-D8EFEC?style=for-the-badge" alt="Contour Detection"></a></td></tr>

<tr><td align="center" style="padding:0; line-height:1.2; font-size:16px;">→</td></tr>

<tr><td align="center" style="padding:4px 0;"><a href="src/segmentation/contour_detection.py"><img src="https://img.shields.io/badge/7. Contour_Selection_Within_ROI-DCEEE9?style=for-the-badge" alt="Contour Selection Within ROI"></a></td></tr>

<tr><td align="center" style="padding:0; line-height:1.2; font-size:16px;">→</td></tr>

<tr><td align="center" style="padding:4px 0;"><a href="src/pixel_selection.py"><img src="https://img.shields.io/badge/8. Select_Pixels_In_Contoured_Area-E2F1ED?style=for-the-badge" alt="Select Pixels In Contoured Area"></a></td></tr>

<tr><td align="center" style="padding:0; line-height:1.2; font-size:16px;">→</td></tr>

<tr><td align="center" style="padding:4px 0;"><a href="src/mapping.py"><img src="https://img.shields.io/badge/9. Map_Pixels_To_Millimeter_Scale-E8F4F0?style=for-the-badge" alt="Map Pixels To Millimeter Scale"></a></td></tr>
</table>

## Sensor axis distortion analysis
> check [here](src/sensor_axis_distortion_analysis/)

## Streamlit app run
- python3 -m streamlit run ./app/streamlit_app.py





