
# 3D_localization

<!-- ![Overview](images/Overview.png) -->

<div style="display: inline-flex; flex-direction: column; align-items: center; gap: 4px;">
	<a href="src/main.py">
		<span style="display:inline-block; padding:8px 14px; border-radius:6px; background:#3F6E8A; color:#000000; font-weight:600; letter-spacing:0.2px;">HSI Input</span>
	</a>
	<span style="font-size: 20px; color: #4b5563; line-height: 1;">&darr;</span>
	<a href="src/visualization.py">
		<span style="display:inline-block; padding:8px 14px; border-radius:6px; background:#4C7F9D; color:#000000; font-weight:600; letter-spacing:0.2px;">Pre Processing</span>
	</a>
	<span style="font-size: 20px; color: #4b5563; line-height: 1;">&darr;</span>
	<a href="https://github.com/TabassumNova/Marker-detection">
		<span style="display:inline-block; padding:8px 14px; border-radius:6px; background:#5B90A8; color:#000000; font-weight:600; letter-spacing:0.2px;">Aruco Marker Detection</span>
	</a>
	<span style="font-size: 20px; color: #4b5563; line-height: 1;">&darr;</span>
	<a href="src/roi_detection.py">
		<span style="display:inline-block; padding:8px 14px; border-radius:6px; background:#6A9FAF; color:#000000; font-weight:600; letter-spacing:0.2px;">Tray ROI Detection</span>
	</a>
	<span style="font-size: 20px; color: #4b5563; line-height: 1;">&darr;</span>
	<a href="src/roi_detection.py">
		<span style="display:inline-block; padding:8px 14px; border-radius:6px; background:#7AAEB5; color:#000000; font-weight:600; letter-spacing:0.2px;">Perspective Warp To Square ROI</span>
	</a>
	<span style="font-size: 20px; color: #4b5563; line-height: 1;">&darr;</span>
	<a href="src/segmentation">
		<span style="display:inline-block; padding:8px 14px; border-radius:6px; background:#89B7B9; color:#000000; font-weight:600; letter-spacing:0.2px;">Contour Detection</span>
	</a>
	<span style="font-size: 20px; color: #4b5563; line-height: 1;">&darr;</span>
	<a href="src/segmentation/contour_detection.py">
		<span style="display:inline-block; padding:8px 14px; border-radius:6px; background:#98C0BE; color:#000000; font-weight:600; letter-spacing:0.2px;">Contour Selection Within ROI</span>
	</a>
	<span style="font-size: 20px; color: #4b5563; line-height: 1;">&darr;</span>
	<a href="src/pixel_selection.py">
		<span style="display:inline-block; padding:8px 14px; border-radius:6px; background:#A8C8C4; color:#000000; font-weight:600; letter-spacing:0.2px;">Select Pixels In Contoured Area</span>
	</a>
	<span style="font-size: 20px; color: #4b5563; line-height: 1;">&darr;</span>
	<a href="src/mapping.py">
		<span style="display:inline-block; padding:8px 14px; border-radius:6px; background:#B8D1CA; color:#000000; font-weight:600; letter-spacing:0.2px;">Map Pixels To Millimeter Scale</span>
	</a>
</div>

## Sensor axis distortion analysis
> check [here](src/sensor_axis_distortion_analysis/)





