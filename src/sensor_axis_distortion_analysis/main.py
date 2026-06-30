
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sensor_axis_distortion_analysis.pose_analysis_pipeline import *

if __name__ == "__main__":
    dataset_path = '/Users/nova98/Documents/Nova/Helios+/FX10/20260616/8cm_Cube'
    TRAY_ARUCOS = [34, 38, 39, 37, 35, 46, 45, 42, 49, 53, 43, 32, 74] # <-- Big black tray
    MEASUREMENT_ARUCOS = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111] # Markers that are in the middle
    TRAY_SIZE = 325 # Actual tray size in mm (for the big black tray)
    CUBE_ARUCO = 7
    CUBE_SQUARE_SIZE = 40 # in mm
    CUBE_MARKER_SIZE = 30 # in mm
    TRAY_SQUARE_SIZE = 25 # in mm  # Applies to both TRAY_ARUCOS and MEASUREMENT_ARUCOS
    TRAY_MARKER_SIZE = 18 # in mm  # Applies to both TRAY_ARUCOS and MEASUREMENT_ARUCOS
    
    pipeline = PoseAnalysisPipeline(
        dataset_path=dataset_path,
        tray_size=TRAY_SIZE,
        tray_aruco_ids=TRAY_ARUCOS,
        measurement_aruco_ids=MEASUREMENT_ARUCOS,
        cube_aruco_id=CUBE_ARUCO,
        cube_square_size=CUBE_SQUARE_SIZE,
        cube_marker_size=CUBE_MARKER_SIZE,
        tray_square_size=TRAY_SQUARE_SIZE,
        tray_marker_size=TRAY_MARKER_SIZE
    )

    pipeline.start_analysis()

    
