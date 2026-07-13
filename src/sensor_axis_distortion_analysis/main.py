
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sensor_axis_distortion_analysis.pose_analysis_pipeline import *

if __name__ == "__main__":
    dataset_path = '/Users/nova98/Documents/Nova/Helios+/FX10/20260708/4cm_Cube'
    # TRAY_ARUCOS = [34, 38, 39, 37, 35, 46, 45, 42, 49, 53, 43, 32, 74] # <-- Big black tray
    # MEASUREMENT_ARUCOS = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111] # Markers that are in the middle
    TRAY_ARUCOS = [65, 59, 60, 61, 58, 62, 57, 56, 70, 71, 72] # <-- Small black tray
    MEASUREMENT_ARUCOS = [103, 104, 105, 106, 107, 108, 109, 110, 111] # <-- Small black tray
    
    # TRAY_SIZE = 325 # Actual tray size in mm (for the big black tray)
    TRAY_SIZE = 247 # Actual tray size in mm (for the small black tray)
    CUBE_ARUCO = 12
    CUBE_SQUARE_SIZE = 20 # in mm
    CUBE_MARKER_SIZE = 18 # in mm
    TRAY_SQUARE_SIZE = 25 # in mm  # Applies to both TRAY_ARUCOS and MEASUREMENT_ARUCOS
    TRAY_MARKER_SIZE = 18 # in mm  # Applies to both TRAY_ARUCOS and MEASUREMENT_ARUCOS

    SEARCH_KEYWORD = "test" # Update this if the pose folders have a different naming convention.
    
    pipeline = PoseAnalysisPipeline(
        dataset_path=dataset_path,
        search_keyword=SEARCH_KEYWORD,
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

    
