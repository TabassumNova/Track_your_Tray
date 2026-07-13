from sys import path
import os
import json
import matplotlib.pyplot as plt

from hylite import io

from src.visualization import *
from src.roi_detection import *
from src.mapping import *

from src.sensor_axis_distortion_analysis.dataloader import *

'''
Aruco detection module
1. Clone the repository:
    - git clone https://github.com/TabassumNova/Marker-detection.git
2. Ensure the path to aruco_detection.py is correct in the import statement below.
'''
# Aruco detection
import importlib.util
spec = importlib.util.spec_from_file_location(
    "aruco_detection",
    "/Users/nova98/Documents/Nova/Marker-detection/src/aruco_detection.py" # Update this path to the actual location of aruco_detection.py
)
aruco_detection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aruco_detection)


def order_corners_top_left_clockwise(corners_xy):
    """Return 4 corners ordered as [top-left, top-right, bottom-right, bottom-left]."""
    pts = np.asarray(corners_xy, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(d)]
    bottom_left = pts[np.argmax(d)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

class PoseAnalysisPipeline:
    def __init__(self, dataset_path, search_keyword, tray_size, tray_aruco_ids, measurement_aruco_ids, 
                 cube_aruco_id, cube_square_size, cube_marker_size, 
                 tray_square_size, tray_marker_size, visualisation=False):
        '''
        Args:
            dataset_path (str): Path to the dataset containing pose folders.
            search_keyword (str): Keyword to identify pose folders (default is "pose").
            tray_size (int): Size of the square shaped tray in mm (width, height).
            tray_aruco_ids (list): List of Aruco IDs for the tray.
            measurement_aruco_ids (list): List of Aruco IDs for measurements.
            cube_aruco_id (int): Aruco ID for the cube.
            cube_square_size (float): Size of the cube square in mm.
            cube_marker_size (float): Size of the cube marker in mm.
            tray_square_size (float): Size of the tray square in mm.
            tray_marker_size (float): Size of the tray marker in mm.
        '''
        self.dataset_path = dataset_path
        self.search_keyword = search_keyword
        self.tray_aruco_ids = tray_aruco_ids
        self.measurement_aruco_ids = measurement_aruco_ids
        self.cube_aruco_id = cube_aruco_id
        self.cube_square_size = cube_square_size
        self.cube_marker_size = cube_marker_size
        self.cube_square_marker_gap = (cube_square_size - cube_marker_size)/2
        self.tray_square_size = tray_square_size
        self.tray_marker_size = tray_marker_size
        self.tray_square_marker_gap = (tray_square_size - tray_marker_size)/2
        self.actual_tray_size = tray_size
        self.considered_tray_size = self.actual_tray_size - 2*self.tray_square_marker_gap # After removing the white 
                                                                                            # border of the aruco
        self.pose_path_dict = load_all_poses_from_folder(self.dataset_path, self.search_keyword)

        #Specific for the given tray. See the reference image. Applies for 12 poses
        offset = self.cube_square_marker_gap - self.tray_square_marker_gap
        # self.true_X = 93 + 2*self.tray_square_size + offset # --> For big tray
        self.true_X = 134 - self.cube_square_size + offset # --> For small tray
        self.true_Y_dict = {1: 0*self.tray_square_size + offset,
                            2: 1*self.tray_square_size + offset,
                            3: 2*self.tray_square_size + offset,
                            4: 3*self.tray_square_size + offset,
                            5: 4*self.tray_square_size + offset,
                            6: 5*self.tray_square_size + offset,
                            7: 6*self.tray_square_size + offset,
                            8: 7*self.tray_square_size + offset,
                            9: 8*self.tray_square_size + offset,
                            10: 9*self.tray_square_size + offset,
                            11: 10*self.tray_square_size + offset,
                            12: 11*self.tray_square_size + offset}
        
        # For storing x and y axis error
        self.pose_error_dict = {} 
        self.visualisation = visualisation
        
    def start_analysis(self):
        for pose_serial_idx, pose_data in self.pose_path_dict.items():
            print(
                f"Processing pose {pose_serial_idx} at path: {pose_data})"
            )
            image = io.load(pose_data)
            img_bgr = plot_hyimage(image, visualisation=self.visualisation)
            
            # aruco marker detction
            marker_dict0 = aruco_detection.getAruco(img_bgr, aruco_dict_id=cv2.aruco.DICT_4X4_1000, visualisation=True)
            CORNER = 'inner_corners'
            marker_dict = {k: v for k, v in marker_dict0.items() if CORNER in v}
            marker_dict = {k: v[CORNER] for k, v in marker_dict.items()}
            
            # roi detection
            roi_pts = find_ROI(img_bgr, marker_dict, considered_markers=self.tray_aruco_ids, visualisation=self.visualisation)
            
            # Crop ROI
            roi_cropped, img_warped, warped_roi_pts, warped_marker_dict = crop_roi_from_image(img_bgr, roi_pts, marker_dict, roi_size_px=1000, visualisation=self.visualisation)
            img_bgr1 = roi_cropped  # For subsequent processing, focus on the cropped ROI

            # Map the 4 corners of the cube Aruco marker in millimeter scale.
            selected_pixels = []
            selected_pixels_for_mapping = []
            marker_ids_found = []
            missing_ids = []
            # for marker_id in aruco_ids:
            corners = warped_marker_dict.get(self.cube_aruco_id)
            if corners is None:
                missing_ids.append(self.cube_aruco_id)
                continue

            ordered_corners = order_corners_top_left_clockwise(corners)
            corners_xy = []
            corners_for_map = []
            for x, y in ordered_corners:
                px = int(round(x))
                py = int(round(y))
                corners_xy.append((px, py))
                corners_for_map.append((px, py, 0))

            selected_pixels.extend(corners_xy)
            selected_pixels_for_mapping.append(corners_for_map)
            marker_ids_found.append(self.cube_aruco_id)

            if missing_ids:
                print(f"Warning: missing cube Aruco IDs in warped ROI: {missing_ids}")

            if not selected_pixels_for_mapping:
                raise ValueError("No cube Aruco corners found in warped_marker_dict.")

            pixels_mm_nested = map_pixels_to_mm(
                warped_roi_pts,
                selected_pixels_for_mapping,
                roi_size_mm=self.considered_tray_size,
                visualisation =self.visualisation,
                img=img_warped,
            )
            pixels_mm = [pt_mm for marker_mm in pixels_mm_nested for pt_mm in marker_mm]
            
            # Considering only the first corner
            x_ = pixels_mm[0][0]
            y_ = pixels_mm[0][1]
            x_error = self.true_X - x_
            y_error = self.true_Y_dict[pose_serial_idx] - y_
            
            self.pose_error_dict[pose_serial_idx] = (x_error, y_error)
        
        self.plot_errors()

    def plot_errors(self):
        output_path = os.path.join(self.dataset_path, 'results')
        os.makedirs(output_path, exist_ok=True)
        # Save the arguments as metadata json
        metadata = {
            "dataset_path": self.dataset_path,
            "tray_aruco_ids": self.tray_aruco_ids,
            "measurement_aruco_ids": self.measurement_aruco_ids,
            "cube_aruco_id": self.cube_aruco_id,
            "cube_square_size": self.cube_square_size,
            "cube_marker_size": self.cube_marker_size,
            "tray_square_size": self.tray_square_size,
            "tray_marker_size": self.tray_marker_size,
            "actual_tray_size": self.actual_tray_size
        }
        metadata_path = os.path.join(output_path, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)


        
        if not self.pose_error_dict:
            raise ValueError("No pose errors found. Run start_analysis() before plot_errors().")

        pose_indices = sorted(self.pose_error_dict.keys())
        x_errors = [self.pose_error_dict[idx][0] for idx in pose_indices]
        y_errors = [self.pose_error_dict[idx][1] for idx in pose_indices]

        # Save the errors in csv file
        csv_path = os.path.join(output_path, 'pose_errors.csv')
        with open(csv_path, 'w') as f:
            f.write("pose_index,x_error,y_error\n")
            for idx in pose_indices:
                f.write(f"{idx},{self.pose_error_dict[idx][0]},{self.pose_error_dict[idx][1]}\n")

        # Plot pose index vs x_error and y_error in seperate 2 plots
        plt.figure(figsize=(9, 5))
        plt.plot(pose_indices, x_errors, marker='o', linewidth=2)
        plt.title('Pose Index vs X Error')
        plt.xlabel('Pose Index')
        plt.ylabel('X Error (mm)')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.xticks(pose_indices)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'x_error_vs_pose.png'), dpi=200)
        plt.close()

        plt.figure(figsize=(9, 5))
        plt.plot(pose_indices, y_errors, marker='o', linewidth=2, color='tab:orange')
        plt.title('Pose Index vs Y Error')
        plt.xlabel('Pose Index')
        plt.ylabel('Y Error (mm)')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.xticks(pose_indices)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'y_error_vs_pose.png'), dpi=200)
        plt.close()
        
