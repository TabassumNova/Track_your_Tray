from dataloader import *

class PoseAnalysisPipeline:
    def __init__(self, dataset_path, tray_aruco_ids, measurement_aruco_ids, 
                 cube_aruco_id, cube_square_size, cube_marker_size, 
                 tray_square_size, tray_marker_size):
        '''
        Args:
            dataset_path (str): Path to the dataset containing pose folders.
            tray_aruco_ids (list): List of Aruco IDs for the tray.
            measurement_aruco_ids (list): List of Aruco IDs for measurements.
            cube_aruco_id (int): Aruco ID for the cube.
            cube_square_size (float): Size of the cube square in mm.
            cube_marker_size (float): Size of the cube marker in mm.
            tray_square_size (float): Size of the tray square in mm.
            tray_marker_size (float): Size of the tray marker in mm.
        '''
        self.dataset_path = dataset_path
        self.tray_aruco_ids = tray_aruco_ids
        self.measurement_aruco_ids = measurement_aruco_ids
        self.cube_aruco_id = cube_aruco_id
        self.cube_square_size = cube_square_size
        self.cube_marker_size = cube_marker_size
        self.cube_square_marker_gap = (cube_square_size - cube_marker_size)/2
        self.tray_square_size = tray_square_size
        self.tray_marker_size = tray_marker_size
        self.tray_square_marker_gap = (tray_square_size - tray_marker_size)/2
        self.pose_path_dict = load_all_poses_from_folder(dataset_path)

        #Specific for the given tray. See the reference image. Applies for 12 poses
        offset = self.cube_square_marker_gap - self.tray_square_marker_gap
        self.true_X = 93 + 2*self.tray_square_size + offset
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