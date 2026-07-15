import open3d as o3d
import cv2
import numpy as np

def dataloader(cloud_path, img_path, visualisation=True):
    '''
    Load point cloud and grayscale image from specified paths.
    Args:
        cloud_path (str): Path to the point cloud file.
        img_path (str): Path to the grayscale image file.
        visualisation (bool): If True, visualize the point cloud and image.
    Returns:
        cloud_points (np.ndarray): Numpy array of point cloud points.
        img (np.ndarray): Grayscale image as a numpy array.
    '''
    # Load point cloud
    pcd = o3d.io.read_point_cloud(cloud_path)
    cloud_points = np.asarray(pcd.points)

    # Load grayscale image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Temporary fix for img
    # Rotate the image by 90 degrees clockwise + flip vertically
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img = cv2.flip(img, 1)  # Flip vertically

    if visualisation:
        # Visualize point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud_points)
        o3d.visualization.draw_geometries([pcd])

        # Visualize image
        cv2.imshow("Grayscale Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return cloud_points, img