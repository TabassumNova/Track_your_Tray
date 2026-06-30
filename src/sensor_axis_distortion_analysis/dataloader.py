'''
- Load all the poses from the folder
- The folder should have following structure:
    dataset_path/
        pose1/
            capture/
                .hdr
        pose2/
        ...
- Each pose folder should contain a capture folder with the HDR image
- pose folder name should be in the format: _poseX_ where X is the pose number (e.g., pose1, pose2, etc.)
'''

from pathlib import Path
import re


def load_all_poses_from_folder(dataset_path):
    '''
    TODO:
    - Search for the 'pose' keyword in the folder name
    - store the folder path in pose_dict
    '''
    pose_path_dict = {}
    pose_entries = []
    dataset_dir = Path(dataset_path)

    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise ValueError(f"Invalid dataset path: {dataset_path}")

    for child in dataset_dir.iterdir():
        if not child.is_dir():
            continue

        match = re.search(r"pose\D*(\d+)", child.name, flags=re.IGNORECASE)
        if match:
            pose_idx = int(match.group(1))
            pose_entries.append((pose_idx, str(child.resolve())))

    pose_entries.sort(key=lambda item: item[0])
    pose_path_dict = {serial_idx: path for serial_idx, (_, path) in enumerate(pose_entries, start=1)}

    return pose_path_dict