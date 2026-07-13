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
from hylite import io


def load_all_poses_from_folder(dataset_path, search_keyword="pose"):
    '''
    Args:
        dataset_path (str): Path to the dataset folder containing pose subfolders.
        search_keyword (str): Keyword to identify pose folders (default is "pose").
    '''
    pose_path_dict = {}
    pose_entries = []
    dataset_dir = Path(dataset_path)

    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise ValueError(f"Invalid dataset path: {dataset_path}")

    for child in dataset_dir.iterdir():
        if not child.is_dir():
            continue

        match = re.search(rf"{search_keyword}\D*(\d+)", child.name, flags=re.IGNORECASE)
        if match:
            pose_idx = int(match.group(1))
            capture_dir = child / "capture"
            hdr_path = capture_dir / f"{child.name}.hdr"

            if not capture_dir.exists() or not capture_dir.is_dir():
                raise FileNotFoundError(f"Missing capture directory: {capture_dir}")

            if not hdr_path.exists():
                raise FileNotFoundError(f"Missing HDR file: {hdr_path}")

            pose_entries.append((pose_idx,str(hdr_path.resolve())))

    pose_entries.sort(key=lambda item: item[0])
    pose_path_dict = {
        serial_idx: pose_data
        for serial_idx, (_, pose_data) in enumerate(pose_entries, start=1)
    }

    return pose_path_dict