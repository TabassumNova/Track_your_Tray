'''
For checking the y-axis error for all the cubes together
'''

import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_pose_y_errors(csv_path):
    """Load pose index and y-axis error columns from a pose_errors.csv file."""
    pose_numbers = []
    y_errors = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pose_key = "pose_index" if "pose_index" in row else "pose_number"
            if pose_key not in row or "y_error" not in row:
                raise ValueError(f"Missing required columns in CSV: {csv_path}")

            pose_numbers.append(int(row[pose_key]))
            y_errors.append(float(row["y_error"]))

    return pose_numbers, y_errors

if __name__ == "__main__":
    cube_4cm = ['/Users/nova98/Documents/Nova/Helios+/FX10/20260708/4cm_Cube/results/pose_errors.csv', 4]
    cube_6cm = ['/Users/nova98/Documents/Nova/Helios+/FX10/20260708/6cm_Cube/results/pose_errors.csv', 6]
    cube_8cm = ['/Users/nova98/Documents/Nova/Helios+/FX10/20260708/8cm_Cube/results/pose_errors.csv', 8]

    cube_sources = [cube_4cm, cube_6cm, cube_8cm]
    styles = [
        ("tab:blue", "o", "-"),
        ("tab:orange", "s", "--"),
        ("tab:green", "^", "-."),
    ]

    plt.figure(figsize=(10, 5.5))

    for i, (csv_path, cube_size_cm) in enumerate(cube_sources):
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        pose_numbers, y_errors = load_pose_y_errors(path)
        normalized_y_errors = [y_err / cube_size_cm for y_err in y_errors]

        color, marker, linestyle = styles[i % len(styles)]
        plt.plot(
            pose_numbers,
            normalized_y_errors,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.8,
            markersize=5,
            label=f"{cube_size_cm} cm cube",
        )

    plt.title("Pose Number vs Normalized Y-Axis Error")
    plt.xlabel("Pose Number")
    plt.ylabel("Y-Axis Error / Cube Size")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    output_path = Path(__file__).resolve().parent / "y_error_normalized_combined.png"
    plt.savefig(output_path, dpi=220)
    plt.show()