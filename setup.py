"""Setup configuration for Track Your Tray package."""

from setuptools import setup, find_packages

setup(
    name="track-your-tray",
    version="0.1.0",
    description="ROI detection and tracking system for trays using ArUco markers",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    python_requires=">=3.8",
    install_requires=[
        "streamlit>=1.0.0",
        "opencv-python>=4.5.0",
        "numpy>=1.20.0",
        "Pillow>=8.0.0",
        # TODO: Add other dependencies from requirements.txt
    ],
    entry_points={
        "console_scripts": [
            "track=track_your_tray.cli:main",
        ],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
