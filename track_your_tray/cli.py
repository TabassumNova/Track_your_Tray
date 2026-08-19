#!/usr/bin/env python3
"""Command-line interface for Track Your Tray."""

import sys
import subprocess
from pathlib import Path


def main():
    """Main entry point for the Track Your Tray CLI."""
    app_dir = Path(__file__).parent
    streamlit_app = app_dir / "pipeline" / "streamlit_app.py"
    
    # Run streamlit app
    subprocess.run(
        ["streamlit", "run", str(streamlit_app)],
        check=False,
    )


if __name__ == "__main__":
    main()
