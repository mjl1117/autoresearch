"""
file: prepare_training_images.py
Preprocess TIFF images for Cellpose training.

Recursively finds all .tif/.tiff files in a directory, extracts the OA-647
channel from the first and last timepoints, applies the same preprocessing
as the inference pipeline (rolling ball, DoG, CLAHE), saves uint16 TIFFs,
and launches the Cellpose GUI pre-loaded with cpsam.

Usage:
    python prepare_training_images.py <input_dir> [options]
    python prepare_training_images.py /data/experiments --pixel-size-um 0.1
    python prepare_training_images.py /data/experiments --no-rolling-ball --no-launch
"""

import argparse
import re
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import tifffile

# Add project root to path so ml_image_analysis imports work regardless of
# the working directory the user invokes the script from.
sys.path.insert(0, str(Path(__file__).parent))

from ml_image_analysis import (
    apply_rolling_ball_background,
    apply_dog_enhancement,
    enhance_contrast_clahe,
    load_tiff_stack,
    read_pixel_size_from_tiff,
)

CPSAM_MODEL_PATH = "/Users/matthew/.cellpose/models/cpsam"
FALLBACK_PYTHON   = "/Users/matthew/miniforge3/envs/membrane-image/bin/python"


def find_tiff_files(root_dir: Path) -> list:
    """
    Recursively find all .tif and .tiff files under root_dir.

    Parameters
    ----------
    root_dir : Path
        Directory to search.

    Returns
    -------
    list of Path
        Sorted list of matching file paths.
    """
    root_dir = Path(root_dir)
    tifs = sorted(root_dir.rglob("*.tif")) + sorted(root_dir.rglob("*.tiff"))
    # rglob("*.tif") won't match *.tiff, so both patterns are needed.
    # De-duplicate and re-sort in case of overlap on case-insensitive filesystems.
    return sorted(set(tifs))
