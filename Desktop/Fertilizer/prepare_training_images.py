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
    tifs = list(root_dir.rglob("*.tif")) + list(root_dir.rglob("*.tiff"))
    # set() guards against symlinks that resolve to the same path.
    return sorted(set(tifs))


def select_frame_indices(n_timepoints: int) -> list:
    """
    Return the frame indices to extract from a TIFF stack.

    Single-frame stacks return [0].
    Time-lapse stacks (T >= 2) return [0, T-1].

    Parameters
    ----------
    n_timepoints : int
        Total number of frames in the stack.

    Returns
    -------
    list of int
    """
    if n_timepoints == 1:
        return [0]
    return [0, n_timepoints - 1]


def compute_auto_params(pixel_size_um, max_bead_diameter_um=50.0) -> dict:
    """
    Compute rolling ball radius and DoG high sigma from physical units.

    Mirrors the auto-parameter logic in ml_image_analysis.py so that
    preprocessing at training time matches preprocessing at inference time.

    Parameters
    ----------
    pixel_size_um : float or None
        Physical pixel size in µm. None or 0 triggers fallback values.
    max_bead_diameter_um : float
        Maximum expected bead/vesicle diameter in µm.

    Returns
    -------
    dict with keys:
        'rb_radius'      – rolling ball radius in pixels (float)
        'dog_sigma_high' – DoG high sigma in pixels (float)
    """
    if pixel_size_um and pixel_size_um > 0:
        max_diam_px = max_bead_diameter_um / pixel_size_um
        rb_radius     = max_diam_px
        dog_sigma_high = max_diam_px / 4.0
    else:
        rb_radius     = 50.0
        dog_sigma_high = 10.0

    return {"rb_radius": rb_radius, "dog_sigma_high": dog_sigma_high}
