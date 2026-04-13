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


def preprocess_frame(frame, rolling_ball, rb_radius,
                     dog, dog_sigma_low, dog_sigma_high,
                     clahe, clahe_clip_limit) -> np.ndarray:
    """
    Apply the inference preprocessing pipeline to a single 2-D frame.

    Steps applied in order (each conditional on its enable flag):
      1. Rolling ball background subtraction
      2. Difference-of-Gaussians enhancement
      3. CLAHE contrast enhancement
      4. Normalise to float32 [0, 1]

    Parameters
    ----------
    frame : array-like, shape (H, W)
        Raw grayscale image (any integer or float dtype).
    rolling_ball : bool
    rb_radius : float
    dog : bool
    dog_sigma_low : float
    dog_sigma_high : float
    clahe : bool
    clahe_clip_limit : float

    Returns
    -------
    np.ndarray, dtype float32, shape (H, W), values in [0, 1]
    """
    image = np.asarray(frame, dtype=np.float32)

    if rolling_ball:
        image = apply_rolling_ball_background(image, radius=rb_radius)
    if dog:
        image = apply_dog_enhancement(image, sigma_low=dog_sigma_low,
                                      sigma_high=dog_sigma_high)
    if clahe:
        image = enhance_contrast_clahe(image, clip_limit=clahe_clip_limit)

    # Normalise to [0, 1]
    img_min, img_max = float(image.min()), float(image.max())
    if img_max > img_min:
        image = (image - img_min) / (img_max - img_min)
    else:
        image = np.zeros_like(image)

    return np.ascontiguousarray(image, dtype=np.float32)


def save_preprocessed_tiff(frame_f32: np.ndarray, output_path: Path) -> None:
    """
    Scale a float32 [0, 1] frame to uint16 and save as a grayscale TIFF.

    Parameters
    ----------
    frame_f32 : np.ndarray, dtype float32, shape (H, W)
        Preprocessed frame with values in [0, 1].
    output_path : Path
        Destination file path. Parent directories are created if absent.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    uint16 = (np.clip(frame_f32, 0.0, 1.0) * 65535).astype(np.uint16)
    tifffile.imwrite(str(output_path), uint16)


def detect_python_executable() -> str:
    """
    Return the Python binary to use when launching the Cellpose GUI.

    If the current process is running inside a conda environment (detected by
    the presence of 'envs/<name>/' in sys.executable), use that Python so the
    GUI runs in the same environment.  Otherwise fall back to the known
    membrane-image environment that has Cellpose installed.

    Returns
    -------
    str
        Absolute path to the Python executable.
    """
    exe = sys.executable
    if re.search(r"envs/[^/]+/", exe):
        return exe
    return FALLBACK_PYTHON


def parse_args(argv=None):
    """
    Parse command-line arguments.

    Parameters
    ----------
    argv : list of str, optional
        Argument list; defaults to sys.argv[1:] when None.
    """
    p = argparse.ArgumentParser(
        description=(
            "Preprocess TIFF images for Cellpose training.\n"
            "Extracts the OA-647 channel from the first and last timepoints,\n"
            "applies the inference preprocessing pipeline, saves uint16 TIFFs,\n"
            "and launches the Cellpose GUI pre-loaded with cpsam."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("input_dir", type=Path,
                   help="Root directory to search recursively for .tif/.tiff files")

    p.add_argument("--output-dir", type=Path, default=Path("training_images"),
                   help="Directory where preprocessed TIFFs are saved (default: training_images/)")
    p.add_argument("--channel", type=int, default=1,
                   help="Channel index to extract — 1 = OA-647 (default: 1)")

    # Rolling ball
    p.add_argument("--no-rolling-ball", action="store_true",
                   help="Disable rolling ball background subtraction")
    p.add_argument("--rolling-ball-radius", type=float, default=None,
                   help="Rolling ball radius in pixels (default: auto from pixel size)")

    # DoG
    p.add_argument("--no-dog", action="store_true",
                   help="Disable Difference-of-Gaussians enhancement")
    p.add_argument("--dog-sigma-low", type=float, default=1.0,
                   help="DoG low sigma in pixels (default: 1.0)")
    p.add_argument("--dog-sigma-high", type=float, default=None,
                   help="DoG high sigma in pixels (default: auto from pixel size)")

    # CLAHE
    p.add_argument("--no-clahe", action="store_true",
                   help="Disable CLAHE contrast enhancement")
    p.add_argument("--clahe-clip-limit", type=float, default=0.01,
                   help="CLAHE clip limit (default: 0.01)")

    # Physical parameters for auto-radius
    p.add_argument("--pixel-size-um", type=float, default=None,
                   help="Pixel size in µm (default: read from TIFF metadata)")
    p.add_argument("--max-bead-diameter-um", type=float, default=50.0,
                   help="Maximum bead diameter in µm for auto rolling ball radius (default: 50.0)")

    # GUI
    p.add_argument("--no-launch", action="store_true",
                   help="Prepare images but do not open the Cellpose GUI")

    return p.parse_args(argv)


def launch_cellpose_gui(output_dir: Path, python_bin: str) -> None:
    """
    Launch the Cellpose GUI non-blocking, pre-loaded with the cpsam model.

    Parameters
    ----------
    output_dir : Path
        Directory containing the preprocessed training TIFFs.
    python_bin : str
        Path to the Python executable in the target conda environment.
    """
    cmd = [
        python_bin, "-m", "cellpose",
        "--image_path", str(output_dir),
        "--pretrained_model", CPSAM_MODEL_PATH,
    ]
    subprocess.Popen(cmd)


def main(argv=None):
    """
    Main entry point.

    Parameters
    ----------
    argv : list of str, optional
        Argument list for testing; defaults to sys.argv[1:] when None.
    """
    args = parse_args(argv)

    # Validate input directory
    if not args.input_dir.is_dir():
        print(f"ERROR: input_dir does not exist or is not a directory: {args.input_dir}")
        sys.exit(1)

    # Validate output directory is writable (create if needed)
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"ERROR: cannot create output directory {args.output_dir}: {exc}")
        sys.exit(1)

    # Discover TIFFs
    tiff_files = find_tiff_files(args.input_dir)
    if not tiff_files:
        print(f"No .tif/.tiff files found under {args.input_dir}")
        sys.exit(0)

    print(f"\nFound {len(tiff_files)} TIFF file(s) under {args.input_dir}")

    frames_written = 0
    skipped = 0

    for tiff_path in tiff_files:
        # Resolve pixel size: flag > TIFF metadata > None
        pixel_size_um = args.pixel_size_um
        if pixel_size_um is None:
            pixel_size_um, _source = read_pixel_size_from_tiff(tiff_path)

        # Compute auto parameters
        auto = compute_auto_params(pixel_size_um, args.max_bead_diameter_um)
        rb_radius     = args.rolling_ball_radius if args.rolling_ball_radius is not None else auto["rb_radius"]
        dog_sigma_high = args.dog_sigma_high      if args.dog_sigma_high      is not None else auto["dog_sigma_high"]

        # Load stack
        try:
            stack, metadata = load_tiff_stack(tiff_path)
        except Exception as exc:
            print(f"  WARNING: failed to load {tiff_path.name}: {exc} — skipping")
            skipped += 1
            continue

        n_timepoints = metadata["n_timepoints"]
        n_channels   = metadata["n_channels"]

        if args.channel >= n_channels:
            print(f"  WARNING: {tiff_path.name} has {n_channels} channel(s); "
                  f"channel index {args.channel} is out of range — skipping")
            skipped += 1
            continue

        frame_indices = select_frame_indices(n_timepoints)

        # Relative path from input_dir for output subdirectory structure
        rel_parent = tiff_path.parent.relative_to(args.input_dir)

        for t in frame_indices:
            raw_frame = stack[t, args.channel, :, :]

            processed = preprocess_frame(
                raw_frame,
                rolling_ball=not args.no_rolling_ball,
                rb_radius=rb_radius,
                dog=not args.no_dog,
                dog_sigma_low=args.dog_sigma_low,
                dog_sigma_high=dog_sigma_high,
                clahe=not args.no_clahe,
                clahe_clip_limit=args.clahe_clip_limit,
            )

            out_name = f"{tiff_path.stem}_t{t:03d}_ch{args.channel}.tif"
            out_path = args.output_dir / rel_parent / out_name
            save_preprocessed_tiff(processed, out_path)
            frames_written += 1

    # Summary
    print(f"\n{'='*60}")
    print("CELLPOSE TRAINING PREP COMPLETE")
    print(f"{'='*60}")
    print(f"Input directory:   {args.input_dir}  (recursive)")
    print(f"TIFFs found:       {len(tiff_files)}")
    if skipped:
        print(f"TIFFs skipped:     {skipped}")
    print(f"Frames written:    {frames_written}")
    print(f"Output directory:  {args.output_dir}")
    print(f"Channel:           {args.channel} (OA-647)")
    print(f"Preprocessing:     rolling_ball={not args.no_rolling_ball}  "
          f"dog={not args.no_dog}  clahe={not args.no_clahe}")

    if not args.no_launch:
        python_bin = detect_python_executable()
        print(f"\nLaunching Cellpose GUI...")
        print(f"  Python:  {python_bin}")
        print(f"  Model:   {CPSAM_MODEL_PATH}")
        print(f"  Images:  {args.output_dir}")
        print(f"{'='*60}\n")
        launch_cellpose_gui(args.output_dir, python_bin)
    else:
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
