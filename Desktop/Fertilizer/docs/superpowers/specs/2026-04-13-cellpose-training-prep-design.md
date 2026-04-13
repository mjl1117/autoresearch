# Design: Cellpose Training Image Preparation Script

**Date:** 2026-04-13
**File:** `prepare_training_images.py`
**Status:** Approved

---

## Overview

A standalone script (`prepare_training_images.py`) that recursively finds all TIFF files in a given directory, extracts the OA-647 channel from the first and last timepoints, applies the same preprocessing pipeline used in `ml_image_analysis.py`, saves the results as grayscale TIFFs, and launches the Cellpose GUI pre-loaded with the `cpsam` model and the output directory.

This ensures training and inference see identical image representations.

---

## CLI Interface

```
python prepare_training_images.py <input_dir> [options]
```

### Positional Arguments

| Argument | Description |
|---|---|
| `input_dir` | Root directory to search recursively for `*.tif` / `*.tiff` files |

### Optional Flags

| Flag | Default | Description |
|---|---|---|
| `--output-dir` | `training_images/` | Directory where preprocessed TIFFs are saved |
| `--channel` | `1` | Channel index to extract (1 = OA-647) |
| `--no-rolling-ball` | off | Disable rolling ball background subtraction |
| `--rolling-ball-radius` | auto | Override rolling ball radius in pixels |
| `--no-dog` | off | Disable Difference of Gaussians enhancement |
| `--dog-sigma-low` | `1.0` | DoG low sigma in pixels |
| `--dog-sigma-high` | auto | DoG high sigma in pixels |
| `--no-clahe` | off | Disable CLAHE contrast enhancement |
| `--clahe-clip-limit` | `0.01` | CLAHE clip limit |
| `--pixel-size-um` | auto from metadata | Pixel size (µm) used for auto-radius computation |
| `--max-bead-diameter-um` | `50.0` | Maximum bead diameter (µm) for auto rolling ball radius |
| `--no-launch` | off | Prepare images but do not open the Cellpose GUI |

---

## Architecture

### Single file: `prepare_training_images.py`

Imports from `ml_image_analysis.py`:
- `apply_rolling_ball_background`
- `apply_dog_enhancement`
- `enhance_contrast_clahe`
- `get_pixel_size_from_tiff`
- `load_tiff_stack` (for reshape/transpose logic)

No other new files are created.

---

## Data Flow

```
input_dir/
  ├── subdir_a/
  │     └── experiment1.tif   →  [T=20, C=2, H, W]
  └── experiment2.tif         →  [T=1,  C=2, H, W]

For each TIFF:
  1. Load stack → (T, C, H, W)
  2. Select frames:
       T == 1  →  [frame 0]
       T >= 2  →  [frame 0, frame T-1]
  3. Extract channel index --channel (default 1)
  4. Apply preprocessing:
       a. Rolling ball background subtraction (Gaussian blur, sigma = radius / sqrt(2))
       b. DoG enhancement
       c. CLAHE contrast enhancement
       d. Normalize to float32 [0, 1]
  5. Scale to uint16 [0, 65535] for TIFF output
  6. Save to output_dir/<relative_subpath>/<stem>_t{N}_ch{channel}.tif

training_images/
  ├── subdir_a/
  │     ├── experiment1_t000_ch1.tif
  │     └── experiment1_t019_ch1.tif
  └── experiment2_t000_ch1.tif
```

---

## Auto-Parameter Computation

Auto-radius computation mirrors `ml_image_analysis.py` exactly:

1. **Pixel size source priority:** `--pixel-size-um` flag → OME metadata from TIFF → `None`
2. **Rolling ball radius:**
   - If pixel size known: `max_bead_diameter_um / pixel_size_um`
   - Fallback: `50.0` px
3. **DoG high sigma:**
   - If pixel size known: `max_bead_diameter_um / pixel_size_um / 4.0`
   - Fallback: `10.0` px

---

## Cellpose GUI Launch

### Environment Detection

At runtime, inspect `sys.executable`:

```python
import sys, re
match = re.search(r"envs/([^/]+)/", sys.executable)
if match:
    python_bin = sys.executable          # use current env
else:
    python_bin = "/Users/matthew/miniforge3/envs/membrane-image/bin/python"
```

### Launch Command

```
<python_bin> -m cellpose \
    --image_path <output_dir> \
    --pretrained_model /Users/matthew/.cellpose/models/cpsam
```

Launched with `subprocess.Popen` (non-blocking). Terminal returns immediately after GUI opens.

---

## Output Summary (printed before launch)

```
============================================================
CELLPOSE TRAINING PREP COMPLETE
============================================================
Input directory:   /path/to/input_dir  (recursive)
TIFFs found:       8
Frames written:    14
Output directory:  training_images/
Channel:           1 (OA-647)
Preprocessing:     rolling_ball=True  dog=True  clahe=True

Launching Cellpose GUI...
  Python:  /Users/matthew/miniforge3/envs/membrane-image/bin/python
  Model:   /Users/matthew/.cellpose/models/cpsam
  Images:  training_images/
============================================================
```

---

## Error Handling

- TIFF fails to load → print warning, skip file, continue
- TIFF has unexpected shape (not 2D/3D/4D) → print warning with shape, skip
- Channel index out of bounds → print warning with available channels, skip
- Output directory not writable → raise with clear message before processing starts
- Cellpose not importable in detected environment → warn and print manual launch command

---

## Non-Goals

- No annotation logic (handled inside the Cellpose GUI)
- No training invocation (handled inside the Cellpose GUI)
- No modification of original TIFF files
- No GFP channel extraction (out of scope for this tool)
