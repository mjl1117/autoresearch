# Cellpose Training Image Preparation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `prepare_training_images.py` — a CLI script that recursively finds TIFFs, extracts the OA-647 channel from the first and last timepoints, applies the pipeline preprocessing, saves uint16 TIFFs, and launches the Cellpose GUI pre-loaded with `cpsam`.

**Architecture:** Standalone script in `/Users/matthew/Desktop/Fertilizer/` that imports preprocessing functions from `ml_image_analysis.py`. Each logical step (discovery, frame selection, preprocessing, saving, launch) is its own function, making each unit independently testable. The `main()` function orchestrates them with `argparse` CLI input.

**Tech Stack:** Python 3.11, numpy, scipy, scikit-image, tifffile, argparse, subprocess — all available in the `membrane-image` conda env at `/Users/matthew/miniforge3/envs/membrane-image/`. Tests run via `conda run -n membrane-image pytest`.

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `prepare_training_images.py` | Create | CLI entry point + all helper functions |
| `tests/test_prepare_training_images.py` | Create | Unit tests for every helper function |

`ml_image_analysis.py` is **not modified** — only imported from.

---

### Task 1: Scaffold and smoke test

**Files:**
- Create: `prepare_training_images.py`
- Create: `tests/test_prepare_training_images.py`

- [ ] **Step 1: Create `prepare_training_images.py` with imports and stubs**

```python
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
```

- [ ] **Step 2: Create `tests/test_prepare_training_images.py` with a smoke test**

```python
"""Tests for prepare_training_images.py"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_module_imports():
    """Script must import without errors."""
    mod = importlib.import_module("prepare_training_images")
    assert hasattr(mod, "CPSAM_MODEL_PATH")
    assert hasattr(mod, "FALLBACK_PYTHON")
```

- [ ] **Step 3: Run the smoke test — expect PASS**

```bash
conda run -n membrane-image pytest tests/test_prepare_training_images.py::test_module_imports -v
```

Expected output contains: `PASSED`

- [ ] **Step 4: Commit**

```bash
git add prepare_training_images.py tests/test_prepare_training_images.py
git commit -m "feat: scaffold prepare_training_images with imports and smoke test"
```

---

### Task 2: Recursive TIFF discovery

**Files:**
- Modify: `prepare_training_images.py` — add `find_tiff_files`
- Modify: `tests/test_prepare_training_images.py` — add tests

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_prepare_training_images.py`:

```python
import pytest
from prepare_training_images import find_tiff_files


def test_find_tiff_files_flat(tmp_path):
    """Finds .tif and .tiff files in the root directory."""
    (tmp_path / "a.tif").touch()
    (tmp_path / "b.tiff").touch()
    (tmp_path / "ignore.png").touch()
    result = find_tiff_files(tmp_path)
    names = {p.name for p in result}
    assert names == {"a.tif", "b.tiff"}


def test_find_tiff_files_recursive(tmp_path):
    """Finds TIFFs in nested subdirectories."""
    sub = tmp_path / "sub" / "deep"
    sub.mkdir(parents=True)
    (tmp_path / "root.tif").touch()
    (sub / "nested.tif").touch()
    result = find_tiff_files(tmp_path)
    names = {p.name for p in result}
    assert names == {"root.tif", "nested.tif"}


def test_find_tiff_files_empty(tmp_path):
    """Returns empty list when no TIFFs exist."""
    assert find_tiff_files(tmp_path) == []
```

- [ ] **Step 2: Run tests — expect FAIL (ImportError)**

```bash
conda run -n membrane-image pytest tests/test_prepare_training_images.py::test_find_tiff_files_flat -v
```

Expected: `FAILED` with `ImportError: cannot import name 'find_tiff_files'`

- [ ] **Step 3: Implement `find_tiff_files` in `prepare_training_images.py`**

Add after the imports block:

```python
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
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
conda run -n membrane-image pytest tests/test_prepare_training_images.py::test_find_tiff_files_flat tests/test_prepare_training_images.py::test_find_tiff_files_recursive tests/test_prepare_training_images.py::test_find_tiff_files_empty -v
```

Expected: all 3 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add prepare_training_images.py tests/test_prepare_training_images.py
git commit -m "feat: add recursive TIFF discovery"
```

---

### Task 3: Frame selection

**Files:**
- Modify: `prepare_training_images.py` — add `select_frame_indices`
- Modify: `tests/test_prepare_training_images.py` — add tests

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_prepare_training_images.py`:

```python
from prepare_training_images import select_frame_indices


def test_select_frame_indices_single():
    """Single-frame TIFF returns [0]."""
    assert select_frame_indices(1) == [0]


def test_select_frame_indices_two():
    """Two-frame TIFF returns [0, 1]."""
    assert select_frame_indices(2) == [0, 1]


def test_select_frame_indices_timelapse():
    """Multi-frame TIFF returns first and last only."""
    assert select_frame_indices(20) == [0, 19]


def test_select_frame_indices_three():
    """Three frames: first and last, not middle."""
    assert select_frame_indices(3) == [0, 2]
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
conda run -n membrane-image pytest tests/test_prepare_training_images.py::test_select_frame_indices_single -v
```

Expected: `FAILED` with `ImportError`

- [ ] **Step 3: Implement `select_frame_indices`**

Add to `prepare_training_images.py`:

```python
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
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
conda run -n membrane-image pytest tests/test_prepare_training_images.py::test_select_frame_indices_single tests/test_prepare_training_images.py::test_select_frame_indices_two tests/test_prepare_training_images.py::test_select_frame_indices_timelapse tests/test_prepare_training_images.py::test_select_frame_indices_three -v
```

Expected: all 4 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add prepare_training_images.py tests/test_prepare_training_images.py
git commit -m "feat: add frame index selection (first + last for time-lapse)"
```

---

### Task 4: Auto-parameter computation

**Files:**
- Modify: `prepare_training_images.py` — add `compute_auto_params`
- Modify: `tests/test_prepare_training_images.py` — add tests

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_prepare_training_images.py`:

```python
from prepare_training_images import compute_auto_params


def test_compute_auto_params_with_pixel_size():
    """Computes radius and sigma from physical units when pixel size is known."""
    params = compute_auto_params(pixel_size_um=0.1, max_bead_diameter_um=50.0)
    assert params["rb_radius"] == pytest.approx(500.0)
    assert params["dog_sigma_high"] == pytest.approx(125.0)


def test_compute_auto_params_fallback():
    """Returns fallback values when pixel size is None."""
    params = compute_auto_params(pixel_size_um=None, max_bead_diameter_um=50.0)
    assert params["rb_radius"] == pytest.approx(50.0)
    assert params["dog_sigma_high"] == pytest.approx(10.0)


def test_compute_auto_params_zero_pixel_size():
    """pixel_size_um=0 is treated as unknown — falls back to defaults."""
    params = compute_auto_params(pixel_size_um=0, max_bead_diameter_um=50.0)
    assert params["rb_radius"] == pytest.approx(50.0)
    assert params["dog_sigma_high"] == pytest.approx(10.0)
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
conda run -n membrane-image pytest tests/test_prepare_training_images.py::test_compute_auto_params_with_pixel_size -v
```

Expected: `FAILED` with `ImportError`

- [ ] **Step 3: Implement `compute_auto_params`**

Add to `prepare_training_images.py`:

```python
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
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
conda run -n membrane-image pytest tests/test_prepare_training_images.py::test_compute_auto_params_with_pixel_size tests/test_prepare_training_images.py::test_compute_auto_params_fallback tests/test_prepare_training_images.py::test_compute_auto_params_zero_pixel_size -v
```

Expected: all 3 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add prepare_training_images.py tests/test_prepare_training_images.py
git commit -m "feat: add auto-parameter computation for rolling ball and DoG"
```

---

### Task 5: Frame preprocessing

**Files:**
- Modify: `prepare_training_images.py` — add `preprocess_frame`
- Modify: `tests/test_prepare_training_images.py` — add tests

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_prepare_training_images.py`:

```python
from prepare_training_images import preprocess_frame


def _synthetic_frame(h=64, w=64):
    rng = np.random.default_rng(42)
    return rng.integers(100, 4000, size=(h, w), dtype=np.uint16)


def test_preprocess_frame_output_range():
    """Output is float32 normalised to [0, 1]."""
    frame = _synthetic_frame()
    result = preprocess_frame(
        frame,
        rolling_ball=True, rb_radius=20.0,
        dog=True, dog_sigma_low=1.0, dog_sigma_high=5.0,
        clahe=True, clahe_clip_limit=0.01,
    )
    assert result.dtype == np.float32
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0


def test_preprocess_frame_shape_preserved():
    """Output shape matches input shape."""
    frame = _synthetic_frame(48, 96)
    result = preprocess_frame(
        frame,
        rolling_ball=False, rb_radius=20.0,
        dog=False, dog_sigma_low=1.0, dog_sigma_high=5.0,
        clahe=False, clahe_clip_limit=0.01,
    )
    assert result.shape == (48, 96)


def test_preprocess_frame_all_disabled():
    """With all steps disabled, output is still float32 [0,1] normalised."""
    frame = _synthetic_frame()
    result = preprocess_frame(
        frame,
        rolling_ball=False, rb_radius=20.0,
        dog=False, dog_sigma_low=1.0, dog_sigma_high=5.0,
        clahe=False, clahe_clip_limit=0.01,
    )
    assert result.dtype == np.float32
    assert float(result.max()) == pytest.approx(1.0, abs=1e-5)
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
conda run -n membrane-image pytest tests/test_prepare_training_images.py::test_preprocess_frame_output_range -v
```

Expected: `FAILED` with `ImportError`

- [ ] **Step 3: Implement `preprocess_frame`**

Add to `prepare_training_images.py`:

```python
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
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
conda run -n membrane-image pytest tests/test_prepare_training_images.py::test_preprocess_frame_output_range tests/test_prepare_training_images.py::test_preprocess_frame_shape_preserved tests/test_prepare_training_images.py::test_preprocess_frame_all_disabled -v
```

Expected: all 3 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add prepare_training_images.py tests/test_prepare_training_images.py
git commit -m "feat: add frame preprocessing (rolling ball, DoG, CLAHE, normalise)"
```

---

### Task 6: Save output TIFF

**Files:**
- Modify: `prepare_training_images.py` — add `save_preprocessed_tiff`
- Modify: `tests/test_prepare_training_images.py` — add tests

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_prepare_training_images.py`:

```python
from prepare_training_images import save_preprocessed_tiff


def test_save_preprocessed_tiff_creates_file(tmp_path):
    """Saves a uint16 TIFF at the given path."""
    frame = np.random.default_rng(0).random((32, 32)).astype(np.float32)
    out = tmp_path / "out.tif"
    save_preprocessed_tiff(frame, out)
    assert out.exists()


def test_save_preprocessed_tiff_dtype(tmp_path):
    """Saved TIFF is uint16."""
    frame = np.linspace(0, 1, 32 * 32, dtype=np.float32).reshape(32, 32)
    out = tmp_path / "out.tif"
    save_preprocessed_tiff(frame, out)
    loaded = tifffile.imread(str(out))
    assert loaded.dtype == np.uint16


def test_save_preprocessed_tiff_creates_parents(tmp_path):
    """Creates intermediate directories if they don't exist."""
    frame = np.zeros((16, 16), dtype=np.float32)
    out = tmp_path / "sub" / "deep" / "frame.tif"
    save_preprocessed_tiff(frame, out)
    assert out.exists()
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
conda run -n membrane-image pytest tests/test_prepare_training_images.py::test_save_preprocessed_tiff_creates_file -v
```

Expected: `FAILED` with `ImportError`

- [ ] **Step 3: Implement `save_preprocessed_tiff`**

Add to `prepare_training_images.py`:

```python
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
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
conda run -n membrane-image pytest tests/test_prepare_training_images.py::test_save_preprocessed_tiff_creates_file tests/test_prepare_training_images.py::test_save_preprocessed_tiff_dtype tests/test_prepare_training_images.py::test_save_preprocessed_tiff_creates_parents -v
```

Expected: all 3 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add prepare_training_images.py tests/test_prepare_training_images.py
git commit -m "feat: add uint16 TIFF output with auto parent directory creation"
```

---

### Task 7: Python environment detection

**Files:**
- Modify: `prepare_training_images.py` — add `detect_python_executable`
- Modify: `tests/test_prepare_training_images.py` — add tests

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_prepare_training_images.py`:

```python
from unittest.mock import patch
from prepare_training_images import detect_python_executable, FALLBACK_PYTHON


def test_detect_python_executable_conda_env():
    """Returns the current Python when running inside a conda env."""
    fake_exe = "/Users/matthew/miniforge3/envs/membrane-image/bin/python"
    with patch("prepare_training_images.sys") as mock_sys:
        mock_sys.executable = fake_exe
        result = detect_python_executable()
    assert result == fake_exe


def test_detect_python_executable_system_python():
    """Falls back to membrane-image Python when not in a conda env."""
    with patch("prepare_training_images.sys") as mock_sys:
        mock_sys.executable = "/opt/homebrew/opt/python@3.14/bin/python3.14"
        result = detect_python_executable()
    assert result == FALLBACK_PYTHON


def test_detect_python_executable_returns_string():
    """Always returns a string."""
    result = detect_python_executable()
    assert isinstance(result, str)
    assert len(result) > 0
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
conda run -n membrane-image pytest tests/test_prepare_training_images.py::test_detect_python_executable_conda_env -v
```

Expected: `FAILED` with `ImportError`

- [ ] **Step 3: Implement `detect_python_executable`**

Add to `prepare_training_images.py`:

```python
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
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
conda run -n membrane-image pytest tests/test_prepare_training_images.py::test_detect_python_executable_conda_env tests/test_prepare_training_images.py::test_detect_python_executable_system_python tests/test_prepare_training_images.py::test_detect_python_executable_returns_string -v
```

Expected: all 3 `PASSED`

- [ ] **Step 5: Commit**

```bash
git add prepare_training_images.py tests/test_prepare_training_images.py
git commit -m "feat: add Python environment detection with conda fallback"
```

---

### Task 8: CLI, main orchestration, and GUI launch

**Files:**
- Modify: `prepare_training_images.py` — add `parse_args`, `launch_cellpose_gui`, `main`
- Modify: `tests/test_prepare_training_images.py` — add integration test

- [ ] **Step 1: Write the failing integration test**

Add to `tests/test_prepare_training_images.py`:

```python
from unittest.mock import patch, MagicMock
from prepare_training_images import main


def _make_fake_tiff(path: Path, n_timepoints=3):
    """Write a minimal (T, C, H, W) TIFF for testing."""
    rng = np.random.default_rng(0)
    stack = rng.integers(100, 4000,
                         size=(n_timepoints * 2, 32, 32),
                         dtype=np.uint16)
    tifffile.imwrite(str(path), stack)


def test_main_creates_output_files(tmp_path):
    """main() writes first+last frame TIFFs into the output directory."""
    input_dir  = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    _make_fake_tiff(input_dir / "exp1.tif", n_timepoints=5)

    with patch("prepare_training_images.subprocess.Popen"), \
         patch("prepare_training_images.read_pixel_size_from_tiff", return_value=None):
        main([str(input_dir),
              "--output-dir", str(output_dir),
              "--no-launch"])

    tifs = list(output_dir.rglob("*.tif"))
    assert len(tifs) == 2                       # frame 0 and frame 4
    names = {p.name for p in tifs}
    assert any("t000" in n for n in names)
    assert any("t004" in n for n in names)


def test_main_single_frame_tiff(tmp_path):
    """Single-frame TIFF produces exactly one output file."""
    input_dir  = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    _make_fake_tiff(input_dir / "single.tif", n_timepoints=1)

    with patch("prepare_training_images.subprocess.Popen"), \
         patch("prepare_training_images.read_pixel_size_from_tiff", return_value=None):
        main([str(input_dir),
              "--output-dir", str(output_dir),
              "--no-launch"])

    tifs = list(output_dir.rglob("*.tif"))
    assert len(tifs) == 1
```

- [ ] **Step 2: Run the integration test — expect FAIL**

```bash
conda run -n membrane-image pytest tests/test_prepare_training_images.py::test_main_creates_output_files -v
```

Expected: `FAILED` with `ImportError` (no `main` yet)

- [ ] **Step 3: Implement `parse_args`, `launch_cellpose_gui`, and `main`**

Add to `prepare_training_images.py`:

```python
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
            pixel_size_um = read_pixel_size_from_tiff(tiff_path)

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
```

- [ ] **Step 4: Run all tests — expect PASS**

```bash
conda run -n membrane-image pytest tests/test_prepare_training_images.py -v
```

Expected: all tests `PASSED`, 0 failures

- [ ] **Step 5: Quick manual smoke test**

```bash
conda run -n membrane-image python prepare_training_images.py --help
```

Expected: argparse help text printed with all flags visible, no errors.

- [ ] **Step 6: Commit**

```bash
git add prepare_training_images.py tests/test_prepare_training_images.py
git commit -m "feat: add CLI, main orchestration, and Cellpose GUI launch"
```

---

## Self-Review Notes

**Spec coverage check:**
- [x] Recursive TIFF discovery → Task 2
- [x] First + last frame selection → Task 3
- [x] Channel extraction (OA-647, configurable) → Task 5 (`preprocess_frame`) + Task 8 (`main`)
- [x] Auto-parameter computation matching inference pipeline → Task 4
- [x] Preprocessing: rolling ball, DoG, CLAHE, normalise → Task 5
- [x] uint16 TIFF output preserving relative subdirectory structure → Task 6
- [x] Environment detection with conda fallback → Task 7
- [x] GUI launch with `--pretrained_model cpsam` → Task 8
- [x] `--no-launch` flag → Task 8 (`parse_args` + `main`)
- [x] All preprocessing flags with defaults → Task 8 (`parse_args`)
- [x] Error handling: bad TIFFs, wrong channel, unwritable output → Task 8 (`main`)
- [x] Summary output → Task 8 (`main`)

**No placeholders found.**

**Type consistency:** `frame_indices` is `list[int]` produced by `select_frame_indices` and consumed as `stack[t, ...]` in `main` — consistent. `compute_auto_params` returns `dict` with keys `rb_radius` and `dog_sigma_high` — both consumed by that exact name in `main`.
