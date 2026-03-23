# Image Analysis Pipeline — Notebook-to-Scripts + MPS Acceleration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert `Image_Analysis_Pipeline.ipynb` and `nadh_matching.ipynb` to Python scripts, add PyTorch MPS batched acceleration for the morphological parameter search, and run the full pipeline over all 14 `.nd2` files in `chemotaxis_by_date/`.

**Architecture:** Seven focused modules under `scripts/pipeline/` are extracted from three notebook cells (cell 3: 1210 lines, cell 8: 3753 lines, cell 99: ~950 lines) plus `nadh_matching.ipynb`. A new `mps_morph.py` module replaces the sequential `int_search(score_mask, ...)` bottleneck with a batched `F.conv2d` call that runs all (erosion, dilation) parameter combinations simultaneously on MPS. `run_pipeline.py` drives all 14 files in sequence with per-file error isolation and a post-loop cross-condition aggregation phase.

**Tech Stack:** Python 3.10+, numpy, scikit-image, scipy, matplotlib (Agg backend), seaborn, pandas, tqdm, nd2, Pillow, torch (optional, for MPS/CUDA/CPU-batched acceleration)

**Spec:** `docs/superpowers/specs/2026-03-23-image-pipeline-scripts-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `scripts/pipeline/__init__.py` | Create | Package marker |
| `scripts/pipeline/utils.py` | Create | `circularity`, `int_search`, `convert_to_8_bit`, `standardize_stack`, `get_image_files` |
| `scripts/pipeline/mps_morph.py` | Create | Device detection, `_make_disk_kernel`, `mps_binary_erosion`, `mps_binary_dilation`, `batched_score_mask`, `_count_valid_circles` |
| `scripts/pipeline/roi_extraction.py` | Create | `score_mask`, `optimize_mask_and_regions`, `regions_in_mask`, `plot_circular_regions`, `save_circular_rois`, `find_next_roi`, `count_path_breaks`, `count_clashes`, `score_drift`, `find_roi_trajectory`, `save_tracked_rois`, `process_nd2_directory`, `process_single_nd2` |
| `scripts/pipeline/analysis.py` | Create | `extract_radial_profile_from_tif`, `average_radial_profiles`, `average_cluster_sizes`, `process_roi_library`, `save_results_to_json` |
| `scripts/pipeline/visualization.py` | Create | All 28 `plot_*` functions + `animate_stack` + frame helpers; add `save` param to 4 missing functions |
| `scripts/pipeline/nadh.py` | Create | `standardize_stack_nadh`, `process_nd2_directory_nadh`, `process_single_nd2_nadh` |
| `scripts/pipeline/nadh_matching.py` | Create | `match_rois_and_export`, `average_droplet_intensity` |
| `scripts/run_pipeline.py` | Create | CLI driver: per-file loop + post-loop cross-condition aggregation |
| `scripts/requirements.txt` | Create | Pinned dependency list |
| `tests/pipeline/test_utils.py` | Create | Unit tests for utils |
| `tests/pipeline/test_mps_morph.py` | Create | Correctness + fallback tests for MPS layer |
| `tests/pipeline/test_roi_extraction.py` | Create | Unit tests for process_single_nd2 wrapper |
| `tests/pipeline/test_analysis.py` | Create | Unit tests for radial profile extraction and averaging |
| `tests/pipeline/test_nadh_matching.py` | Create | Unit tests for ROI matching and intensity averaging |
| `tests/pipeline/test_run_pipeline.py` | Create | Integration smoke test for run_pipeline |

---

## Task 1: Scaffold — directories, `__init__.py`, requirements

**Files:**
- Create: `scripts/pipeline/__init__.py`
- Create: `scripts/requirements.txt`
- Create: `tests/__init__.py`
- Create: `tests/pipeline/__init__.py`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p scripts/pipeline tests/pipeline
touch scripts/pipeline/__init__.py tests/__init__.py tests/pipeline/__init__.py
```

- [ ] **Step 2: Write `scripts/requirements.txt`**

```
nd2
numpy
scikit-image
scipy
matplotlib
pandas
seaborn
tqdm
Pillow
colorama
pytest
torch
```

- [ ] **Step 3: Verify imports available in current environment**

```bash
cd /Users/matthew/Desktop/Chemotaxis && python3 -c "import nd2, numpy, skimage, scipy, matplotlib, pandas, seaborn, tqdm, PIL; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add scripts/ tests/
git commit -m "chore: scaffold scripts/pipeline package and test directories"
```

---

## Task 2: `utils.py` — core utility functions

**Source:** Notebook cell 3 (lines 0–175: `get_image_files`, `convert_to_8_bit`, `circularity`, `int_search`, `get_zoomed_roi`, `standardize_stack`)

**Files:**
- Create: `scripts/pipeline/utils.py`
- Create: `tests/pipeline/test_utils.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/pipeline/test_utils.py
import sys; sys.path.insert(0, "scripts")
import numpy as np
import pytest
from unittest.mock import MagicMock
from pipeline.utils import circularity, convert_to_8_bit, standardize_stack, int_search


def test_circularity_perfect_circle():
    region = MagicMock()
    region.area = np.pi * 10**2
    region.perimeter = 2 * np.pi * 10
    assert abs(circularity(region) - 1.0) < 0.01


def test_circularity_zero_perimeter():
    region = MagicMock()
    region.perimeter = 0
    assert circularity(region) == 0.0


def test_convert_to_8bit_range():
    img = np.array([[0, 500], [1000, 2000]], dtype=np.uint16)
    out = convert_to_8_bit(img)
    assert out.dtype == np.uint8
    assert out.max() == 255
    assert out.min() == 0


def test_standardize_stack_2d():
    img = np.zeros((64, 64))
    out = standardize_stack(img)
    assert out.shape == (64, 64)


def test_standardize_stack_tchw():
    stack = np.zeros((10, 2, 64, 64))
    out = standardize_stack(stack, layout="tchw")
    assert out.shape == (10, 64, 64)


def test_int_search_finds_best():
    def score_fn(params, img):
        return -abs(params[0] - 3)  # best at x=3
    best_params, best_score = int_search(score_fn, None, [1], [6])
    assert best_params[0] == 3
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/matthew/Desktop/Chemotaxis && python3 -m pytest tests/pipeline/test_utils.py -v 2>&1 | head -30
```
Expected: `ModuleNotFoundError: No module named 'pipeline'`

- [ ] **Step 3: Create `scripts/pipeline/utils.py`**

Port from notebook cell 3 lines 0–175. The file should contain:
- `get_image_files(path, extension, recursive)` — unchanged from notebook
- `convert_to_8_bit(image)` — unchanged
- `circularity(region)` — keep only one copy (drop the duplicate at line 256)
- `int_search(score_function, my_image, minimum, maximum, skips, verbose)` — unchanged
- `get_zoomed_roi(my_image, region, zoom_factor, pad, contrast_thresh)` — unchanged
- `standardize_stack(stack, channel, layout)` — unchanged

Top of file:
```python
import itertools
import numpy as np
from pathlib import Path
from skimage import img_as_ubyte, transform
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/matthew/Desktop/Chemotaxis && python3 -m pytest tests/pipeline/test_utils.py -v
```
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/utils.py tests/pipeline/test_utils.py
git commit -m "feat: add pipeline/utils.py with core utility functions"
```

---

## Task 3: `mps_morph.py` — batched MPS morphological acceleration

**Files:**
- Create: `scripts/pipeline/mps_morph.py`
- Create: `tests/pipeline/test_mps_morph.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/pipeline/test_mps_morph.py
import sys; sys.path.insert(0, "scripts")
import numpy as np
import pytest
from skimage.morphology import binary_erosion, binary_dilation, disk


def make_circle_image(size=64, radius=10):
    """Synthetic binary image with one circle."""
    img = np.zeros((size, size), dtype=bool)
    cy, cx = size // 2, size // 2
    Y, X = np.ogrid[:size, :size]
    img[(Y - cy)**2 + (X - cx)**2 <= radius**2] = True
    return img


def test_mps_erosion_matches_skimage():
    from pipeline.mps_morph import mps_binary_erosion
    img = make_circle_image()
    expected = binary_erosion(img, disk(3)).astype(bool)
    result = mps_binary_erosion(img, 3).astype(bool)
    # Allow small boundary differences due to padding
    assert np.mean(result == expected) > 0.98


def test_mps_dilation_matches_skimage():
    from pipeline.mps_morph import mps_binary_dilation
    img = make_circle_image()
    expected = binary_dilation(img, disk(3)).astype(bool)
    result = mps_binary_dilation(img, 3).astype(bool)
    assert np.mean(result == expected) > 0.98


def test_batched_score_mask_returns_tuple():
    from pipeline.mps_morph import batched_score_mask
    img = make_circle_image().astype(np.uint8) * 200
    result = batched_score_mask(img, minimum=[1, 1], maximum=[5, 5])
    assert len(result) == 2
    assert len(result[0]) == 2


def test_device_fallback_no_crash():
    """mps_morph imports without crash even if torch missing."""
    import importlib
    import pipeline.mps_morph as m
    assert hasattr(m, 'batched_score_mask')
    assert hasattr(m, 'mps_binary_erosion')
    assert hasattr(m, 'mps_binary_dilation')
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/matthew/Desktop/Chemotaxis && python3 -m pytest tests/pipeline/test_mps_morph.py -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError: No module named 'pipeline.mps_morph'`

- [ ] **Step 3: Create `scripts/pipeline/mps_morph.py`**

```python
# scripts/pipeline/mps_morph.py
import warnings
import numpy as np
from skimage import measure

try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    warnings.warn(
        "torch not installed — using skimage fallback for morphological ops. "
        "Install torch to enable MPS/CUDA/batched-CPU acceleration.",
        stacklevel=2
    )


def _get_device():
    if not _TORCH_AVAILABLE:
        return None
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


_DEVICE = _get_device()


def _make_disk_kernel(radius: int, max_radius: int) -> "torch.Tensor":
    """Disk kernel of given radius, zero-padded to (2*max_radius+1)^2."""
    from skimage.morphology import disk as sk_disk
    kernel = sk_disk(radius).astype(np.float32)
    max_size = 2 * max_radius + 1
    pad_total = max_size - kernel.shape[0]
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before
    padded = np.pad(kernel, ((pad_before, pad_after), (pad_before, pad_after)),
                    mode="constant", constant_values=0)
    padded = padded[:max_size, :max_size]  # ensure exact size
    return torch.from_numpy(padded).unsqueeze(0).unsqueeze(0)  # [1, 1, kH, kW]


def _check_mps_memory(device) -> "torch.device":
    """Fall back to CPU if MPS memory >80% full."""
    if not _TORCH_AVAILABLE or device is None or device.type != "mps":
        return device
    try:
        current = torch.mps.current_allocated_memory()
        maximum = torch.mps.recommended_max_memory()
        if maximum > 0 and current > 0.8 * maximum:
            warnings.warn("MPS memory >80% full; falling back to CPU for this frame.")
            return torch.device("cpu")
    except Exception:
        pass
    return device


def mps_binary_erosion(image: np.ndarray, radius: int) -> np.ndarray:
    """Binary erosion with disk(radius). Uses MPS/CUDA/CPU torch if available."""
    if not _TORCH_AVAILABLE or _DEVICE is None:
        from skimage.morphology import binary_erosion, disk
        return binary_erosion(image.astype(bool), disk(radius))

    device = _DEVICE
    kernel = _make_disk_kernel(radius, radius).to(device)
    kernel_sum = float(kernel.sum())
    img_t = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    out = F.conv2d(img_t, kernel, padding=radius)
    return (out >= kernel_sum - 1e-6).squeeze().cpu().numpy().astype(bool)


def mps_binary_dilation(image: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation with disk(radius). Uses MPS/CUDA/CPU torch if available."""
    if not _TORCH_AVAILABLE or _DEVICE is None:
        from skimage.morphology import binary_dilation, disk
        return binary_dilation(image.astype(bool), disk(radius))

    device = _DEVICE
    kernel = _make_disk_kernel(radius, radius).to(device)
    img_t = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    out = F.conv2d(img_t, kernel, padding=radius)
    return (out > 0.5).squeeze().cpu().numpy().astype(bool)


def _count_valid_circles(mask: np.ndarray, circ_thresh=0.9, min_area=200,
                          base_min_perimeter=50, base_max_perimeter=200, **_) -> int:
    """Count valid circular regions in a binary mask. CPU-only (requires CC labeling)."""
    labeled = measure.label(mask, connectivity=2)
    props = measure.regionprops(labeled)
    count = 0
    for r in props:
        if r.perimeter < 1e-9:
            continue
        circ = (4 * np.pi * r.area) / (r.perimeter ** 2)
        if (circ >= circ_thresh
                and r.area >= min_area
                and r.perimeter >= base_min_perimeter
                and (base_max_perimeter is None or r.perimeter <= base_max_perimeter)):
            count += 1
    return count


def batched_score_mask(image: np.ndarray, minimum: list, maximum: list,
                        skips: list = None, **score_kwargs) -> tuple:
    """
    Batched replacement for int_search(score_mask, image, minimum, maximum, skips).

    Runs all (erosion_r, dilation_r) combinations simultaneously on MPS/CUDA/CPU
    using padded F.conv2d kernels, then scores each result on CPU.

    Returns
    -------
    (best_params, best_score) : same contract as int_search
    """
    if skips is None:
        skips = [1, 1]

    erosion_radii = list(range(minimum[0], maximum[0], skips[0]))
    dilation_radii = list(range(minimum[1], maximum[1], skips[1]))

    if not erosion_radii or not dilation_radii:
        return (minimum[0], minimum[1]), 0

    if not _TORCH_AVAILABLE:
        # Full skimage fallback
        from pipeline.utils import int_search
        from pipeline.roi_extraction import score_mask
        return int_search(score_mask, image, minimum, maximum, skips)

    device = _check_mps_memory(_DEVICE)
    max_r = max(max(erosion_radii), max(dilation_radii))
    padding = max_r

    # Stack all erosion kernels: [N_e, 1, kH, kW]
    erosion_kernels = torch.cat(
        [_make_disk_kernel(r, max_r) for r in erosion_radii], dim=0
    ).to(device)
    erosion_sums = erosion_kernels.view(len(erosion_radii), -1).sum(dim=1)

    # Stack all dilation kernels: [N_d, 1, kH, kW]
    dilation_kernels = torch.cat(
        [_make_disk_kernel(r, max_r) for r in dilation_radii], dim=0
    ).to(device)

    img_t = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    # Batch-erode: [1, N_e, H, W]
    eroded_conv = F.conv2d(img_t, erosion_kernels, padding=padding)
    eroded_batch = (eroded_conv >= erosion_sums.view(1, -1, 1, 1) - 1e-6).float()

    best_score = -np.inf
    best_params = (erosion_radii[0], dilation_radii[0])

    for i, e_r in enumerate(erosion_radii):
        # Extract one eroded image: [1, 1, H, W]
        eroded_img = eroded_batch[:, i:i+1, :, :]
        # Batch-dilate: [1, N_d, H, W]
        dilated_conv = F.conv2d(eroded_img, dilation_kernels, padding=padding)
        dilated_batch = (dilated_conv > 0.5).cpu().numpy()  # [1, N_d, H, W]

        for j, d_r in enumerate(dilation_radii):
            mask = dilated_batch[0, j]
            score = _count_valid_circles(mask, **score_kwargs)
            if score > best_score:
                best_score = score
                best_params = (e_r, d_r)

    return best_params, float(best_score)
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/matthew/Desktop/Chemotaxis && python3 -m pytest tests/pipeline/test_mps_morph.py -v
```
Expected: 4 passed (`test_batched_score_mask_matches_sequential` is deferred to Task 4 where `roi_extraction` is available)

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/mps_morph.py tests/pipeline/test_mps_morph.py
git commit -m "feat: add pipeline/mps_morph.py with batched MPS morphological acceleration"
```

---

## Task 4: `roi_extraction.py` — ROI detection, tracking, and per-file wrapper

**Source:** Notebook cell 3 lines 256–1210: `score_mask`, `show_image_channels`, `regions_in_mask`, `optimize_mask_and_regions`, `plot_circular_regions`, `save_circular_rois`, `find_next_roi`, `count_path_breaks`, `count_clashes`, `score_drift`, `find_roi_trajectory`, `save_tracked_rois`, `process_nd2_directory`

**Files:**
- Create: `scripts/pipeline/roi_extraction.py`
- Create: `tests/pipeline/test_roi_extraction.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/pipeline/test_roi_extraction.py
import sys; sys.path.insert(0, "scripts")
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


def make_mask_with_circle(size=128, r=15):
    mask = np.zeros((size, size), dtype=bool)
    cy, cx = size // 2, size // 2
    Y, X = np.ogrid[:size, :size]
    mask[(Y - cy)**2 + (X - cx)**2 <= r**2] = True
    return mask


def test_score_mask_detects_circle():
    from pipeline.roi_extraction import score_mask
    mask = make_mask_with_circle().astype(np.uint8) * 200
    score = score_mask((3, 3), mask, circ_thresh=0.85, min_area=100,
                       base_min_perimeter=30, base_max_perimeter=500)
    assert score >= 1


def test_regions_in_mask_returns_list():
    from pipeline.roi_extraction import regions_in_mask
    mask = make_mask_with_circle()
    results = regions_in_mask(mask)
    assert isinstance(results, list)
    assert len(results) >= 1
    assert "circularity" in results[0]
    assert "centroid" in results[0]


def test_optimize_mask_and_regions_returns_params():
    from pipeline.roi_extraction import optimize_mask_and_regions
    img = make_mask_with_circle().astype(np.uint8) * 200
    best_params, regions = optimize_mask_and_regions(
        img, minimum=[1, 1], maximum=[6, 6], verbose=0
    )
    assert len(best_params) == 2
    assert isinstance(regions, list)


def test_process_single_nd2_exists():
    from pipeline.roi_extraction import process_single_nd2
    import inspect
    sig = inspect.signature(process_single_nd2)
    assert "nd2_path" in sig.parameters
    assert "save_path" in sig.parameters


def test_find_next_roi_matches_closest():
    from pipeline.roi_extraction import find_next_roi
    last = [(10, 10), (50, 50)]
    current = [
        {"centroid": (11, 11), "label": 1},
        {"centroid": (51, 51), "label": 2},
    ]
    matched = find_next_roi(last, current, max_distance=10)
    assert matched[0]["centroid"] == (11, 11)
    assert matched[1]["centroid"] == (51, 51)


def test_batched_score_mask_matches_sequential():
    """Verify batched MPS scoring returns same score as sequential skimage scoring.
    Uses explicit score_kwargs so both paths use identical parameters."""
    from pipeline.mps_morph import batched_score_mask
    from pipeline.roi_extraction import score_mask
    from pipeline.utils import int_search
    score_kwargs = dict(circ_thresh=0.9, min_area=200,
                        base_min_perimeter=50, base_max_perimeter=200)
    img = make_mask_with_circle(size=128, r=20).astype(np.uint8) * 200
    batched_params, batched_score = batched_score_mask(
        img, minimum=[1, 1], maximum=[8, 8], **score_kwargs
    )
    seq_params, seq_score = int_search(
        score_mask, img, minimum=[1, 1], maximum=[8, 8],
        skips=None, verbose=0, **score_kwargs
    )
    # Scores must match (params may differ if multiple combos tie)
    assert int(batched_score) == int(seq_score)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/matthew/Desktop/Chemotaxis && python3 -m pytest tests/pipeline/test_roi_extraction.py -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError: No module named 'pipeline.roi_extraction'`

- [ ] **Step 3: Create `scripts/pipeline/roi_extraction.py`**

Port all functions from cell 3 lines 256–1210. Key changes from the notebook:

1. **Top of file imports:**
```python
import nd2
import numpy as np
from pathlib import Path
from skimage import filters, img_as_ubyte, io, measure, morphology, transform
from skimage.morphology import disk, binary_dilation, binary_erosion
from scipy.spatial.distance import euclidean
from pipeline.utils import (
    circularity, int_search, standardize_stack, get_image_files, get_zoomed_roi
)
from pipeline.mps_morph import batched_score_mask, mps_binary_erosion, mps_binary_dilation
```

2. **Replace `int_search(score_mask, ...)` calls** in `optimize_mask_and_regions` with `batched_score_mask(...)`.

3. **Replace `binary_erosion`/`binary_dilation`** calls inside `score_drift` and `find_roi_trajectory` with `mps_binary_erosion`/`mps_binary_dilation`.

4. **Add `process_single_nd2`** at the bottom:
```python
def process_single_nd2(nd2_path: Path, save_path: Path = None,
                        circ_thresh: float = 0.9, max_distance: int = 40,
                        minimum_mask: list = None, maximum_mask: list = None,
                        skips: list = None, channel: int = 0,
                        pad: int = 0, verbose: int = 1):
    """
    Process a single .nd2 file: detect and track ROIs, save crops to
    save_path/tagged_rois/<stem>/.

    Parameters mirror process_nd2_directory. Creates a one-element list
    and delegates to the inner processing loop.
    """
    if minimum_mask is None:
        minimum_mask = [1, 1]
    if maximum_mask is None:
        maximum_mask = [20, 20]
    nd2_path = Path(nd2_path)
    if save_path is None:
        save_path = nd2_path.parent
    save_path = Path(save_path)

    stack = nd2.imread(str(nd2_path))
    stack = standardize_stack(stack, channel=channel)
    image_name = nd2_path.stem

    best_params, regions = optimize_mask_and_regions(
        stack[0] if stack.ndim == 3 else stack,
        minimum=minimum_mask, maximum=maximum_mask,
        skips=skips, verbose=verbose, circ_thresh=circ_thresh
    )
    if not regions:
        if verbose:
            print(f"No ROIs found in {nd2_path.name}")
        return

    tracked = find_roi_trajectory(
        stack, best_params, channel=channel,
        circ_thresh=circ_thresh, max_distance=max_distance, verbose=verbose
    )
    save_tracked_rois(stack, tracked,
                      save_root=save_path / "tagged_rois" / image_name,
                      image_name=image_name, pad=pad)
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/matthew/Desktop/Chemotaxis && python3 -m pytest tests/pipeline/test_roi_extraction.py tests/pipeline/test_mps_morph.py -v
```
Expected: all pass (now includes `test_batched_score_mask_matches_sequential` from Task 3 that was deferred here)

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/roi_extraction.py tests/pipeline/test_roi_extraction.py
git commit -m "feat: add pipeline/roi_extraction.py with MPS-accelerated optimize_mask_and_regions and process_single_nd2"
```

---

## Task 5: `analysis.py` — radial profile extraction and averaging

**Source:** Notebook cell 8 lines 0–465: `extract_radial_profile_from_tif`, `average_radial_profiles`, `average_cluster_sizes`, `process_roi_library`, `save_results_to_json`

**Files:**
- Create: `scripts/pipeline/analysis.py`
- Create: `tests/pipeline/test_analysis.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/pipeline/test_analysis.py
import sys; sys.path.insert(0, "scripts")
import numpy as np
import pytest
import json
import tempfile
from pathlib import Path
from skimage import io


def make_radial_tif(tmp_path, radius=20, size=64):
    """Synthetic circular ROI tif — bright center, dark edge."""
    img = np.zeros((size, size), dtype=np.uint16)
    cy, cx = size // 2, size // 2
    Y, X = np.ogrid[:size, :size]
    dist = np.sqrt((Y - cy)**2 + (X - cx)**2)
    img = np.clip(1000 - dist * 20, 0, 1000).astype(np.uint16)
    p = tmp_path / "stack_0.tif"
    io.imsave(str(p), img)
    return p


def test_extract_radial_profile_returns_arrays(tmp_path):
    from pipeline.analysis import extract_radial_profile_from_tif
    tif = make_radial_tif(tmp_path)
    radii, intensities = extract_radial_profile_from_tif(tif)
    assert len(radii) > 0
    assert len(radii) == len(intensities)


def test_average_radial_profiles_structure():
    from pipeline.analysis import average_radial_profiles
    results = [
        {"image": "img1", "roi": "roi_10_10", "stack_index": 0,
         "radii": [0.0, 1.0, 2.0], "intensities": [1.0, 0.8, 0.5]},
        {"image": "img1", "roi": "roi_10_10", "stack_index": 0,
         "radii": [0.0, 1.0, 2.0], "intensities": [1.0, 0.9, 0.6]},
    ]
    averaged = average_radial_profiles(results, n_common=3)
    assert len(averaged) > 0
    assert "mean" in averaged[0]


def test_save_results_to_json(tmp_path):
    from pipeline.analysis import save_results_to_json
    data = [{"image": "x", "roi": "r", "mean": [1.0]}]
    save_results_to_json(data, "test.json", save_path=tmp_path)
    out = json.loads((tmp_path / "test.json").read_text())
    assert out[0]["image"] == "x"


def test_process_roi_library_empty_dir(tmp_path):
    from pipeline.analysis import process_roi_library
    # Empty library should return empty list without crashing
    result = process_roi_library(str(tmp_path))
    assert isinstance(result, list)
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /Users/matthew/Desktop/Chemotaxis && python3 -m pytest tests/pipeline/test_analysis.py -v 2>&1 | head -20
```

- [ ] **Step 3: Create `scripts/pipeline/analysis.py`**

Port from cell 8 lines 0–465. Top of file:
```python
import json
import numpy as np
from pathlib import Path
from skimage import io
from scipy.interpolate import interp1d
```

Use the 4-arg `save_results_to_json` exclusively (drop the 2-arg version from cell 3).

- [ ] **Step 4: Run tests**

```bash
cd /Users/matthew/Desktop/Chemotaxis && python3 -m pytest tests/pipeline/test_analysis.py -v
```
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/analysis.py tests/pipeline/test_analysis.py
git commit -m "feat: add pipeline/analysis.py with radial profile extraction and averaging"
```

---

## Task 6: `visualization.py` — all 28 plot functions

**Source:** Notebook cell 8 lines 466–3753

**Files:**
- Create: `scripts/pipeline/visualization.py`
- Create: `tests/pipeline/test_visualization.py`

- [ ] **Step 1: Write smoke tests**

```python
# tests/pipeline/test_visualization.py
import sys; sys.path.insert(0, "scripts")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import tempfile
from pathlib import Path


def make_averaged_profiles(n_stacks=3, n_radii=10):
    return [
        {
            "image": "img1",
            "roi": f"roi_{i*10}_{i*10}",
            "stack_index": s,
            "radii": list(np.linspace(0, 1, n_radii)),
            "mean": list(np.random.rand(n_radii)),
            "stderr": list(np.random.rand(n_radii) * 0.1),
            "x_max": float(np.random.rand()),
            "cluster_areas": list(np.random.rand(5) * 100),
        }
        for i in range(3) for s in range(n_stacks)
    ]


def test_plot_mean_profiles_no_crash(tmp_path):
    from pipeline.visualization import plot_mean_profiles
    profiles = make_averaged_profiles()
    fig = plot_mean_profiles(
        averaged_profiles=profiles,
        dirs=[0, 1, 2],
        labels=["t=0", "t=1", "t=2"],
        save=True,
        save_path=tmp_path,
        title="test"
    )
    plt.close("all")


def test_plot_max_intensities_by_time_no_crash(tmp_path):
    from pipeline.visualization import plot_max_intensities_by_time
    profiles = make_averaged_profiles()
    plot_max_intensities_by_time(
        averaged_profiles=profiles, dirs=np.arange(3),
        error="sem", title="test", save=True, save_path=tmp_path
    )
    plt.close("all")


def test_animate_stack_creates_gif(tmp_path):
    from pipeline.visualization import animate_stack, plot_mean_profiles_frame
    profiles = make_averaged_profiles(n_stacks=5)
    gif_path = tmp_path / "test.gif"
    animate_stack(plot_mean_profiles_frame, stack=profiles,
                  save_path=gif_path, interval=100)
    assert gif_path.exists()
    plt.close("all")
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /Users/matthew/Desktop/Chemotaxis && python3 -m pytest tests/pipeline/test_visualization.py -v 2>&1 | head -20
```

- [ ] **Step 3: Create `scripts/pipeline/visualization.py`**

Port all 28 functions from cell 8 lines 466–3753. Key changes:

**Top of file — set Agg backend before any matplotlib import:**
```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.collections as mcoll
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
```

**Add `save` + `save_path` parameters to 4 functions** that currently lack them:
- `plot_radial_profile_circle(profile, ..., save=False, save_path=None)`
- `plot_correlation_dynamics(averaged_profiles, ..., save=False, save_path=None)`
- `plot_mean_profiles_frame(ax, frame, stack, ..., save=False, save_path=None)`
- `plot_radial_profile_circle_frame(axs, frame, stack, ..., save=False, save_path=None)`

Each gets this block before `plt.show()` / `return`:
```python
if save:
    save_path = Path(save_path) if save_path else Path("graphs")
    save_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path / f"{title or 'plot'}.png", dpi=150, bbox_inches="tight")
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/matthew/Desktop/Chemotaxis && python3 -m pytest tests/pipeline/test_visualization.py -v
```
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline/visualization.py tests/pipeline/test_visualization.py
git commit -m "feat: add pipeline/visualization.py with all 28 plot functions and animate_stack"
```

---

## Task 7: `nadh.py` + `nadh_matching.py`

**Source:** Cell 99 (~950 lines) → `nadh.py`; `nadh_matching.ipynb` → `nadh_matching.py`

**Files:**
- Create: `scripts/pipeline/nadh.py`
- Create: `scripts/pipeline/nadh_matching.py`
- Create: `tests/pipeline/test_nadh_matching.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/pipeline/test_nadh_matching.py
import sys; sys.path.insert(0, "scripts")
import json
import tempfile
from pathlib import Path
import pytest


def test_match_rois_and_export_basic(tmp_path):
    from pipeline.nadh_matching import match_rois_and_export
    # Create minimal tagged_rois structure
    chem_dir = tmp_path / "tagged_rois" / "img1" / "roi_10_10"
    chem_dir.mkdir(parents=True)
    nadh_dir = tmp_path / "nadh" / "tagged_rois" / "img1" / "roi_11_11"
    nadh_dir.mkdir(parents=True)

    results, out_path = match_rois_and_export(
        chemotaxis_root=str(tmp_path / "tagged_rois"),
        nadh_root=str(tmp_path / "nadh" / "tagged_rois"),
        output_dir=str(tmp_path / "out"),
    )
    assert isinstance(results, list)
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert len(data) == 1
    assert data[0]["image"] == "img1"


def test_average_droplet_intensity_basic():
    from pipeline.nadh_matching import average_droplet_intensity
    results = [
        {"image": "img1", "roi": "roi_0_0", "stack_index": 0, "mean": 1.0},
        {"image": "img1", "roi": "roi_0_0", "stack_index": 0, "mean": 2.0},
    ]
    averaged = average_droplet_intensity(results)
    assert len(averaged) == 1
    assert abs(averaged[0]["mean"] - 1.5) < 1e-6


def test_average_droplet_intensity_discards_negative():
    from pipeline.nadh_matching import average_droplet_intensity
    results = [
        {"image": "img1", "roi": "roi_0_0", "mean": -1.0},
        {"image": "img1", "roi": "roi_0_0", "mean": 2.0},
    ]
    averaged = average_droplet_intensity(results, negative_policy="discard")
    assert averaged[0]["mean"] == 2.0


def test_process_single_nd2_nadh_exists():
    from pipeline.nadh import process_single_nd2_nadh
    import inspect
    sig = inspect.signature(process_single_nd2_nadh)
    assert "nd2_path" in sig.parameters
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /Users/matthew/Desktop/Chemotaxis && python3 -m pytest tests/pipeline/test_nadh_matching.py -v 2>&1 | head -20
```

- [ ] **Step 3: Create `scripts/pipeline/nadh_matching.py`**

Port directly from `nadh_matching.ipynb` cells. Functions: `parse_roi_name`, `euclidean`, `match_rois_and_export`, `average_droplet_intensity`.

```python
# scripts/pipeline/nadh_matching.py
import json
import re
import numpy as np
from collections import defaultdict
from math import sqrt
from pathlib import Path
```

- [ ] **Step 4: Create `scripts/pipeline/nadh.py`**

Port from cell 99. Top of file:
```python
import nd2
import numpy as np
from pathlib import Path
from skimage import filters, measure, morphology
from pipeline.utils import standardize_stack
from pipeline.roi_extraction import (
    optimize_mask_and_regions, find_roi_trajectory, save_tracked_rois
)
```

Add `process_single_nd2_nadh` at the bottom, mirroring `process_single_nd2` from `roi_extraction.py` but selecting the NADH channel:

```python
def process_single_nd2_nadh(nd2_path, save_path=None, channel=None,
                              circ_thresh=0.9, max_distance=40,
                              minimum_mask=None, maximum_mask=None,
                              pad=5, verbose=1):
    """
    Process NADH channel from a single .nd2 file.
    channel=None → auto-detect NADH channel (typically last channel).
    Saves crops to save_path/nadh/tagged_rois/<stem>/.
    """
    if minimum_mask is None:
        minimum_mask = [1, 1]
    if maximum_mask is None:
        maximum_mask = [20, 20]
    nd2_path = Path(nd2_path)
    if save_path is None:
        save_path = nd2_path.parent
    save_path = Path(save_path)

    raw = nd2.imread(str(nd2_path))
    if raw.ndim < 3:
        if verbose:
            print(f"No NADH channel in {nd2_path.name} (2D image)")
        return

    # Auto-detect: NADH is typically the last channel
    nadh_channel = channel if channel is not None else (raw.shape[1] - 1 if raw.ndim == 4 else raw.shape[0] - 1)
    stack = standardize_stack(raw, channel=nadh_channel)

    image_name = nd2_path.stem
    best_params, regions = optimize_mask_and_regions(
        stack[0] if stack.ndim == 3 else stack,
        minimum=minimum_mask, maximum=maximum_mask,
        verbose=verbose, circ_thresh=circ_thresh
    )
    if not regions:
        return

    tracked = find_roi_trajectory(
        stack, best_params, channel=0,
        circ_thresh=circ_thresh, max_distance=max_distance, verbose=verbose
    )
    save_tracked_rois(stack, tracked,
                      save_root=save_path / "nadh" / "tagged_rois" / image_name,
                      image_name=image_name, pad=pad)
```

- [ ] **Step 5: Run tests**

```bash
cd /Users/matthew/Desktop/Chemotaxis && python3 -m pytest tests/pipeline/test_nadh_matching.py -v
```
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add scripts/pipeline/nadh.py scripts/pipeline/nadh_matching.py tests/pipeline/test_nadh_matching.py
git commit -m "feat: add pipeline/nadh.py and pipeline/nadh_matching.py"
```

---

## Task 8: `run_pipeline.py` — batch driver

**Files:**
- Create: `scripts/run_pipeline.py`
- Create: `tests/pipeline/test_run_pipeline.py`

- [ ] **Step 1: Write failing smoke test**

```python
# tests/pipeline/test_run_pipeline.py
import sys; sys.path.insert(0, "scripts")
import subprocess
import pytest


def test_run_pipeline_help():
    """run_pipeline.py --help exits 0 and prints usage."""
    result = subprocess.run(
        ["python3", "scripts/run_pipeline.py", "--help"],
        capture_output=True, text=True,
        cwd="/Users/matthew/Desktop/Chemotaxis"
    )
    assert result.returncode == 0
    assert "--file" in result.stdout or "--file" in result.stderr


def test_run_pipeline_missing_dir(tmp_path):
    """run_pipeline.py exits non-zero when nd2 dir does not exist."""
    result = subprocess.run(
        ["python3", "scripts/run_pipeline.py",
         "--nd2-dir", str(tmp_path / "nonexistent")],
        capture_output=True, text=True,
        cwd="/Users/matthew/Desktop/Chemotaxis"
    )
    assert result.returncode != 0
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /Users/matthew/Desktop/Chemotaxis && python3 -m pytest tests/pipeline/test_run_pipeline.py -v 2>&1 | head -20
```

- [ ] **Step 3: Create `scripts/run_pipeline.py`**

```python
#!/usr/bin/env python3
"""
run_pipeline.py — Batch chemotaxis image analysis pipeline.

Usage:
    python run_pipeline.py
    python run_pipeline.py --file 20250518_Active_in_vitro.nd2
    python run_pipeline.py --nd2-dir chemotaxis_by_date
    python run_pipeline.py --skip-plots
    python run_pipeline.py --skip-nadh
    python run_pipeline.py --skip-cross-condition
"""
import matplotlib
matplotlib.use("Agg")  # must be before any other matplotlib import

import argparse
import json
import sys
import time
import traceback
import numpy as np
from pathlib import Path

# Pipeline modules
from pipeline.roi_extraction import process_single_nd2
from pipeline.analysis import (
    process_roi_library, average_radial_profiles,
    average_cluster_sizes, save_results_to_json
)
from pipeline.nadh import process_single_nd2_nadh
from pipeline.nadh_matching import match_rois_and_export, average_droplet_intensity
from pipeline.visualization import (
    plot_mean_profiles, plot_max_intensities_by_time, plot_starting_profile_versus_max,
    plot_profile_differences, plot_max_velocity, plot_max_scatter_vs_time,
    plot_cluster_size_by_time, plot_cluster_size_boxplot, plot_trace_correlation,
    plot_max_positions_across_conditions, plot_max_scatter_across_conditions,
    plot_edge_intensity_across_conditions, plot_cluster_fold_change_across_conditions,
    plot_cluster_scatter_vs_time, animate_stack, plot_mean_profiles_frame,
    plot_radial_profile_circle_from_profiles, plot_droplet_max_x_dynamics,
)


ROOT = Path(__file__).parent.parent  # Chemotaxis/
DEFAULT_ND2_DIR = ROOT / "chemotaxis_by_date"
DEFAULT_OUT_DIR = ROOT / "processed_data"
UM_PER_PX = 0.31074
TIME_BETWEEN_FRAMES = 0.5  # minutes


def label_from_stem(stem: str) -> str:
    """'20250518_Active_in_vitro' → 'Active in vitro (2025-05-18)'"""
    parts = stem.split("_", 1)
    if len(parts) == 2:
        date_str = f"{parts[0][:4]}-{parts[0][4:6]}-{parts[0][6:]}"
        condition = parts[1].replace("_", " ")
        return f"{condition} ({date_str})"
    return stem


def has_nadh_channel(nd2_path: Path) -> bool:
    try:
        import nd2
        raw = nd2.imread(str(nd2_path))
        return raw.ndim >= 3 and (raw.shape[0] >= 2 or (raw.ndim == 4 and raw.shape[1] >= 2))
    except Exception:
        return False


def process_file(nd2_path: Path, out_dir: Path, skip_plots: bool,
                 skip_nadh: bool) -> dict:
    """Process a single .nd2 file. Returns status dict."""
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "graphs").mkdir(exist_ok=True)
    (out_dir / "animations").mkdir(exist_ok=True)

    # --- ROI extraction ---
    process_single_nd2(nd2_path, save_path=out_dir, verbose=1)
    tagged_rois_dir = out_dir / "tagged_rois"

    # --- Radial profile analysis ---
    results = process_roi_library(str(tagged_rois_dir))
    if not results:
        return {"status": "warned", "reason": "no ROIs found",
                "duration_seconds": time.time() - t0}

    # Warn if >50% of frames have zero ROIs
    frame_roi_counts = {}
    for entry in results:
        frame_roi_counts[entry.get("stack_index", 0)] = \
            frame_roi_counts.get(entry.get("stack_index", 0), 0) + 1
    total_frames = max(frame_roi_counts.keys(), default=0) + 1
    empty_frames = sum(1 for f in range(total_frames) if frame_roi_counts.get(f, 0) == 0)
    warn_flag = empty_frames > total_frames * 0.5

    averaged = average_radial_profiles(results, n_common=100)
    clusters = average_cluster_sizes(results, min_cluster_area=10)

    save_results_to_json(results, "radial_profiles.json",
                         save_path=out_dir, combine_all=True)
    save_results_to_json(averaged, "radial_profiles_averaged.json",
                         save_path=out_dir, combine_all=True)
    save_results_to_json(clusters, "cluster_sizes_averaged.json",
                         save_path=out_dir, combine_all=True)

    n_frames = max((e.get("stack_index", 0) for e in averaged), default=0) + 1
    dirs = np.arange(0, n_frames)
    label = label_from_stem(nd2_path.stem)
    roi_count = len({(e["image"], e["roi"]) for e in results})

    # --- NADH processing ---
    if not skip_nadh and has_nadh_channel(nd2_path):
        process_single_nd2_nadh(nd2_path, save_path=out_dir, verbose=1)
        nadh_results = process_roi_library(str(out_dir / "nadh" / "tagged_rois"))
        if nadh_results:
            avg_nadh = average_droplet_intensity(nadh_results)
            save_results_to_json(avg_nadh, "nadh_intensity.json",
                                  save_path=out_dir, combine_all=True)
            match_rois_and_export(
                chemotaxis_root=str(tagged_rois_dir),
                nadh_root=str(out_dir / "nadh" / "tagged_rois"),
                output_dir=str(out_dir),
            )

    # --- Visualization ---
    if not skip_plots:
        _run_per_file_plots(averaged, clusters, results, dirs, label,
                             out_dir / "graphs", out_dir / "animations")

    try:
        import torch
        device = "mps" if torch.backends.mps.is_available() else \
                 "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    return {
        "status": "warned" if warn_flag else "success",
        "reason": f"{empty_frames}/{total_frames} frames had 0 ROIs" if warn_flag else None,
        "roi_count": roi_count,
        "frames_processed": int(n_frames),
        "duration_seconds": round(time.time() - t0, 1),
        "device_used": device,
    }


def _run_per_file_plots(averaged, clusters, raw_results, dirs, label,
                         graphs_dir, animations_dir):
    import matplotlib.pyplot as plt

    kw = dict(save=True, save_path=graphs_dir)

    try:
        plot_mean_profiles(averaged, dirs=list(dirs[[0, len(dirs)//2, -1]]),
                           labels=[label], title=label, **kw)
    except Exception as e:
        print(f"  [warn] plot_mean_profiles: {e}")

    try:
        plot_max_intensities_by_time(averaged, dirs=dirs,
                                      error="sem", title=label, **kw)
    except Exception as e:
        print(f"  [warn] plot_max_intensities_by_time: {e}")

    try:
        plot_cluster_size_by_time(clusters, scale=3.2181,
                                   title=label, min_cluster_area=5, **kw)
    except Exception as e:
        print(f"  [warn] plot_cluster_size_by_time: {e}")

    try:
        plot_max_velocity(raw_results, dirs=dirs, title=label,
                          um_per_px=UM_PER_PX, time_between_images=TIME_BETWEEN_FRAMES,
                          **kw)
    except Exception as e:
        print(f"  [warn] plot_max_velocity: {e}")

    plt.close("all")


def run_cross_condition_plots(all_averaged: dict, out_dir: Path):
    """Group files by Active/Passive and run multi-condition comparison plots."""
    import matplotlib.pyplot as plt
    graphs_dir = out_dir / "cross_condition_graphs"
    graphs_dir.mkdir(exist_ok=True)
    kw = dict(save=True, save_path=graphs_dir)

    active = [(stem, profiles) for stem, profiles in all_averaged.items() if "Active" in stem]
    passive = [(stem, profiles) for stem, profiles in all_averaged.items() if "Passive" in stem]
    all_groups = active + passive

    if len(all_groups) < 2:
        return

    all_profiles = [g[1] for g in all_groups]
    all_labels = [label_from_stem(g[0]) for g in all_groups]
    n_frames = max(
        (max((e.get("stack_index", 0) for e in p), default=0) + 1 for p in all_profiles),
        default=1
    )
    dirs_list = [np.arange(0, n_frames)] * len(all_groups)

    try:
        plot_max_positions_across_conditions(all_profiles, dirs_list,
                                              condition_labels=all_labels,
                                              min_traces=1, show_points=True, **kw)
    except Exception as e:
        print(f"  [warn] plot_max_positions_across_conditions: {e}")

    try:
        plot_max_scatter_across_conditions(all_profiles, dirs_list,
                                            condition_labels=all_labels,
                                            min_traces=1, **kw)
    except Exception as e:
        print(f"  [warn] plot_max_scatter_across_conditions: {e}")

    try:
        plot_edge_intensity_across_conditions(all_profiles, dirs_list,
                                              condition_labels=all_labels,
                                              min_traces=1, **kw)
    except Exception as e:
        print(f"  [warn] plot_edge_intensity_across_conditions: {e}")

    try:
        # Load cluster data for the remaining two plots
        all_cluster_data = []
        for stem in all_averaged:
            cluster_json = out_dir / stem / "cluster_sizes_averaged.json"
            if cluster_json.exists():
                import json as _j
                all_cluster_data.append(_j.loads(cluster_json.read_text()))
            else:
                all_cluster_data.append([])
        plot_cluster_fold_change_across_conditions(
            all_cluster_data, dirs_list,
            condition_labels=all_labels, min_traces=1, **kw)
    except Exception as e:
        print(f"  [warn] plot_cluster_fold_change_across_conditions: {e}")

    try:
        plot_cluster_scatter_vs_time(
            all_cluster_data, dirs_list,
            condition_labels=all_labels, min_traces=1,
            scale=3.2181 * 5, **kw)
    except Exception as e:
        print(f"  [warn] plot_cluster_scatter_vs_time: {e}")

    plt.close("all")


def main():
    parser = argparse.ArgumentParser(description="Chemotaxis image analysis pipeline")
    parser.add_argument("--nd2-dir", default=str(DEFAULT_ND2_DIR),
                        help="Directory containing .nd2 files")
    parser.add_argument("--file", default=None,
                        help="Process a single .nd2 file by name")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                        help="Root output directory")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-nadh", action="store_true")
    parser.add_argument("--skip-cross-condition", action="store_true")
    args = parser.parse_args()

    nd2_dir = Path(args.nd2_dir)
    if not nd2_dir.exists():
        print(f"ERROR: nd2 directory not found: {nd2_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)

    if args.file:
        nd2_files = [nd2_dir / args.file]
    else:
        nd2_files = sorted(nd2_dir.glob("*.nd2"))

    if not nd2_files:
        print(f"No .nd2 files found in {nd2_dir}", file=sys.stderr)
        sys.exit(1)

    summary = {}
    all_averaged = {}

    for nd2_path in nd2_files:
        print(f"\n{'='*60}")
        print(f"Processing: {nd2_path.name}")
        file_out = out_dir / nd2_path.stem
        try:
            status = process_file(nd2_path, file_out,
                                   skip_plots=args.skip_plots,
                                   skip_nadh=args.skip_nadh)
            summary[nd2_path.stem] = status
            # Load averaged profiles for cross-condition plots
            avg_json = file_out / "radial_profiles_averaged.json"
            if avg_json.exists():
                import json as _json
                all_averaged[nd2_path.stem] = _json.loads(avg_json.read_text())
        except Exception:
            print(f"  [ERROR] {nd2_path.name} failed:")
            traceback.print_exc()
            summary[nd2_path.stem] = {"status": "failed", "error": traceback.format_exc()}

    # Cross-condition aggregation
    if not args.skip_cross_condition and len(all_averaged) > 1:
        print(f"\n{'='*60}")
        print("Running cross-condition plots...")
        try:
            run_cross_condition_plots(all_averaged, out_dir)
        except Exception:
            print("  [warn] Cross-condition plots failed:")
            traceback.print_exc()

    # Write summary
    summary_path = out_dir / "pipeline_run_summary.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    import json as _json
    summary_path.write_text(_json.dumps(summary, indent=2))
    print(f"\nSummary written to {summary_path}")

    n_ok = sum(1 for v in summary.values() if v.get("status") == "success")
    print(f"Done: {n_ok}/{len(summary)} files succeeded.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run smoke tests**

```bash
cd /Users/matthew/Desktop/Chemotaxis && python3 -m pytest tests/pipeline/test_run_pipeline.py -v
```
Expected: both pass

- [ ] **Step 5: Commit**

```bash
git add scripts/run_pipeline.py tests/pipeline/test_run_pipeline.py
git commit -m "feat: add run_pipeline.py batch driver with CLI and cross-condition aggregation"
```

---

## Task 9: End-to-end run on one `.nd2` file

- [ ] **Step 1: Run pipeline on a single file**

```bash
cd /Users/matthew/Desktop/Chemotaxis && python3 scripts/run_pipeline.py \
  --file 20250716_Active_in_vitro.nd2 \
  --skip-cross-condition \
  2>&1 | tee /tmp/pipeline_run.log
```

- [ ] **Step 2: Verify outputs exist**

```bash
ls processed_data/20250716_Active_in_vitro/
# Expected: tagged_rois/  radial_profiles.json  radial_profiles_averaged.json
#           cluster_sizes_averaged.json  graphs/  animations/

ls processed_data/20250716_Active_in_vitro/graphs/ | head -10
```

- [ ] **Step 3: Check summary**

```bash
cat processed_data/pipeline_run_summary.json
# Expected: {"20250716_Active_in_vitro": {"status": "success", "roi_count": N, ...}}
```

- [ ] **Step 4: Run on all 14 files**

```bash
cd /Users/matthew/Desktop/Chemotaxis && python3 scripts/run_pipeline.py 2>&1 | tee /tmp/pipeline_full_run.log
```

- [ ] **Step 5: Check final summary**

```bash
python3 -c "
import json
s = json.load(open('processed_data/pipeline_run_summary.json'))
for k, v in s.items():
    print(f'{k}: {v[\"status\"]} | ROIs: {v.get(\"roi_count\",\"?\")}'  \
          f' | {v.get(\"duration_seconds\",\"?\")}s | {v.get(\"device_used\",\"?\")}')
"
```

- [ ] **Step 6: Commit**

```bash
git add processed_data/pipeline_run_summary.json
git commit -m "data: run full pipeline on chemotaxis_by_date (14 files)"
```

---

## Run All Tests

```bash
cd /Users/matthew/Desktop/Chemotaxis && python3 -m pytest tests/ -v --tb=short
```
