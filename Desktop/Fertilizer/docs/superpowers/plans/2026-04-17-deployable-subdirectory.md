# Deployable Subdirectory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `deployable/` containing a generalizable, config-driven version of the 2D time-lapse and 3D z-stack image analysis pipelines for collaborator use.

**Architecture:** Shared core in `ml_image_analysis.py` (preprocessing, segmentation, tracking, TIFF I/O); `analyze_intensities.py` for 2D temporal analysis; `analyze_zstack_intensities.py` for 3D leakage metrics. All project-specific hardcoded values (channel names, indices, paths) replaced with `config.yaml` lookups.

**Tech Stack:** Python 3.10+, cellpose, scikit-image, scipy, numpy, tifffile, pyyaml, pandas, matplotlib, seaborn, tqdm, jupyter

---

## File Map

| File | Action | Key change |
|---|---|---|
| `deployable/config.yaml` | Create | Full config with NO₃⁻ project as commented example |
| `deployable/requirements.txt` | Create | All pip deps |
| `deployable/.gitignore` | Create | Ignore output/, __pycache__, .ipynb_checkpoints |
| `deployable/ml_image_analysis.py` | Copy + adapt | Add `load_config()`; `seg_channel` param in tracker; generic channel names in intensity extractor |
| `deployable/analyze_intensities.py` | Copy + adapt | `time_interval_minutes` param; generic signal/reference column names |
| `deployable/analyze_zstack_intensities.py` | Copy + adapt | Channel indices from config, not hardcoded |
| `deployable/prepare_training_images.py` | Copy + adapt | Remove hardcoded `CPSAM_MODEL_PATH` and `FALLBACK_PYTHON` |
| `deployable/validate_tracking.py` | Copy | No logic changes needed |
| `deployable/notebooks/2D_timelapse_analysis.ipynb` | Create | Clean, unrun |
| `deployable/notebooks/zstack_analysis.ipynb` | Create | Clean, unrun |
| `deployable/README.md` | Create | Install, config, usage, human-in-the-loop section |
| `deployable/output/processed/.gitkeep` | Create | Placeholder dir |
| `deployable/output/qc_plots/.gitkeep` | Create | Placeholder dir |

---

## Task 1: Scaffold + config.yaml + requirements.txt

**Files:**
- Create: `deployable/config.yaml`
- Create: `deployable/requirements.txt`
- Create: `deployable/.gitignore`
- Create: `deployable/output/processed/.gitkeep`
- Create: `deployable/output/qc_plots/.gitkeep`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p deployable/notebooks deployable/output/processed deployable/output/qc_plots
touch deployable/output/processed/.gitkeep deployable/output/qc_plots/.gitkeep
```

- [ ] **Step 2: Write `deployable/config.yaml`**

```yaml
# ── Channels ────────────────────────────────────────────────────────────────
# Signal channel = the channel used for ROI segmentation AND as the primary
# fluorescence reporter. Reference channel = ratiometric normalization partner.
# Example below: NO3-sensor project (GFP signal, OA-647 reference).
channels:
  signal: 0            # channel index for primary signal (e.g. GFP)
  reference: 1         # channel index for reference / normalization (e.g. OA-647)
  signal_name: "GFP"   # human-readable name — used as column header in output JSON
  reference_name: "OA-647"

# ── Segmentation ────────────────────────────────────────────────────────────
segmentation:
  cellpose_model: "cyto3"   # model name or absolute path to a custom .pt file
  diameter_um: null         # expected object diameter in µm; null = auto from metadata
  min_diameter_um: 5.0      # objects smaller than this (µm) are rejected
  max_diameter_um: 50.0     # objects larger than this (µm) are rejected
  min_circularity: 0.7      # 4πA/P²; 1.0 = perfect circle
  flow_threshold: 0.4       # Cellpose: higher = accept more masks
  cellprob_threshold: 0.0   # Cellpose: lower = detect dimmer objects
  border_margin: 0          # exclude ROIs within N px of any image edge

# ── Preprocessing ────────────────────────────────────────────────────────────
# All preprocessing steps apply only to the detection copy sent to Cellpose.
# Raw pixel values used for intensity measurements are NEVER modified.
preprocessing:
  rolling_ball: true
  rolling_ball_radius_px: null   # null = auto (max_diameter_um / pixel_size_um)
  dog: true                      # Difference-of-Gaussians band-pass filter
  dog_sigma_low: 1.0             # narrow Gaussian sigma (noise suppression), px
  dog_sigma_high: null           # null = auto (~half expected object radius)
  clahe: true                    # CLAHE contrast enhancement
  clahe_clip_limit: 0.01         # 0–1; higher = more contrast, more noise

# ── Pixel size ───────────────────────────────────────────────────────────────
pixel_size:
  um_per_px: null   # null = read from TIFF metadata (OME-TIFF / ImageJ / XRes tag)

# ── Tracking (2D time-lapse only) ────────────────────────────────────────────
tracking:
  max_displacement_px: 20    # max centroid shift between frames for a valid match
  frame_interval_minutes: 5.0
  max_gap_minutes: 10.0      # ROI missing longer than this is permanently terminated
  min_track_length: 3        # discard tracks shorter than N timepoints

# ── Output ───────────────────────────────────────────────────────────────────
output:
  processed_dir: "output/processed"
  qc_plots_dir:  "output/qc_plots"
```

- [ ] **Step 3: Write `deployable/requirements.txt`**

```
cellpose>=3.0
scikit-image>=0.21
scipy>=1.11
numpy>=1.24
tifffile>=2023.1
pyyaml>=6.0
pandas>=2.0
matplotlib>=3.7
seaborn>=0.13
tqdm>=4.66
jupyter>=1.0
ipykernel>=6.0
```

- [ ] **Step 4: Write `deployable/.gitignore`**

```
# Pipeline output
output/processed/*
!output/processed/.gitkeep
output/qc_plots/*
!output/qc_plots/.gitkeep

# Python
__pycache__/
*.pyc
*.pyo
.ipynb_checkpoints/

# OS
.DS_Store
```

- [ ] **Step 5: Commit scaffold**

```bash
cd /Users/matthew/Desktop/Fertilizer
git add deployable/
git commit -m "feat: scaffold deployable/ directory with config.yaml and requirements.txt"
```

---

## Task 2: Adapt `ml_image_analysis.py`

**Files:**
- Create: `deployable/ml_image_analysis.py` (copy + adapt from `ml_image_analysis.py`)

Three targeted changes to the copy:
1. Add `load_config()` at the top of the file.
2. Add `seg_channel` parameter to `track_rois_across_frames` (replaces hardcoded `stack[t, 1, :]`).
3. Add `channel_names` parameter to `extract_multichannel_intensities` (replaces hardcoded `gfp_mean`/`oa647_mean`).
4. Add `seg_channel` and `channel_names` parameters to `process_tiff_directory` and `test_segmentation_on_first_frame`.
5. Make `visualize_segmentation_qc` accept generic `channel_names`.

- [ ] **Step 1: Copy the file**

```bash
cp /Users/matthew/Desktop/Fertilizer/ml_image_analysis.py \
   /Users/matthew/Desktop/Fertilizer/deployable/ml_image_analysis.py
```

- [ ] **Step 2: Write failing test for `load_config`**

Create `deployable/tests/test_ml_image_analysis.py`:

```python
import pytest
import yaml
from pathlib import Path

def test_load_config_reads_yaml(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "channels:\n  signal: 0\n  reference: 1\n"
        "  signal_name: TestSignal\n  reference_name: TestRef\n"
    )
    import sys
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from ml_image_analysis import load_config
    cfg = load_config(cfg_path)
    assert cfg["channels"]["signal"] == 0
    assert cfg["channels"]["signal_name"] == "TestSignal"


def test_load_config_missing_file_raises():
    from ml_image_analysis import load_config
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.yaml")
```

- [ ] **Step 3: Run test — expect FAIL**

```bash
cd /Users/matthew/Desktop/Fertilizer
python -m pytest deployable/tests/test_ml_image_analysis.py -v
```

Expected: `ImportError` or `AttributeError` — `load_config` not defined yet.

- [ ] **Step 4: Add `load_config` to `deployable/ml_image_analysis.py`**

Insert after the existing imports block (after the `warnings` import, before `# Cellpose imports`):

```python
import yaml


def load_config(config_path="config.yaml"):
    """
    Load pipeline configuration from a YAML file.

    Parameters
    ----------
    config_path : str or Path
        Path to config.yaml.

    Returns
    -------
    dict
        Parsed configuration.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
```

- [ ] **Step 5: Run test — expect PASS**

```bash
python -m pytest deployable/tests/test_ml_image_analysis.py::test_load_config_reads_yaml \
                 deployable/tests/test_ml_image_analysis.py::test_load_config_missing_file_raises -v
```

Expected: both PASS.

- [ ] **Step 6: Write failing test for `seg_channel` param in tracker**

Add to `deployable/tests/test_ml_image_analysis.py`:

```python
import numpy as np

def test_track_rois_uses_seg_channel():
    """track_rois_across_frames should index stack with seg_channel, not hardcoded 1."""
    import sys
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from ml_image_analysis import track_rois_across_frames

    # 2 timepoints, 2 channels, 32×32 image
    # channel 0 is bright (signal), channel 1 is dark (reference)
    stack = np.zeros((2, 2, 32, 32), dtype=np.float32)
    stack[:, 0, 12:18, 12:18] = 1000.0   # bright blob in channel 0

    # Provide an initial_roi at the blob location
    mask = np.zeros((32, 32), dtype=bool)
    mask[12:18, 12:18] = True
    initial_rois = [{
        'roi_id': 0,
        'centroid': (15.0, 15.0),
        'area': 36,
        'circularity': 0.9,
        'mask': mask,
    }]

    # A segmenter that always returns the same mask (no Cellpose needed)
    from skimage import measure
    def dummy_segmenter(frame):
        labeled = mask.astype(np.int32)
        m = [{'segmentation': mask, 'area': 36, 'bbox': (12, 12, 18, 18),
              'circularity': 0.9, 'diameter_um': None,
              'predicted_iou': 1.0, 'stability_score': 1.0}]
        return m, labeled, None

    # seg_channel=0: segmenter should be called with channel 0 slice
    result = track_rois_across_frames(
        stack, initial_rois,
        max_distance=20,
        segmenter=dummy_segmenter,
        seg_channel=0,
    )
    # If seg_channel was ignored (hardcoded to 1), the call would use the dark
    # channel; the test passes as long as no exception is raised and tracks exist.
    assert len(result) >= 1
```

- [ ] **Step 7: Run test — verify current behavior (no seg_channel param yet)**

```bash
python -m pytest deployable/tests/test_ml_image_analysis.py::test_track_rois_uses_seg_channel -v
```

Expected: `TypeError` — `track_rois_across_frames` does not accept `seg_channel`.

- [ ] **Step 8: Add `seg_channel` param to `track_rois_across_frames` in `deployable/ml_image_analysis.py`**

Find the function signature (currently around line 1168):
```python
def track_rois_across_frames(stack, initial_rois,
                             max_distance=20, similarity_weight=0.3,
                             segmenter=None,
                             frame_interval_minutes=5.0,
                             max_gap_minutes=10.0,
                             velocity_history=3,
                             velocity_weight=0.2):
```

Replace with:
```python
def track_rois_across_frames(stack, initial_rois,
                             max_distance=20, similarity_weight=0.3,
                             segmenter=None,
                             frame_interval_minutes=5.0,
                             max_gap_minutes=10.0,
                             velocity_history=3,
                             velocity_weight=0.2,
                             seg_channel=1):
```

Then find the hardcoded line inside the function body (currently `frame = stack[t, 1, :, :]`):
```python
        # Use OA-647 channel (index 1) for segmentation
        frame = stack[t, 1, :, :]
```

Replace with:
```python
        frame = stack[t, seg_channel, :, :]
```

- [ ] **Step 9: Run test — expect PASS**

```bash
python -m pytest deployable/tests/test_ml_image_analysis.py::test_track_rois_uses_seg_channel -v
```

Expected: PASS.

- [ ] **Step 10: Write failing test for generic channel names in intensity extractor**

Add to `deployable/tests/test_ml_image_analysis.py`:

```python
def test_extract_multichannel_intensities_generic_names():
    from ml_image_analysis import extract_multichannel_intensities

    stack = np.zeros((1, 2, 10, 10), dtype=np.float32)
    stack[0, 0, 3:6, 3:6] = 500.0
    stack[0, 1, 3:6, 3:6] = 200.0

    mask = np.zeros((10, 10), dtype=bool)
    mask[3:6, 3:6] = True

    tracked = [{
        'roi_id': 0,
        'timepoints': [0],
        'centroids': [(4.5, 4.5)],
        'masks': [mask],
        'areas': [9],
        'circularities': [0.9],
    }]

    data = extract_multichannel_intensities(
        stack, tracked, channel_names=["signal_mean", "ref_mean"]
    )
    assert len(data) == 1
    assert "signal_mean" in data[0]
    assert "ref_mean" in data[0]
    assert abs(data[0]["signal_mean"] - 500.0) < 1.0
    assert abs(data[0]["ref_mean"] - 200.0) < 1.0
    # Old hardcoded names must NOT appear
    assert "gfp_mean" not in data[0]
    assert "oa647_mean" not in data[0]
```

- [ ] **Step 11: Run test — expect FAIL**

```bash
python -m pytest deployable/tests/test_ml_image_analysis.py::test_extract_multichannel_intensities_generic_names -v
```

Expected: `TypeError` — no `channel_names` param yet.

- [ ] **Step 12: Adapt `extract_multichannel_intensities` in `deployable/ml_image_analysis.py`**

Find the function signature:
```python
def extract_multichannel_intensities(stack, tracked_rois):
```

Replace with:
```python
def extract_multichannel_intensities(stack, tracked_rois, channel_names=None):
    """
    ...existing docstring...

    channel_names : list of str or None
        Column names for each channel in the output dicts.
        e.g. ["gfp_mean", "oa647_mean"] or ["signal_mean", "ref_mean"].
        Defaults to ["ch0_mean", "ch1_mean", ...] when None.
    """
    n_channels = stack.shape[1]
    if channel_names is None:
        channel_names = [f"ch{i}_mean" for i in range(n_channels)]
    if len(channel_names) != n_channels:
        raise ValueError(
            f"channel_names length ({len(channel_names)}) must match "
            f"stack channels ({n_channels})"
        )
```

Then replace the intensity extraction body (the lines that extract `gfp_image`, `oa647_image`, `gfp_mean`, `oa647_mean` and build the dict):

Old code:
```python
            # Extract intensities from both channels
            gfp_image = stack[t, 0, :, :]
            oa647_image = stack[t, 1, :, :]
            
            gfp_mean = np.mean(gfp_image[mask])
            oa647_mean = np.mean(oa647_image[mask])
            
            intensity_data.append({
                'roi_id': roi_id,
                'timepoint': t,
                'gfp_mean': float(gfp_mean),
                'oa647_mean': float(oa647_mean),
                'centroid': tuple(float(c) for c in centroid),
                'area': int(area)
            })
```

New code:
```python
            entry = {
                'roi_id': roi_id,
                'timepoint': t,
                'centroid': tuple(float(c) for c in centroid),
                'area': int(area),
            }
            for ch_idx, ch_name in enumerate(channel_names):
                entry[ch_name] = float(np.mean(stack[t, ch_idx, :, :][mask]))
            intensity_data.append(entry)
```

- [ ] **Step 13: Run test — expect PASS**

```bash
python -m pytest deployable/tests/test_ml_image_analysis.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 14: Add `seg_channel` and `channel_names` to `process_tiff_directory`**

In `deployable/ml_image_analysis.py`, find `process_tiff_directory` signature and add two parameters at the end:

```python
def process_tiff_directory(...,
                           dog_sigma_high=None,
                           seg_channel=1,
                           channel_names=None):
```

Inside the function, replace the two hardcoded lines:
```python
            first_frame_oa647 = stack[0, 1, :, :]
            first_frame_gfp   = stack[0, 0, :, :]
```

With:
```python
            first_frame_seg   = stack[0, seg_channel, :, :]
            other_ch = 1 - seg_channel if stack.shape[1] == 2 else 0
            first_frame_other = stack[0, other_ch, :, :]
```

Replace call to `segment_rois_with_cellpose` — change `first_frame_oa647` → `first_frame_seg`.

Replace call to `visualize_segmentation_qc`:
```python
            visualize_segmentation_qc(
                first_frame_other, first_frame_seg, initial_rois,
                save_path=qc_save_path,
                channel_names=channel_names,
            )
```

Replace call to `track_rois_across_frames` — add `seg_channel=seg_channel`.

Replace call to `extract_multichannel_intensities`:
```python
            intensity_data = extract_multichannel_intensities(
                stack, tracked_rois, channel_names=channel_names
            )
```

Also find `test_segmentation_on_first_frame` and replace hardcoded `stack[0, 1, :, :]` / `stack[0, 0, :, :]`:

Old:
```python
    first_frame_oa647 = stack[0, 1, :, :]
    first_frame_gfp = stack[0, 0, :, :]
```

New (add `seg_channel=1` to the function signature):
```python
def test_segmentation_on_first_frame(tiff_path, ..., seg_channel=1):
    ...
    first_frame_seg   = stack[0, seg_channel, :, :]
    other_ch = 1 - seg_channel if stack.shape[1] == 2 else 0
    first_frame_other = stack[0, other_ch, :, :]
```

Update the print statements in `test_segmentation_on_first_frame` to use generic names (replace "OA-647" and "GFP" labels with channel indices).

- [ ] **Step 15: Update `visualize_segmentation_qc` to accept `channel_names`**

Find the function signature:
```python
def visualize_segmentation_qc(image_gfp, image_oa647, rois, save_path=None):
```

Replace with:
```python
def visualize_segmentation_qc(image_ch0, image_ch1, rois, save_path=None,
                               channel_names=None):
    if channel_names is None or len(channel_names) < 2:
        ch0_label = "Channel 0"
        ch1_label = "Channel 1"
    else:
        ch0_label = channel_names[0].replace("_mean", "")
        ch1_label = channel_names[1].replace("_mean", "")
```

Replace all uses of `image_gfp` → `image_ch0` and `image_oa647` → `image_ch1` inside the function body.

Replace hardcoded titles:
```python
    axes[0].set_title('GFP Channel', fontsize=14)
    ...
    axes[1].set_title(f'OA-647 with {len(rois)} ROIs', fontsize=14)
```

With:
```python
    axes[0].set_title(ch0_label, fontsize=14)
    ...
    axes[1].set_title(f'{ch1_label} with {len(rois)} ROIs', fontsize=14)
```

- [ ] **Step 16: Run all tests**

```bash
python -m pytest deployable/tests/ -v
```

Expected: all PASS.

- [ ] **Step 17: Commit**

```bash
git add deployable/ml_image_analysis.py deployable/tests/
git commit -m "feat: add deployable ml_image_analysis.py with load_config and generic channel params"
```

---

## Task 3: Adapt `analyze_intensities.py`

**Files:**
- Create: `deployable/analyze_intensities.py` (copy + adapt)

Two changes: (1) `time_interval_minutes` parameter replaces hardcoded `timepoint * 5.0`; (2) ratio computation uses configurable signal/reference column names.

- [ ] **Step 1: Copy file**

```bash
cp /Users/matthew/Desktop/Fertilizer/analyze_intensities.py \
   /Users/matthew/Desktop/Fertilizer/deployable/analyze_intensities.py
```

- [ ] **Step 2: Write failing test**

Create `deployable/tests/test_analyze_intensities.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd
from analyze_intensities import organize_data_by_condition, compute_condition_statistics


def _make_fake_json(tmp_path, condition, n_timepoints=3, signal_col="signal_mean",
                    ref_col="ref_mean"):
    import json
    data = []
    for t in range(n_timepoints):
        data.append({
            "roi_id": 0,
            "timepoint": t,
            signal_col: float(100 + t * 10),
            ref_col: float(50.0),
            "centroid": [5.0, 5.0],
            "area": 36,
        })
    payload = {"metadata": {"filename": "test.tif"}, "data": data}
    p = tmp_path / f"{condition}.json"
    p.write_text(json.dumps(payload))
    return str(p)


def test_organize_data_custom_time_interval(tmp_path):
    f = _make_fake_json(tmp_path, "condA")
    data, unit = organize_data_by_condition(
        [f], ["condA"],
        time_interval_minutes=2.0,
        signal_col="signal_mean",
        reference_col="ref_mean",
    )
    df = data["condA"][0]
    assert list(df["time_min"]) == [0.0, 2.0, 4.0]
    assert unit == "minutes"


def test_compute_condition_statistics_generic_ratio(tmp_path):
    f = _make_fake_json(tmp_path, "condA")
    data, _ = organize_data_by_condition(
        [f], ["condA"],
        time_interval_minutes=5.0,
        signal_col="signal_mean",
        reference_col="ref_mean",
    )
    stats = compute_condition_statistics(
        data, channel="ratio",
        signal_col="signal_mean",
        reference_col="ref_mean",
    )
    assert not stats.empty
    # ratio at t=0: 100/50 = 2.0
    row = stats[stats["time_min"] == 0.0].iloc[0]
    assert abs(row["mean"] - 2.0) < 0.01
```

- [ ] **Step 3: Run test — expect FAIL**

```bash
python -m pytest deployable/tests/test_analyze_intensities.py -v
```

Expected: `TypeError` — `organize_data_by_condition` does not accept `time_interval_minutes`.

- [ ] **Step 4: Adapt `organize_data_by_condition` in `deployable/analyze_intensities.py`**

Find the function signature:
```python
def organize_data_by_condition(json_files, conditions):
```

Replace with:
```python
def organize_data_by_condition(json_files, conditions,
                                time_interval_minutes=5.0,
                                signal_col="gfp_mean",
                                reference_col="oa647_mean"):
```

Inside the function, find:
```python
            df['time_min'] = df['timepoint'] * 5.0
```

Replace with:
```python
            df['time_min'] = df['timepoint'] * time_interval_minutes
```

- [ ] **Step 5: Adapt `compute_condition_statistics` in `deployable/analyze_intensities.py`**

Find the function signature:
```python
def compute_condition_statistics(organized_data, channel='gfp_mean'):
```

Replace with:
```python
def compute_condition_statistics(organized_data, channel='signal_mean',
                                  signal_col='gfp_mean',
                                  reference_col='oa647_mean'):
```

Inside the function, find the ratio computation:
```python
                    if channel == 'ratio':
                        # Compute GFP/OA647 ratio
                        gfp_vals = time_data['gfp_mean'].values
                        oa647_vals = time_data['oa647_mean'].values
                        # Avoid division by zero
                        ratio_vals = np.where(oa647_vals > 0, gfp_vals / oa647_vals, np.nan)
                        values.extend(ratio_vals[~np.isnan(ratio_vals)])
```

Replace with:
```python
                    if channel == 'ratio':
                        sig_vals = time_data[signal_col].values
                        ref_vals = time_data[reference_col].values
                        ratio_vals = np.where(ref_vals > 0, sig_vals / ref_vals, np.nan)
                        values.extend(ratio_vals[~np.isnan(ratio_vals)])
```

- [ ] **Step 6: Run tests — expect PASS**

```bash
python -m pytest deployable/tests/test_analyze_intensities.py -v
```

Expected: both tests PASS.

- [ ] **Step 7: Commit**

```bash
git add deployable/analyze_intensities.py deployable/tests/test_analyze_intensities.py
git commit -m "feat: add deployable analyze_intensities.py with configurable time interval and channel names"
```

---

## Task 4: Adapt `analyze_zstack_intensities.py`

**Files:**
- Create: `deployable/analyze_zstack_intensities.py` (copy + adapt)

Replace hardcoded `channel_gfp=0`, `channel_oa647=1` defaults with parameters that `load_config()` can populate.

- [ ] **Step 1: Copy file**

```bash
cp /Users/matthew/Desktop/Fertilizer/analyze_zstack_intensities.py \
   /Users/matthew/Desktop/Fertilizer/deployable/analyze_zstack_intensities.py
```

- [ ] **Step 2: Write failing test**

Create `deployable/tests/test_analyze_zstack.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np


def test_load_zstack_tiff_channel_params(tmp_path):
    """load_zstack_tiff should use caller-supplied channel indices."""
    import tifffile
    from analyze_zstack_intensities import load_zstack_tiff

    # Write a minimal (2z, 2c, 8, 8) TIFF
    data = np.zeros((2, 2, 8, 8), dtype=np.uint16)
    data[:, 0, :, :] = 100   # channel 0 = signal
    data[:, 1, :, :] = 200   # channel 1 = reference
    tif_path = tmp_path / "test.tif"
    tifffile.imwrite(str(tif_path), data, imagej=True)

    zstack, meta = load_zstack_tiff(str(tif_path), n_channels=2,
                                    channel_signal=0, channel_reference=1)
    assert zstack.shape == (2, 2, 8, 8)
    assert meta["channel_signal"] == 0
    assert meta["channel_reference"] == 1
```

- [ ] **Step 3: Run test — expect FAIL**

```bash
python -m pytest deployable/tests/test_analyze_zstack.py -v
```

Expected: `TypeError` — `load_zstack_tiff` does not accept `channel_signal`/`channel_reference`.

- [ ] **Step 4: Adapt `load_zstack_tiff` signature in `deployable/analyze_zstack_intensities.py`**

Find:
```python
def load_zstack_tiff(file_path, n_channels=2, channel_gfp=0, channel_oa647=1):
```

Replace with:
```python
def load_zstack_tiff(file_path, n_channels=2,
                     channel_signal=0, channel_reference=1,
                     channel_gfp=None, channel_oa647=None):
    # Backward-compat aliases
    if channel_gfp is not None:
        channel_signal = channel_gfp
    if channel_oa647 is not None:
        channel_reference = channel_oa647
```

Inside `load_zstack_tiff`, find the metadata dict construction and add channel info:
```python
    metadata = {
        ...existing keys...,
        "channel_signal": channel_signal,
        "channel_reference": channel_reference,
    }
```

Find all other functions in the file that use hardcoded `channel_gfp=0` / `channel_oa647=1` as defaults (e.g., `analyze_zstack_batch`, `compute_vesicle_metrics_zstack`) and add `channel_signal=0, channel_reference=1` params in the same pattern.

Find every internal use of `channel_gfp` or `channel_oa647` inside these functions and replace with `channel_signal` / `channel_reference`.

- [ ] **Step 5: Run test — expect PASS**

```bash
python -m pytest deployable/tests/test_analyze_zstack.py -v
```

Expected: PASS.

- [ ] **Step 6: Run all tests**

```bash
python -m pytest deployable/tests/ -v
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add deployable/analyze_zstack_intensities.py deployable/tests/test_analyze_zstack.py
git commit -m "feat: add deployable analyze_zstack_intensities.py with configurable channel indices"
```

---

## Task 5: Adapt `prepare_training_images.py`

**Files:**
- Create: `deployable/prepare_training_images.py` (copy + adapt)

Remove two hardcoded constants and replace with CLI flags or `sys.executable`.

- [ ] **Step 1: Copy file**

```bash
cp /Users/matthew/Desktop/Fertilizer/prepare_training_images.py \
   /Users/matthew/Desktop/Fertilizer/deployable/prepare_training_images.py
```

- [ ] **Step 2: Remove hardcoded constants**

In `deployable/prepare_training_images.py`, find and delete:

```python
CPSAM_MODEL_PATH = "/Users/matthew/.cellpose/models/cpsam"
FALLBACK_PYTHON   = "/Users/matthew/miniforge3/envs/membrane-image/bin/python"
```

- [ ] **Step 3: Add `--model-path` CLI argument**

In the `argparse` section, add a new argument after `--no-launch`:

```python
parser.add_argument(
    "--model-path",
    default=None,
    help=(
        "Path to a Cellpose model file or built-in model name "
        "(e.g. 'cyto3', 'cpsam'). Defaults to Cellpose's built-in default."
    ),
)
```

- [ ] **Step 4: Replace the environment-detection block**

Find the environment-detection block that uses `FALLBACK_PYTHON`:

```python
match = re.search(r"envs/([^/]+)/", sys.executable)
if match:
    python_bin = sys.executable
else:
    python_bin = "/Users/matthew/miniforge3/envs/membrane-image/bin/python"
```

Replace with:

```python
python_bin = sys.executable
```

- [ ] **Step 5: Replace the Cellpose launch command**

Find the subprocess launch that uses `CPSAM_MODEL_PATH`:

```python
cmd = [
    python_bin, "-m", "cellpose",
    "--image_path", str(output_dir),
    "--pretrained_model", CPSAM_MODEL_PATH,
]
```

Replace with:

```python
cmd = [python_bin, "-m", "cellpose", "--image_path", str(output_dir)]
if args.model_path:
    cmd += ["--pretrained_model", args.model_path]
```

- [ ] **Step 6: Verify script runs with `--help`**

```bash
cd /Users/matthew/Desktop/Fertilizer/deployable
python prepare_training_images.py --help
```

Expected: usage printed without errors; `--model-path` listed as optional argument.

- [ ] **Step 7: Commit**

```bash
git add deployable/prepare_training_images.py
git commit -m "feat: add deployable prepare_training_images.py; remove hardcoded paths"
```

---

## Task 6: Copy `validate_tracking.py`

**Files:**
- Create: `deployable/validate_tracking.py` (copy, no logic changes)

- [ ] **Step 1: Copy file**

```bash
cp /Users/matthew/Desktop/Fertilizer/validate_tracking.py \
   /Users/matthew/Desktop/Fertilizer/deployable/validate_tracking.py
```

- [ ] **Step 2: Verify import**

```bash
cd /Users/matthew/Desktop/Fertilizer/deployable
python -c "import validate_tracking; print('OK')"
```

Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add deployable/validate_tracking.py
git commit -m "feat: add deployable validate_tracking.py"
```

---

## Task 7: Create Jupyter Notebooks

**Files:**
- Create: `deployable/notebooks/2D_timelapse_analysis.ipynb`
- Create: `deployable/notebooks/zstack_analysis.ipynb`

Both are clean (no cell outputs). They demonstrate the full workflow so collaborators can run cell-by-cell and verify the pipeline on their machine.

- [ ] **Step 1: Create `deployable/notebooks/2D_timelapse_analysis.ipynb`**

```bash
cat > /Users/matthew/Desktop/Fertilizer/deployable/notebooks/2D_timelapse_analysis.ipynb << 'NBEOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# 2D Time-Lapse Analysis\n\nThis notebook walks through the full 2D time-lapse pipeline:\n1. Load `config.yaml`\n2. Segment ROIs in the first frame\n3. Track ROIs across all timepoints\n4. Extract multi-channel intensities\n5. Plot intensity traces\n\n**Before running:** edit `../config.yaml` to match your image channels and bead sizes."]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys, functools\n",
    "from pathlib import Path\n",
    "sys.path.insert(0, str(Path('../').resolve()))\n",
    "\n",
    "from ml_image_analysis import (\n",
    "    load_config,\n",
    "    load_tiff_stack,\n",
    "    segment_rois_with_cellpose,\n",
    "    extract_roi_properties,\n",
    "    track_rois_across_frames,\n",
    "    extract_multichannel_intensities,\n",
    "    visualize_segmentation_qc,\n",
    "    save_intensity_data_json,\n",
    ")\n",
    "from analyze_intensities import organize_data_by_condition, compute_condition_statistics\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Load config ──────────────────────────────────────────────────────────────\n",
    "cfg = load_config('../config.yaml')\n",
    "\n",
    "seg_ch    = cfg['channels']['signal']          # channel used for segmentation\n",
    "ch_names  = [\n",
    "    cfg['channels']['signal_name'] + '_mean',\n",
    "    cfg['channels']['reference_name'] + '_mean',\n",
    "]\n",
    "model     = cfg['segmentation']['cellpose_model']\n",
    "min_diam  = cfg['segmentation']['min_diameter_um']\n",
    "max_diam  = cfg['segmentation']['max_diameter_um']\n",
    "min_circ  = cfg['segmentation']['min_circularity']\n",
    "max_dist  = cfg['tracking']['max_displacement_px']\n",
    "frame_int = cfg['tracking']['frame_interval_minutes']\n",
    "max_gap   = cfg['tracking']['max_gap_minutes']\n",
    "out_dir   = Path('../') / cfg['output']['processed_dir']\n",
    "qc_dir    = Path('../') / cfg['output']['qc_plots_dir']\n",
    "\n",
    "print('Config loaded OK')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Set your TIFF path ───────────────────────────────────────────────────────\n",
    "tiff_path = Path('YOUR_TIFF_FILE.tif')   # <-- edit this\n",
    "\n",
    "stack, meta = load_tiff_stack(tiff_path)\n",
    "print(f'Stack shape: {stack.shape}  (T, C, H, W)')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Segment first frame ──────────────────────────────────────────────────────\n",
    "import numpy as np\n",
    "min_area = int(np.pi * (min_diam / 2) ** 2)\n",
    "max_area = int(np.pi * (max_diam / 2) ** 2)\n",
    "\n",
    "masks, labeled, px_um = segment_rois_with_cellpose(\n",
    "    stack[0, seg_ch, :, :],\n",
    "    model_type=model,\n",
    "    min_area=min_area,\n",
    "    max_area=max_area,\n",
    "    min_circularity=min_circ,\n",
    "    tiff_path=tiff_path,\n",
    "    min_bead_diameter_um=min_diam,\n",
    "    max_bead_diameter_um=max_diam,\n",
    ")\n",
    "initial_rois = extract_roi_properties(masks, labeled)\n",
    "print(f'Detected {len(initial_rois)} ROIs')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── QC plot ──────────────────────────────────────────────────────────────────\n",
    "other_ch = 1 - seg_ch if stack.shape[1] == 2 else 0\n",
    "visualize_segmentation_qc(\n",
    "    stack[0, other_ch, :, :],\n",
    "    stack[0, seg_ch, :, :],\n",
    "    initial_rois,\n",
    "    channel_names=ch_names,\n",
    ")\n",
    "# HUMAN-IN-THE-LOOP: inspect the overlay above.\n",
    "# Verify that ROI outlines match actual objects before proceeding."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Track ROIs ───────────────────────────────────────────────────────────────\n",
    "segmenter = functools.partial(\n",
    "    segment_rois_with_cellpose,\n",
    "    model_type=model,\n",
    "    min_area=min_area,\n",
    "    max_area=max_area,\n",
    "    min_circularity=min_circ,\n",
    "    min_bead_diameter_um=min_diam,\n",
    "    max_bead_diameter_um=max_diam,\n",
    ")\n",
    "\n",
    "tracked = track_rois_across_frames(\n",
    "    stack, initial_rois,\n",
    "    max_distance=max_dist,\n",
    "    segmenter=segmenter,\n",
    "    frame_interval_minutes=frame_int,\n",
    "    max_gap_minutes=max_gap,\n",
    "    seg_channel=seg_ch,\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Extract intensities and save ─────────────────────────────────────────────\n",
    "intensities = extract_multichannel_intensities(stack, tracked, channel_names=ch_names)\n",
    "\n",
    "out_dir.mkdir(parents=True, exist_ok=True)\n",
    "save_intensity_data_json(intensities, meta, out_dir / f'{tiff_path.stem}_intensities.json')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Quick intensity plot ─────────────────────────────────────────────────────\n",
    "signal_col = cfg['channels']['signal_name'] + '_mean'\n",
    "ref_col    = cfg['channels']['reference_name'] + '_mean'\n",
    "\n",
    "import pandas as pd\n",
    "df = pd.DataFrame(intensities)\n",
    "df['time_min'] = df['timepoint'] * frame_int\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(10, 4))\n",
    "for roi_id, grp in df.groupby('roi_id'):\n",
    "    ax.plot(grp['time_min'], grp[signal_col], alpha=0.4, linewidth=1)\n",
    "ax.set_xlabel('Time (min)')\n",
    "ax.set_ylabel(signal_col)\n",
    "ax.set_title('Per-ROI signal intensity over time')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.10.0"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
NBEOF
```

- [ ] **Step 2: Create `deployable/notebooks/zstack_analysis.ipynb`**

```bash
cat > /Users/matthew/Desktop/Fertilizer/deployable/notebooks/zstack_analysis.ipynb << 'NBEOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# Z-Stack Analysis\n\nThis notebook walks through the 3D z-stack pipeline:\n1. Load `config.yaml`\n2. Load a multi-channel z-stack TIFF\n3. Segment vesicle ROIs on the MIP (maximum intensity projection)\n4. Compute per-vesicle leakage metrics (signal fraction inside vs. outside)\n5. Plot summary statistics\n\n**Before running:** edit `../config.yaml` to match your image channels."]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "from pathlib import Path\n",
    "sys.path.insert(0, str(Path('../').resolve()))\n",
    "\n",
    "from ml_image_analysis import load_config, segment_rois_with_cellpose, extract_roi_properties\n",
    "from analyze_zstack_intensities import load_zstack_tiff, compute_vesicle_metrics_zstack\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Load config ──────────────────────────────────────────────────────────────\n",
    "cfg = load_config('../config.yaml')\n",
    "\n",
    "sig_ch   = cfg['channels']['signal']\n",
    "ref_ch   = cfg['channels']['reference']\n",
    "model    = cfg['segmentation']['cellpose_model']\n",
    "min_diam = cfg['segmentation']['min_diameter_um']\n",
    "max_diam = cfg['segmentation']['max_diameter_um']\n",
    "min_circ = cfg['segmentation']['min_circularity']\n",
    "print('Config loaded OK')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Load z-stack ─────────────────────────────────────────────────────────────\n",
    "tiff_path = Path('YOUR_ZSTACK.tif')   # <-- edit this\n",
    "\n",
    "zstack, meta = load_zstack_tiff(\n",
    "    str(tiff_path),\n",
    "    n_channels=2,\n",
    "    channel_signal=sig_ch,\n",
    "    channel_reference=ref_ch,\n",
    ")\n",
    "print(f'Z-stack shape: {zstack.shape}  (Z, C, H, W)')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Segment ROIs on MIP ──────────────────────────────────────────────────────\n",
    "mip = zstack[:, sig_ch, :, :].max(axis=0)\n",
    "\n",
    "min_area = int(np.pi * (min_diam / 2) ** 2)\n",
    "max_area = int(np.pi * (max_diam / 2) ** 2)\n",
    "\n",
    "masks, labeled, px_um = segment_rois_with_cellpose(\n",
    "    mip,\n",
    "    model_type=model,\n",
    "    min_area=min_area,\n",
    "    max_area=max_area,\n",
    "    min_circularity=min_circ,\n",
    "    tiff_path=tiff_path,\n",
    "    min_bead_diameter_um=min_diam,\n",
    "    max_bead_diameter_um=max_diam,\n",
    ")\n",
    "rois = extract_roi_properties(masks, labeled)\n",
    "print(f'Detected {len(rois)} vesicle ROIs')\n",
    "\n",
    "# HUMAN-IN-THE-LOOP: verify ROI detection before computing metrics."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Compute leakage metrics ──────────────────────────────────────────────────\n",
    "metrics_df = compute_vesicle_metrics_zstack(\n",
    "    zstack, masks,\n",
    "    channel_signal=sig_ch,\n",
    "    channel_reference=ref_ch,\n",
    "    pixel_size_um=px_um,\n",
    ")\n",
    "print(metrics_df.describe())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Plot ─────────────────────────────────────────────────────────────────────\n",
    "fig, axes = plt.subplots(1, 2, figsize=(10, 4))\n",
    "axes[0].hist(metrics_df['n_vesicles'], bins=10)\n",
    "axes[0].set_xlabel('Vesicles per z-slice')\n",
    "axes[1].hist(metrics_df['signal_fraction_in'].dropna(), bins=20)\n",
    "axes[1].set_xlabel('Signal fraction inside vesicles')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.10.0"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
NBEOF
```

- [ ] **Step 3: Verify notebooks are valid JSON**

```bash
python -c "import json; json.load(open('deployable/notebooks/2D_timelapse_analysis.ipynb')); print('2D OK')"
python -c "import json; json.load(open('deployable/notebooks/zstack_analysis.ipynb')); print('zstack OK')"
```

Expected: both print `OK`.

- [ ] **Step 4: Commit**

```bash
git add deployable/notebooks/
git commit -m "feat: add clean unrun analysis notebooks to deployable"
```

---

## Task 8: Write `README.md`

**Files:**
- Create: `deployable/README.md`

- [ ] **Step 1: Write `deployable/README.md`**

```bash
cat > /Users/matthew/Desktop/Fertilizer/deployable/README.md << 'MDEOF'
# Image Analysis Pipeline — Deployable

Cellpose-based segmentation and intensity extraction for round fluorescent objects (vesicles, beads, droplets) in multi-channel TIFF time-lapses and z-stacks.

## Installation

```bash
pip install -r requirements.txt
```

For GPU acceleration:
- **Apple Silicon (M1–M4):** MPS is used automatically — no extra steps.
- **CUDA (Linux/Windows):** `pip install cellpose[gpu]`
- **CPU only:** the default install works on all platforms.

## Configuration

Edit `config.yaml` before running. Key fields:

| Section | Field | Description |
|---|---|---|
| `channels` | `signal` / `reference` | Channel indices for segmentation and normalization |
| `channels` | `signal_name` / `reference_name` | Human-readable labels (used as column headers in output) |
| `segmentation` | `cellpose_model` | `"cyto3"` for round objects; path to custom `.pt` file |
| `segmentation` | `min/max_diameter_um` | Expected object size range in micrometres |
| `tracking` | `frame_interval_minutes` | Time between frames |
| `output` | `processed_dir` / `qc_plots_dir` | Where outputs are saved |

The `config.yaml` shipped here uses a NO₃⁻ sensor / GFP/OA-647 vesicle experiment as a worked example. Replace values to match your experiment.

## Usage

### 2D time-lapse

Open `notebooks/2D_timelapse_analysis.ipynb`, set your TIFF path, and run cells top to bottom.

Or via script:

```python
from ml_image_analysis import load_config, process_tiff_directory

cfg = load_config("config.yaml")
process_tiff_directory(
    input_dir="your_tiff_folder/",
    seg_channel=cfg["channels"]["signal"],
    channel_names=[
        cfg["channels"]["signal_name"] + "_mean",
        cfg["channels"]["reference_name"] + "_mean",
    ],
    cellpose_model=cfg["segmentation"]["cellpose_model"],
    min_bead_diameter_um=cfg["segmentation"]["min_diameter_um"],
    max_bead_diameter_um=cfg["segmentation"]["max_diameter_um"],
    frame_interval_minutes=cfg["tracking"]["frame_interval_minutes"],
    output_dir=cfg["output"]["processed_dir"],
    qc_dir=cfg["output"]["qc_plots_dir"],
)
```

### Z-stack

Open `notebooks/zstack_analysis.ipynb` and follow the cell instructions.

### Preparing training images for a custom Cellpose model

```bash
python prepare_training_images.py /path/to/raw/tiffs \
    --channel 0 \
    --model-path cyto3 \
    --output-dir training_images/
```

This preprocesses TIFFs identically to the inference pipeline and opens the Cellpose GUI for annotation.

---

## ⚠️ Human-in-the-Loop: Required Validation

**This pipeline is a starting point, not a complete solution.** Automated image segmentation requires human judgment at every stage. Please read this section before treating pipeline outputs as scientific results.

### ROI detection is image-dependent

Cellpose performs well on round, well-separated objects with moderate signal-to-noise. It may:
- Miss dim or partially-occluded objects
- Merge touching objects (watershed recovery helps but is not guaranteed)
- Over-segment debris, aggregates, or out-of-focus objects
- Fail entirely on image types it was not trained on

For images with unusual morphology, non-circular objects, or low SNR, **traditional methods may outperform ML segmentation**: intensity thresholding, connected-component labeling, active contours, or manual annotation in tools like Fiji/ImageJ, OMERO, or Napari are all valid alternatives. The right tool depends on your specific imaging conditions.

### Parameters require domain expertise

The pipeline exposes parameters (`min_diameter_um`, `max_diameter_um`, `min_circularity`, `flow_threshold`, `cellprob_threshold`) that have no universal correct values. **Meaningful parameter choices depend on:**
- The physical size of your objects under your microscope optics
- Your acquisition SNR and background fluorescence
- What constitutes a "valid" ROI in your experimental context (e.g., an intact vesicle vs. a ruptured one)

Do not use default values without verifying they match your data.

### Verify before analyzing

Before running the full pipeline on a dataset:
1. Run `test_segmentation_on_first_frame()` on representative images and inspect the QC overlay.
2. Visually confirm that ROI boundaries align with actual objects — not debris, not merged clusters.
3. Check tracking quality using `validate_tracking.py` on a subset of your data.
4. Only proceed with quantitative analysis once ROI detection is visually validated.

The QC plots generated in `output/qc_plots/` show ROI outlines overlaid on raw images. **Review these plots.** A pipeline that runs without errors is not the same as a pipeline that is correctly segmenting your data.

---

## Output Format

Intensity data is saved as JSON with this structure:

```json
{
  "metadata": { "filename": "...", "n_timepoints": 73, "pixel_size_um": 0.325 },
  "data": [
    { "roi_id": 0, "timepoint": 0, "GFP_mean": 450.2, "OA-647_mean": 312.1,
      "centroid": [124.5, 87.3], "area": 284 },
    ...
  ]
}
```

Column names in `data` match `signal_name + "_mean"` and `reference_name + "_mean"` from `config.yaml`.

## File Overview

| File | Purpose |
|---|---|
| `ml_image_analysis.py` | Core pipeline: preprocessing, Cellpose segmentation, tracking, intensity extraction |
| `analyze_intensities.py` | Load JSON output, compute statistics, plot time series |
| `analyze_zstack_intensities.py` | Z-stack pipeline: per-slice segmentation, leakage metrics |
| `prepare_training_images.py` | Preprocess TIFFs and open Cellpose GUI for model training |
| `validate_tracking.py` | Tracking quality diagnostics |
| `config.yaml` | All tunable parameters |
| `notebooks/` | Interactive worked examples (run these to verify your install) |
MDEOF
```

- [ ] **Step 2: Verify file exists and is readable**

```bash
wc -l deployable/README.md
```

Expected: non-zero line count.

- [ ] **Step 3: Commit**

```bash
git add deployable/README.md
git commit -m "docs: add README with install, config, usage, and human-in-the-loop section"
```

---

## Task 9: Final verification and commit

- [ ] **Step 1: Run the full test suite**

```bash
python -m pytest deployable/tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 2: Verify deployable structure**

```bash
find deployable/ -type f | sort
```

Expected output includes:
```
deployable/.gitignore
deployable/README.md
deployable/analyze_intensities.py
deployable/analyze_zstack_intensities.py
deployable/config.yaml
deployable/ml_image_analysis.py
deployable/notebooks/2D_timelapse_analysis.ipynb
deployable/notebooks/zstack_analysis.ipynb
deployable/output/processed/.gitkeep
deployable/output/qc_plots/.gitkeep
deployable/prepare_training_images.py
deployable/requirements.txt
deployable/tests/test_analyze_intensities.py
deployable/tests/test_analyze_zstack.py
deployable/tests/test_ml_image_analysis.py
deployable/validate_tracking.py
```

- [ ] **Step 3: Verify notebooks are unrun (no outputs)**

```bash
python -c "
import json
for nb in ['deployable/notebooks/2D_timelapse_analysis.ipynb',
           'deployable/notebooks/zstack_analysis.ipynb']:
    data = json.load(open(nb))
    outputs = [o for c in data['cells'] for o in c.get('outputs', [])]
    assert len(outputs) == 0, f'{nb} has outputs'
    print(f'{nb}: clean')
"
```

Expected: both print `clean`.

- [ ] **Step 4: Final commit**

```bash
git add deployable/
git commit -m "feat: complete deployable/ subdirectory for collaborator use"
```
