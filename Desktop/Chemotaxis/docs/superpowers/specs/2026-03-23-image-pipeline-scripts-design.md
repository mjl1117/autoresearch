# Image Analysis Pipeline — Notebook-to-Scripts + MPS Acceleration

**Date:** 2026-03-23
**Status:** Approved

---

## Goal

Convert `Image_Analysis_Pipeline.ipynb` and `nadh_matching.ipynb` from Jupyter notebooks into
runnable Python scripts, add PyTorch MPS acceleration for the morphological/convolution hot path,
and run the full pipeline over all 14 `.nd2` files in `chemotaxis_by_date/`.

---

## Source notebooks

| Notebook | Key cells | Lines of code |
|---|---|---|
| `Image_Analysis_Pipeline.ipynb` | Cell 3 (utils + ROI extraction, 1210 lines), Cell 8 (analysis + viz, 3753 lines), Cell 99 (NADH processing, ~950 lines) | ~5900 |
| `nadh_matching.ipynb` | match_rois_and_export, average_droplet_intensity | ~150 |

---

## Required dependencies

```
nd2          # non-standard; install as `pip install nd2`
numpy
scikit-image
scipy
matplotlib
pandas
seaborn
tqdm
torch        # optional; enables MPS/CUDA acceleration
Pillow       # for GIF animation output via PillowWriter
colorama
```

---

## Module layout

```
scripts/
├── pipeline/
│   ├── __init__.py
│   ├── utils.py          # circularity, int_search, convert_to_8_bit,
│   │                     # standardize_stack, get_image_files
│   ├── mps_morph.py      # MPS-accelerated binary_erosion/dilation +
│   │                     # batched score_mask
│   ├── roi_extraction.py # optimize_mask_and_regions, find_roi_trajectory,
│   │                     # save_tracked_rois, process_nd2_directory,
│   │                     # process_single_nd2 (new wrapper)
│   ├── analysis.py       # extract_radial_profile_from_tif,
│   │                     # average_radial_profiles, average_cluster_sizes,
│   │                     # process_roi_library, save_results_to_json
│   ├── visualization.py  # all 28 plot_* functions + animate_stack
│   │                     # (includes plot_nadh_consumption_over_time and
│   │                     #  plot_max_chemotaxis_vs_nadh_rate_colored_by_velocity)
│   ├── nadh.py           # process_nd2_directory_nadh,
│   │                     # standardize_stack_nadh (from cell 99)
│   └── nadh_matching.py  # match_rois_and_export, average_droplet_intensity
└── run_pipeline.py       # batch driver
```

---

## `save_results_to_json` — name collision resolution

Two versions exist in the notebook (cell 3: 2-arg, cell 8: 4-arg). Use only the 4-arg
version from cell 8 throughout:

```python
save_results_to_json(results, file_name, save_path=None, combine_all=False)
```

The 2-arg version from cell 3 is dropped.

---

## MPS acceleration (`mps_morph.py`)

### Problem

`score_mask` is called inside `int_search` with up to `range(minimum[0], maximum[0])` ×
`range(minimum[1], maximum[1])` parameter combinations per frame. With the default call
`minimum=[1,1], maximum=[20,20], skips=[1,1]` this is 19×19 = 361 combinations. Batch size
scales dynamically with `minimum`, `maximum`, and `skips` — the implementation must not
assume a fixed 361.

With 14 files × ~60 frames = ~840 frames, this is ~300K sequential morphological operation
pairs on CPU.

### Solution — batched GPU scoring

Binary morphological operations are equivalent to convolutions:
- **Erosion** with disk(r): convolve with disk kernel, output pixel = 1 iff == kernel_sum
- **Dilation** with disk(r): convolve with disk kernel, output pixel = 1 iff > 0

`skimage.morphology.disk(r)` produces a `(2r+1) × (2r+1)` kernel. Kernels of different
radii have different spatial sizes, so they cannot be directly stacked into a single
`F.conv2d` weight tensor. **Fix:** pad all kernels to the size of the largest one
(determined by `max(maximum)`) with zeros before stacking. The `F.conv2d` padding value
must equal `max_kernel_size // 2` to preserve `(H, W)` after convolution.

**Batching strategy:**
1. Compute `unique_erosion_radii = range(minimum[0], maximum[0], skips[0])` and
   `unique_dilation_radii = range(minimum[1], maximum[1], skips[1])`
2. Pad all disk kernels to `(2*max_r+1, 2*max_r+1)` and stack into weight tensors
3. Run one batched `F.conv2d` call for all unique erosion kernels → `[N_erosion, H, W]`
4. For each eroded image, run one batched `F.conv2d` for all dilation kernels →
   `[N_erosion * N_dilation, H, W]`
5. Transfer all binary masks to CPU as a batch
6. Run `measure.label` + circularity filtering per mask (must remain CPU; no PyTorch CC)
7. Return `(best_params, best_score)` matching `int_search`'s return contract

### Public API

```python
# Drop-in replacement for int_search(score_mask, ...) in roi_extraction.py:
batched_score_mask(
    image: np.ndarray,
    minimum: list[int],
    maximum: list[int],
    skips: list[int] | None = None,
    **score_kwargs,
) -> tuple[tuple[int, int], float]   # (best_params, best_score)

# Single-image ops used inside find_roi_trajectory:
mps_binary_erosion(image: np.ndarray, radius: int) -> np.ndarray
mps_binary_dilation(image: np.ndarray, radius: int) -> np.ndarray
```

### Device fallback chain

1. `torch.backends.mps.is_available()` → use MPS
2. `torch.cuda.is_available()` → use CUDA
3. torch installed but no GPU → use CPU torch (still faster due to batching)
4. torch not installed → use original skimage functions (import-time warning printed once)

### MPS memory guard

Before each batched call, check `torch.mps.current_allocated_memory()` (public API).
If >80% of `torch.mps.recommended_max_memory()`, fall back to CPU torch for that frame
and log a warning.

---

## `roi_extraction.py` — new `process_single_nd2` wrapper

The existing `process_nd2_directory(root, ...)` takes a directory path and finds all `.nd2`
files inside it. `run_pipeline.py` needs to process one file at a time. Add:

```python
def process_single_nd2(nd2_path: Path, save_path: Path | None = None, **kwargs):
    """Process a single .nd2 file. Wraps process_nd2_directory on a temp single-file dir,
    or re-implements the inner loop for one file."""
```

Same signature extension applies to `process_nd2_directory_nadh` → add
`process_single_nd2_nadh`.

---

## `run_pipeline.py` — batch driver

### Input

All `.nd2` files in `chemotaxis_by_date/`, named `{date}_{Active|Passive}_{in_vitro|minimal}.nd2`.

### Output structure

```
processed_data/
├── 20250518_Active_in_vitro/
│   ├── tagged_rois/                  ← ROI crops from process_nd2_directory
│   ├── nadh/tagged_rois/             ← NADH ROI crops (if NADH channel present)
│   ├── radial_profiles.json
│   ├── radial_profiles_averaged.json
│   ├── cluster_sizes_averaged.json
│   ├── nadh_intensity.json
│   ├── matched_rois.json
│   ├── graphs/                       ← PNG/SVG plots
│   └── animations/                   ← .gif outputs from animate_stack
├── 20250518_Active_minimal/
│   └── ...
└── pipeline_run_summary.json
```

### Run order per file

1. `process_single_nd2(nd2_path, save_path=out_dir)` → `tagged_rois/`
2. `process_roi_library(out_dir / "tagged_rois/")` → raw radial profiles
3. `average_radial_profiles(results)` + `average_cluster_sizes(results)`
4. `save_results_to_json(...)` → JSON outputs
5. *(if NADH channel detected)*:
   a. `process_single_nd2_nadh(nd2_path, save_path=out_dir)` → `nadh/tagged_rois/`
   b. `process_roi_library(out_dir / "nadh/tagged_rois/")` → NADH radial profiles
   c. `average_droplet_intensity(nadh_results)` → averaged NADH intensities
   d. `save_results_to_json(average_nadh, "nadh_intensity.json", ...)`
   e. `match_rois_and_export(chemotaxis_root=..., nadh_root=..., output_dir=...)`
6. Per-file visualization (all `plot_*` with `save=True`, output to `graphs/` or `animations/`)

### `dirs` inference

`dirs` for time-lapse plots is inferred from the number of frames in the stack:
```python
n_frames = len(averaged_profiles)  # or read from nd2 metadata
dirs = np.arange(0, n_frames)
```
Hardcoded `dirs = np.arange(0, 61)` in the notebook is replaced by this dynamic inference.

### Plot functions lacking `save` parameter

`plot_radial_profile_circle`, `plot_correlation_dynamics`, `plot_mean_profiles_frame`, and
`plot_radial_profile_circle_frame` currently have no `save` parameter. During porting,
add `save: bool = False` and a `plt.savefig(...)` call to each. This is in-scope for the
conversion.

### Post-loop cross-condition aggregation

After the per-file loop completes, `run_pipeline.py` runs a second pass that:
1. Loads all `radial_profiles_averaged.json` files from `processed_data/`
2. Groups by `Active` vs `Passive` (parsed from filename)
3. Calls cross-condition plot functions:
   - `plot_max_positions_across_conditions`
   - `plot_max_scatter_across_conditions`
   - `plot_edge_intensity_across_conditions`
   - `plot_cluster_fold_change_across_conditions`
   - `plot_cluster_scatter_vs_time`
4. Saves outputs to `processed_data/cross_condition_graphs/`

### Condition labels

Inferred from filename:
- `20250518_Active_in_vitro` → `"Active in vitro (2025-05-18)"`
- `20250518_Passive_minimal` → `"Passive minimal (2025-05-18)"`

### CLI

```bash
python run_pipeline.py                                     # all 14 files
python run_pipeline.py --file 20250518_Active_in_vitro.nd2 # single file
python run_pipeline.py --skip-plots                        # ROI extraction + analysis only
python run_pipeline.py --skip-nadh                         # skip NADH processing
python run_pipeline.py --skip-cross-condition              # skip post-loop aggregation plots
```

---

## Error handling

| Scenario | Behavior |
|---|---|
| Single file fails (corrupt, no ROIs) | Log error, continue to next file |
| MPS OOM on a frame | Fall back to CPU torch for that frame, log warning |
| >50% of frames in a file have 0 ROIs | Flag file as `warned` in summary JSON |
| torch not installed | Import-time warning; use skimage fallback throughout |
| NADH channel absent in .nd2 | Skip NADH stage for that file silently |

`pipeline_run_summary.json` records per-file: `status` (success/failed/warned), `roi_count`,
`frames_processed`, `duration_seconds`, `device_used` (mps/cuda/cpu).

---

## Matplotlib backend

All scripts set `matplotlib.use('Agg')` before any other matplotlib import. `save=True` is
the default for all plot calls in `run_pipeline.py`.

---

## Out of scope

- Re-implementing connected-component labeling on MPS (no native PyTorch CC; skimage on CPU is adequate)
- Parallelizing across multiple `.nd2` files simultaneously (sequential per-file is sufficient)
