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
│   │                     # save_tracked_rois, process_nd2_directory
│   ├── analysis.py       # extract_radial_profile_from_tif,
│   │                     # average_radial_profiles, average_cluster_sizes,
│   │                     # process_roi_library, save_results_to_json
│   ├── visualization.py  # all 22+ plot_* functions + animate_stack
│   ├── nadh.py           # process_nd2_directory_nadh,
│   │                     # standardize_stack_nadh (from cell 99)
│   └── nadh_matching.py  # match_rois_and_export, average_droplet_intensity
└── run_pipeline.py       # batch driver
```

---

## MPS acceleration (`mps_morph.py`)

### Problem

`score_mask` is called inside `int_search` with up to 19×19 = 361 (erosion_r, dilation_r)
parameter combinations per frame. With 14 files × ~60 frames = ~840 frames, this is ~300K
sequential morphological operation pairs on CPU.

### Solution — batched GPU scoring

Binary morphological operations are equivalent to convolutions:
- **Erosion** with disk(r): `F.conv2d(img, disk_kernel_r)`, threshold at `kernel_sum`
- **Dilation** with disk(r): `F.conv2d(img, disk_kernel_r)`, threshold at `> 0`

**Batching strategy (38 GPU ops instead of 361):**
1. Pre-compute all unique erosions (up to 19) in a single batched `F.conv2d` call → `[19, H, W]`
2. For each eroded image, apply all unique dilations in one batched call → `[19, 19, H, W]`
3. Transfer all 361 binary masks back to CPU as a batch
4. Run `measure.label` + circularity filtering (must remain CPU; no native PyTorch CC labeling)
5. Return scores for all 361 combos; caller picks the best

### Public API

```python
# Used by roi_extraction.py in place of int_search + score_mask:
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

1. MPS available (`torch.backends.mps.is_available()`) → use MPS
2. MPS not available but CUDA available → use CUDA
3. Neither → use CPU torch (still faster than skimage loop due to batching)
4. torch not installed → fall back to original skimage functions (import-time warning)

### MPS memory guard

Before each batched call, check `torch.mps.driver_allocated_memory()`. If >80% of device
memory is in use, fall back to CPU torch for that frame and log a warning.

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
│   └── graphs/                       ← all plot_* outputs (PNG/SVG)
├── 20250518_Active_minimal/
│   └── ...
└── pipeline_run_summary.json         ← per-file status, ROI counts, timing
```

### Run order per file

1. `process_nd2_directory(nd2_path)` → `tagged_rois/`
2. `process_nd2_directory_nadh(nd2_path)` → `nadh/tagged_rois/` (skipped if no NADH channel)
3. `process_roi_library("tagged_rois/")` → raw radial profiles
4. `average_radial_profiles(results)` + `average_cluster_sizes(results)`
5. `save_results_to_json(...)` → JSON outputs
6. `match_rois_and_export()` (skipped if no NADH)
7. All `plot_*` functions with `save=True` (matplotlib backend set to `Agg`)

### Condition labels

Inferred from filename:
- `20250518_Active_in_vitro` → `"Active in vitro (2025-05-18)"`
- `20250518_Passive_minimal` → `"Passive minimal (2025-05-18)"`

Cross-condition comparison plots (`plot_max_positions_across_conditions`, etc.) group files
by `Active` vs `Passive` across all dates.

### CLI

```bash
python run_pipeline.py                                     # all 14 files
python run_pipeline.py --file 20250518_Active_in_vitro.nd2 # single file
python run_pipeline.py --skip-plots                        # ROI extraction + analysis only
python run_pipeline.py --skip-nadh                         # skip NADH processing
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

All scripts set `matplotlib.use('Agg')` before any other matplotlib import. This prevents
display errors when running as terminal scripts (no `plt.show()` calls in scripts).
`save=True` is the default for all plot calls in `run_pipeline.py`.

---

## Out of scope

- Re-implementing connected-component labeling on MPS (no native PyTorch CC; skimage on CPU is adequate)
- Parallelizing across multiple `.nd2` files simultaneously (sequential per-file is sufficient)
- Porting the 22+ visualization functions to MPS (pure matplotlib/numpy, not a bottleneck)
