# Design: Deployable Subdirectory

**Date:** 2026-04-17
**Status:** Approved

---

## Overview

Create a `deployable/` subdirectory containing a minimal, generalizable version of the image analysis pipeline for collaborator use. The deployable strips project-specific hardcoded values (paths, condition names, channel labels) into a `config.yaml` and refactors shared segmentation/tracking logic into a common module imported by both the 2D time-lapse and 3D z-stack pipelines.

---

## Directory Structure

```
deployable/
├── README.md
├── requirements.txt
├── config.yaml                        # example based on NO3 project, fully commented
├── ml_image_analysis.py               # core pipeline: preprocessing, segmentation, intensity extraction, 2D time tracking
├── analyze_zstack_intensities.py      # 3D z-stack: per-slice segmentation + vesicle leakage metrics
├── analyze_intensities.py             # 2D time-lapse: load JSON output, compute stats, plot
├── prepare_training_images.py         # CLI: preprocess TIFFs and launch Cellpose GUI for annotation
├── validate_tracking.py               # tracking QC diagnostics and visualization
├── notebooks/
│   ├── 2D_timelapse_analysis.ipynb    # clean, unrun example notebook
│   └── zstack_analysis.ipynb          # clean, unrun example notebook
└── output/                            # placeholder dirs (gitignored)
    ├── processed/
    └── qc_plots/
```

---

## Modularity

### Shared core in `ml_image_analysis.py`

Extract the following functions so both pipelines import from a single source of truth:

- `enhance_contrast_clahe` — CLAHE contrast enhancement
- `apply_rolling_ball_background` — Gaussian-approximated rolling ball subtraction
- `apply_dog_enhancement` — Difference-of-Gaussians band-pass filter
- `segment_rois_with_cellpose` — Cellpose call + watershed splitting + circularity/area filtering
- `track_rois_hungarian` — Hungarian-algorithm cross-frame ROI matching
- `load_tiff_stack` — load multi-dimensional TIFF to (T, C, H, W)
- `read_pixel_size_from_tiff` — extract µm/px from OME/ImageJ metadata

### 2D time-lapse (`ml_image_analysis.py` main pipeline)

Adds a temporal loop over `segment_rois_with_cellpose` and `track_rois_hungarian` to produce per-ROI intensity traces saved as JSON.

### 3D z-stack (`analyze_zstack_intensities.py`)

Uses `segment_rois_with_cellpose` per z-slice (no temporal tracking). Computes:
- n_vesicles per slice
- GFP/reference ratio inside vesicle ROIs
- Reference channel fraction inside vesicles vs. puncta vs. diffuse background
- Vesicle cross-sectional area distribution (µm²)

---

## Config File (`config.yaml`)

All pipeline parameters are read from `config.yaml`. Hardcoded project-specific values are replaced with config lookups. The file ships with the NO₃⁻ project settings as commented examples so collaborators understand each field.

### Top-level sections

```yaml
# ── Channels ────────────────────────────────────────────────────────────────
channels:
  signal: 0          # index of primary signal channel (e.g. GFP)
  reference: 1       # index of reference channel (e.g. OA-647)
  signal_name: "GFP"
  reference_name: "OA-647"

# ── Segmentation ────────────────────────────────────────────────────────────
segmentation:
  cellpose_model: "cyto3"          # model name or absolute path to custom model
  diameter_um: null                # expected object diameter in µm; null = auto
  min_diameter_um: 1.0
  max_diameter_um: 50.0
  min_circularity: 0.7
  flow_threshold: 0.4
  cellprob_threshold: 0.0

# ── Preprocessing ────────────────────────────────────────────────────────────
preprocessing:
  rolling_ball: true
  rolling_ball_radius_px: null     # null = auto from pixel size + max_diameter_um
  dog: true
  dog_sigma_low: 1.0
  dog_sigma_high: null             # null = auto
  clahe: true
  clahe_clip_limit: 0.01

# ── Pixel size ───────────────────────────────────────────────────────────────
pixel_size:
  um_per_px: null                  # null = read from TIFF metadata

# ── Tracking (2D time-lapse only) ────────────────────────────────────────────
tracking:
  max_displacement_px: 20          # max centroid shift between frames
  min_track_length: 3              # discard tracks shorter than N timepoints

# ── Output ───────────────────────────────────────────────────────────────────
output:
  processed_dir: "output/processed"
  qc_plots_dir:  "output/qc_plots"
```

---

## README Sections

1. **Overview** — what the pipeline does (Cellpose-based ROI segmentation, multi-channel intensity extraction, optional time tracking)
2. **Installation** — `pip install -r requirements.txt`, Cellpose GPU/MPS notes
3. **Configuration** — edit `config.yaml`; field-by-field descriptions
4. **Running the pipeline** — which script/notebook for which use case
5. **Human-in-the-loop (critical)** — see below
6. **Outputs** — JSON structure, QC plot descriptions
7. **Training a custom model** — `prepare_training_images.py` usage

### Human-in-the-loop section (key message)

This section must communicate:

- ROI identification in fluorescence images is fundamentally problem-dependent. Cellpose (cyto3/cpsam) works well for round, well-separated objects but may under-segment touching objects, over-segment debris, or miss dim ROIs depending on signal-to-noise and object morphology.
- Traditional methods (intensity thresholding, connected-component labeling, manual annotation, watershed on domain-specific markers) may outperform ML segmentation for specific image types and should be considered as alternatives or complements.
- The scientifically meaningful parameters — which channels carry signal, what intensity ratio is biologically interpretable, what constitutes a valid ROI in context — require domain expertise and cannot be determined by the pipeline. These must be validated by the researcher before treating pipeline output as ground truth.
- QC plots and tracking validation outputs are tools to support this judgment, not substitutes for it. Researchers should visually inspect ROI overlays on representative images before committing to a parameter set.

---

## Notebooks

Both notebooks ship with all cells cleared (no output). They demonstrate the full workflow end-to-end so collaborators can run cell-by-cell and verify the pipeline executes correctly on their machine with their data.

- `2D_timelapse_analysis.ipynb`: load TIFF stack → configure → segment → track → extract intensities → plot traces
- `zstack_analysis.ipynb`: load z-stack TIFF → configure → segment per slice → compute leakage metrics → plot

---

## Non-Goals

- No project-specific condition names or analysis scripts (`analyze_ratio.py` stays in the main project only)
- No hardcoded file paths
- No annotation or Cellpose training invocation (covered by `prepare_training_images.py` launching the GUI)
- No automatic parameter optimization
