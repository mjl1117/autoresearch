from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class FeatureFrame:
    amplitude: float          # peak amplitude, normalized 0–1
    rms: float                # RMS energy, normalized 0–1
    spectral_centroid: float  # spectral centroid, normalized 0–1
    onset_strength: float     # onset strength, normalized 0–1
    dissonance_raw: float     # raw dissonance (for shader transient effects)
    dissonance_smooth: float  # EMA-smoothed dissonance (for style classifier)
    chroma: np.ndarray        # shape (12,) chroma vector
    bpm: float                # beats per minute
