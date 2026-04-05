from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto


class Style(Enum):
    GEOMETRIC = auto()
    ORGANIC = auto()
    COSMIC = auto()


@dataclass
class ContextConfig:
    dissonance_threshold: float = 0.65
    tempo_threshold: float = 0.35       # bpm / 200.0 threshold
    style_hold_seconds: float = 4.0
    blend_duration_seconds: float = 3.0
    ema_alpha: float = 0.15


@dataclass
class RenderParams:
    active_style: Style
    blend_target: Style        # equals active_style when not transitioning
    blend_weight: float        # 0.0 = fully active_style, 1.0 = fully blend_target
    intensity: float           # RMS-driven, 0–1
    brightness: float          # spectral centroid-driven, 0–1
    pulse: float               # onset strength (raw), 0–1
    dissonance_raw: float      # raw dissonance for glitch micro-effects
    color_a: tuple[float, float, float]   # primary RGB
    color_b: tuple[float, float, float]   # secondary RGB
