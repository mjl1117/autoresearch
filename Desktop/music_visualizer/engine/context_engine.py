from __future__ import annotations
import math

import numpy as np

from audio.feature_frame import FeatureFrame
from engine.palette import Palette
from engine.render_params import ContextConfig, RenderParams, Style


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _blend_ease(t: float) -> float:
    """Map t in [0,1] through a centred sigmoid for smooth ease-in-out."""
    return _sigmoid((t - 0.5) * 12.0)


def _classify(dissonance_smooth: float, bpm: float, onset_smooth: float, cfg: ContextConfig) -> Style:
    tempo_norm = min(bpm / 200.0, 1.0)
    if dissonance_smooth > cfg.dissonance_threshold:
        return Style.GEOMETRIC
    # COSMIC requires slow tempo AND low onset activity — prevents fast-but-quiet music
    # from going cosmic just because the bpm is under the threshold
    if tempo_norm < cfg.tempo_threshold and onset_smooth < cfg.onset_threshold:
        return Style.COSMIC
    return Style.ORGANIC


class ContextEngine:
    def __init__(self, config: ContextConfig) -> None:
        self._cfg = config
        self._palette = Palette()

        self._current_style = Style.ORGANIC
        self._blend_target = Style.ORGANIC
        self._blend_weight = 0.0

        self._candidate_style = Style.ORGANIC
        self._candidate_hold = 0.0

        self._blend_elapsed = 0.0
        self._in_transition = False

        self._onset_smooth = 0.0

        self._prev_color_a: tuple[float, float, float] | None = None
        self._palette_alpha = 1.0

    def update(self, frame: FeatureFrame, dt: float) -> RenderParams:
        self._onset_smooth = (
            self._cfg.ema_alpha * frame.onset_strength
            + (1.0 - self._cfg.ema_alpha) * self._onset_smooth
        )
        classified = _classify(frame.dissonance_smooth, frame.bpm, self._onset_smooth, self._cfg)

        if classified == self._candidate_style:
            self._candidate_hold += dt
        else:
            self._candidate_style = classified
            self._candidate_hold = 0.0

        if (
            not self._in_transition
            and self._candidate_hold >= self._cfg.style_hold_seconds
            and self._candidate_style != self._current_style
        ):
            self._in_transition = True
            self._blend_target = self._candidate_style
            self._blend_elapsed = 0.0
            self._blend_weight = 0.0

        if self._in_transition:
            self._blend_elapsed += dt
            t = min(self._blend_elapsed / self._cfg.blend_duration_seconds, 1.0)
            self._blend_weight = _blend_ease(t)
            if t >= 1.0:
                self._current_style = self._blend_target
                self._blend_weight = 0.0  # reset — renderer uses active_style exclusively
                self._in_transition = False

        self._palette_alpha = min(self._palette_alpha + dt * 0.5, 1.0)
        color_a, color_b, _ = self._palette.get_palette(
            frame.chroma,
            prev_color_a=self._prev_color_a,
            alpha=self._palette_alpha,
        )
        self._prev_color_a = color_a

        return RenderParams(
            active_style=self._current_style,
            blend_target=self._blend_target,
            blend_weight=self._blend_weight,
            intensity=frame.rms,
            brightness=frame.spectral_centroid,
            pulse=frame.onset_strength,
            dissonance_raw=frame.dissonance_raw,
            color_a=color_a,
            color_b=color_b,
        )
