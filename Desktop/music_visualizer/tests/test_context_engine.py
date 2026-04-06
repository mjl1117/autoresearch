import numpy as np
import pytest
from audio.feature_frame import FeatureFrame
from engine.context_engine import ContextEngine
from engine.render_params import ContextConfig, RenderParams, Style


def _make_frame(dissonance_smooth: float = 0.0, bpm: float = 120.0) -> FeatureFrame:
    return FeatureFrame(
        amplitude=0.5,
        rms=0.5,
        spectral_centroid=0.5,
        onset_strength=0.2,
        dissonance_raw=dissonance_smooth,
        dissonance_smooth=dissonance_smooth,
        chroma=np.zeros(12),
        bpm=bpm,
    )


def test_default_style_is_organic():
    engine = ContextEngine(ContextConfig())
    params = engine.update(_make_frame(), dt=1.0 / 24)
    assert params.active_style == Style.ORGANIC


def test_high_dissonance_stays_organic_until_hold():
    """Must hold for 4s before transitioning away from organic."""
    cfg = ContextConfig(style_hold_seconds=4.0, blend_duration_seconds=3.0)
    engine = ContextEngine(cfg)
    frame = _make_frame(dissonance_smooth=0.9)
    for _ in range(int(3.9 * 24)):
        params = engine.update(frame, dt=1.0 / 24)
    assert params.active_style == Style.ORGANIC
    assert params.blend_weight < 1.0


def test_sustained_dissonance_triggers_geometric():
    """After 4s hold + blend start, style should eventually be GEOMETRIC."""
    cfg = ContextConfig(style_hold_seconds=4.0, blend_duration_seconds=3.0)
    engine = ContextEngine(cfg)
    frame = _make_frame(dissonance_smooth=0.9)
    params = None
    for _ in range(int(7.1 * 24)):
        params = engine.update(frame, dt=1.0 / 24)
    assert params.active_style == Style.GEOMETRIC
    assert params.blend_weight == pytest.approx(0.0, abs=0.01)  # reset after transition completes


def test_low_tempo_triggers_cosmic():
    cfg = ContextConfig(style_hold_seconds=2.0, blend_duration_seconds=1.0)
    engine = ContextEngine(cfg)
    frame = _make_frame(dissonance_smooth=0.1, bpm=50.0)  # 50/200 = 0.25 < 0.35
    params = None
    for _ in range(int(3.1 * 24)):
        params = engine.update(frame, dt=1.0 / 24)
    assert params.active_style == Style.COSMIC


def test_blend_weight_increases_during_transition():
    cfg = ContextConfig(style_hold_seconds=0.1, blend_duration_seconds=2.0)
    engine = ContextEngine(cfg)
    frame = _make_frame(dissonance_smooth=0.9)
    weights = []
    for _ in range(int(2.5 * 24)):
        params = engine.update(frame, dt=1.0 / 24)
        weights.append(params.blend_weight)
    increasing = [weights[i] <= weights[i + 1] for i in range(len(weights) - 1)]
    assert sum(increasing) > len(increasing) * 0.9


def test_render_params_has_valid_colors():
    engine = ContextEngine(ContextConfig())
    chroma = np.zeros(12)
    chroma[0] = 1.0
    frame = FeatureFrame(
        amplitude=0.5, rms=0.5, spectral_centroid=0.5, onset_strength=0.2,
        dissonance_raw=0.0, dissonance_smooth=0.0, chroma=chroma, bpm=120.0,
    )
    params = engine.update(frame, dt=1.0 / 24)
    assert all(0.0 <= c <= 1.0 for c in params.color_a)
    assert all(0.0 <= c <= 1.0 for c in params.color_b)
