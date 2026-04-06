import numpy as np
import pytest
from audio.feature_frame import FeatureFrame
from engine.render_params import ContextConfig, RenderParams, Style


def test_feature_frame_fields():
    ff = FeatureFrame(
        amplitude=0.5,
        rms=0.3,
        spectral_centroid=0.4,
        onset_strength=0.2,
        dissonance_raw=0.1,
        dissonance_smooth=0.1,
        chroma=np.zeros(12),
        bpm=120.0,
    )
    assert ff.amplitude == 0.5
    assert ff.chroma.shape == (12,)


def test_render_params_defaults():
    params = RenderParams(
        active_style=Style.ORGANIC,
        blend_target=Style.ORGANIC,
        blend_weight=0.0,
        intensity=0.5,
        brightness=0.5,
        pulse=0.0,
        dissonance_raw=0.0,
        color_a=(0.31, 0.76, 0.63),
        color_b=(0.48, 0.41, 0.93),
    )
    assert params.active_style == Style.ORGANIC
    assert params.blend_weight == 0.0


def test_context_config_defaults():
    cfg = ContextConfig()
    assert cfg.dissonance_threshold == 0.40
    assert cfg.tempo_threshold == 0.40
    assert cfg.onset_threshold == 0.25
    assert cfg.style_hold_seconds == 2.0
    assert cfg.blend_duration_seconds == 2.0
    assert cfg.ema_alpha == 0.25


def test_style_enum_values():
    assert Style.GEOMETRIC != Style.ORGANIC
    assert Style.ORGANIC != Style.COSMIC
    assert len(Style) == 3
