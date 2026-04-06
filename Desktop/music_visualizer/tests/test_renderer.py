import numpy as np
import pytest
from engine.render_params import ContextConfig, RenderParams, Style
from renderer.renderer import Renderer


def _default_params() -> RenderParams:
    return RenderParams(
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


def test_renderer_creates_without_error(gl_ctx):
    r = Renderer(gl_ctx, width=320, height=240)
    assert r is not None
    r.release()


def test_read_pixels_returns_correct_shape(gl_ctx):
    r = Renderer(gl_ctx, width=320, height=240)
    r.render_frame(_default_params(), elapsed_time=0.0)
    pixels = r.read_pixels()
    assert pixels.shape == (240, 320, 3)
    assert pixels.dtype == np.uint8
    r.release()


def test_frame_is_not_all_black(gl_ctx):
    r = Renderer(gl_ctx, width=320, height=240)
    r.render_frame(_default_params(), elapsed_time=1.0)
    pixels = r.read_pixels()
    assert pixels.max() > 10
    r.release()


def test_geometric_style_renders(gl_ctx):
    params = RenderParams(
        active_style=Style.GEOMETRIC,
        blend_target=Style.GEOMETRIC,
        blend_weight=0.0,
        intensity=0.8,
        brightness=0.6,
        pulse=0.5,
        dissonance_raw=0.7,
        color_a=(1.0, 0.0, 0.12),
        color_b=(0.0, 1.0, 0.88),
    )
    r = Renderer(gl_ctx, width=320, height=240)
    r.render_frame(params, elapsed_time=1.0)
    pixels = r.read_pixels()
    assert pixels.max() > 10
    r.release()


def test_cosmic_style_renders(gl_ctx):
    params = RenderParams(
        active_style=Style.COSMIC,
        blend_target=Style.COSMIC,
        blend_weight=0.0,
        intensity=0.3,
        brightness=0.2,
        pulse=0.1,
        dissonance_raw=0.05,
        color_a=(0.55, 0.36, 0.96),
        color_b=(0.96, 0.62, 0.04),
    )
    r = Renderer(gl_ctx, width=320, height=240)
    r.render_frame(params, elapsed_time=2.0)
    pixels = r.read_pixels()
    assert pixels.max() > 10
    r.release()
