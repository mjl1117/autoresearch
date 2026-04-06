"""End-to-end: analyze a wav, run through engine + renderer, collect frames."""
import numpy as np
import pytest
from audio.analyzer import PrerecordedAnalyzer
from engine.context_engine import ContextEngine
from engine.render_params import ContextConfig
from export.exporter import Exporter
from renderer.renderer import Renderer


def test_full_pipeline_produces_frames(gl_ctx, sine_wav):
    fps = 24
    analyzer = PrerecordedAnalyzer()
    analysis = analyzer.analyze(sine_wav, fps=fps)
    assert len(analysis.frames) > 0

    renderer = Renderer(gl_ctx, width=160, height=90)
    engine = ContextEngine(ContextConfig())
    exporter = Exporter()

    frames = exporter.collect_frames(analysis, renderer, engine, fps=fps)

    assert len(frames) == len(analysis.frames)
    assert frames[0].shape == (90, 160, 3)
    assert frames[0].dtype == np.uint8
    assert not np.array_equal(frames[0], frames[-1])

    renderer.release()


def test_organic_frames_not_all_black(gl_ctx, sine_wav):
    """440 Hz sine with moderate tempo stays in ORGANIC — output must be non-black."""
    analyzer = PrerecordedAnalyzer()
    analysis = analyzer.analyze(sine_wav, fps=24)
    renderer = Renderer(gl_ctx, width=160, height=90)
    engine = ContextEngine(ContextConfig())
    exporter = Exporter()
    frames = exporter.collect_frames(analysis, renderer, engine, fps=24)
    mid = frames[len(frames) // 2]
    assert mid.max() > 20
    renderer.release()
