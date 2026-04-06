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


def test_headless_export_writes_mp4(gl_ctx, sine_wav, tmp_path):
    """Full pipeline: analyze → render frames → assemble MP4 on disk."""
    pytest.importorskip("moviepy", reason="moviepy not installed")

    output = str(tmp_path / "out.mp4")
    analyzer = PrerecordedAnalyzer()
    analysis = analyzer.analyze(sine_wav, fps=24)
    renderer = Renderer(gl_ctx, width=160, height=90)
    engine = ContextEngine(ContextConfig())
    exporter = Exporter()

    exporter.export_headless(
        analysis, renderer, engine,
        source_audio_path=sine_wav,
        output_path=output,
        fps=24,
    )
    renderer.release()

    import os
    assert os.path.exists(output), "MP4 file was not created"
    assert os.path.getsize(output) > 1024, "MP4 file is suspiciously small"


def test_progress_callback_fires(gl_ctx, sine_wav):
    """collect_frames calls on_progress once per frame with increasing values."""
    analyzer = PrerecordedAnalyzer()
    analysis = analyzer.analyze(sine_wav, fps=24)
    renderer = Renderer(gl_ctx, width=160, height=90)
    engine = ContextEngine(ContextConfig())
    exporter = Exporter()

    progress_vals: list[float] = []
    exporter.collect_frames(analysis, renderer, engine, fps=24,
                            on_progress=progress_vals.append)
    renderer.release()

    assert len(progress_vals) == len(analysis.frames)
    assert progress_vals[0] > 0.0
    assert progress_vals[-1] == pytest.approx(1.0)
    assert progress_vals == sorted(progress_vals)
