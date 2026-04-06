import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from audio.analyzer import AnalysisResult
from audio.feature_frame import FeatureFrame
from engine.context_engine import ContextEngine
from engine.render_params import ContextConfig
from export.exporter import Exporter


def _make_analysis(n_frames: int = 12) -> AnalysisResult:
    sr = 22050
    chroma = np.zeros(12)
    chroma[0] = 1.0
    frames = [
        FeatureFrame(
            amplitude=0.5, rms=0.5, spectral_centroid=0.5,
            onset_strength=0.2, dissonance_raw=0.1, dissonance_smooth=0.1,
            chroma=chroma, bpm=120.0,
        )
        for _ in range(n_frames)
    ]
    return AnalysisResult(
        bpm=120.0,
        duration=n_frames / 24.0,
        sr=sr,
        audio=np.zeros(sr, dtype=np.float32),
        frames=frames,
    )


def test_collect_frames_returns_correct_count(gl_ctx):
    from renderer.renderer import Renderer
    r = Renderer(gl_ctx, width=64, height=48)
    engine = ContextEngine(ContextConfig())
    exporter = Exporter()
    analysis = _make_analysis(n_frames=6)
    frames = exporter.collect_frames(analysis, r, engine, fps=24)
    assert len(frames) == 6
    assert frames[0].shape == (48, 64, 3)
    r.release()


def test_export_live_calls_moviepy(tmp_path, sine_wav):
    frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(12)]
    with patch("moviepy.ImageSequenceClip") as mock_clip:
        mock_audio = MagicMock()
        instance = MagicMock()
        mock_clip.return_value = instance
        instance.set_audio.return_value = instance
        with patch("moviepy.AudioFileClip", return_value=mock_audio):
            exporter = Exporter()
            out = str(tmp_path / "out.mp4")
            exporter.export_live(frames, sine_wav, out, fps=24)
            mock_clip.assert_called_once()
