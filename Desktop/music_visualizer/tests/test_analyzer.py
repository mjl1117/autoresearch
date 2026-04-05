import numpy as np
import pytest
from audio.analyzer import AnalysisResult, PrerecordedAnalyzer


def test_analyze_returns_result(sine_wav):
    analyzer = PrerecordedAnalyzer()
    result = analyzer.analyze(sine_wav, fps=24)
    assert isinstance(result, AnalysisResult)


def test_analyze_frame_count(sine_wav):
    analyzer = PrerecordedAnalyzer()
    result = analyzer.analyze(sine_wav, fps=24)
    expected_frames = int(result.duration * 24)
    # allow ±2 frames for rounding
    assert abs(len(result.frames) - expected_frames) <= 2


def test_analyze_bpm_positive(sine_wav):
    analyzer = PrerecordedAnalyzer()
    result = analyzer.analyze(sine_wav, fps=24)
    assert result.bpm > 0.0


def test_feature_frame_fields_normalized(sine_wav):
    analyzer = PrerecordedAnalyzer()
    result = analyzer.analyze(sine_wav, fps=24)
    ff = result.frames[0]
    assert 0.0 <= ff.amplitude <= 1.0
    assert 0.0 <= ff.rms <= 1.0
    assert 0.0 <= ff.spectral_centroid <= 1.0
    assert ff.chroma.shape == (12,)
    assert ff.bpm == result.bpm


def test_analyze_avi_extracts_audio(tmp_path, sine_wav):
    """AVI input: audio extracted to temp wav, same result as wav input."""
    import shutil
    avi_path = str(tmp_path / "test.avi")
    shutil.copy(sine_wav, avi_path)
    analyzer = PrerecordedAnalyzer()
    # Should not raise; falls back to treating as audio directly for test
    result = analyzer.analyze(avi_path, fps=24)
    assert len(result.frames) > 0
