import numpy as np
import pytest
from unittest.mock import patch
from audio.live_analyzer import LiveAnalyzer
from audio.feature_frame import FeatureFrame


@pytest.fixture
def mock_devices():
    devices = [
        {"name": "Built-in Microphone", "max_input_channels": 2, "index": 0},
        {"name": "Focusrite USB", "max_input_channels": 8, "index": 1},
        {"name": "Built-in Output", "max_input_channels": 0, "index": 2},
    ]
    with patch("sounddevice.query_devices", return_value=devices):
        yield devices


def test_list_input_devices(mock_devices):
    devices = LiveAnalyzer.list_input_devices()
    assert len(devices) == 2  # output-only device filtered out
    assert devices[0]["name"] == "Built-in Microphone"
    assert devices[1]["name"] == "Focusrite USB"


def test_get_frame_returns_feature_frame():
    analyzer = LiveAnalyzer(device_index=0)
    ff = analyzer.get_frame()
    assert isinstance(ff, FeatureFrame)
    assert ff.chroma.shape == (12,)


def test_process_chunk_updates_frame():
    analyzer = LiveAnalyzer(device_index=0)
    sr = 44100
    chunk_size = 512
    t = np.linspace(0, chunk_size / sr, chunk_size)
    chunk = np.sin(2 * np.pi * 440.0 * t).astype(np.float32).reshape(-1, 1)
    analyzer._process_chunk(chunk, None, None, None)
    ff = analyzer.get_frame()
    assert ff.rms > 0.0


def test_get_recording_returns_array():
    analyzer = LiveAnalyzer(device_index=0)
    chunk = np.ones((512, 1), dtype=np.float32) * 0.1
    for _ in range(10):
        analyzer._process_chunk(chunk, None, None, None)
    audio = analyzer.get_recording()
    assert isinstance(audio, np.ndarray)
    assert audio.ndim == 1
    assert len(audio) == 512 * 10
