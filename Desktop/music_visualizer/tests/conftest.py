# tests/conftest.py
import wave
import numpy as np
import pytest


def _write_wav(path: str, sr: int, samples: np.ndarray) -> None:
    """Write float32 samples as 16-bit mono WAV using stdlib only."""
    int16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int16.tobytes())


@pytest.fixture
def sine_wav(tmp_path):
    """3-second 440 Hz sine wave at 22050 Hz."""
    sr = 22050
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = (np.sin(2 * np.pi * 440.0 * t) * 0.5).astype(np.float32)
    path = str(tmp_path / "test.wav")
    _write_wav(path, sr, audio)
    return path


@pytest.fixture
def gl_ctx():
    """Headless ModernGL context for renderer tests."""
    import moderngl
    ctx = moderngl.create_standalone_context()
    yield ctx
    ctx.release()
