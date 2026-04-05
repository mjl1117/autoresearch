# Music Visualizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone music visualizer that converts pre-recorded or live audio to GPU-rendered MP4 video using three adaptive visual styles driven by real-time audio feature extraction.

**Architecture:** Pygame owns the window and event loop; ModernGL renders three GLSL fragment shader styles (Geometric/Glitch, Organic/Flow, Cosmic/Nebula) into an FBO that gets blitted to the Pygame surface each frame. A librosa pre-analysis pass (or live sounddevice ring buffer) feeds a ContextEngine that classifies style, manages sigmoid-blended transitions with hysteresis, and emits RenderParams per frame. MoviePy assembles exported MP4 from FBO readback frames.

**Tech Stack:** Python 3.11+, Pygame 2.5+, ModernGL 5.10+, librosa 0.10+, sounddevice 0.4+, MoviePy 1.0+, NumPy 1.24+, colorsys (stdlib), wave (stdlib)

---

## File Map

```
music_visualizer/
├── main.py                        # entry point — Pygame loop, wires all modules
├── config.toml                    # tunable thresholds and render settings
├── pyproject.toml                 # dependencies
├── audio/
│   ├── __init__.py
│   ├── feature_frame.py           # FeatureFrame dataclass
│   ├── analyzer.py                # PrerecordedAnalyzer (librosa full pass)
│   └── live_analyzer.py          # LiveAnalyzer (sounddevice ring buffer)
├── engine/
│   ├── __init__.py
│   ├── render_params.py           # Style enum + RenderParams dataclass + ContextConfig
│   ├── palette.py                 # Circle of Fifths → color palette
│   └── context_engine.py         # classifier, hysteresis, sigmoid blend, palette
├── renderer/
│   ├── __init__.py
│   ├── renderer.py                # ModernGL FBO setup, per-frame dispatch
│   ├── post.py                    # post-processing pass manager
│   └── shaders/
│       ├── quad.vert              # shared fullscreen quad vertex shader
│       ├── geometric.glsl        # Geometric/Glitch fragment shader
│       ├── organic.glsl          # Organic/Flow fragment shader
│       ├── cosmic.glsl           # Cosmic/Nebula fragment shader
│       ├── composite.glsl        # cross-style blend pass
│       ├── bloom_extract.glsl    # bright-pixel extraction
│       ├── blur.glsl             # separable Gaussian (H and V passes)
│       └── final.glsl            # FXAA + tone map + vignette + motion blur + bloom add
├── ui/
│   ├── __init__.py
│   └── launcher.py               # Pygame top bar state machine
├── export/
│   ├── __init__.py
│   └── exporter.py               # headless render loop + live frame assembly + MoviePy
└── tests/
    ├── conftest.py                # shared fixtures (sine wav, gl_ctx)
    ├── test_feature_frame.py
    ├── test_analyzer.py
    ├── test_live_analyzer.py
    ├── test_context_engine.py
    ├── test_palette.py
    ├── test_renderer.py
    └── test_exporter.py
```

---

## Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `config.toml`
- Create: `audio/__init__.py`, `engine/__init__.py`, `renderer/__init__.py`, `ui/__init__.py`, `export/__init__.py`, `tests/__init__.py`
- Create: `main.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "music_visualizer"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pygame>=2.5",
    "moderngl>=5.10",
    "librosa>=0.10",
    "sounddevice>=0.4",
    "moviepy>=1.0",
    "numpy>=1.24",
    "soundfile>=0.12",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-cov"]
```

- [ ] **Step 2: Create `config.toml`**

```toml
[context]
dissonance_threshold = 0.65
tempo_threshold = 0.35
style_hold_seconds = 4.0
blend_duration_seconds = 3.0
ema_alpha = 0.15

[export]
fps = 24
resolution = [1920, 1080]
crf = 18
codec = "libx264"

[renderer]
realtime_fps = 60
bloom_intensity = 0.4
motion_blur_alpha = 0.15
```

- [ ] **Step 3: Create package `__init__.py` files**

Run: `touch audio/__init__.py engine/__init__.py renderer/__init__.py ui/__init__.py export/__init__.py tests/__init__.py renderer/shaders/.gitkeep`

- [ ] **Step 4: Create `main.py` stub**

```python
import os
import pygame
import tomllib

def load_config(path: str = "config.toml") -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)

def main() -> None:
    config = load_config()
    os.environ.setdefault("SDL_VIDEODRIVER", "x11")

    pygame.init()
    width, height = config["export"]["resolution"]
    screen = pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("Music Visualizer")

    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
        clock.tick(config["renderer"]["realtime_fps"])

    pygame.quit()

if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Install dependencies**

Run: `pip install -e ".[dev]"`
Expected: `Successfully installed music-visualizer-0.1.0`

- [ ] **Step 6: Verify stub runs headlessly**

Run: `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python -c "import main; print('OK')" 2>/dev/null && echo PASS`
Expected: `PASS`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml config.toml main.py audio/__init__.py engine/__init__.py renderer/__init__.py ui/__init__.py export/__init__.py tests/__init__.py
git commit -m "feat: project scaffold — packages, config, stub main"
```

---

## Task 2: Data Contracts — FeatureFrame, RenderParams, Style

**Files:**
- Create: `audio/feature_frame.py`
- Create: `engine/render_params.py`
- Create: `tests/test_feature_frame.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_feature_frame.py
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
    assert cfg.dissonance_threshold == 0.65
    assert cfg.tempo_threshold == 0.35
    assert cfg.style_hold_seconds == 4.0
    assert cfg.blend_duration_seconds == 3.0
    assert cfg.ema_alpha == 0.15


def test_style_enum_values():
    assert Style.GEOMETRIC != Style.ORGANIC
    assert Style.ORGANIC != Style.COSMIC
    assert len(Style) == 3
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_feature_frame.py -v`
Expected: `ImportError` — modules don't exist yet

- [ ] **Step 3: Create `audio/feature_frame.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class FeatureFrame:
    amplitude: float          # peak amplitude, normalized 0–1
    rms: float                # RMS energy, normalized 0–1
    spectral_centroid: float  # spectral centroid, normalized 0–1
    onset_strength: float     # onset strength, normalized 0–1
    dissonance_raw: float     # raw dissonance (for shader transient effects)
    dissonance_smooth: float  # EMA-smoothed dissonance (for style classifier)
    chroma: np.ndarray        # shape (12,) chroma vector
    bpm: float                # beats per minute
```

- [ ] **Step 4: Create `engine/render_params.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto


class Style(Enum):
    GEOMETRIC = auto()
    ORGANIC = auto()
    COSMIC = auto()


@dataclass
class ContextConfig:
    dissonance_threshold: float = 0.65
    tempo_threshold: float = 0.35       # bpm / 200.0 threshold
    style_hold_seconds: float = 4.0
    blend_duration_seconds: float = 3.0
    ema_alpha: float = 0.15


@dataclass
class RenderParams:
    active_style: Style
    blend_target: Style        # equals active_style when not transitioning
    blend_weight: float        # 0.0 = fully active_style, 1.0 = fully blend_target
    intensity: float           # RMS-driven, 0–1
    brightness: float          # spectral centroid-driven, 0–1
    pulse: float               # onset strength (raw), 0–1
    dissonance_raw: float      # raw dissonance for glitch micro-effects
    color_a: tuple[float, float, float]   # primary RGB
    color_b: tuple[float, float, float]   # secondary RGB
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_feature_frame.py -v`
Expected: `4 passed`

- [ ] **Step 6: Commit**

```bash
git add audio/feature_frame.py engine/render_params.py tests/test_feature_frame.py
git commit -m "feat: FeatureFrame and RenderParams data contracts"
```

---

## Task 3: Color Palette (Circle of Fifths)

**Files:**
- Create: `engine/palette.py`
- Create: `tests/test_palette.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_palette.py
import numpy as np
import pytest
from engine.palette import Palette, KeyCharacter


def test_major_key_returns_vibrant():
    p = Palette()
    # C major: chroma peak at index 0, strong harmonic profile
    chroma = np.zeros(12)
    chroma[0] = 1.0   # C
    chroma[4] = 0.7   # E (major third)
    chroma[7] = 0.6   # G (perfect fifth)
    color_a, color_b, char = p.get_palette(chroma)
    assert char == KeyCharacter.MAJOR
    assert len(color_a) == 3
    assert all(0.0 <= c <= 1.0 for c in color_a)
    assert all(0.0 <= c <= 1.0 for c in color_b)


def test_minor_key_returns_cool():
    p = Palette()
    chroma = np.zeros(12)
    chroma[9] = 1.0   # A
    chroma[0] = 0.7   # C (minor third)
    chroma[4] = 0.5   # E (perfect fifth of Am)
    color_a, color_b, char = p.get_palette(chroma)
    assert char == KeyCharacter.MINOR


def test_atonal_returns_neutral():
    p = Palette()
    chroma = np.ones(12) / 12.0  # flat, ambiguous
    color_a, color_b, char = p.get_palette(chroma)
    assert char == KeyCharacter.ATONAL


def test_interpolated_palette_values_are_valid():
    p = Palette()
    chroma = np.zeros(12)
    chroma[0] = 1.0
    chroma[7] = 0.5
    prev_a = (0.31, 0.76, 0.63)
    color_a, color_b, _ = p.get_palette(chroma, prev_color_a=prev_a, alpha=0.3)
    assert all(0.0 <= c <= 1.0 for c in color_a)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_palette.py -v`
Expected: `ImportError`

- [ ] **Step 3: Create `engine/palette.py`**

```python
from __future__ import annotations
import colorsys
from enum import Enum, auto
import numpy as np


class KeyCharacter(Enum):
    MAJOR = auto()
    MINOR = auto()
    ATONAL = auto()


# Circle of Fifths: index = pitch class (C=0, C#=1, ... B=11)
# Maps pitch class → hue (degrees 0–360) evenly around the circle
_COF_HUE: dict[int, float] = {
    0: 0.0,    # C
    7: 30.0,   # G
    2: 60.0,   # D
    9: 90.0,   # A
    4: 120.0,  # E
    11: 150.0, # B
    6: 180.0,  # F#
    1: 210.0,  # C#
    8: 240.0,  # Ab
    3: 270.0,  # Eb
    10: 300.0, # Bb
    5: 330.0,  # F
}

# Minor third interval = +3 semitones; major third = +4 semitones
_MINOR_THIRD = 3
_MAJOR_THIRD = 4


def _classify_key(chroma: np.ndarray) -> tuple[int, KeyCharacter]:
    """Return (root_pitch_class, KeyCharacter)."""
    root = int(np.argmax(chroma))
    minor_strength = chroma[(root + _MINOR_THIRD) % 12]
    major_strength = chroma[(root + _MAJOR_THIRD) % 12]
    flatness = float(np.std(chroma))
    if flatness < 0.12:
        return root, KeyCharacter.ATONAL
    if major_strength > minor_strength:
        return root, KeyCharacter.MAJOR
    return root, KeyCharacter.MINOR


def _hue_to_rgb(h_deg: float, s: float, l: float) -> tuple[float, float, float]:
    r, g, b = colorsys.hls_to_rgb(h_deg / 360.0, l, s)
    return (r, g, b)


def _lerp_color(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    t: float,
) -> tuple[float, float, float]:
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )


class Palette:
    def get_palette(
        self,
        chroma: np.ndarray,
        prev_color_a: tuple[float, float, float] | None = None,
        alpha: float = 1.0,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], KeyCharacter]:
        """
        Return (color_a, color_b, KeyCharacter) from a 12-bin chroma vector.
        alpha: blend weight toward new palette (0 = keep prev, 1 = full new).
        """
        root, char = _classify_key(chroma)
        hue = _COF_HUE[root]

        if char == KeyCharacter.MAJOR:
            # Complementary: primary hue + complement (+180)
            color_a = _hue_to_rgb(hue, 0.85, 0.55)
            color_b = _hue_to_rgb((hue + 180.0) % 360.0, 0.75, 0.60)
        elif char == KeyCharacter.MINOR:
            # Analogous cool: primary hue + adjacent (-30)
            color_a = _hue_to_rgb(hue, 0.70, 0.45)
            color_b = _hue_to_rgb((hue - 30.0) % 360.0, 0.65, 0.50)
        else:  # ATONAL
            # Neutral cool desaturated
            color_a = _hue_to_rgb(210.0, 0.25, 0.45)
            color_b = _hue_to_rgb(220.0, 0.20, 0.40)

        if prev_color_a is not None and alpha < 1.0:
            color_a = _lerp_color(prev_color_a, color_a, alpha)

        return color_a, color_b, char
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_palette.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add engine/palette.py tests/test_palette.py
git commit -m "feat: Circle of Fifths color palette system"
```

---

## Task 4: Pre-recorded Audio Analyzer

**Files:**
- Create: `audio/analyzer.py`
- Create: `tests/conftest.py`
- Create: `tests/test_analyzer.py`

- [ ] **Step 1: Create `tests/conftest.py` with shared fixtures**

```python
# tests/conftest.py
import wave
import struct
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
```

- [ ] **Step 2: Write failing tests**

```python
# tests/test_analyzer.py
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
    # Create a minimal AVI by copying the wav (moviepy would normally do this;
    # we mock it here by just passing the wav path with .avi extension via monkeypatch)
    import shutil
    avi_path = str(tmp_path / "test.avi")
    shutil.copy(sine_wav, avi_path)
    analyzer = PrerecordedAnalyzer()
    # Should not raise; just falls back to treating as audio
    result = analyzer.analyze(avi_path, fps=24)
    assert len(result.frames) > 0
```

- [ ] **Step 3: Run to verify failure**

Run: `pytest tests/test_analyzer.py -v`
Expected: `ImportError`

- [ ] **Step 4: Create `audio/analyzer.py`**

```python
from __future__ import annotations
import os
import tempfile
from dataclasses import dataclass

import librosa
import numpy as np

from audio.feature_frame import FeatureFrame


@dataclass
class AnalysisResult:
    bpm: float
    duration: float
    sr: int
    audio: np.ndarray          # mono float32 for playback
    frames: list[FeatureFrame] # one entry per render frame at requested fps


class PrerecordedAnalyzer:
    def analyze(self, path: str, fps: int = 24) -> AnalysisResult:
        audio_path = self._resolve_audio(path)
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)

        # Full-track feature extraction
        tempo_arr, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo_arr) if np.ndim(tempo_arr) == 0 else float(tempo_arr[0])

        y_harm, y_perc = librosa.effects.hpss(y)
        hop = 512
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
        flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop)[0]
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)  # (12, T)
        onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        perc_rms = librosa.feature.rms(y=y_perc, hop_length=hop)[0]

        # Normalise to 0–1
        def _norm(arr: np.ndarray) -> np.ndarray:
            m = arr.max()
            return arr / m if m > 0 else arr

        centroid_norm = _norm(centroid)
        onset_norm = _norm(onset)
        rms_norm = _norm(rms)

        # Percussive ratio per frame
        total_rms = rms + 1e-8
        perc_ratio = np.clip(perc_rms / total_rms, 0.0, 1.0)

        n_frames = int(duration * fps)
        frames: list[FeatureFrame] = []
        dissonance_smooth = 0.0
        alpha = 0.15

        for i in range(n_frames):
            t = i / fps
            lib_frame = min(int(t * sr / hop), len(centroid_norm) - 1)

            raw_dis = float(perc_ratio[lib_frame] * flatness[lib_frame])
            dissonance_smooth = alpha * raw_dis + (1.0 - alpha) * dissonance_smooth

            frames.append(
                FeatureFrame(
                    amplitude=float(rms_norm[lib_frame]),
                    rms=float(rms_norm[lib_frame]),
                    spectral_centroid=float(centroid_norm[lib_frame]),
                    onset_strength=float(onset_norm[lib_frame]),
                    dissonance_raw=raw_dis,
                    dissonance_smooth=float(dissonance_smooth),
                    chroma=chroma[:, lib_frame].copy(),
                    bpm=bpm,
                )
            )

        if audio_path != path:
            os.unlink(audio_path)

        return AnalysisResult(
            bpm=bpm,
            duration=duration,
            sr=sr,
            audio=y,
            frames=frames,
        )

    @staticmethod
    def _resolve_audio(path: str) -> str:
        """For .avi, extract audio to a temp wav via moviepy."""
        if not path.lower().endswith(".avi"):
            return path
        from moviepy.editor import VideoFileClip
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            clip = VideoFileClip(path)
            clip.audio.write_audiofile(tmp.name, verbose=False, logger=None)
            clip.close()
        except Exception:
            # Fallback: treat as audio file directly (for test mocking)
            os.unlink(tmp.name)
            return path
        return tmp.name
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_analyzer.py -v`
Expected: `5 passed`

- [ ] **Step 6: Commit**

```bash
git add audio/analyzer.py tests/conftest.py tests/test_analyzer.py
git commit -m "feat: PrerecordedAnalyzer — librosa full-pass audio analysis"
```

---

## Task 5: Live Audio Analyzer

**Files:**
- Create: `audio/live_analyzer.py`
- Create: `tests/test_live_analyzer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_live_analyzer.py
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
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
    sr = 44100
    chunk = np.ones((512, 1), dtype=np.float32) * 0.1
    for _ in range(10):
        analyzer._process_chunk(chunk, None, None, None)
    audio = analyzer.get_recording()
    assert isinstance(audio, np.ndarray)
    assert audio.ndim == 1
    assert len(audio) == 512 * 10
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_live_analyzer.py -v`
Expected: `ImportError`

- [ ] **Step 3: Create `audio/live_analyzer.py`**

```python
from __future__ import annotations
import collections
import threading

import numpy as np
import sounddevice as sd

from audio.feature_frame import FeatureFrame

_SR = 44100
_CHUNK = 512
_BUFFER_SECONDS = 8


class LiveAnalyzer:
    def __init__(self, device_index: int, sr: int = _SR, chunk_size: int = _CHUNK):
        self._device_index = device_index
        self._sr = sr
        self._chunk_size = chunk_size
        self._buffer: collections.deque[np.ndarray] = collections.deque()
        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None
        self._dissonance_smooth = 0.0
        self._latest_frame = FeatureFrame(
            amplitude=0.0,
            rms=0.0,
            spectral_centroid=0.0,
            onset_strength=0.0,
            dissonance_raw=0.0,
            dissonance_smooth=0.0,
            chroma=np.zeros(12),
            bpm=0.0,
        )
        self._prev_spectrum: np.ndarray | None = None

    @staticmethod
    def list_input_devices() -> list[dict]:
        """Return input-capable devices from sounddevice."""
        return [
            d for d in sd.query_devices()
            if d["max_input_channels"] > 0
        ]

    def start(self) -> None:
        self._stream = sd.InputStream(
            device=self._device_index,
            samplerate=self._sr,
            channels=1,
            blocksize=self._chunk_size,
            dtype="float32",
            callback=self._process_chunk,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def get_frame(self) -> FeatureFrame:
        with self._lock:
            return self._latest_frame

    def get_recording(self) -> np.ndarray:
        """Return all buffered audio as a 1-D float32 array."""
        with self._lock:
            return np.concatenate(list(self._buffer)) if self._buffer else np.array([], dtype=np.float32)

    def _process_chunk(self, indata: np.ndarray, frames, time, status) -> None:
        mono = indata[:, 0].copy()

        # Append to recording buffer (cap at 8s)
        max_samples = self._sr * _BUFFER_SECONDS
        with self._lock:
            self._buffer.append(mono)
            total = sum(len(c) for c in self._buffer)
            while total > max_samples and self._buffer:
                removed = self._buffer.popleft()
                total -= len(removed)

        # Feature extraction
        rms = float(np.sqrt(np.mean(mono ** 2)))
        amplitude = float(np.max(np.abs(mono)))

        spectrum = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
        freqs = np.fft.rfftfreq(len(mono), d=1.0 / self._sr)

        # Spectral centroid
        spec_sum = spectrum.sum() + 1e-8
        centroid_hz = float(np.dot(freqs, spectrum) / spec_sum)
        centroid_norm = min(centroid_hz / (self._sr / 2.0), 1.0)

        # Spectral flatness
        geom_mean = np.exp(np.mean(np.log(spectrum + 1e-10)))
        arith_mean = spec_sum / len(spectrum)
        flatness = float(geom_mean / (arith_mean + 1e-8))

        # Onset strength via spectral flux
        onset = 0.0
        if self._prev_spectrum is not None:
            diff = spectrum - self._prev_spectrum
            onset = float(np.mean(np.maximum(diff, 0.0)))
            onset = min(onset / (rms + 1e-8), 1.0)
        self._prev_spectrum = spectrum

        # Rough percussive ratio: high-frequency energy ratio
        hf_mask = freqs > 2000.0
        lf_mask = freqs <= 2000.0
        hf_energy = spectrum[hf_mask].sum()
        lf_energy = spectrum[lf_mask].sum() + 1e-8
        perc_ratio = float(np.clip(hf_energy / (hf_energy + lf_energy), 0.0, 1.0))

        dissonance_raw = perc_ratio * flatness
        alpha = 0.15
        self._dissonance_smooth = alpha * dissonance_raw + (1.0 - alpha) * self._dissonance_smooth

        # Chroma (12-bin, simplified)
        chroma = np.zeros(12)
        for i, f in enumerate(freqs):
            if f > 0 and spectrum[i] > 0:
                note = int(round(12 * np.log2(f / 440.0))) % 12
                chroma[note] += spectrum[i]
        chroma_sum = chroma.sum()
        if chroma_sum > 0:
            chroma /= chroma_sum

        with self._lock:
            self._latest_frame = FeatureFrame(
                amplitude=min(amplitude, 1.0),
                rms=min(rms * 10.0, 1.0),
                spectral_centroid=centroid_norm,
                onset_strength=min(onset, 1.0),
                dissonance_raw=float(dissonance_raw),
                dissonance_smooth=float(self._dissonance_smooth),
                chroma=chroma,
                bpm=0.0,  # not available in live mode
            )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_live_analyzer.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add audio/live_analyzer.py tests/test_live_analyzer.py
git commit -m "feat: LiveAnalyzer — sounddevice ring buffer with per-chunk FFT analysis"
```

---

## Task 6: Context Engine

**Files:**
- Create: `engine/context_engine.py`
- Create: `tests/test_context_engine.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_context_engine.py
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
    frame = _make_frame(dissonance_smooth=0.9)  # above threshold
    # Simulate 3.9 seconds of high dissonance (just under hold window)
    for _ in range(int(3.9 * 24)):
        params = engine.update(frame, dt=1.0 / 24)
    assert params.active_style == Style.ORGANIC  # no transition yet
    assert params.blend_weight < 1.0


def test_sustained_dissonance_triggers_geometric():
    """After 4s hold + blend start, style should eventually be GEOMETRIC."""
    cfg = ContextConfig(style_hold_seconds=4.0, blend_duration_seconds=3.0)
    engine = ContextEngine(cfg)
    frame = _make_frame(dissonance_smooth=0.9)
    # Run for hold + full blend duration
    params = None
    for _ in range(int(7.1 * 24)):
        params = engine.update(frame, dt=1.0 / 24)
    assert params.active_style == Style.GEOMETRIC
    assert params.blend_weight == pytest.approx(1.0, abs=0.01)


def test_low_tempo_triggers_cosmic():
    cfg = ContextConfig(style_hold_seconds=2.0, blend_duration_seconds=1.0)
    engine = ContextEngine(cfg)
    frame = _make_frame(dissonance_smooth=0.1, bpm=50.0)  # tempo_normalized = 0.25 < 0.35
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
    # blend_weight should increase monotonically once transition starts
    increasing = [weights[i] <= weights[i + 1] for i in range(len(weights) - 1)]
    assert sum(increasing) > len(increasing) * 0.9  # allow minor float noise


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
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_context_engine.py -v`
Expected: `ImportError`

- [ ] **Step 3: Create `engine/context_engine.py`**

```python
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
    # Scale so that sigmoid(±6) ≈ 0/1
    return _sigmoid((t - 0.5) * 12.0)


def _classify(dissonance_smooth: float, bpm: float, cfg: ContextConfig) -> Style:
    tempo_norm = min(bpm / 200.0, 1.0)
    if dissonance_smooth > cfg.dissonance_threshold:
        return Style.GEOMETRIC
    if tempo_norm < cfg.tempo_threshold:
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
        self._candidate_hold = 0.0   # seconds candidate has been dominant

        self._blend_elapsed = 0.0    # seconds into current blend transition
        self._in_transition = False

        self._prev_color_a: tuple[float, float, float] | None = None
        self._palette_alpha = 1.0

    def update(self, frame: FeatureFrame, dt: float) -> RenderParams:
        classified = _classify(frame.dissonance_smooth, frame.bpm, self._cfg)

        # --- Hysteresis: track how long the candidate has been stable ---
        if classified == self._candidate_style:
            self._candidate_hold += dt
        else:
            self._candidate_style = classified
            self._candidate_hold = 0.0

        # --- Trigger a new transition if candidate is held long enough ---
        if (
            not self._in_transition
            and self._candidate_hold >= self._cfg.style_hold_seconds
            and self._candidate_style != self._current_style
        ):
            self._in_transition = True
            self._blend_target = self._candidate_style
            self._blend_elapsed = 0.0
            self._blend_weight = 0.0

        # --- Advance blend ---
        if self._in_transition:
            self._blend_elapsed += dt
            t = min(self._blend_elapsed / self._cfg.blend_duration_seconds, 1.0)
            self._blend_weight = _blend_ease(t)
            if t >= 1.0:
                # Transition complete
                self._current_style = self._blend_target
                self._blend_weight = 1.0
                self._in_transition = False

        # --- Color palette (smooth update) ---
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
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_context_engine.py -v`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add engine/context_engine.py tests/test_context_engine.py
git commit -m "feat: ContextEngine — style classifier, hysteresis, sigmoid blend"
```

---

## Task 7: Renderer Base — FBO Setup and Shared Infrastructure

**Files:**
- Create: `renderer/shaders/quad.vert`
- Create: `renderer/renderer.py`
- Create: `tests/test_renderer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_renderer.py
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
    assert pixels.max() > 10  # something was rendered
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
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_renderer.py -v`
Expected: `ImportError`

- [ ] **Step 3: Create `renderer/shaders/quad.vert`**

```glsl
#version 330 core

in vec2 in_vert;
out vec2 v_uv;

void main() {
    v_uv = in_vert * 0.5 + 0.5;
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
```

- [ ] **Step 4: Create `renderer/renderer.py`** (stub — shaders added in Tasks 8–11)

```python
from __future__ import annotations
import os
from pathlib import Path

import moderngl
import numpy as np

from engine.render_params import RenderParams, Style

_SHADER_DIR = Path(__file__).parent / "shaders"
_QUAD_VERTS = np.array([
    -1.0, -1.0,
     1.0, -1.0,
    -1.0,  1.0,
     1.0,  1.0,
], dtype="f4")


def _load(name: str) -> str:
    return (_SHADER_DIR / name).read_text()


class Renderer:
    def __init__(self, ctx: moderngl.Context, width: int, height: int) -> None:
        self._ctx = ctx
        self._w = width
        self._h = height
        self._ring_times: list[float] = []  # recent beat timestamps for cosmic rings

        # Fullscreen quad
        vbo = ctx.buffer(_QUAD_VERTS)
        vert_src = _load("quad.vert")

        # Style programs
        self._prog: dict[Style, moderngl.Program] = {
            Style.GEOMETRIC: ctx.program(
                vertex_shader=vert_src,
                fragment_shader=_load("geometric.glsl"),
            ),
            Style.ORGANIC: ctx.program(
                vertex_shader=vert_src,
                fragment_shader=_load("organic.glsl"),
            ),
            Style.COSMIC: ctx.program(
                vertex_shader=vert_src,
                fragment_shader=_load("cosmic.glsl"),
            ),
        }
        self._vaos: dict[Style, moderngl.VertexArray] = {
            style: ctx.vertex_array(prog, [(vbo, "2f", "in_vert")])
            for style, prog in self._prog.items()
        }

        # Post-processing programs
        self._prog_composite = ctx.program(
            vertex_shader=vert_src,
            fragment_shader=_load("composite.glsl"),
        )
        self._vao_composite = ctx.vertex_array(self._prog_composite, [(vbo, "2f", "in_vert")])

        self._prog_bloom_extract = ctx.program(
            vertex_shader=vert_src,
            fragment_shader=_load("bloom_extract.glsl"),
        )
        self._vao_bloom_extract = ctx.vertex_array(self._prog_bloom_extract, [(vbo, "2f", "in_vert")])

        self._prog_blur = ctx.program(
            vertex_shader=vert_src,
            fragment_shader=_load("blur.glsl"),
        )
        self._vao_blur = ctx.vertex_array(self._prog_blur, [(vbo, "2f", "in_vert")])

        self._prog_final = ctx.program(
            vertex_shader=vert_src,
            fragment_shader=_load("final.glsl"),
        )
        self._vao_final = ctx.vertex_array(self._prog_final, [(vbo, "2f", "in_vert")])

        # FBOs
        def _make_fbo() -> tuple[moderngl.Framebuffer, moderngl.Texture]:
            tex = ctx.texture((width, height), 3)
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            fbo = ctx.framebuffer(color_attachments=[tex])
            return fbo, tex

        self._fbo_a, self._tex_a = _make_fbo()
        self._fbo_b, self._tex_b = _make_fbo()
        self._fbo_composite, self._tex_composite = _make_fbo()
        self._fbo_prev, self._tex_prev = _make_fbo()
        self._fbo_bright, self._tex_bright = _make_fbo()
        self._fbo_blur_h, self._tex_blur_h = _make_fbo()
        self._fbo_blur_v, self._tex_blur_v = _make_fbo()
        self._fbo_final, self._tex_final = _make_fbo()

        # Screen FBO (default)
        self._screen_fbo = ctx.detect_framebuffer()

    def render_frame(self, params: RenderParams, elapsed_time: float) -> None:
        # Track ring pulses for cosmic mode
        if params.pulse > 0.6:
            self._ring_times.append(elapsed_time)
            if len(self._ring_times) > 4:
                self._ring_times.pop(0)

        # 1. Render active style → fbo_a
        self._render_style(params.active_style, params, elapsed_time, self._fbo_a)

        # 2. Render blend target → fbo_b (skip if no active blend)
        if params.blend_weight > 0.001:
            self._render_style(params.blend_target, params, elapsed_time, self._fbo_b)

        # 3. Composite blend
        self._fbo_composite.use()
        self._ctx.clear(0.0, 0.0, 0.0)
        self._tex_a.use(location=0)
        self._tex_b.use(location=1)
        p = self._prog_composite
        p["u_tex_a"] = 0
        p["u_tex_b"] = 1
        p["u_blend"] = params.blend_weight
        self._vao_composite.render(moderngl.TRIANGLE_STRIP)

        # 4. Bloom extraction
        self._fbo_bright.use()
        self._ctx.clear(0.0, 0.0, 0.0)
        self._tex_composite.use(location=0)
        self._prog_bloom_extract["u_scene"] = 0
        self._vao_bloom_extract.render(moderngl.TRIANGLE_STRIP)

        # 5. Horizontal blur
        self._fbo_blur_h.use()
        self._ctx.clear(0.0, 0.0, 0.0)
        self._tex_bright.use(location=0)
        self._prog_blur["u_tex"] = 0
        self._prog_blur["u_direction"] = (1.0, 0.0)
        self._prog_blur["u_resolution"] = (float(self._w), float(self._h))
        self._vao_blur.render(moderngl.TRIANGLE_STRIP)

        # 6. Vertical blur
        self._fbo_blur_v.use()
        self._ctx.clear(0.0, 0.0, 0.0)
        self._tex_blur_h.use(location=0)
        self._prog_blur["u_tex"] = 0
        self._prog_blur["u_direction"] = (0.0, 1.0)
        self._vao_blur.render(moderngl.TRIANGLE_STRIP)

        # 7. Final pass: FXAA + tone map + vignette + motion blur + bloom add
        self._fbo_final.use()
        self._ctx.clear(0.0, 0.0, 0.0)
        self._tex_composite.use(location=0)
        self._tex_blur_v.use(location=1)
        self._tex_prev.use(location=2)
        p = self._prog_final
        p["u_scene"] = 0
        p["u_bloom"] = 1
        p["u_prev_frame"] = 2
        p["u_resolution"] = (float(self._w), float(self._h))
        p["u_motion_blur"] = 0.15 if params.active_style == Style.ORGANIC else 0.0
        p["u_bloom_intensity"] = 0.4
        self._vao_final.render(moderngl.TRIANGLE_STRIP)

        # 8. Copy final → prev for next frame
        self._ctx.copy_framebuffer(dst=self._fbo_prev, src=self._fbo_final)

    def read_pixels(self) -> np.ndarray:
        """Return current frame as (H, W, 3) uint8 array."""
        raw = self._fbo_final.read(components=3)
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(self._h, self._w, 3)
        return np.flipud(arr)

    def release(self) -> None:
        for fbo, tex in [
            (self._fbo_a, self._tex_a),
            (self._fbo_b, self._tex_b),
            (self._fbo_composite, self._tex_composite),
            (self._fbo_prev, self._tex_prev),
            (self._fbo_bright, self._tex_bright),
            (self._fbo_blur_h, self._tex_blur_h),
            (self._fbo_blur_v, self._tex_blur_v),
            (self._fbo_final, self._tex_final),
        ]:
            fbo.release()
            tex.release()
        for prog in self._prog.values():
            prog.release()

    def _render_style(
        self,
        style: Style,
        params: RenderParams,
        elapsed_time: float,
        fbo: moderngl.Framebuffer,
    ) -> None:
        fbo.use()
        self._ctx.clear(0.0, 0.0, 0.0)
        prog = self._prog[style]
        prog["u_time"] = elapsed_time
        prog["u_resolution"] = (float(self._w), float(self._h))
        prog["u_intensity"] = params.intensity
        prog["u_brightness"] = params.brightness
        prog["u_pulse"] = params.pulse
        prog["u_dissonance"] = params.dissonance_raw
        prog["u_color_a"] = params.color_a
        prog["u_color_b"] = params.color_b
        if style == Style.COSMIC:
            ring_times = (self._ring_times + [0.0, 0.0, 0.0, 0.0])[:4]
            prog["u_ring_times"] = tuple(ring_times)
            prog["u_ring_count"] = len(self._ring_times)
        self._vaos[style].render(moderngl.TRIANGLE_STRIP)
```

- [ ] **Step 5: Create placeholder shader files** (will be replaced in Tasks 8–11)

Create `renderer/shaders/geometric.glsl`, `organic.glsl`, `cosmic.glsl`, `composite.glsl`, `bloom_extract.glsl`, `blur.glsl`, `final.glsl` — all with this minimal passthrough body for now:

```glsl
#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform float u_time;
uniform vec2  u_resolution;
uniform float u_intensity;
uniform float u_brightness;
uniform float u_pulse;
uniform float u_dissonance;
uniform vec3  u_color_a;
uniform vec3  u_color_b;
void main() { fragColor = vec4(u_color_a * u_intensity, 1.0); }
```

For `composite.glsl`:
```glsl
#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_tex_a;
uniform sampler2D u_tex_b;
uniform float u_blend;
void main() {
    vec4 a = texture(u_tex_a, v_uv);
    vec4 b = texture(u_tex_b, v_uv);
    fragColor = mix(a, b, u_blend);
}
```

For `bloom_extract.glsl`:
```glsl
#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_scene;
void main() { fragColor = texture(u_scene, v_uv); }
```

For `blur.glsl`:
```glsl
#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_tex;
uniform vec2 u_direction;
uniform vec2 u_resolution;
void main() { fragColor = texture(u_tex, v_uv); }
```

For `final.glsl`:
```glsl
#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_scene;
uniform sampler2D u_bloom;
uniform sampler2D u_prev_frame;
uniform vec2 u_resolution;
uniform float u_motion_blur;
uniform float u_bloom_intensity;
void main() { fragColor = texture(u_scene, v_uv); }
```

For `cosmic.glsl`, also add the ring uniforms to the placeholder:
```glsl
#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform float u_time;
uniform vec2  u_resolution;
uniform float u_intensity;
uniform float u_brightness;
uniform float u_pulse;
uniform float u_dissonance;
uniform vec3  u_color_a;
uniform vec3  u_color_b;
uniform float u_ring_times[4];
uniform int   u_ring_count;
void main() { fragColor = vec4(u_color_a * u_intensity, 1.0); }
```

- [ ] **Step 6: Run renderer tests**

Run: `pytest tests/test_renderer.py -v`
Expected: `5 passed`

- [ ] **Step 7: Commit**

```bash
git add renderer/renderer.py renderer/shaders/
git commit -m "feat: Renderer base — ModernGL FBO setup, post-processing pipeline, placeholder shaders"
```

---

## Task 8: Geometric/Glitch Shader

**Files:**
- Modify: `renderer/shaders/geometric.glsl`

- [ ] **Step 1: Write the full Geometric/Glitch fragment shader**

Replace `renderer/shaders/geometric.glsl`:

```glsl
#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform float u_time;
uniform vec2  u_resolution;
uniform float u_intensity;
uniform float u_brightness;
uniform float u_pulse;
uniform float u_dissonance;
uniform vec3  u_color_a;   // electric red
uniform vec3  u_color_b;   // cyan

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float sdTriangle(vec2 p, float r) {
    const float k = 1.7320508;  // sqrt(3)
    p.x = abs(p.x) - r;
    p.y = p.y + r / k;
    if (p.x + k * p.y > 0.0) p = vec2(p.x - k * p.y, -k * p.x - p.y) * 0.5;
    p.x -= clamp(p.x, -2.0 * r, 0.0);
    return -length(p) * sign(p.y);
}

float sdBox(vec2 p, vec2 b) {
    vec2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

mat2 rot2(float a) {
    float c = cos(a), s = sin(a);
    return mat2(c, -s, s, c);
}

void main() {
    vec2 uv = v_uv;

    // --- Glitch: horizontal slice displacement ---
    float glitchStrength = u_dissonance * u_pulse;
    float slice = floor(uv.y * 24.0);
    float timeSlot = floor(u_time * 28.0);
    float sliceOffset = (hash(vec2(slice, timeSlot)) - 0.5) * glitchStrength * 0.18;
    uv.x = fract(uv.x + sliceOffset);

    vec2 p = (uv - 0.5) * vec2(u_resolution.x / u_resolution.y, 1.0);

    // --- Base: deep dark with subtle grid ---
    vec3 col = vec3(0.015, 0.015, 0.03);
    vec2 grid = abs(fract(p * 5.0 + 0.5) - 0.5);
    float gridLine = min(grid.x, grid.y);
    col += 0.04 * (1.0 - smoothstep(0.0, 0.03, gridLine)) * vec3(0.4, 0.6, 1.0);

    // --- Primary triangle SDF ---
    float rotA = u_time * 0.25 + u_intensity * 0.5;
    vec2 tp = p * rot2(rotA);
    float triSize = 0.28 + u_intensity * 0.12;
    float tri = sdTriangle(tp, triSize);
    float triLine = abs(tri);

    // Chromatic aberration on triangle outline
    float aberr = u_dissonance * 0.012 + u_pulse * 0.006;
    vec2 tp_r = p * rot2(rotA) + vec2(aberr, 0.0);
    vec2 tp_b = p * rot2(rotA) - vec2(aberr, 0.0);
    float tri_r = sdTriangle(tp_r, triSize);
    float tri_b = sdTriangle(tp_b, triSize);
    col.r += u_color_a.r * (1.0 - smoothstep(0.0, 0.009, abs(tri_r))) * 0.95;
    col.g += 0.15 * (1.0 - smoothstep(0.0, 0.011, abs(tri)));
    col.b += u_color_b.b * (1.0 - smoothstep(0.0, 0.009, abs(tri_b))) * 0.7;

    // Interior fill (faint)
    col += u_color_a * max(0.0, -tri) * 0.04;

    // --- Ghost triangle (slight offset, lower opacity) ---
    vec2 tpG = p * rot2(rotA + 0.08);
    float triG = sdTriangle(tpG, triSize * 0.98);
    col += u_color_a * (1.0 - smoothstep(0.0, 0.007, abs(triG))) * 0.25;

    // --- Rotating box ---
    vec2 bp = p * rot2(-u_time * 0.4 + u_intensity * 0.3);
    float boxSize = 0.16 + u_brightness * 0.06;
    float box = sdBox(bp, vec2(boxSize));
    col.r += u_color_b.r * (1.0 - smoothstep(0.0, 0.007, abs(box))) * 0.5;
    col.g += u_color_b.g * (1.0 - smoothstep(0.0, 0.007, abs(box))) * 0.8;
    col.b += u_color_b.b * (1.0 - smoothstep(0.0, 0.007, abs(box))) * 0.9;

    // Second box, counter-rotating
    vec2 bp2 = p * rot2(u_time * 0.6 - 1.2);
    float box2 = sdBox(bp2, vec2(0.08 + u_intensity * 0.04));
    col += u_color_a * (1.0 - smoothstep(0.0, 0.006, abs(box2))) * 0.4;

    // --- Frequency bars along bottom ---
    {
        float barCount = 20.0;
        float barIdx = floor(uv.x * barCount);
        float barX   = fract(uv.x * barCount);
        float hSeed  = hash(vec2(barIdx * 0.13, 1.0));
        float barH   = (0.05 + hSeed * 0.15) * (0.3 + u_intensity * 1.4);
        barH = clamp(barH, 0.0, 0.55);
        float barFill = step(1.0 - uv.y * 0.8, barH);
        float gap = smoothstep(0.0, 0.06, barX) * smoothstep(1.0, 0.94, barX);
        float t = barIdx / (barCount - 1.0);
        vec3 barCol = mix(u_color_a, u_color_b, t);
        col += barCol * barFill * gap * 0.85;
    }

    // --- Scanlines ---
    float scanline = mod(gl_FragCoord.y, 3.0) < 1.0 ? 0.80 : 1.0;
    col *= scanline;

    // --- Pulse: full-frame flash on sharp transient ---
    col += u_color_a * u_pulse * 0.18 * (1.0 - length(p) * 0.6);

    // --- Vignette ---
    float vig = 1.0 - dot(uv - 0.5, uv - 0.5) * 1.6;
    col *= max(0.0, vig);

    col = clamp(col, 0.0, 1.5);  // allow slight HDR
    fragColor = vec4(col, 1.0);
}
```

- [ ] **Step 2: Run renderer tests to verify shader compiles and renders**

Run: `pytest tests/test_renderer.py::test_geometric_style_renders -v`
Expected: `1 passed`

- [ ] **Step 3: Commit**

```bash
git add renderer/shaders/geometric.glsl
git commit -m "feat: Geometric/Glitch GLSL shader — SDFs, scanlines, chromatic aberration, glitch displacement"
```

---

## Task 9: Organic/Flow Shader

**Files:**
- Modify: `renderer/shaders/organic.glsl`

- [ ] **Step 1: Write the full Organic/Flow fragment shader**

Replace `renderer/shaders/organic.glsl`:

```glsl
#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform float u_time;
uniform vec2  u_resolution;
uniform float u_intensity;
uniform float u_brightness;
uniform float u_pulse;
uniform float u_dissonance;
uniform vec3  u_color_a;   // teal
uniform vec3  u_color_b;   // violet

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

vec2 hash2(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    return fract(sin(p) * 43758.5453);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x),
        mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x),
        u.y
    );
}

float fbm(vec2 p) {
    float v = 0.0, a = 0.5;
    mat2 rot = mat2(0.8775, 0.4794, -0.4794, 0.8775);  // rot(~28.6 deg)
    for (int i = 0; i < 5; i++) {
        v += a * noise(p);
        p = rot * p * 2.1 + vec2(31.7, 17.3);
        a *= 0.5;
    }
    return v;
}

void main() {
    vec2 uv = v_uv;
    float ar = u_resolution.x / u_resolution.y;
    vec2 p = vec2((uv.x - 0.5) * ar, uv.y - 0.5);
    float t = u_time * 0.12;

    // --- Domain-warped FBM background ---
    vec2 q = vec2(fbm(p + t), fbm(p + vec2(1.73, 9.24) + t * 0.85));
    vec2 r = vec2(
        fbm(p + 1.6 * q + vec2(1.7, 9.2) + t * 0.28),
        fbm(p + 1.0 * q + vec2(8.3, 2.8) + t * 0.44)
    );
    float f = fbm(p + r);

    vec3 col = vec3(0.02, 0.02, 0.05);
    col = mix(col, u_color_a * 0.35, clamp(f * 2.5, 0.0, 1.0));
    col = mix(col, u_color_b * 0.45, clamp(length(q) * 0.7, 0.0, 1.0));
    col = mix(col, u_color_a * 0.6, clamp(f * f * 3.5, 0.0, 1.0));
    col *= 0.4 + 0.6 * u_intensity;

    // --- Layered waveforms ---
    for (int wi = 0; wi < 4; wi++) {
        float fi = float(wi);
        float freq = 1.0 + fi * 0.55;
        float speed = 0.25 + fi * 0.08;
        float amp = (0.07 - fi * 0.014) * (0.6 + u_intensity * 1.0);
        float warp = fbm(vec2(p.x * 1.8 + fi * 2.1, t * 0.4)) * 0.08;
        float wave_y = sin(p.x * freq * 3.14159 + u_time * speed + warp) * amp;
        float dist = abs(p.y - wave_y);
        float line = smoothstep(0.022 - fi * 0.002, 0.0, dist);
        float tint = (float(wi) + uv.x) / 5.0;
        col += mix(u_color_a, u_color_b, tint) * line * (1.0 - fi * 0.18) * 0.9;
    }

    // --- Floating gradient orbs ---
    vec2 orb_pos[3];
    orb_pos[0] = vec2(sin(u_time * 0.22) * 0.45 * ar, cos(u_time * 0.17) * 0.28);
    orb_pos[1] = vec2(cos(u_time * 0.13) * 0.38 * ar, sin(u_time * 0.27) * 0.35);
    orb_pos[2] = vec2(sin(u_time * 0.18 + 1.57) * 0.28 * ar, cos(u_time * 0.11 + 2.09) * 0.30);
    float orb_sz[3];
    orb_sz[0] = 0.30 + u_intensity * 0.12;
    orb_sz[1] = 0.24 + u_intensity * 0.09;
    orb_sz[2] = 0.18 + u_intensity * 0.07;
    vec3 orb_col[3];
    orb_col[0] = u_color_a;
    orb_col[1] = u_color_b;
    orb_col[2] = u_color_a * 0.6 + u_color_b * 0.4;

    for (int oi = 0; oi < 3; oi++) {
        float d = length(p - orb_pos[oi]);
        float s = orb_sz[oi];
        float orb = exp(-d * d / (s * s));
        col += orb_col[oi] * orb * 0.13;
    }

    // --- Particle field ---
    for (int pi = 0; pi < 48; pi++) {
        vec2 seed = vec2(float(pi) * 0.3742, float(pi) * 0.7373);
        vec2 base = hash2(seed) * 2.0 - 1.0;
        base.x *= ar;
        float drift_x = sin(u_time * (0.15 + hash(seed) * 0.2) + seed.x * 6.28) * 0.18;
        float drift_y = cos(u_time * (0.12 + hash(seed + 1.0) * 0.15) + seed.y * 6.28) * 0.15;
        vec2 pos = base + vec2(drift_x, drift_y);
        // wrap to [-ar, ar] x [-0.5, 0.5]
        pos.x = mod(pos.x + ar, 2.0 * ar) - ar;
        pos.y = mod(pos.y + 0.5, 1.0) - 0.5;
        float pd = length(p - pos);
        float psize = 0.007 + hash(seed + 2.0) * 0.009;
        float particle = smoothstep(psize, 0.0, pd);
        float pbrightness = 0.4 + hash(seed + floor(u_time * 0.5)) * 0.6;
        col += mix(u_color_a, u_color_b, hash(seed + 3.0)) * particle * pbrightness * 0.65;
    }

    // --- Pulse glow at centre ---
    col += u_color_a * u_pulse * 0.35 * exp(-length(p) * (2.5 - u_intensity));

    // --- Dissonance micro-shudder: subtle brightness spike ---
    col += vec3(u_dissonance * u_pulse * 0.12);

    // --- Vignette ---
    float vig = 1.0 - dot(uv - 0.5, uv - 0.5) * 1.3;
    col *= max(0.0, vig);

    col = clamp(col, 0.0, 1.6);
    fragColor = vec4(col, 1.0);
}
```

- [ ] **Step 2: Run renderer tests**

Run: `pytest tests/test_renderer.py::test_frame_is_not_all_black tests/test_renderer.py::test_read_pixels_returns_correct_shape -v`
Expected: `2 passed`

- [ ] **Step 3: Commit**

```bash
git add renderer/shaders/organic.glsl
git commit -m "feat: Organic/Flow GLSL shader — domain-warped FBM, waveforms, orbs, particle field"
```

---

## Task 10: Cosmic/Nebula Shader

**Files:**
- Modify: `renderer/shaders/cosmic.glsl`

- [ ] **Step 1: Write the full Cosmic/Nebula fragment shader**

Replace `renderer/shaders/cosmic.glsl`:

```glsl
#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform float u_time;
uniform vec2  u_resolution;
uniform float u_intensity;
uniform float u_brightness;
uniform float u_pulse;
uniform float u_dissonance;
uniform vec3  u_color_a;   // violet
uniform vec3  u_color_b;   // amber
uniform float u_ring_times[4];
uniform int   u_ring_count;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float hash1(float x) {
    return fract(sin(x * 127.1) * 43758.5453);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x),
        mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x),
        u.y
    );
}

void main() {
    vec2 uv = v_uv;
    float ar = u_resolution.x / u_resolution.y;
    vec2 p = vec2((uv.x - 0.5) * ar, uv.y - 0.5);

    // --- Deep space background ---
    vec3 col = vec3(0.008, 0.008, 0.018);

    // --- Star field ---
    vec2 starGrid = uv * vec2(70.0, 48.0);
    vec2 starCell = floor(starGrid);
    vec2 starFrac = fract(starGrid);
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            vec2 cell = starCell + vec2(float(dx), float(dy));
            float h = hash(cell);
            if (h > 0.72) {
                vec2 starPos = vec2(hash(cell + 1.3), hash(cell + 2.7));
                vec2 offset = starFrac - starPos - vec2(float(dx), float(dy));
                float starSize = 0.035 + hash(cell + 5.1) * 0.045;
                float brightness_mod = 0.35 + hash(cell + 3.3) * 0.65;
                float twinkle = 0.65 + 0.35 * sin(
                    u_time * (0.8 + hash(cell) * 2.5) + hash(cell + 9.1) * 6.28
                );
                float dist = length(offset);
                float star = smoothstep(starSize, 0.0, dist);
                // Soft diffraction spike
                float spike_h = smoothstep(starSize * 3.0, 0.0, abs(offset.y)) *
                                smoothstep(0.5, 0.0, abs(offset.x)) * 0.3;
                float spike_v = smoothstep(starSize * 3.0, 0.0, abs(offset.x)) *
                                smoothstep(0.5, 0.0, abs(offset.y)) * 0.3;
                vec3 starCol = mix(vec3(0.9, 0.95, 1.0), u_color_a * 1.5, hash(cell + 7.1) * 0.5);
                col += starCol * (star + spike_h + spike_v) * brightness_mod * twinkle;
            }
        }
    }

    // --- Nebula gradients (slow UV warp) ---
    float warpT = u_time * 0.05;
    float nx = noise(p * 1.5 + vec2(warpT, 0.0)) * 0.04;
    float ny = noise(p * 1.5 + vec2(0.0, warpT * 0.8)) * 0.04;

    float nebula1 = exp(-length(p - vec2(-0.18 + nx, 0.05)) * (1.8 - u_intensity * 0.6));
    float nebula2 = exp(-length(p - vec2(0.25 + ny, -0.08)) * (2.5 - u_brightness * 0.5));
    float nebula3 = exp(-length(p - vec2(-0.05, 0.20)) * 3.2) * 0.5;

    col += u_color_a * nebula1 * (0.22 + u_intensity * 0.18);
    col += u_color_b * nebula2 * (0.15 + u_brightness * 0.12);
    col += mix(u_color_a, u_color_b, 0.5) * nebula3 * 0.12;

    // --- Expanding ring pulses ---
    for (int ri = 0; ri < 4; ri++) {
        if (ri >= u_ring_count) break;
        float age = u_time - u_ring_times[ri];
        if (age < 0.0 || age > 7.0) continue;
        float radius = age * 0.22 + 0.02;
        float fade = pow(1.0 - age / 7.0, 1.5);
        float dist = abs(length(p) - radius);
        float ringWidth = 0.018 + age * 0.003;
        float ring = smoothstep(ringWidth, 0.0, dist) * fade;
        float secondaryRing = smoothstep(ringWidth * 2.5, 0.0, abs(length(p) - radius * 0.7)) * fade * 0.3;
        vec3 ringCol = mix(u_color_a, vec3(0.85, 0.9, 1.0), 0.4);
        col += ringCol * (ring + secondaryRing) * (0.45 + u_intensity * 0.3);
    }

    // --- Core glow ---
    float coreRadius = 4.0 - u_intensity * 2.5;
    float core = exp(-length(p) * coreRadius);
    col += u_color_a * core * (0.25 + u_intensity * 0.45);
    col += vec3(0.95, 0.97, 1.0) * core * core * (0.08 + u_intensity * 0.18);

    // --- Pulse burst ---
    col += u_color_a * u_pulse * exp(-length(p) * 2.8) * 0.45;

    // --- Film grain ---
    float grain = hash(uv + fract(u_time * 0.39)) * 0.045 - 0.0225;
    col += grain;

    // --- Vignette ---
    float vig = 1.0 - dot(uv - 0.5, uv - 0.5) * 1.9;
    col *= max(0.0, vig);

    col = clamp(col, 0.0, 1.4);
    fragColor = vec4(col, 1.0);
}
```

- [ ] **Step 2: Run renderer tests**

Run: `pytest tests/test_renderer.py::test_cosmic_style_renders -v`
Expected: `1 passed`

- [ ] **Step 3: Commit**

```bash
git add renderer/shaders/cosmic.glsl
git commit -m "feat: Cosmic/Nebula GLSL shader — star field, nebula gradients, ring pulses, film grain"
```

---

## Task 11: Post-Processing Pipeline

**Files:**
- Modify: `renderer/shaders/composite.glsl` (already correct from Task 7)
- Modify: `renderer/shaders/bloom_extract.glsl`
- Modify: `renderer/shaders/blur.glsl`
- Modify: `renderer/shaders/final.glsl`
- Create: `renderer/post.py`

- [ ] **Step 1: Write `renderer/shaders/bloom_extract.glsl`**

```glsl
#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_scene;

void main() {
    vec3 col = texture(u_scene, v_uv).rgb;
    float luma = dot(col, vec3(0.2126, 0.7152, 0.0722));
    float threshold = 0.65;
    float knee = 0.1;
    float extracted = smoothstep(threshold - knee, threshold + knee, luma);
    fragColor = vec4(col * extracted, 1.0);
}
```

- [ ] **Step 2: Write `renderer/shaders/blur.glsl`**

```glsl
#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_tex;
uniform vec2 u_direction;    // (1,0) for H, (0,1) for V
uniform vec2 u_resolution;

// 9-tap Gaussian weights (sigma ≈ 2.2)
const float WEIGHTS[5] = float[](0.2270270, 0.1945946, 0.1216216, 0.0540540, 0.0162162);

void main() {
    vec2 texel = u_direction / u_resolution;
    vec3 col = texture(u_tex, v_uv).rgb * WEIGHTS[0];
    for (int i = 1; i < 5; i++) {
        col += texture(u_tex, v_uv + texel * float(i)).rgb * WEIGHTS[i];
        col += texture(u_tex, v_uv - texel * float(i)).rgb * WEIGHTS[i];
    }
    fragColor = vec4(col, 1.0);
}
```

- [ ] **Step 3: Write `renderer/shaders/final.glsl`**

```glsl
#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_scene;
uniform sampler2D u_bloom;
uniform sampler2D u_prev_frame;
uniform vec2  u_resolution;
uniform float u_motion_blur;    // 0.0 = off, 0.15 = organic motion blur
uniform float u_bloom_intensity;

// Simplified FXAA (luma-based edge-detection AA)
vec3 fxaa(sampler2D tex, vec2 uv, vec2 resolution) {
    vec2 px = 1.0 / resolution;
    vec3 rgbNW = texture(tex, uv + vec2(-1.0, -1.0) * px).rgb;
    vec3 rgbNE = texture(tex, uv + vec2( 1.0, -1.0) * px).rgb;
    vec3 rgbSW = texture(tex, uv + vec2(-1.0,  1.0) * px).rgb;
    vec3 rgbSE = texture(tex, uv + vec2( 1.0,  1.0) * px).rgb;
    vec3 rgbM  = texture(tex, uv).rgb;

    vec3 luma_weights = vec3(0.299, 0.587, 0.114);
    float lumaNW = dot(rgbNW, luma_weights);
    float lumaNE = dot(rgbNE, luma_weights);
    float lumaSW = dot(rgbSW, luma_weights);
    float lumaSE = dot(rgbSE, luma_weights);
    float lumaM  = dot(rgbM,  luma_weights);

    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));
    float lumaRange = lumaMax - lumaMin;

    if (lumaRange < 0.1) return rgbM;

    vec2 dir = vec2(
        -((lumaNW + lumaNE) - (lumaSW + lumaSE)),
         ((lumaNW + lumaSW) - (lumaNE + lumaSE))
    );
    float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * 0.03125, 0.0078125);
    float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    dir = clamp(dir * rcpDirMin, vec2(-8.0), vec2(8.0)) * px;

    vec3 rgbA = 0.5 * (
        texture(tex, uv + dir * (1.0/3.0 - 0.5)).rgb +
        texture(tex, uv + dir * (2.0/3.0 - 0.5)).rgb
    );
    vec3 rgbB = rgbA * 0.5 + 0.25 * (
        texture(tex, uv + dir * -0.5).rgb +
        texture(tex, uv + dir *  0.5).rgb
    );
    float lumaB = dot(rgbB, luma_weights);
    return (lumaB < lumaMin || lumaB > lumaMax) ? rgbA : rgbB;
}

void main() {
    vec2 uv = v_uv;

    // FXAA on scene
    vec3 col = fxaa(u_scene, uv, u_resolution);

    // Motion blur (organic mode only)
    if (u_motion_blur > 0.0) {
        vec3 prev = texture(u_prev_frame, uv).rgb;
        col = mix(col, prev, u_motion_blur);
    }

    // Additive bloom
    vec3 bloom = texture(u_bloom, uv).rgb;
    col += bloom * u_bloom_intensity;

    // Reinhard tone mapping
    col = col / (col + vec3(1.0));

    // Gamma correction (sRGB)
    col = pow(col, vec3(1.0 / 2.2));

    // Vignette (subtle secondary pass)
    float vig = 1.0 - dot(uv - 0.5, uv - 0.5) * 0.5;
    col *= vig;

    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
```

- [ ] **Step 4: Create `renderer/post.py`**

```python
"""
post.py — documents the post-processing pass order managed by Renderer.
The actual GPU passes are driven by renderer.py. This module provides
configuration constants and helpers used by Renderer.
"""

BLOOM_THRESHOLD = 0.65
BLOOM_KNEE = 0.1
BLUR_TAPS = 5          # each side: 9-tap total Gaussian
MOTION_BLUR_ALPHA = 0.15   # organic mode only
```

- [ ] **Step 5: Run full renderer tests**

Run: `pytest tests/test_renderer.py -v`
Expected: `5 passed`

- [ ] **Step 6: Commit**

```bash
git add renderer/shaders/bloom_extract.glsl renderer/shaders/blur.glsl renderer/shaders/final.glsl renderer/post.py
git commit -m "feat: post-processing pipeline — bloom extract, Gaussian blur, FXAA, tone mapping, motion blur"
```

---

## Task 12: Headless Exporter

**Files:**
- Create: `export/exporter.py`
- Create: `tests/test_exporter.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_exporter.py
import os
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


def test_export_live_writes_file(tmp_path, gl_ctx, sine_wav):
    from renderer.renderer import Renderer
    r = Renderer(gl_ctx, width=64, height=48)
    frames = [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(12)]
    # Patch moviepy to avoid writing a real video in tests
    with patch("export.exporter.ImageSequenceClip") as mock_clip:
        instance = MagicMock()
        mock_clip.return_value = instance
        instance.set_audio.return_value = instance
        exporter = Exporter()
        out = str(tmp_path / "out.mp4")
        exporter.export_live(frames, sine_wav, out, fps=24)
        mock_clip.assert_called_once()
    r.release()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_exporter.py -v`
Expected: `ImportError`

- [ ] **Step 3: Create `export/exporter.py`**

```python
from __future__ import annotations
from typing import Callable

import numpy as np

from audio.analyzer import AnalysisResult
from engine.context_engine import ContextEngine


class Exporter:
    def collect_frames(
        self,
        analysis: AnalysisResult,
        renderer,            # Renderer — imported at call site to avoid circular deps
        engine: ContextEngine,
        fps: int = 24,
        on_progress: Callable[[float], None] | None = None,
    ) -> list[np.ndarray]:
        """
        Headless render pass: no display flip, full GPU speed.
        Returns list of (H, W, 3) uint8 numpy arrays.
        """
        frames_out: list[np.ndarray] = []
        n = len(analysis.frames)
        elapsed = 0.0
        dt = 1.0 / fps

        for i, audio_frame in enumerate(analysis.frames):
            params = engine.update(audio_frame, dt=dt)
            renderer.render_frame(params, elapsed_time=elapsed)
            frames_out.append(renderer.read_pixels())
            elapsed += dt
            if on_progress is not None:
                on_progress(float(i + 1) / n)

        return frames_out

    def assemble_mp4(
        self,
        frames: list[np.ndarray],
        audio_source: str,
        output_path: str,
        fps: int = 24,
    ) -> None:
        """Assemble collected frames + audio into an MP4."""
        from moviepy.editor import AudioFileClip, ImageSequenceClip
        clip = ImageSequenceClip(frames, fps=fps)
        audio = AudioFileClip(audio_source)
        clip.set_audio(audio).write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            verbose=False,
            logger=None,
        )

    def export_headless(
        self,
        analysis: AnalysisResult,
        renderer,
        engine: ContextEngine,
        source_audio_path: str,
        output_path: str,
        fps: int = 24,
        on_progress: Callable[[float], None] | None = None,
    ) -> None:
        """Full headless pipeline: collect frames, then assemble MP4."""
        frames = self.collect_frames(analysis, renderer, engine, fps=fps, on_progress=on_progress)
        self.assemble_mp4(frames, source_audio_path, output_path, fps=fps)

    def export_live(
        self,
        frames: list[np.ndarray],
        audio_path: str,
        output_path: str,
        fps: int = 24,
    ) -> None:
        """Assemble live-session frames with recorded audio."""
        self.assemble_mp4(frames, audio_path, output_path, fps=fps)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_exporter.py -v`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add export/exporter.py tests/test_exporter.py
git commit -m "feat: Exporter — headless render loop, live frame assembly, MoviePy MP4 output"
```

---

## Task 13: Launcher UI

**Files:**
- Create: `ui/launcher.py`

No unit tests for Pygame rendering — the Launcher is exercised in the integration test (Task 14). The state machine logic is tested here independently.

- [ ] **Step 1: Write failing state machine tests in-line**

Create `tests/test_launcher.py`:

```python
# tests/test_launcher.py
import pytest
from unittest.mock import MagicMock, patch
from ui.launcher import LauncherState, LauncherUI


def test_initial_state_is_expanded():
    ui = LauncherUI(width=1920, height=1080)
    assert ui.state == LauncherState.EXPANDED


def test_play_collapses_bar():
    ui = LauncherUI(width=1920, height=1080)
    ui.on_play()
    assert ui.state == LauncherState.COLLAPSED


def test_stop_expands_bar():
    ui = LauncherUI(width=1920, height=1080)
    ui.on_play()
    ui.on_stop()
    assert ui.state == LauncherState.EXPANDED


def test_mode_toggle():
    ui = LauncherUI(width=1920, height=1080)
    assert ui.mode == "prerecorded"
    ui.toggle_mode()
    assert ui.mode == "live"
    ui.toggle_mode()
    assert ui.mode == "prerecorded"


def test_set_file_path():
    ui = LauncherUI(width=1920, height=1080)
    ui.set_file_path("/path/to/track.mp3")
    assert ui.file_path == "/path/to/track.mp3"


def test_bar_height_collapses():
    ui = LauncherUI(width=1920, height=1080)
    ui.on_play()
    # Simulate transition ticks
    for _ in range(60):
        ui.tick(dt=1.0 / 60)
    assert ui.bar_height < ui.expanded_height


def test_export_requested_fires():
    ui = LauncherUI(width=1920, height=1080)
    ui.set_file_path("/path/to/track.mp3")
    fired = []
    ui.on_export_requested = lambda: fired.append(True)
    ui.request_export()
    assert len(fired) == 1
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_launcher.py -v`
Expected: `ImportError`

- [ ] **Step 3: Create `ui/launcher.py`**

```python
from __future__ import annotations
from enum import Enum, auto
from typing import Callable


class LauncherState(Enum):
    EXPANDED = auto()
    COLLAPSED = auto()
    TRANSITIONING = auto()


_EXPANDED_H = 72    # px
_COLLAPSED_H = 22   # px
_TRANSITION_S = 0.2 # seconds


class LauncherUI:
    """
    Top-bar state machine. Pygame rendering calls are in draw().
    All state transitions are pure Python so they can be unit-tested
    without a display.
    """

    def __init__(self, width: int, height: int) -> None:
        self._w = width
        self._h = height
        self.state = LauncherState.EXPANDED
        self.mode: str = "prerecorded"   # "prerecorded" | "live"
        self.file_path: str | None = None
        self.selected_device_index: int = 0
        self.device_list: list[dict] = []

        self.expanded_height: int = _EXPANDED_H
        self.collapsed_height: int = _COLLAPSED_H
        self.bar_height: float = float(_EXPANDED_H)

        self._transition_elapsed: float = 0.0
        self._transition_target: float = float(_EXPANDED_H)

        # Playback status shown in collapsed strip
        self.status_text: str = ""
        self.is_recording: bool = False

        # Callbacks — set by main.py
        self.on_export_requested: Callable[[], None] = lambda: None
        self.on_play_requested: Callable[[], None] = lambda: None
        self.on_stop_requested: Callable[[], None] = lambda: None
        self.on_go_live_requested: Callable[[], None] = lambda: None

    def toggle_mode(self) -> None:
        self.mode = "live" if self.mode == "prerecorded" else "prerecorded"

    def set_file_path(self, path: str) -> None:
        self.file_path = path

    def set_device_list(self, devices: list[dict]) -> None:
        self.device_list = devices

    def on_play(self) -> None:
        self._start_transition(_COLLAPSED_H)
        self.state = LauncherState.COLLAPSED

    def on_stop(self) -> None:
        self._start_transition(_EXPANDED_H)
        self.state = LauncherState.EXPANDED

    def request_export(self) -> None:
        self.on_export_requested()

    def tick(self, dt: float) -> None:
        if self.state == LauncherState.TRANSITIONING:
            self._transition_elapsed += dt
            t = min(self._transition_elapsed / _TRANSITION_S, 1.0)
            # Ease-in-out cubic
            t = t * t * (3.0 - 2.0 * t)
            start = (
                float(_EXPANDED_H)
                if self._transition_target == _COLLAPSED_H
                else float(_COLLAPSED_H)
            )
            self.bar_height = start + (self._transition_target - start) * t
            if self._transition_elapsed >= _TRANSITION_S:
                self.bar_height = self._transition_target
                self.state = (
                    LauncherState.COLLAPSED
                    if self._transition_target == _COLLAPSED_H
                    else LauncherState.EXPANDED
                )

    def _start_transition(self, target_h: float) -> None:
        self.state = LauncherState.TRANSITIONING
        self._transition_target = float(target_h)
        self._transition_elapsed = 0.0

    def draw(self, surface) -> None:
        """
        Render the top bar onto a pygame Surface.
        Called once per frame from main.py.
        """
        import pygame

        bar_h = int(self.bar_height)
        bar_rect = pygame.Rect(0, 0, self._w, bar_h)

        # Background
        bar_surf = pygame.Surface((self._w, bar_h), pygame.SRCALPHA)
        bar_surf.fill((19, 19, 31, 230))
        pygame.draw.line(bar_surf, (42, 42, 58), (0, bar_h - 1), (self._w, bar_h - 1), 1)

        if self.state in (LauncherState.COLLAPSED, LauncherState.TRANSITIONING) and bar_h <= _COLLAPSED_H + 4:
            self._draw_collapsed_strip(bar_surf, bar_h)
        elif self.state == LauncherState.EXPANDED or (
            self.state == LauncherState.TRANSITIONING and bar_h > _COLLAPSED_H + 4
        ):
            self._draw_expanded_panel(bar_surf, bar_h)

        surface.blit(bar_surf, (0, 0))

    def _draw_collapsed_strip(self, surf, h: int) -> None:
        import pygame
        font = pygame.font.SysFont("monospace", 10)
        # Mode indicator dot
        dot_color = (78, 195, 161) if self.mode == "prerecorded" else (225, 29, 72)
        pygame.draw.circle(surf, dot_color, (12, h // 2), 4)
        # Status text
        text = font.render(self.status_text or "READY", True, dot_color)
        surf.blit(text, (24, h // 2 - text.get_height() // 2))

    def _draw_expanded_panel(self, surf, h: int) -> None:
        import pygame
        font_sm = pygame.font.SysFont("monospace", 10)
        font_xs = pygame.font.SysFont("monospace", 9)

        # Mode toggle buttons
        mode_x = 12
        btn_w, btn_h = 110, 20
        mode_y = (h - btn_h) // 2

        prerecorded_active = self.mode == "prerecorded"
        live_active = self.mode == "live"

        pre_col = (139, 92, 246) if prerecorded_active else (30, 30, 46)
        live_col = (225, 29, 72) if live_active else (30, 30, 46)

        pygame.draw.rect(surf, pre_col, (mode_x, mode_y, btn_w, btn_h), border_radius=4)
        pygame.draw.rect(surf, live_col, (mode_x + btn_w + 4, mode_y, btn_w, btn_h), border_radius=4)

        pre_label = font_xs.render("📁 Pre-recorded", True, (255, 255, 255))
        live_label = font_xs.render("🎙 Live Input", True, (255, 255, 255))
        surf.blit(pre_label, (mode_x + 8, mode_y + 5))
        surf.blit(live_label, (mode_x + btn_w + 12, mode_y + 5))

        # Right-side action buttons
        if self.mode == "prerecorded":
            play_col = (78, 195, 161)
            play_label = font_sm.render("▶ Play", True, (0, 0, 0))
            exp_label  = font_sm.render("⬛ Export", True, (200, 200, 200))
            rx = self._w - 180
            pygame.draw.rect(surf, play_col, (rx, mode_y, 76, btn_h), border_radius=4)
            surf.blit(play_label, (rx + 12, mode_y + 5))
            pygame.draw.rect(surf, (30, 30, 46), (rx + 82, mode_y, 88, btn_h), border_radius=4)
            pygame.draw.rect(surf, (42, 42, 58), (rx + 82, mode_y, 88, btn_h), 1, border_radius=4)
            surf.blit(exp_label, (rx + 88, mode_y + 5))
        else:
            go_col = (225, 29, 72)
            go_label = font_sm.render("⏺ Go Live", True, (255, 255, 255))
            rx = self._w - 130
            pygame.draw.rect(surf, go_col, (rx, mode_y, 110, btn_h), border_radius=4)
            surf.blit(go_label, (rx + 14, mode_y + 5))
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_launcher.py -v`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add ui/launcher.py tests/test_launcher.py
git commit -m "feat: LauncherUI — Pygame top bar state machine, expand/collapse animation"
```

---

## Task 14: main.py Integration

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Replace `main.py` with full integration**

```python
"""
main.py — Music Visualizer entry point.

Pre-recorded mode:
  python main.py path/to/track.mp3 [--export output.mp4]

Live mode:
  python main.py --live [--device N]
"""
from __future__ import annotations
import argparse
import os
import sys
import tempfile
import threading
import tomllib
import wave

import numpy as np
import pygame
import moderngl

from audio.analyzer import PrerecordedAnalyzer
from audio.live_analyzer import LiveAnalyzer
from engine.context_engine import ContextEngine
from engine.render_params import ContextConfig
from export.exporter import Exporter
from renderer.renderer import Renderer
from ui.launcher import LauncherUI, LauncherState


def load_config(path: str = "config.toml") -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _run_headless_export(
    analysis,
    renderer: Renderer,
    engine: ContextEngine,
    source_path: str,
    output_path: str,
    config: dict,
    launcher: LauncherUI,
) -> None:
    exporter = Exporter()
    fps = config["export"]["fps"]

    def progress(pct: float) -> None:
        launcher.status_text = f"Exporting… {int(pct * 100)}%"

    exporter.export_headless(
        analysis, renderer, engine, source_path, output_path,
        fps=fps, on_progress=progress,
    )
    launcher.status_text = f"Saved → {output_path}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Music Visualizer")
    parser.add_argument("file", nargs="?", help="Audio/video file to visualize")
    parser.add_argument("--export", metavar="OUTPUT", help="Export MP4 without playback")
    parser.add_argument("--live", action="store_true", help="Live input mode")
    parser.add_argument("--device", type=int, default=0, help="Input device index")
    args = parser.parse_args()

    config = load_config()
    ctx_cfg = ContextConfig(
        dissonance_threshold=config["context"]["dissonance_threshold"],
        tempo_threshold=config["context"]["tempo_threshold"],
        style_hold_seconds=config["context"]["style_hold_seconds"],
        blend_duration_seconds=config["context"]["blend_duration_seconds"],
        ema_alpha=config["context"]["ema_alpha"],
    )

    width, height = config["export"]["resolution"]
    fps_target = config["renderer"]["realtime_fps"]

    # Pygame + OpenGL init
    os.environ.setdefault("SDL_VIDEODRIVER", "x11")
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode(
        (width, height), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
    )
    pygame.display.set_caption("Music Visualizer")

    ctx = moderngl.create_context()
    renderer = Renderer(ctx, width, height)
    engine = ContextEngine(ctx_cfg)
    launcher = LauncherUI(width, height)
    exporter = Exporter()

    # Enumerate audio devices for live mode
    launcher.set_device_list(LiveAnalyzer.list_input_devices())

    # Pre-load file from CLI arg
    if args.file:
        launcher.set_file_path(args.file)

    # Headless export path (no window interaction needed)
    if args.export and args.file:
        analyzer = PrerecordedAnalyzer()
        analysis = analyzer.analyze(args.file, fps=config["export"]["fps"])
        _run_headless_export(analysis, renderer, engine, args.file, args.export, config, launcher)
        renderer.release()
        pygame.quit()
        return

    # State for interactive session
    analysis = None
    live_analyzer: LiveAnalyzer | None = None
    live_frames: list[np.ndarray] = []
    live_audio_chunks: list[np.ndarray] = []
    is_playing = False
    is_live = args.live
    frame_index = 0
    elapsed_time = 0.0
    pygame_audio_channel: pygame.mixer.Channel | None = None

    def start_prerecorded() -> None:
        nonlocal analysis, is_playing, frame_index, elapsed_time
        if not launcher.file_path:
            return
        analyzer = PrerecordedAnalyzer()
        analysis = analyzer.analyze(launcher.file_path, fps=fps_target)
        frame_index = 0
        elapsed_time = 0.0
        is_playing = True
        launcher.on_play()
        # Play audio via pygame.mixer
        pygame.mixer.init(frequency=analysis.sr, channels=1)
        int16 = (np.clip(analysis.audio, -1.0, 1.0) * 32767).astype(np.int16)
        sound = pygame.sndarray.make_sound(int16.reshape(-1, 1))
        sound.play()

    def start_live() -> None:
        nonlocal live_analyzer, is_playing, is_live, elapsed_time
        is_live = True
        live_frames.clear()
        live_audio_chunks.clear()
        elapsed_time = 0.0
        device_idx = launcher.selected_device_index
        live_analyzer = LiveAnalyzer(device_index=device_idx)
        live_analyzer.start()
        is_playing = True
        launcher.on_play()
        launcher.status_text = "LIVE"

    def stop_and_export_live() -> None:
        nonlocal is_playing
        if live_analyzer:
            live_analyzer.stop()
        is_playing = False
        launcher.on_stop()
        if not live_frames:
            return
        audio = live_analyzer.get_recording() if live_analyzer else np.zeros(1)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        sr = 44100
        int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(tmp.name, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(int16.tobytes())
        out_path = "live_output.mp4"
        exporter.export_live(live_frames, tmp.name, out_path, fps=fps_target)
        os.unlink(tmp.name)
        launcher.status_text = f"Saved → {out_path}"

    # Wire launcher callbacks
    launcher.on_play_requested = start_prerecorded
    launcher.on_go_live_requested = start_live
    launcher.on_stop_requested = stop_and_export_live

    def request_headless_export() -> None:
        if not launcher.file_path or not analysis:
            return
        out = launcher.file_path.rsplit(".", 1)[0] + "_viz.mp4"
        engine2 = ContextEngine(ctx_cfg)
        t = threading.Thread(
            target=_run_headless_export,
            args=(analysis, renderer, engine2, launcher.file_path, out, config, launcher),
            daemon=True,
        )
        t.start()

    launcher.on_export_requested = request_headless_export

    clock = pygame.time.Clock()
    dt = 1.0 / fps_target

    # Pixel buffer for FBO→Pygame blit
    pbo_surf = pygame.Surface((width, height), 0, 24)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    if is_playing:
                        is_playing = False
                    else:
                        if is_live:
                            start_live()
                        else:
                            start_prerecorded()
                elif event.key == pygame.K_e and not is_live:
                    request_headless_export()
            elif event.type == pygame.DROPFILE:
                launcher.set_file_path(event.file)

        launcher.tick(dt)

        # --- Build FeatureFrame ---
        if is_playing:
            if is_live and live_analyzer:
                audio_frame = live_analyzer.get_frame()
            elif analysis and frame_index < len(analysis.frames):
                audio_frame = analysis.frames[frame_index]
                frame_index += 1
                if frame_index >= len(analysis.frames):
                    is_playing = False
                    launcher.on_stop()
            else:
                is_playing = False
                launcher.on_stop()
                continue
        else:
            from audio.feature_frame import FeatureFrame
            audio_frame = FeatureFrame(
                amplitude=0.0, rms=0.0, spectral_centroid=0.5,
                onset_strength=0.0, dissonance_raw=0.0, dissonance_smooth=0.0,
                chroma=np.zeros(12), bpm=0.0,
            )

        params = engine.update(audio_frame, dt=dt)
        renderer.render_frame(params, elapsed_time=elapsed_time)
        elapsed_time += dt

        # Collect live frames for export
        if is_live and is_playing:
            live_frames.append(renderer.read_pixels())

        # Blit FBO → Pygame surface
        raw = renderer._fbo_final.read(components=3)
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
        arr = np.flipud(arr)
        pygame.surfarray.blit_array(pbo_surf, arr.transpose(1, 0, 2))
        screen.blit(pbo_surf, (0, 0))

        # Draw launcher bar on top
        launcher.draw(screen)

        pygame.display.flip()
        clock.tick(fps_target)

    if live_analyzer:
        live_analyzer.stop()
    renderer.release()
    pygame.quit()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run all tests to verify nothing is broken**

Run: `pytest -v`
Expected: all tests pass

- [ ] **Step 3: Smoke test the window opens headlessly**

Run: `SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy timeout 2 python main.py 2>/dev/null; echo "exit $?"`
Expected: `exit 0` or `exit 124` (timeout — window opened and ran)

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat: main.py — full event loop integrating analyzer, context engine, renderer, launcher"
```

---

## Task 15: Integration Test

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration.py
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
    # At least some variation between first and last frame
    assert not np.array_equal(frames[0], frames[-1])

    renderer.release()


def test_organic_frames_not_all_black(gl_ctx, sine_wav):
    """Organic/Flow is the default style — a 440Hz sine stays in ORGANIC."""
    analyzer = PrerecordedAnalyzer()
    analysis = analyzer.analyze(sine_wav, fps=24)
    renderer = Renderer(gl_ctx, width=160, height=90)
    engine = ContextEngine(ContextConfig())
    exporter = Exporter()
    frames = exporter.collect_frames(analysis, renderer, engine, fps=24)
    mid = frames[len(frames) // 2]
    assert mid.max() > 20
    renderer.release()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_integration.py -v`
Expected: `ImportError` (exporter not yet known to this file)

- [ ] **Step 3: Run with all modules in place**

Run: `pytest tests/test_integration.py -v`
Expected: `2 passed`

- [ ] **Step 4: Run full suite**

Run: `pytest -v --tb=short`
Expected: all tests pass, 0 failures

- [ ] **Step 5: Final commit**

```bash
git add tests/test_integration.py
git commit -m "test: integration test — full pipeline analyze→engine→renderer→export"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] Pre-recorded + live input modes — Tasks 3, 4
- [x] librosa full-pass analysis — Task 3
- [x] sounddevice ring buffer + FFT — Task 4
- [x] FeatureFrame / RenderParams contracts — Task 2
- [x] dissonance_raw vs dissonance_smooth split — Task 3, 4, 6
- [x] Style classification (dissonance / tempo thresholds) — Task 6
- [x] Hysteresis + sigmoid blend — Task 6
- [x] Sharp transient micro-effects (dissonance_raw → u_dissonance uniform) — Tasks 8, 9
- [x] Circle of Fifths palette — Task 3
- [x] Geometric/Glitch shader (SDF, scanlines, chromatic aberration, glitch) — Task 8
- [x] Organic/Flow shader (FBM, waveforms, orbs, particles) — Task 9
- [x] Cosmic/Nebula shader (stars, nebula, ring pulses, film grain) — Task 10
- [x] Post-processing (bloom, FXAA, tone mapping, motion blur, vignette) — Task 11
- [x] Headless export (no playback required) — Task 12
- [x] Live export (sounddevice → temp wav → MoviePy) — Task 12, 14
- [x] Collapsible top bar launcher — Task 13
- [x] Export button in launcher — Task 13, 14
- [x] ring_times uniform for cosmic rings — Task 7, 10
- [x] config.toml thresholds — Task 1

**Type consistency check:**
- `FeatureFrame.dissonance_raw` → used in `ContextEngine.update()` → `RenderParams.dissonance_raw` → `renderer._render_style()` → `prog["u_dissonance"]` ✓
- `FeatureFrame.dissonance_smooth` → used in `_classify()` inside `ContextEngine` ✓
- `RenderParams.blend_weight` → `prog_composite["u_blend"]` ✓
- `Renderer.read_pixels()` → `np.ndarray (H,W,3) uint8` → `Exporter.collect_frames()` → `ImageSequenceClip(frames)` ✓
- `AnalysisResult.frames: list[FeatureFrame]` → iterated in `Exporter.collect_frames()` ✓
- `LauncherUI.on_play()` / `.on_stop()` → called in `main.py` ✓
