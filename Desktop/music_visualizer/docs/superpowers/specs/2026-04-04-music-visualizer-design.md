# Music Visualizer — Design Spec
**Date:** 2026-04-04  
**Status:** Approved for implementation

---

## 1. Overview

A standalone Python music visualizer that accepts pre-recorded audio/video files or live audio interface input and renders GPU-accelerated visuals driven by real-time audio feature extraction. Primary output is exported MP4 video. Real-time playback supports live collaboration with the user's Neuroaesthetic_Music generative music system.

**Core pipeline:**
```
LauncherUI → AudioPipeline → ContextEngine → Renderer (ModernGL) → Exporter (MoviePy)
```

---

## 2. Input Modes

### Pre-recorded
- Accepted formats: `.mp3`, `.wav`, `.avi`
- `.avi` files: audio extracted to temp `.wav` via MoviePy before analysis
- Full-track librosa pre-analysis pass before playback or export begins
- Supports headless export (no playback required — renders all frames at full GPU speed with vsync disabled, then muxes audio)

### Live Input
- Audio interface enumerated at launch via `sounddevice.query_devices()`
- Device selected from dropdown in launcher
- Ring buffer captures rolling audio window (8 seconds, float32 mono)
- On "Stop + Export": buffer flushed to temp `.wav`, frames assembled via MoviePy

---

## 3. Audio Pipeline

### Pre-recorded Analysis (`AudioAnalyzer`)
Full librosa pass on load:

| Feature | librosa call | Output |
|---|---|---|
| Tempo / BPM | `beat_track` | `bpm: float` |
| Harmonic/Percussive | `hpss` | `harmonic`, `percussive` arrays |
| Spectral centroid | `spectral_centroid` | per-frame brightness proxy |
| Chroma (key) | `chroma_cqt` | 12-bin chroma vector per frame |
| Onset strength | `onset_strength` | per-frame pulse proxy |
| Spectral flatness | `spectral_flatness` | noise/tone ratio |

Pre-computed results stored in a frame-indexed lookup for O(1) retrieval during playback/export.

### Live Analysis (`LiveAnalyzer`)
Per-chunk (512 samples at 44.1 kHz):
- `np.fft.rfft` → magnitude spectrum → spectral centroid, flatness
- RMS amplitude
- Onset detection via successive difference of spectral flux
- 8-second rolling buffer fed to rolling style classifier (updated every 2s)

### `FeatureFrame` dataclass
Per-frame struct passed from AudioPipeline to ContextEngine:

```python
@dataclass
class FeatureFrame:
    amplitude: float        # peak amplitude 0–1
    rms: float              # RMS energy 0–1
    spectral_centroid: float  # Hz, normalized 0–1
    onset_strength: float   # 0–1
    dissonance_raw: float   # for shader uniforms (transient-sensitive)
    dissonance_smooth: float  # EMA(~2s) for style classifier only
    chroma: np.ndarray      # shape (12,) for key estimation
    bpm: float              # beats per minute
```

**Dissonance formula:**
```python
dissonance_raw = percussive_ratio * spectral_flatness
# EMA smoothing for classifier:
dissonance_smooth = alpha * dissonance_raw + (1 - alpha) * prev_smooth
# alpha ≈ 0.15 → ~2s settling time at 30fps
```

---

## 4. Context Engine

### Style Classification
Three visual styles, selected by dominant audio character:

```python
if dissonance_smooth > 0.65:
    style = GEOMETRIC
elif tempo_normalized < 0.35:   # bpm / 200.0, clamped 0–1
    style = COSMIC
else:
    style = ORGANIC             # default / highest priority
```

Thresholds configurable in `config.toml`:
```toml
[context]
dissonance_threshold = 0.65
tempo_threshold = 0.35
style_hold_seconds = 4.0      # must hold before transition starts
blend_duration_seconds = 3.0  # cross-fade duration
```

### Transition Logic
**Two-layer protection against abrupt style changes:**

1. **Classifier hysteresis:** new style must be dominant for `style_hold_seconds` before triggering a transition. Single sharp dissonance spikes never start the countdown — they are absorbed by the EMA and expressed as micro-transitions within the current style instead.

2. **Sigmoid cross-fade:** 3-second blend between styles. Sigmoid shape makes first/last ~0.5s imperceptible; visual shift concentrates in the middle. Both shader programs render simultaneously during blend; composite pass lerps by `u_blend`.

**Sharp transients within current style:** `dissonance_raw` drives `u_pulse` directly. A sudden glitch hit causes a visible impulse (scanline flash, waveform shudder, brightness spike) within whatever style is active — without triggering a style switch.

### Color Palette (Circle of Fifths)
Chroma vector → argmax → key index (0–11) → palette lookup:

| Key character | Palette |
|---|---|
| Minor key | Analogous cool tones (teal → violet) |
| Major key | Complementary vibrant (warm accent + cool base) |
| Atonal / ambiguous | Neutral cool (desaturated blue-grey) |

Color transitions use `colorsys` for perceptually smooth HSL interpolation. Palette updates smoothly across key changes, not instantly.

### `RenderParams` dataclass
Output from ContextEngine to Renderer:

```python
@dataclass
class RenderParams:
    active_style: Style         # GEOMETRIC | ORGANIC | COSMIC
    blend_target: Style         # style being blended toward; equals active_style when no transition is active
    blend_weight: float         # 0.0 = fully active_style, 1.0 = fully blend_target; 0.0 when not transitioning
    intensity: float            # driven by RMS
    brightness: float           # driven by spectral centroid
    pulse: float                # driven by onset_strength (raw, for transients)
    dissonance_raw: float       # for glitch micro-effects
    color_a: tuple[float,float,float]  # primary RGB
    color_b: tuple[float,float,float]  # secondary RGB
```

---

## 5. Renderer (Pygame + ModernGL)

Pygame owns the window, event loop, and audio sync. ModernGL renders all visuals into an OpenGL FBO. Each frame: FBO texture → blit to Pygame surface → `display.flip()`.

### Shared Shader Uniforms
All three shader programs receive:
```glsl
uniform float u_time;
uniform vec2  u_resolution;
uniform float u_intensity;
uniform float u_brightness;
uniform float u_pulse;
uniform float u_dissonance;    // raw, for transient glitch effects
uniform float u_blend;
uniform vec3  u_color_a;
uniform vec3  u_color_b;
```

### Geometric/Glitch Shader
Target aesthetic: sharp, electric, dissonant.
- Polygon outlines via signed distance functions (SDF) — crisp at any resolution
- Frequency bar columns as rect SDFs
- Scanline overlay: `mod(uv.y * u_resolution.y, stride) < 1.0`
- Chromatic aberration: RGB channels offset by `u_dissonance * u_pulse`
- On sharp transient (`u_pulse` spike): full-frame glitch flash — random horizontal slice displacement driven by hash noise
- Palette: electric red (#ff003c), cyan (#00ffe0) on near-black

### Organic/Flow Shader
Target aesthetic: fluid, breathing, melodic. Highest graphical investment.
- Multi-octave domain-warped noise for waveform paths (fbm / layered sine with phase drift)
- Soft radial gradient orbs with position drift over time, scale modulated by `u_intensity`
- Particle field: hash-noise dot positions, slow drift velocity, fade in/out
- Reaction-diffusion-style texture overlay at low amplitude for micro-texture
- Soft bloom pass: luminance threshold extract → Gaussian blur → additive composite
- Motion blur: accumulation buffer blend (current frame × 0.85 + prev × 0.15)
- Palette: teal (#4fc3a1) → violet (#7b68ee) analogous cool

### Cosmic/Nebula Shader
Target aesthetic: deep, slow, spatial.
- Star field: stable per-pixel hash positions, twinkle via `sin(u_time + seed * 7.3)`
- Dual radial nebula gradients (violet + amber) with slow UV distortion over time
- Expanding ring pulses: emit on beat onset, radius grows with `u_time` from emit point, fade alpha by distance
- Core glow: soft bright point at canvas center, intensity driven by `u_intensity`
- Film grain overlay for texture depth
- Vignette: darkened edges pull focus to center
- Palette: violet (#8b5cf6), blue (#3b82f6), amber (#f59e0b) on deep black (#020208)

### Post-Processing Passes (all styles)
Applied after the primary style composite:
- **Bloom:** extract highlights → dual Kawase blur → additive blend (intensity tuned per style)
- **FXAA:** fast anti-aliasing pass for crisp edges
- **Tone mapping:** Reinhard operator for HDR-like feel without true HDR
- **Vignette:** subtle edge darkening (adjustable per style)

### Cross-Style Blend
During transition: both `active_style` and `blend_target` shaders render into separate FBO color attachments. Composite pass lerps by `u_blend` (sigmoid-eased). At `blend_weight == 0` or `1`, only one shader runs (no wasted GPU work).

---

## 6. Export Pipeline

### Headless Export (Pre-recorded)
No playback required. Render loop runs at full GPU speed (vsync off, no `display.flip()`). A minimal Pygame window is still created to provide the OpenGL context — it can be minimized or hidden during export. A progress bar in the launcher strip shows render completion %.

1. Seek audio analysis to frame 0
2. For each frame: compute `FeatureFrame` → `RenderParams` → render to FBO → `glReadPixels` → append numpy array to frame buffer
3. After all frames: `ImageSequenceClip(frames, fps=fps).set_audio(AudioFileClip(source)).write_videofile(output, codec="libx264")`

FBO readback cost: ~8ms/frame at 1080p. Export render pass targets 24fps to keep export time manageable.

### Live Export
During live session: frames accumulated in memory list, sounddevice records to ring buffer written to temp `.wav`. On "Stop + Export": same MoviePy assembly as pre-recorded, using temp `.wav` as audio source.

### Output
- Codec: H.264 (libx264), CRF 18 (visually lossless)
- Container: `.mp4`
- Resolution: matches source or configurable (default 1920×1080)
- Framerate: 24fps export, 60fps realtime

---

## 7. Launcher UI

Implemented entirely in Pygame — no separate GUI framework. Top bar rendered as a fixed strip above the canvas each frame.

### States
**Expanded (on launch):**
- Mode toggle: Pre-recorded / Live Input
- Pre-recorded: drag-and-drop zone (`pygame.DROPFILE`) + file browser fallback
- Live: device dropdown (populated from `sounddevice.query_devices()` at launch)
- Style selector pills: Auto / Geo / Org / Cos (Auto = hybrid engine default)
- Buttons: ▶ Play + ⏺ Record (pre-recorded) / ⏺ Go Live + ⏹ Stop+Export (live)
- Export without playback: "Export" button visible in pre-recorded mode — triggers headless render pass immediately

**Collapsed (during playback/live):**
- Thin status strip: mode indicator dot, track name or "LIVE", active style, timestamp
- Pause / Record / Expand controls in strip
- Hover strip → expand animation

**Transition:**
- 200ms height animation (ease-in-out) between expanded and collapsed

### Color Coding
| State | Accent color |
|---|---|
| Pre-recorded mode | Purple `#8b5cf6` |
| Live mode | Red `#e11d48` |
| Playing / active | Teal `#4fc3a1` |

---

## 8. Project Structure

```
music_visualizer/
├── main.py                  # entry point, Pygame init, event loop
├── config.toml              # tunable thresholds and defaults
├── audio/
│   ├── analyzer.py          # PrerecordedAnalyzer + LiveAnalyzer
│   └── feature_frame.py     # FeatureFrame dataclass
├── engine/
│   ├── context_engine.py    # style classifier, hysteresis, blend state
│   └── render_params.py     # RenderParams dataclass
├── renderer/
│   ├── renderer.py          # ModernGL FBO setup, frame dispatch
│   ├── post.py              # bloom, FXAA, tone map, vignette passes
│   └── shaders/
│       ├── geometric.glsl
│       ├── organic.glsl
│       ├── cosmic.glsl
│       └── composite.glsl   # cross-blend + post-processing
├── ui/
│   └── launcher.py          # top bar state machine, Pygame rendering
├── export/
│   └── exporter.py          # headless render loop + MoviePy assembly
└── docs/
    └── superpowers/specs/
        └── 2026-04-04-music-visualizer-design.md
```

---

## 9. Dependencies

```toml
[dependencies]
pygame = ">=2.5"
moderngl = ">=5.10"
librosa = ">=0.10"
sounddevice = ">=0.4"
moviepy = ">=1.0"
numpy = ">=1.24"
colorsys = "stdlib"
```

---

## 10. Configuration (`config.toml`)

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
