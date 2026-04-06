"""
Music Visualizer entry point.

Pre-recorded mode:
  python main.py path/to/track.mp3

Headless export (no playback):
  python main.py path/to/track.mp3 --export output.mp4

Live mode:
  python main.py --live [--device N]
"""
from __future__ import annotations
import argparse
import os
import tempfile
import threading
import wave
from pathlib import Path

import numpy as np
import pygame
import moderngl

from audio.analyzer import PrerecordedAnalyzer
from audio.feature_frame import FeatureFrame
from audio.live_analyzer import LiveAnalyzer
from engine.context_engine import ContextEngine
from engine.render_params import ContextConfig
from export.exporter import Exporter
from renderer.renderer import Renderer
from ui.launcher import LauncherUI


def load_config(path: Path | str | None = None) -> dict:
    import tomllib
    if path is None:
        path = Path(__file__).parent / "config.toml"
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

    launcher.set_device_list(LiveAnalyzer.list_input_devices())

    if args.file:
        launcher.set_file_path(args.file)

    # Headless export — no interactive loop needed
    if args.export and args.file:
        analyzer = PrerecordedAnalyzer()
        analysis = analyzer.analyze(args.file, fps=config["export"]["fps"])
        _run_headless_export(analysis, renderer, engine, args.file, args.export, config, launcher)
        renderer.release()
        pygame.quit()
        return

    # --- Interactive session state ---
    analysis = None
    live_analyzer: LiveAnalyzer | None = None
    live_frames: list[np.ndarray] = []
    is_playing = False
    is_live = args.live
    frame_index = 0
    elapsed_time = 0.0

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
        pygame.mixer.init(frequency=analysis.sr, channels=1)
        int16 = (np.clip(analysis.audio, -1.0, 1.0) * 32767).astype(np.int16)
        sound = pygame.sndarray.make_sound(int16.reshape(-1, 1))
        sound.play()

    def start_live() -> None:
        nonlocal live_analyzer, is_playing, is_live, elapsed_time
        is_live = True
        live_frames.clear()
        elapsed_time = 0.0
        live_analyzer = LiveAnalyzer(device_index=launcher.selected_device_index)
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
        int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(tmp.name, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(int16.tobytes())
        out_path = "live_output.mp4"
        exporter.export_live(live_frames, tmp.name, out_path, fps=fps_target)
        os.unlink(tmp.name)
        launcher.status_text = f"Saved → {out_path}"

    def request_headless_export() -> None:
        if not launcher.file_path or not analysis:
            return
        out = launcher.file_path.rsplit(".", 1)[0] + "_viz.mp4"
        engine2 = ContextEngine(ctx_cfg)
        threading.Thread(
            target=_run_headless_export,
            args=(analysis, renderer, engine2, launcher.file_path, out, config, launcher),
            daemon=True,
        ).start()

    launcher.on_play_requested = start_prerecorded
    launcher.on_go_live_requested = start_live
    launcher.on_stop_requested = stop_and_export_live
    launcher.on_export_requested = request_headless_export

    clock = pygame.time.Clock()
    dt = 1.0 / fps_target
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
                        launcher.on_stop()
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

        # Build FeatureFrame for this render frame
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
                audio_frame = FeatureFrame(
                    amplitude=0.0, rms=0.0, spectral_centroid=0.5,
                    onset_strength=0.0, dissonance_raw=0.0, dissonance_smooth=0.0,
                    chroma=np.zeros(12), bpm=0.0,
                )
        else:
            audio_frame = FeatureFrame(
                amplitude=0.0, rms=0.0, spectral_centroid=0.5,
                onset_strength=0.0, dissonance_raw=0.0, dissonance_smooth=0.0,
                chroma=np.zeros(12), bpm=0.0,
            )

        params = engine.update(audio_frame, dt=dt)
        renderer.render_frame(params, elapsed_time=elapsed_time)
        elapsed_time += dt

        if is_live and is_playing:
            live_frames.append(renderer.read_pixels())

        # Blit FBO → Pygame surface
        raw = renderer._fbo_final.read(components=3)
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
        arr = np.flipud(arr)
        pygame.surfarray.blit_array(pbo_surf, arr.transpose(1, 0, 2))
        screen.blit(pbo_surf, (0, 0))
        launcher.draw(screen)
        pygame.display.flip()
        clock.tick(fps_target)

    if live_analyzer:
        live_analyzer.stop()
    renderer.release()
    pygame.quit()


if __name__ == "__main__":
    main()
