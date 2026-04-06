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
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)
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
        launcher.status_text = "Analyzing…"
        analyzer = PrerecordedAnalyzer()
        analysis = analyzer.analyze(launcher.file_path, fps=fps_target)
        frame_index = 0
        elapsed_time = 0.0
        is_playing = True
        launcher.on_play()
        try:
            pygame.mixer.quit()
            pygame.mixer.init(frequency=analysis.sr, channels=1)
            int16 = (np.clip(analysis.audio, -1.0, 1.0) * 32767).astype(np.int16)
            sound = pygame.sndarray.make_sound(int16)
            sound.play()
        except Exception:
            pass  # audio playback is best-effort

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

    _export_pending = False

    def request_headless_export() -> None:
        nonlocal _export_pending
        if not launcher.file_path or not analysis:
            return
        _export_pending = True
        launcher.status_text = "Export queued…"

    launcher.on_play_requested = start_prerecorded
    launcher.on_go_live_requested = start_live
    launcher.on_stop_requested = stop_and_export_live
    launcher.on_export_requested = request_headless_export

    clock = pygame.time.Clock()
    dt = 1.0 / fps_target
    ui_surf = pygame.Surface((width, height), pygame.SRCALPHA, 32)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                launcher.handle_click(*event.pos)
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

        # Run pending export synchronously on the main thread (safe: single GL context)
        if _export_pending and analysis and launcher.file_path:
            _export_pending = False
            out = launcher.file_path.rsplit(".", 1)[0] + "_viz.mp4"
            _run_headless_export(analysis, renderer, ContextEngine(ctx_cfg),
                                 launcher.file_path, out, config, launcher)

        params = engine.update(audio_frame, dt=dt)
        renderer.render_frame(params, elapsed_time=elapsed_time)
        elapsed_time += dt

        if is_live and is_playing:
            live_frames.append(renderer.read_pixels())

        # Copy final FBO → default framebuffer, then composite UI overlay
        ctx.copy_framebuffer(dst=ctx.screen, src=renderer._fbo_final)
        ui_surf.fill((0, 0, 0, 0))
        launcher.draw(ui_surf)
        renderer.render_overlay(pygame.image.tobytes(ui_surf, "RGBA", False))
        pygame.display.flip()
        clock.tick(fps_target)

    if live_analyzer:
        live_analyzer.stop()
    renderer.release()
    pygame.quit()


if __name__ == "__main__":
    main()
