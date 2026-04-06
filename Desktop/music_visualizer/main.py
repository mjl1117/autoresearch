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
    export_w, export_h = config["export"]["resolution"]
    fps_target = config["renderer"]["realtime_fps"]

    pygame.init()
    pygame.font.init()

    # Cap window to actual screen size so buttons stay on-screen
    desktop_sizes = pygame.display.get_desktop_sizes()
    scr_w, scr_h = desktop_sizes[0] if desktop_sizes else (export_w, export_h)
    width = min(export_w, scr_w)
    height = min(export_h, scr_h)

    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)
    screen = pygame.display.set_mode(
        (width, height), pygame.OPENGL | pygame.DOUBLEBUF
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
    _analyzing = False
    _pending_audio: tuple[np.ndarray, int] | None = None  # queued for main-thread playback

    import sounddevice as sd

    def start_prerecorded() -> None:
        nonlocal _analyzing
        if not launcher.file_path or _analyzing:
            return
        _analyzing = True
        launcher.status_text = "Analyzing…"
        file_path = launcher.file_path

        def _analyze_thread() -> None:
            nonlocal analysis, is_playing, frame_index, elapsed_time, _analyzing, _pending_audio
            try:
                result = PrerecordedAnalyzer().analyze(file_path, fps=fps_target)
                analysis = result
                frame_index = 0
                elapsed_time = 0.0
                is_playing = True
                launcher.on_play()
                # Queue audio for playback on the main thread (Core Audio requires it)
                _pending_audio = (result.audio.astype(np.float32), result.sr)
            except Exception as e:
                print(f"[analyze] {e}")
                launcher.status_text = f"Error: {e}"
            finally:
                _analyzing = False

        threading.Thread(target=_analyze_thread, daemon=True).start()

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
        try:
            sd.stop()
        except Exception:
            pass
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

    _exporting = False

    def request_headless_export() -> None:
        nonlocal _exporting
        if not launcher.file_path or not analysis or _exporting:
            return
        _exporting = True
        out = launcher.file_path.rsplit(".", 1)[0] + "_viz.mp4"
        file_path = launcher.file_path
        fps = config["export"]["fps"]
        launcher.status_text = "Exporting…"

        def _export_thread() -> None:
            nonlocal _exporting
            try:
                # Own standalone GL context — never touches the display context
                import moderngl as mgl
                exp_ctx = mgl.create_standalone_context()
                exp_renderer = Renderer(exp_ctx, export_w, export_h)
                exp_engine = ContextEngine(ctx_cfg)
                exporter = Exporter()

                def _progress(pct: float) -> None:
                    launcher.status_text = f"Exporting {int(pct * 100)}%…"

                exporter.export_headless(
                    analysis, exp_renderer, exp_engine,
                    source_audio_path=file_path,
                    output_path=out,
                    fps=fps,
                    on_progress=_progress,
                )
                exp_renderer.release()
                exp_ctx.release()
                launcher.status_text = f"Saved → {out}"
            except Exception as e:
                print(f"[export] {e}")
                launcher.status_text = f"Export failed: {e}"
            finally:
                _exporting = False

        threading.Thread(target=_export_thread, daemon=True).start()

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
                        try:
                            sd.stop()
                        except Exception:
                            pass
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

        # Start queued audio on the main thread (Core Audio requires main-thread init)
        if _pending_audio is not None:
            audio, sr = _pending_audio
            _pending_audio = None
            try:
                sd.stop()
                sd.play(audio, samplerate=sr)
            except Exception as e:
                print(f"[audio] {e}")

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
