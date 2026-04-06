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
        Headless render pass — no display flip, full GPU speed.
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
        from moviepy import AudioFileClip, ImageSequenceClip
        clip = ImageSequenceClip(frames, fps=fps)
        audio = AudioFileClip(audio_source)
        # moviepy 2.x uses with_audio(); fall back to set_audio() for older installs
        attach = getattr(clip, "with_audio", None) or clip.set_audio
        attach(audio).write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            ffmpeg_params=["-crf", "18"],
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
        """Full headless pipeline: collect frames then assemble MP4."""
        frames = self.collect_frames(
            analysis, renderer, engine, fps=fps, on_progress=on_progress
        )
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
