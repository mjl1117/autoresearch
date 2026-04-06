from __future__ import annotations
import os
import subprocess
import tempfile
from typing import Callable

import numpy as np

from audio.analyzer import AnalysisResult
from engine.context_engine import ContextEngine


class Exporter:
    def collect_frames(
        self,
        analysis: AnalysisResult,
        renderer,
        engine: ContextEngine,
        fps: int = 24,
        on_progress: Callable[[float], None] | None = None,
    ) -> list[np.ndarray]:
        """
        Headless render pass — returns list of (H, W, 3) uint8 arrays.
        Used by tests; for production export use export_headless() which streams to ffmpeg.
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
        """Assemble pre-collected frames + audio into an MP4 (used by tests and live export)."""
        from moviepy import AudioFileClip, ImageSequenceClip
        clip = ImageSequenceClip(frames, fps=fps)
        audio = AudioFileClip(audio_source)
        attach = getattr(clip, "with_audio", None) or clip.set_audio
        attach(audio).write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            ffmpeg_params=["-crf", "18"],
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
        """
        Stream frames one-at-a-time directly into ffmpeg — constant ~6 MB RAM
        regardless of track length, instead of accumulating all frames first.
        """
        w = renderer._w
        h = renderer._h

        # Step 1: encode video-only stream, piping raw RGB frames to ffmpeg
        tmp_video = tempfile.mktemp(suffix=".mp4")
        cmd_video = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}",
            "-r", str(fps),
            "-i", "pipe:0",
            "-vf", "vflip",          # OpenGL bottom-up → top-down
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "fast",
            "-an",
            tmp_video,
        ]
        proc = subprocess.Popen(cmd_video, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        n = len(analysis.frames)
        elapsed = 0.0
        dt = 1.0 / fps

        try:
            for i, audio_frame in enumerate(analysis.frames):
                params = engine.update(audio_frame, dt=dt)
                renderer.render_frame(params, elapsed_time=elapsed)
                # Read raw bytes directly — no numpy allocation per frame
                raw = renderer._fbo_final.read(components=3)
                proc.stdin.write(raw)
                elapsed += dt
                if on_progress is not None:
                    on_progress(float(i + 1) / n)
        finally:
            proc.stdin.close()
            proc.wait()

        # Step 2: mux video + audio with a single fast ffmpeg pass
        cmd_mux = [
            "ffmpeg", "-y",
            "-i", tmp_video,
            "-i", source_audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            output_path,
        ]
        subprocess.run(cmd_mux, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.unlink(tmp_video)

    def export_live(
        self,
        frames: list[np.ndarray],
        audio_path: str,
        output_path: str,
        fps: int = 24,
    ) -> None:
        """Assemble live-session frames with recorded audio."""
        self.assemble_mp4(frames, audio_path, output_path, fps=fps)
