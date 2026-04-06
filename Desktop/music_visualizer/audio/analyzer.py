from __future__ import annotations
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

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
        # Fallback: beat_track returns 0.0 for non-percussive material (e.g. pure tones)
        if bpm <= 0.0:
            fallback = librosa.feature.tempo(y=y, sr=sr)
            bpm = float(fallback[0]) if len(fallback) > 0 else 120.0

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

    # Formats that need conversion to WAV before librosa can load them reliably
    _FFMPEG_CONVERT = {".aif", ".aiff", ".m4a", ".mp4", ".flac", ".ogg", ".opus", ".wma"}

    @staticmethod
    def _resolve_audio(path: str) -> str:
        """Convert non-native formats to a temp WAV using ffmpeg, return path to use."""
        ext = Path(path).suffix.lower()

        if ext == ".avi":
            from moviepy import VideoFileClip
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            try:
                clip = VideoFileClip(path)
                clip.audio.write_audiofile(tmp.name, verbose=False, logger=None)
                clip.close()
                return tmp.name
            except Exception:
                os.unlink(tmp.name)
                return path

        if ext in PrerecordedAnalyzer._FFMPEG_CONVERT:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", path, "-vn", "-ac", "1", tmp.name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if result.returncode == 0:
                return tmp.name
            os.unlink(tmp.name)
            # ffmpeg failed — let librosa try the original (may work on some systems)
            return path

        return path
