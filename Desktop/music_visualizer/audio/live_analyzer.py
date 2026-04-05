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
        hf_energy = spectrum[hf_mask].sum()
        lf_energy = spectrum[~hf_mask].sum() + 1e-8
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
