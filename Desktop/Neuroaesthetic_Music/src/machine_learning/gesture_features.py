"""
gesture_features.py
Canonical 37-feature extractor shared by all V/A model training scripts
and the feedback-window prediction path.

Feature layout (37 elements total)
───────────────────────────────────
[0:24]   Spectral — 6 per-event stats × 4 aggregations (mean, std, min, max)
         rms, centroid, rolloff, bandwidth, flatness, zcr (placeholder, =0)
[24:28]  Temporal — bpm, mean_beats, std_beats, n_events
[28:37]  Harmonic transitions — 3 delta stats × 3 aggregations (mean, std, max)
         semitone_jump  : absolute pitch jump between consecutive events
         centroid_delta : brightness shift between consecutive events
         rms_delta      : loudness shift between consecutive events

Why transitions matter for arousal
───────────────────────────────────
Arousal correlates with tempo and rate-of-change rather than the timbre of
any single event.  Rapid semitone jumps and large spectral-centroid shifts
between events signal dynamism and harmonic tension that static per-event
statistics cannot capture.

MJL Neuroaesthetic Music Research — 2026
"""
from __future__ import annotations

import numpy as np

# ── Pitch helpers ─────────────────────────────────────────────────────────────

_NOTE_SEMITONES: dict[str, float] = {
    'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11,
}
_ACC_OFFSETS: dict[str, float] = {
    'bb': -2, 'b': -1, '3qb': -1.5, 'qb': -0.5,
    '': 0,
    'q#': 0.5, '#': 1, '3q#': 1.5, 'x': 2,
}

N_SPECTRAL   = 24
N_TEMPORAL   = 4
N_TRANSITION = 9
N_FEATURES   = N_SPECTRAL + N_TEMPORAL + N_TRANSITION  # 37


def _hz(note: str, accidental: str, octave: int) -> float:
    semi = _NOTE_SEMITONES.get(note.upper(), 9) + _ACC_OFFSETS.get(accidental, 0.0)
    midi = (octave + 1) * 12 + semi
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def _semitone(note: str, accidental: str, octave: int) -> float:
    """Absolute semitone value (C-1 = 0, A4 = 69)."""
    semi = _NOTE_SEMITONES.get(note.upper(), 9) + _ACC_OFFSETS.get(accidental, 0.0)
    return (octave + 1) * 12.0 + semi


# ── Main extractor ────────────────────────────────────────────────────────────

def extract_gesture_features(events: list[dict],
                              bpm: float = 80.0) -> np.ndarray | None:
    """Extract a 37-element feature vector from a gesture event list.

    Parameters
    ----------
    events : list[dict]
        The 'events' list from a gesture JSON file.
    bpm : float
        Gesture tempo in beats-per-minute.

    Returns
    -------
    np.ndarray of shape (37,), or None if no valid non-rest events exist.
    """
    spectral_rows: list[list[float]] = []
    semitones:     list[float] = []
    centroids:     list[float] = []
    rmss:          list[float] = []
    beat_list:     list[float] = []

    for ev in events:
        if ev.get('is_rest', False):
            continue

        note       = ev.get('note', 'A')
        accidental = ev.get('accidental', '')
        octave     = int(ev.get('octave', 4))

        freq    = _hz(note, accidental, octave)
        weights = np.array([ev.get('partials', {}).get(f'w{i}', 1.0)
                            for i in range(1, 17)], dtype=float)
        weights = np.clip(weights, 0.0, None)
        total   = weights.sum()
        if total < 1e-9:
            continue

        freqs    = np.array([freq * i for i in range(1, 17)])
        rms      = float(np.sqrt(np.mean(weights ** 2)))
        centroid = float(np.dot(weights, freqs) / total)
        bw       = float(np.sqrt(np.dot(weights, (freqs - centroid) ** 2) / total))
        cumsum   = np.cumsum(weights ** 2)
        rolloff  = float(freqs[min(np.searchsorted(cumsum, 0.85 * cumsum[-1]), 15)])
        gm       = float(np.exp(np.mean(np.log(weights + 1e-9))))
        flatness = float(gm / (total / 16 + 1e-9))

        spectral_rows.append([rms, centroid, rolloff, bw, flatness, 0.0])
        semitones.append(_semitone(note, accidental, octave))
        centroids.append(centroid)
        rmss.append(rms)
        beat_list.append(float(ev.get('beats', 1)))

    if not spectral_rows:
        return None

    # ── Spectral block [0:24] ─────────────────────────────────────────────────
    arr          = np.array(spectral_rows)
    spectral_vec = np.concatenate([arr.mean(0), arr.std(0), arr.min(0), arr.max(0)])

    # ── Temporal block [24:28] ────────────────────────────────────────────────
    beats_arr    = np.array(beat_list)
    temporal_vec = np.array([
        float(bpm),
        float(beats_arr.mean()),
        float(beats_arr.std()) if len(beats_arr) > 1 else 0.0,
        float(len(spectral_rows)),
    ])

    # ── Transition block [28:37] ──────────────────────────────────────────────
    n = len(spectral_rows)
    if n > 1:
        semi_d = np.abs(np.diff(np.array(semitones)))
        cent_d = np.abs(np.diff(np.array(centroids)))
        rms_d  = np.abs(np.diff(np.array(rmss)))

        def _agg(arr: np.ndarray) -> list[float]:
            return [float(arr.mean()),
                    float(arr.std()) if len(arr) > 1 else 0.0,
                    float(arr.max())]

        transition_vec = np.array(_agg(semi_d) + _agg(cent_d) + _agg(rms_d))
    else:
        transition_vec = np.zeros(N_TRANSITION)

    return np.concatenate([spectral_vec, temporal_vec, transition_vec])
