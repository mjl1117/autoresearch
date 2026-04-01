"""
chord_predictor.py
Loads the scored chord library and provides nearest-neighbour queries
in valence/arousal space.

The raw RF predictions are normalised to [0, 100] at load time so that
the user's crosshair target (0–100 on each axis) maps meaningfully onto
the full spread of the chord data — the raw predictions cluster in a
narrow band (~33–38 / ~41–48) due to limited training data.

MJL + Claude — Neuroaesthetic Music Research 2026
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default path relative to project root
_DEFAULT_JSONL = Path(__file__).parent.parent.parent.parent / 'models' / 'chord_predictions.jsonl'


class ChordPredictor:
    """
    Nearest-neighbour chord selector in normalised valence/arousal space.

    Attributes
    ----------
    chords : list[dict]
        All chord records (original fields + 'norm_valence', 'norm_arousal').
    va_matrix : np.ndarray  shape (N, 2)
        Normalised V/A coordinates for every chord, used for distance queries.
    """

    def __init__(self, jsonl_path: Optional[Path] = None):
        path = Path(jsonl_path) if jsonl_path else _DEFAULT_JSONL
        self.chords: list[dict] = []
        self.va_matrix: np.ndarray = np.empty((0, 2))
        self._load(path)

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self, path: Path) -> None:
        if not path.exists():
            logger.warning(f"ChordPredictor: {path} not found — empty library")
            return

        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        if not records:
            logger.warning("ChordPredictor: chord_predictions.jsonl is empty")
            return

        raw_v = np.array([r['predicted_valence'] for r in records], dtype=float)
        raw_a = np.array([r['predicted_arousal'] for r in records], dtype=float)

        # Normalise to [0, 100].  Guard against degenerate (flat) distributions.
        def _norm(arr: np.ndarray) -> np.ndarray:
            lo, hi = arr.min(), arr.max()
            if hi - lo < 1e-6:
                return np.full_like(arr, 50.0)
            return (arr - lo) / (hi - lo) * 100.0

        norm_v = _norm(raw_v)
        norm_a = _norm(raw_a)

        for i, rec in enumerate(records):
            rec['norm_valence'] = float(norm_v[i])
            rec['norm_arousal'] = float(norm_a[i])

        self.chords = records
        self.va_matrix = np.column_stack([norm_v, norm_a])
        logger.info(f"ChordPredictor: loaded {len(records)} chords "
                    f"(V range {raw_v.min():.1f}–{raw_v.max():.1f}, "
                    f"A range {raw_a.min():.1f}–{raw_a.max():.1f})")

    # ── Queries ───────────────────────────────────────────────────────────────

    def find_nearest(self, valence: float, arousal: float,
                     n: int = 1,
                     exclude_ids: Optional[set] = None,
                     participant_id: str = '',
                     ranker=None) -> list[dict]:
        """
        Return the n closest chords to (valence, arousal) in normalised V/A space.

        When participant_id and ranker are provided, each chord's effective distance
        is multiplied by a rating factor so that higher-rated chords are preferred:
            factor = 1.0 / (1.0 + mean_stars * 0.2)
        A 5-star chord gets factor ≈ 0.50 (distance halved); unrated stays at 1.0.

        Parameters
        ----------
        valence, arousal : float
            Target coordinates in [0, 100].
        n : int
            Number of results to return.
        exclude_ids : set[str] | None
            chord_id values to skip (avoids repeating the same chord).
        participant_id : str
            Participant ID for per-participant rating weighting.
        ranker : Any
            LibraryRanker instance with get_participant_chord_rating() method.

        Returns
        -------
        list[dict]  — each dict is a chord record with 'norm_valence' / 'norm_arousal'.
        """
        if len(self.chords) == 0:
            return []

        query = np.array([valence, arousal], dtype=float)
        dists = np.linalg.norm(self.va_matrix - query, axis=1).copy()

        if exclude_ids:
            for i, rec in enumerate(self.chords):
                if rec['chord_id'] in exclude_ids:
                    dists[i] = np.inf

        if participant_id and ranker is not None:
            for i, rec in enumerate(self.chords):
                if np.isinf(dists[i]):
                    continue
                stars = ranker.get_participant_chord_rating(
                    rec['chord_id'], participant_id)
                if stars is not None and stars > 0:
                    dists[i] *= 1.0 / (1.0 + stars * 0.2)

        idx = np.argsort(dists)
        results = []
        for i in idx:
            if len(results) >= n:
                break
            results.append(self.chords[i])
        return results

    def find_path(self, start_va: tuple[float, float],
                  end_va: tuple[float, float],
                  steps: int = 6) -> list[dict]:
        """
        Return a sequence of chords that traverses from start_va to end_va.

        Linearly interpolates `steps` waypoints between start and end, then
        picks the nearest chord to each waypoint.  Consecutive duplicates are
        removed so the same chord is never played back-to-back.

        Parameters
        ----------
        start_va, end_va : (valence, arousal) tuples in [0, 100].
        steps : int
            Number of chords to return.

        Returns
        -------
        list[dict]  — ordered sequence of chord records (length ≤ steps).
        """
        if len(self.chords) == 0:
            return []

        steps = max(1, steps)
        sv, sa = float(start_va[0]), float(start_va[1])
        ev, ea = float(end_va[0]), float(end_va[1])

        path: list[dict] = []
        last_id: Optional[str] = None

        for i in range(steps):
            t = i / max(steps - 1, 1)
            wv = sv + t * (ev - sv)
            wa = sa + t * (ea - sa)
            exclude = {last_id} if last_id else None
            candidates = self.find_nearest(wv, wa, n=3, exclude_ids=exclude)
            if not candidates:
                continue
            chosen = candidates[0]
            path.append(chosen)
            last_id = chosen['chord_id']

        return path
