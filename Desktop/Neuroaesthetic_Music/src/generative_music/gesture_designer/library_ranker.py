"""
library_ranker.py
Updates per-participant star scores in gesture JSON files and a chord rating sidecar.

Ratings are stored per-participant and never aggregated across participants,
preserving individual taste profiles.

MJL Neuroaesthetic Music Research — 2026
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_DEFAULT_GESTURE_DIR  = Path(__file__).parent.parent.parent.parent / 'data' / 'gesture_library'
_DEFAULT_FEEDBACK_DIR = Path(__file__).parent.parent.parent.parent / 'data' / 'feedback'


def _sanitise(name: str) -> str:
    s = re.sub(r'[^\w\s\-]', '', name).strip()
    s = re.sub(r'\s+', '_', s)
    return s[:64] or 'gesture'


class LibraryRanker:
    """Updates per-participant ratings in gesture JSON files and chord sidecar."""

    def __init__(self, gesture_dir: Optional[Path] = None,
                 feedback_dir: Optional[Path] = None):
        self.gesture_dir  = Path(gesture_dir)  if gesture_dir  else _DEFAULT_GESTURE_DIR
        self.feedback_dir = Path(feedback_dir) if feedback_dir else _DEFAULT_FEEDBACK_DIR
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self._chord_path = self.feedback_dir / 'chord_ratings.jsonl'
        # {chord_id: {participant_id: [stars, ...]}}
        self._chord_cache: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list))
        self._load_chord_cache()

    # ── Chord ratings ─────────────────────────────────────────────────────────

    def _load_chord_cache(self) -> None:
        if not self._chord_path.exists():
            return
        with open(self._chord_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    self._chord_cache[r['chord_id']][r['participant_id']].append(
                        float(r['stars']))
                except (json.JSONDecodeError, KeyError):
                    pass

    def update_chord_rating(self, chord_id: str, participant_id: str, stars: int) -> None:
        """Append one chord rating to sidecar JSONL and update the in-memory cache."""
        record = {
            'chord_id': chord_id,
            'participant_id': participant_id,
            'stars': int(stars),
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }
        with open(self._chord_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
        self._chord_cache[chord_id][participant_id].append(float(stars))

    def get_participant_chord_rating(self, chord_id: str,
                                     participant_id: str) -> Optional[float]:
        """Mean star rating for this chord by this participant, or None if unrated."""
        ratings = self._chord_cache.get(chord_id, {}).get(participant_id)
        if ratings:
            return sum(ratings) / len(ratings)
        return None

    # ── Gesture ratings ───────────────────────────────────────────────────────

    def _gesture_path(self, name: str) -> Optional[Path]:
        """Return path to gesture JSON file, or None if not found."""
        p = self.gesture_dir / f'{_sanitise(name)}.json'
        return p if p.exists() else None

    def update_gesture_rating(self, gesture_name: str,
                              participant_id: str, stars: int) -> None:
        """Update the running per-participant mean in the gesture JSON file."""
        path = self._gesture_path(gesture_name)
        if path is None:
            return
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        ratings = data.get('ratings', {})
        counts  = data.get('_rating_counts', {})
        prev    = ratings.get(participant_id)
        n       = counts.get(participant_id, 0)

        if prev is None:
            ratings[participant_id] = float(stars)
            counts[participant_id]  = 1
        else:
            ratings[participant_id] = (prev * n + float(stars)) / (n + 1)
            counts[participant_id]  = n + 1

        data['ratings']        = ratings
        data['_rating_counts'] = counts
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def get_participant_gesture_rating(self, gesture_name: str,
                                       participant_id: str) -> Optional[float]:
        """Read the per-participant rating from gesture JSON, or None if unrated."""
        path = self._gesture_path(gesture_name)
        if path is None:
            return None
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('ratings', {}).get(participant_id)
