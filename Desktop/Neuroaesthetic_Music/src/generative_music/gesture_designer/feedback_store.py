"""
feedback_store.py
Persists human feedback ratings to data/feedback/ratings.jsonl.

MJL Neuroaesthetic Music Research — 2026
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_DEFAULT_DIR = Path(__file__).parent.parent.parent.parent / 'data' / 'feedback'


class FeedbackStore:
    """Read/write per-submission rating records to JSONL."""

    def __init__(self, directory: Optional[Path] = None):
        self.directory = Path(directory) if directory else _DEFAULT_DIR
        self.directory.mkdir(parents=True, exist_ok=True)
        self._ratings_path = self.directory / 'ratings.jsonl'

    def save_rating(self, participant_id: str, item_type: str, item_id: str,
                    user_valence: float, user_arousal: float, user_stars: int,
                    ml_valence: float, ml_arousal: float) -> dict:
        """Append one rating record and return it."""
        record = {
            'participant_id': participant_id,
            'item_type': item_type,          # 'gesture' | 'chord' | 'music'
            'item_id': item_id,
            'user_valence': float(user_valence),
            'user_arousal': float(user_arousal),
            'user_stars': int(user_stars),
            'ml_valence': float(ml_valence),
            'ml_arousal': float(ml_arousal),
            'delta_valence': float(ml_valence - user_valence),
            'delta_arousal': float(ml_arousal - user_arousal),
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }
        with open(self._ratings_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
        return record

    def load_ratings(self) -> list[dict]:
        """Return all rating records from disk."""
        if not self._ratings_path.exists():
            return []
        records = []
        with open(self._ratings_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

    def export_for_pipeline(self) -> list[dict]:
        """Return records from participants who rated >= 3 items (engagement filter).

        Each record gains sample_weight = 0.3 so the ML pipeline down-weights
        subjective ratings relative to biosignal-derived labels (weight 1.0).
        """
        all_records = self.load_ratings()
        by_participant: defaultdict[str, list] = defaultdict(list)
        for r in all_records:
            by_participant[r['participant_id']].append(r)
        result = []
        for records in by_participant.values():
            if len(records) >= 3:
                for r in records:
                    result.append({**r, 'sample_weight': 0.3})
        return result
