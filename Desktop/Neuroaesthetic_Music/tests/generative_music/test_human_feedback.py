"""Tests for the DBTL human feedback loop backend."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import pytest
from pathlib import Path

from src.generative_music.gesture_designer.feedback_store import FeedbackStore


# ── FeedbackStore tests ───────────────────────────────────────────────────────

def test_feedback_store_write_read(tmp_path):
    """A submission round-trips through FeedbackStore correctly."""

    store = FeedbackStore(directory=tmp_path)
    record = store.save_rating(
        participant_id='alice',
        item_type='gesture',
        item_id='Four_Note_Sweep',
        user_valence=63.0,
        user_arousal=28.0,
        user_stars=4,
        ml_valence=71.0,
        ml_arousal=34.0,
    )

    assert record['participant_id'] == 'alice'
    assert record['user_valence'] == 63.0
    assert record['user_stars'] == 4
    assert abs(record['delta_valence'] - 8.0) < 1e-6   # ml - user = 71 - 63
    assert abs(record['delta_arousal'] - 6.0) < 1e-6

    loaded = store.load_ratings()
    assert len(loaded) == 1
    assert loaded[0]['item_id'] == 'Four_Note_Sweep'
    assert 'timestamp' in loaded[0]


def test_feedback_store_session_threshold(tmp_path):
    """Sessions with fewer than 3 ratings are excluded from pipeline export."""

    store = FeedbackStore(directory=tmp_path)

    # alice rates 2 items — below threshold
    for i in range(2):
        store.save_rating('alice', 'gesture', f'g{i}', 50, 50, 3, 55, 55)

    # bob rates 3 items — meets threshold
    for i in range(3):
        store.save_rating('bob', 'gesture', f'g{i}', 50, 50, 3, 55, 55)

    exported = store.export_for_pipeline()
    pids = {r['participant_id'] for r in exported}
    assert 'alice' not in pids
    assert 'bob' in pids
    assert len(exported) == 3
