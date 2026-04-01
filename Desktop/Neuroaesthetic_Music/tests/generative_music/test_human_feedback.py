"""Tests for the DBTL human feedback loop backend."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import pytest
from pathlib import Path

from src.generative_music.gesture_designer.feedback_store import FeedbackStore
from src.generative_music.gesture_designer.library_ranker import LibraryRanker


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


# ── LibraryRanker tests ───────────────────────────────────────────────────────

def test_library_ranker_gesture_update(tmp_path):
    """Star rating is written into gesture JSON under correct participant key."""
    # Create a minimal gesture JSON file
    gesture_dir = tmp_path / 'gestures'
    gesture_dir.mkdir()
    gesture_path = gesture_dir / 'sweep.json'
    gesture_path.write_text(json.dumps({'name': 'sweep', 'bpm': 80, 'events': []}))

    ranker = LibraryRanker(gesture_dir=gesture_dir, feedback_dir=tmp_path / 'feedback')
    ranker.update_gesture_rating('sweep', 'alice', 5)

    data = json.loads(gesture_path.read_text())
    assert data['ratings']['alice'] == 5.0


def test_library_ranker_chord_sidecar(tmp_path):
    """Chord rating written to sidecar JSONL; per-participant mean computed correctly."""
    feedback_dir = tmp_path / 'feedback'
    ranker = LibraryRanker(gesture_dir=tmp_path / 'gestures', feedback_dir=feedback_dir)

    ranker.update_chord_rating('chord_001', 'bob', 4)
    ranker.update_chord_rating('chord_001', 'bob', 2)

    mean = ranker.get_participant_chord_rating('chord_001', 'bob')
    assert abs(mean - 3.0) < 1e-6


def test_library_ranker_no_cross_participant_bleed(tmp_path):
    """Alice's ratings do not affect Bob's weighted draw."""
    gesture_dir = tmp_path / 'gestures'
    gesture_dir.mkdir()
    (gesture_dir / 'sweep.json').write_text(json.dumps({'name': 'sweep', 'bpm': 80, 'events': []}))

    ranker = LibraryRanker(gesture_dir=gesture_dir, feedback_dir=tmp_path / 'feedback')
    ranker.update_gesture_rating('sweep', 'alice', 5)

    # Bob has not rated — should return None, not Alice's rating
    bob_rating = ranker.get_participant_gesture_rating('sweep', 'bob')
    assert bob_rating is None


def test_gesture_library_weighted_random(tmp_path):
    """Under-rated / unrated items surface at higher probability than 5-star items."""
    import random
    from src.generative_music.gesture_designer.gesture_library import GestureLibrary

    gesture_dir = tmp_path / 'gestures'
    gesture_dir.mkdir()

    # Create 3 gesture files: unrated, 1-star, 5-star
    for name, ratings in [('alpha', {}), ('beta', {'alice': 1.0}), ('gamma', {'alice': 5.0})]:
        (gesture_dir / f'{name}.json').write_text(json.dumps(
            {'name': name, 'bpm': 80, 'events': [], 'ratings': ratings}
        ))

    lib = GestureLibrary(directory=gesture_dir)

    # Sample many times and count how often gamma (5-star) is drawn vs alpha (unrated)
    random.seed(42)
    counts = {'alpha': 0, 'beta': 0, 'gamma': 0}
    for _ in range(300):
        item = lib.weighted_random(participant_id='alice')
        counts[item['name']] += 1

    # gamma (5-star) should appear significantly less than alpha (unrated)
    assert counts['gamma'] < counts['alpha'], (
        f"5-star item appeared {counts['gamma']}x vs unrated {counts['alpha']}x — "
        "exploration weighting not working"
    )
