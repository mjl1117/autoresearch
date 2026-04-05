# Human Feedback Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix invisible star ratings, make the Chord Gesture tab play gestural chord sequences, and rewrite the Music tab to collect ML training data on gesture congruency via simultaneous layered playback.

**Architecture:** All changes are confined to `src/generative_music/gesture_designer/human_feedback.py`. Two new module-level helper functions (`_compute_star_states`, `_build_chord_gesture`) are extracted so their logic is testable without a QApplication. The Music tab spawns one `GesturePlayer` per layer and calls `play_gesture()` on each — `GesturePlayer` already manages its own background thread, so layers run in parallel with no extra threading code.

**Tech Stack:** PyQt6, python-osc (via GesturePlayer), existing `ChordPredictor.find_path()`, `GestureLibrary`, `LibraryRanker`, `FeedbackStore`.

---

## Files

| Action | Path |
|--------|------|
| Modify | `src/generative_music/gesture_designer/human_feedback.py` |
| Modify | `tests/generative_music/test_human_feedback.py` |

---

## Task 1: Fix Star Rendering

Replace the five `QPushButton('☆')` star buttons with `_ClickableLabel` instances (a tiny `QLabel` subclass). Extract `_compute_star_states` as a pure function so the fill/unfill logic is unit-testable without a QApplication.

**Files:**
- Modify: `src/generative_music/gesture_designer/human_feedback.py`
- Modify: `tests/generative_music/test_human_feedback.py`

- [ ] **Step 1: Write failing tests for `_compute_star_states`**

Add to `tests/generative_music/test_human_feedback.py` (after the existing tests):

```python
# ── Star state logic ──────────────────────────────────────────────────────────

def test_star_states_all_empty():
    from src.generative_music.gesture_designer.human_feedback import _compute_star_states
    result = _compute_star_states(0)
    assert [t for t, _ in result] == ['☆', '☆', '☆', '☆', '☆']

def test_star_states_three_filled():
    from src.generative_music.gesture_designer.human_feedback import _compute_star_states
    result = _compute_star_states(3)
    assert [t for t, _ in result] == ['★', '★', '★', '☆', '☆']

def test_star_states_click_down():
    """After clicking 5 then 2, stars 3-5 must unfill."""
    from src.generative_music.gesture_designer.human_feedback import _compute_star_states
    result = _compute_star_states(2)
    assert [t for t, _ in result] == ['★', '★', '☆', '☆', '☆']

def test_star_states_all_filled():
    from src.generative_music.gesture_designer.human_feedback import _compute_star_states
    result = _compute_star_states(5)
    assert all(t == '★' for t, _ in result)

def test_star_states_colors():
    from src.generative_music.gesture_designer.human_feedback import _compute_star_states
    result = _compute_star_states(2)
    colors = [c for _, c in result]
    assert colors == ['#C0A020', '#C0A020', '#B8B5A4', '#B8B5A4', '#B8B5A4']
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/matthew/Desktop/Neuroaesthetic_Music
python -m pytest tests/generative_music/test_human_feedback.py::test_star_states_all_empty -v
```

Expected: `ImportError` or `AttributeError` — `_compute_star_states` does not exist yet.

- [ ] **Step 3: Add `_compute_star_states` and `_ClickableLabel` to `human_feedback.py`**

In `human_feedback.py`, add after the `_sep()` function (around line 76), before the `# ── Helper functions` comment:

```python
# ── Star widget helpers ───────────────────────────────────────────────────────

def _compute_star_states(n: int, total: int = 5) -> list[tuple[str, str]]:
    """Return (character, color) for each star position given n filled stars.

    Filled stars use gold; empty stars use muted gray.
    Pure function — no Qt dependency — so it can be unit-tested directly.
    """
    return [('★', '#C0A020') if i < n else ('☆', '#B8B5A4')
            for i in range(total)]


class _ClickableLabel(QLabel):
    """A QLabel that fires a callback when clicked — used for star ratings."""

    def __init__(self, n: int, callback, parent=None):
        super().__init__('☆', parent)
        self._n = n
        self._callback = callback
        self.setFixedSize(40, 40)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            f'color:#B8B5A4; font-size:20pt; background:{PANEL_BG};')

    def mousePressEvent(self, event):
        self._callback(self._n)
        super().mousePressEvent(event)
```

- [ ] **Step 4: Replace QPushButton stars with `_ClickableLabel` in `_EvalWidget.__init__`**

In `_EvalWidget.__init__`, find the star-building loop (currently lines 243–254):

```python
        self._star_btns: list[QPushButton] = []
        for i in range(1, 6):
            b = QPushButton('☆')
            b.setFixedSize(40, 40)
            b.setStyleSheet(
                f'QPushButton {{ background:transparent; color:#B8B5A4; '
                f'border:none; font-size:20pt; }}'
                f'QPushButton:hover {{ color:{ACCENT}; }}'
            )
            b.clicked.connect(lambda _, n=i: self._set_stars(n))
            star_row.addWidget(b)
            self._star_btns.append(b)
```

Replace it with:

```python
        self._star_btns: list[_ClickableLabel] = []
        for i in range(1, 6):
            lbl = _ClickableLabel(i, self._set_stars)
            star_row.addWidget(lbl)
            self._star_btns.append(lbl)
```

- [ ] **Step 5: Update `_set_stars` to update labels instead of buttons**

Find `_set_stars` (currently lines 339–349):

```python
    def _set_stars(self, n: int):
        self._stars = n
        for i, b in enumerate(self._star_btns):
            filled = i < n
            b.setText('★' if filled else '☆')
            b.setStyleSheet(
                f'QPushButton {{ background:transparent; '
                f'color:{"#C0A020" if filled else "#B8B5A4"}; '
                f'border:none; font-size:20pt; }}'
                f'QPushButton:hover {{ color:{ACCENT}; }}'
            )
```

Replace it with:

```python
    def _set_stars(self, n: int):
        self._stars = n
        for lbl, (char, color) in zip(self._star_btns, _compute_star_states(n)):
            lbl.setText(char)
            lbl.setStyleSheet(
                f'color:{color}; font-size:20pt; background:{PANEL_BG};')
```

- [ ] **Step 6: Run the star tests**

```bash
cd /Users/matthew/Desktop/Neuroaesthetic_Music
python -m pytest tests/generative_music/test_human_feedback.py -k "star" -v
```

Expected output:
```
test_star_states_all_empty PASSED
test_star_states_three_filled PASSED
test_star_states_click_down PASSED
test_star_states_all_filled PASSED
test_star_states_colors PASSED
```

- [ ] **Step 7: Run full test suite to check no regressions**

```bash
python -m pytest tests/generative_music/test_human_feedback.py -v
```

Expected: all previously-passing tests still pass.

- [ ] **Step 8: Commit**

```bash
git add src/generative_music/gesture_designer/human_feedback.py \
        tests/generative_music/test_human_feedback.py
git commit -m "feat: fix star rating rendering with QLabel-based clickable stars"
```

---

## Task 2: Chord Gesture Builder

Extract `_build_chord_gesture` as a pure module-level function. Test it. Then rewrite `_ChordTab` to use `find_path()` targeting a single V/A point and play the resulting `Gesture`.

**Files:**
- Modify: `src/generative_music/gesture_designer/human_feedback.py`
- Modify: `tests/generative_music/test_human_feedback.py`

- [ ] **Step 1: Write the failing test for `_build_chord_gesture`**

Add to `tests/generative_music/test_human_feedback.py`:

```python
# ── Chord gesture builder ─────────────────────────────────────────────────────

def test_build_chord_gesture_event_count():
    """Each chord record produces exactly one NoteEvent."""
    from src.generative_music.gesture_designer.human_feedback import _build_chord_gesture
    chords = [
        {'chord_id': 'A', 'chord_type': 0, 'num_voices': 3,
         'balance': 0.7, 'inversion': 0, 'weights': []},
        {'chord_id': 'B', 'chord_type': 1, 'num_voices': 3,
         'balance': 0.7, 'inversion': 0, 'weights': []},
        {'chord_id': 'C', 'chord_type': 7, 'num_voices': 4,
         'balance': 0.7, 'inversion': 0, 'weights': []},
    ]
    g = _build_chord_gesture(chords, bpm=80.0)
    assert len(g.events) == 3

def test_build_chord_gesture_chord_enabled():
    """All events must have chord.enabled = True."""
    from src.generative_music.gesture_designer.human_feedback import _build_chord_gesture
    chords = [
        {'chord_id': 'X', 'chord_type': 6, 'num_voices': 4,
         'balance': 0.6, 'inversion': 1, 'weights': []},
    ]
    g = _build_chord_gesture(chords, bpm=72.0)
    assert g.events[0].chord.enabled is True
    assert g.events[0].chord.chord_type == 6
    assert g.events[0].chord.num_voices == 4
    assert g.events[0].chord.inversion == 1

def test_build_chord_gesture_beat_durations():
    """All beat durations must come from [1, 2, 3]."""
    from src.generative_music.gesture_designer.human_feedback import _build_chord_gesture
    chords = [
        {'chord_id': str(i), 'chord_type': 0, 'num_voices': 3,
         'balance': 0.7, 'inversion': 0, 'weights': []}
        for i in range(6)
    ]
    g = _build_chord_gesture(chords, bpm=80.0)
    assert all(ev.beats in (1, 2, 3) for ev in g.events)

def test_build_chord_gesture_bpm():
    """BPM is preserved on the returned Gesture."""
    from src.generative_music.gesture_designer.human_feedback import _build_chord_gesture
    g = _build_chord_gesture(
        [{'chord_id': 'Z', 'chord_type': 0, 'num_voices': 3,
          'balance': 0.7, 'inversion': 0, 'weights': []}],
        bpm=120.0,
    )
    assert g.bpm == 120.0

def test_build_chord_gesture_empty_input():
    """Empty chord list returns a Gesture with no events."""
    from src.generative_music.gesture_designer.human_feedback import _build_chord_gesture
    g = _build_chord_gesture([], bpm=80.0)
    assert g.events == []
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/generative_music/test_human_feedback.py -k "chord_gesture" -v
```

Expected: `ImportError` — `_build_chord_gesture` not defined yet.

- [ ] **Step 3: Add `_build_chord_gesture` to `human_feedback.py`**

Add after `_weighted_random_chord` (after line ~158), before `_play_chord_via_player`:

```python
def _build_chord_gesture(chords: list[dict], bpm: float = 72.0) -> 'Gesture':
    """Convert a sequence of chord records into a playable gestural Gesture.

    Each chord becomes one NoteEvent with chord.enabled=True.
    Beat durations cycle through [1, 2, 3] to give varied, gesture-like rhythm.
    The root is always A4 (440 Hz) — matching the existing chord preview behaviour.
    """
    from .gesture_model import Gesture, NoteEvent, ChordConfig, PartialWeights
    BEAT_CYCLE = [1, 2, 3, 2]
    events = []
    for i, chord in enumerate(chords):
        pw = PartialWeights()
        for j, w in enumerate(chord.get('weights', [1.0] * 16)[:16]):
            pw.set_index(j, float(w))
        cc = ChordConfig(
            enabled=True,
            chord_type=int(chord.get('chord_type', 0)),
            num_voices=int(chord.get('num_voices', 3)),
            balance=float(chord.get('balance', 0.7)),
            inversion=int(chord.get('inversion', 0)),
        )
        ev = NoteEvent(
            note='A', accidental='', octave=4,
            amplitude=0.25, brightness=0.5,
            beats=BEAT_CYCLE[i % len(BEAT_CYCLE)],
            partials=pw,
            chord=cc,
        )
        events.append(ev)
    return Gesture(name='chord_gesture', bpm=bpm, events=events)
```

- [ ] **Step 4: Run chord gesture tests**

```bash
python -m pytest tests/generative_music/test_human_feedback.py -k "chord_gesture" -v
```

Expected: all five tests pass.

- [ ] **Step 5: Rewrite `_ChordTab`**

Replace the entire `_ChordTab` class (lines ~428–458) with:

```python
class _ChordTab(_EvalWidget):
    def __init__(self, store, ranker, pid_fn, player: GesturePlayer,
                 predictor: ChordPredictor, parent=None):
        self._player = player
        self._predictor = predictor
        self._current_gesture = None
        super().__init__(store, ranker, pid_fn, 'chord', parent)

    def _load_item(self) -> Optional[dict]:
        if not self._predictor.chords:
            return None
        tv = random.uniform(10.0, 90.0)
        ta = random.uniform(10.0, 90.0)
        self._ml_valence = tv
        self._ml_arousal = ta
        steps = random.randint(4, 6)
        chords = self._predictor.find_path((tv, ta), (tv, ta), steps=steps)
        if not chords:
            return None
        self._current_gesture = _build_chord_gesture(chords)
        chord_ids = ' · '.join(c['chord_id'] for c in chords[:3])
        return {'name': f'Chord Gesture  (V={tv:.0f} A={ta:.0f})  {chord_ids}',
                'chords': chords}

    def _play_item(self, item: dict):
        if self._current_gesture:
            self._player.play_gesture(self._current_gesture)

    def _stop_item(self):
        self._player.stop_gesture()

    def _item_id(self, item: dict) -> str:
        return item.get('name', 'chord_gesture')

    def _item_display_name(self, item: dict) -> str:
        return item.get('name', 'Chord Gesture')

    def _after_submit(self, item_id, participant_id, stars):
        pass  # chord gesture ratings stored by item_id; no per-chord ranker update
```

- [ ] **Step 6: Run the full test suite**

```bash
python -m pytest tests/generative_music/test_human_feedback.py -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/generative_music/gesture_designer/human_feedback.py \
        tests/generative_music/test_human_feedback.py
git commit -m "feat: chord gesture tab now plays gestural chord sequences via find_path()"
```

---

## Task 3: Music Tab — Layered Congruency Composition

Rewrite `_MusicTab` to: pick one V/A target, select 2–3 layers (gesture library or chord gesture) weighted toward that target, play all layers simultaneously via separate `GesturePlayer` instances, and store the rating as `'music_layer'` in `FeedbackStore`. Pass `ChordPredictor` to `_MusicTab` from `HumanFeedbackWindow`.

**Files:**
- Modify: `src/generative_music/gesture_designer/human_feedback.py`
- Modify: `tests/generative_music/test_human_feedback.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/generative_music/test_human_feedback.py`:

```python
# ── Music layer helpers ───────────────────────────────────────────────────────

def test_music_layer_rating_stored_as_music_layer_type(tmp_path):
    """Music congruency ratings are stored with item_type='music_layer'."""
    store = FeedbackStore(directory=tmp_path)
    record = store.save_rating(
        participant_id='alice',
        item_type='music_layer',
        item_id='sweep+chord_42+arp',
        user_valence=60.0,
        user_arousal=40.0,
        user_stars=4,
        ml_valence=65.0,
        ml_arousal=38.0,
    )
    assert record['item_type'] == 'music_layer'
    assert record['item_id'] == 'sweep+chord_42+arp'

def test_music_layer_item_id_format(tmp_path):
    """item_id is '+'-joined layer names."""
    store = FeedbackStore(directory=tmp_path)
    record = store.save_rating(
        participant_id='bob',
        item_type='music_layer',
        item_id='gesture_a+chord_001+gesture_b',
        user_valence=50.0, user_arousal=50.0, user_stars=3,
        ml_valence=50.0, ml_arousal=50.0,
    )
    parts = record['item_id'].split('+')
    assert len(parts) == 3

def test_music_layer_pipeline_export_included(tmp_path):
    """music_layer records are included in pipeline export (same threshold as others)."""
    store = FeedbackStore(directory=tmp_path)
    for i in range(3):
        store.save_rating('alice', 'music_layer', f'a+b+c_{i}',
                          50, 50, 3, 50, 50)
    exported = store.export_for_pipeline()
    types = {r['item_type'] for r in exported}
    assert 'music_layer' in types
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/generative_music/test_human_feedback.py -k "music_layer" -v
```

Expected: `ImportError` on the `_MusicTab` import path tests, but the `FeedbackStore` tests should pass (they test the data layer only, which already accepts any `item_type` string). Verify the FeedbackStore tests pass.

- [ ] **Step 3: Add `_select_music_layers` to `human_feedback.py`**

Add after `_build_chord_gesture` (and its helper `_play_chord_via_player` can be removed if unused — leave it for now):

```python
def _select_music_layers(lib: 'GestureLibrary', predictor: ChordPredictor,
                          ranker: LibraryRanker, pid: str,
                          target_va: tuple[float, float], n: int) -> list[dict]:
    """Pick n layers near target_va from the gesture library and chord predictor.

    Each layer is a dict with keys:
        'name'    : str — display name and contribution to item_id
        'gesture' : Gesture — ready to play

    Selection is 50/50 gesture vs chord gesture when both sources are available,
    falling back to gesture-only when the chord predictor has no data.
    The gesture library's weighted_random() already applies LibraryRanker weights,
    so under-explored items surface more often — no extra weighting needed here.
    """
    tv, ta = target_va
    layers: list[dict] = []

    for _ in range(n):
        use_chord = bool(predictor.chords) and random.random() < 0.5
        if use_chord:
            steps = random.randint(3, 5)
            chords = predictor.find_path((tv, ta), (tv, ta), steps=steps)
            if chords:
                gesture = _build_chord_gesture(chords)
                layers.append({
                    'name': f'chord_{chords[0]["chord_id"]}',
                    'gesture': gesture,
                })
        else:
            item = lib.weighted_random(participant_id=pid)
            if item:
                gesture = lib.load(item['path'])
                if gesture:
                    layers.append({'name': item['name'], 'gesture': gesture})

    return layers
```

- [ ] **Step 4: Rewrite `_MusicTab`**

Replace the entire `_MusicTab` class (lines ~463–516) with:

```python
class _MusicTab(_EvalWidget):
    """Plays 2–3 simultaneous gesture/chord layers targeted at one V/A point.

    Stores ratings as item_type='music_layer' so the ML can learn combination
    congruency independently of single-gesture quality.
    """

    def __init__(self, store, ranker, pid_fn, player: GesturePlayer,
                 predictor: ChordPredictor, parent=None):
        self._player = player          # kept for _stop_item fallback
        self._predictor = predictor
        self._lib = GestureLibrary()
        self._layers: list[dict] = []  # [{name, gesture}]
        self._layer_players: list[GesturePlayer] = []
        self._target_va: tuple[float, float] = (50.0, 50.0)
        super().__init__(store, ranker, pid_fn, 'music_layer', parent)

    def _load_item(self) -> Optional[dict]:
        pid = self._get_pid()
        tv = random.uniform(10.0, 90.0)
        ta = random.uniform(10.0, 90.0)
        self._target_va = (tv, ta)
        self._ml_valence = tv
        self._ml_arousal = ta

        n_layers = random.randint(2, 3)
        self._layers = _select_music_layers(
            self._lib, self._predictor, self._ranker, pid, (tv, ta), n_layers)

        if not self._layers:
            return None

        layer_names = ',  '.join(layer['name'] for layer in self._layers)
        return {
            'name': (f'V={tv:.0f}  A={ta:.0f}\n'
                     f'Layers:  {layer_names}'),
        }

    def _play_item(self, item: dict):
        self._stop_item()
        self._layer_players = []
        for layer in self._layers:
            p = GesturePlayer()
            self._layer_players.append(p)
            p.play_gesture(layer['gesture'])   # GesturePlayer runs its own thread

    def _stop_item(self):
        for p in self._layer_players:
            p.stop_gesture()
        self._layer_players = []

    def _item_id(self, item: dict) -> str:
        return '+'.join(layer['name'] for layer in self._layers)

    def _item_display_name(self, item: dict) -> str:
        return item.get('name', '—')

    def _after_submit(self, item_id, participant_id, stars):
        pass   # congruency stored as music_layer; no per-item ranker update
```

- [ ] **Step 5: Update `HumanFeedbackWindow` to pass `predictor` to `_MusicTab`**

In `HumanFeedbackWindow.__init__`, find the tab construction (around line 586–588):

```python
        tabs.addTab(_GestureTab(store, ranker, pid_fn, player), 'Gesture')
        tabs.addTab(_ChordTab(store, ranker, pid_fn, player, predictor), 'Chord Gesture')
        tabs.addTab(_MusicTab(store, ranker, pid_fn, player), 'Music')
```

Change the last line to:

```python
        tabs.addTab(_GestureTab(store, ranker, pid_fn, player), 'Gesture')
        tabs.addTab(_ChordTab(store, ranker, pid_fn, player, predictor), 'Chord Gesture')
        tabs.addTab(_MusicTab(store, ranker, pid_fn, player, predictor), 'Music')
```

- [ ] **Step 6: Run the full test suite**

```bash
python -m pytest tests/generative_music/test_human_feedback.py -v
```

Expected: all tests pass, including the three new `music_layer` tests.

- [ ] **Step 7: Commit**

```bash
git add src/generative_music/gesture_designer/human_feedback.py \
        tests/generative_music/test_human_feedback.py
git commit -m "feat: music tab plays simultaneous layers for congruency ML data collection"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task covering it |
|------------------|-----------------|
| Stars start as outlines, fill dynamically on click | Task 1 (_ClickableLabel + _compute_star_states) |
| Click 3 → 3 filled; click 5 → 5 filled; click 2 → 3 unfill | Task 1 (_set_stars via _compute_star_states) |
| Chord Gesture tab plays gesture of chords, not single chord | Task 2 (_build_chord_gesture + rewired _ChordTab) |
| Chord gesture uses find_path() targeting one V/A | Task 2 (find_path with same start/end) |
| Music tab composes a short piece with simultaneous layers | Task 3 (_MusicTab rewrite) |
| Music tab stores rating as music_layer, separate from gesture/chord | Task 3 (item_type='music_layer') |
| Rule-based V/A-targeted selection (not random noise) | Task 3 (_select_music_layers with weighted_random + find_path) |
| FeedbackStore unchanged | Out of scope — confirmed no schema change needed |

**Placeholder scan:** None found.

**Type consistency:**
- `_build_chord_gesture` takes `list[dict], float` → returns `Gesture` ✓ Used in Task 2 and Task 3
- `_select_music_layers` returns `list[dict]` with keys `name`, `gesture` ✓ Consumed by `_MusicTab._play_item` and `_item_id`
- `_compute_star_states` returns `list[tuple[str, str]]` ✓ Consumed by `_set_stars`
- `_ClickableLabel` callback takes `int` ✓ `_set_stars(n: int)` matches
- `_MusicTab.__init__` now takes `predictor: ChordPredictor` ✓ `HumanFeedbackWindow` updated to pass it
