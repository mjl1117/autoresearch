# Human Feedback Window Redesign

**Date:** 2026-04-05
**File:** `src/generative_music/gesture_designer/human_feedback.py`

---

## 1. Star Rating Widget

**Problem:** QPushButton with `background:transparent` drops Unicode text on some platforms, making stars invisible.

**Fix:** Replace the five `QPushButton('☆')` instances in `_EvalWidget` with five `QLabel` instances. Each label is given a fixed size, centered alignment, and a large font. Mouse press events on the label call `_set_stars(n)`. The `_set_stars` logic is unchanged: stars 1→n show `★` in gold (`#C0A020`), stars n+1→5 show `☆` in muted gray (`#B8B5A4`). Behaviour: clicking 3 fills 1–3; clicking 5 fills all five; clicking 2 unfills 3–5.

---

## 2. Chord Gesture Tab

**Problem:** `_ChordTab` plays a single chord via a one-event `Gesture`, which is musically flat and not a gesture.

**Redesign:**

- On `_load_item()`, pick a single random V/A target `(tv, ta)` in [0, 100].
- Call `predictor.find_path(start_va=(tv, ta), end_va=(tv, ta), steps=random.randint(4,6))` — clustering near one emotional point rather than traversing, so each item maps cleanly to one ML label.
- Convert each chord record in the path to a `NoteEvent` with `chord.enabled=True`, using the chord library's root frequency, and a beat duration drawn randomly from `[1, 2, 3]`.
- Assemble into a single `Gesture` and play via the existing `GesturePlayer`.
- Store `tv` / `ta` as `ml_valence` / `ml_arousal` for the reveal panel.

---

## 3. Music Tab — Layered Congruency Composition

**Problem:** `_MusicTab` concatenates random gestures sequentially; this teaches nothing about gesture congruency, and the item type `'music'` conflates congruency ratings with individual gesture quality.

**Redesign:**

### Item selection (rule-based, not dataset-driven)

- Pick one random V/A target `(tv, ta)`.
- Draw 2–3 items from a mix of the gesture library and chord predictor, both weighted toward `(tv, ta)` via `LibraryRanker`. Weighting ensures ratings are semantically coherent rather than random noise, so the ML receives sparse but meaningful signal.

### Simultaneous playback

- Spawn one `GesturePlayer` per layer (2–3 instances). Start all layers in parallel using `threading.Thread` with no explicit synchronisation — layers run freely, allowing emergent rhythmic interaction. Store layer players on the tab so `_stop_item()` can stop all of them.

### Display

- Show the V/A target and the names of all layers so the user knows what they are rating.

### Storage

- On submit, save one record to `FeedbackStore` with:
  - `item_type = 'music_layer'` (distinct from `'gesture'` and `'chord'`)
  - `item_id` = `'+'`-joined layer names (e.g. `"gesture_a+chord_42+gesture_b"`)
  - `ml_valence` / `ml_arousal` = V/A target `(tv, ta)`
  - `user_valence` / `user_arousal` / `user_stars` from sliders and star widget
  - Signed `delta_valence` / `delta_arousal` as computed by existing `FeedbackStore.save_rating()`
- No per-layer ratings — one congruency rating covers the whole combination. This keeps individual gesture ratings clean so the ML can separately learn single-gesture quality vs. combination quality.

### ML intent

The `music_layer` item type gives the ML a training signal for *combination congruency*: given a set of gesture/chord IDs and a V/A context, predict how well they sound together. This is trained separately from single-gesture valence/arousal models.

---

## 4. FeedbackStore

No schema changes required. The existing `save_rating()` signature and JSONL fields accommodate `music_layer` as a new value of `item_type`. The `export_for_pipeline()` engagement filter (≥3 items per participant) applies equally.

---

## Out of Scope

- No changes to `LibraryRanker` — existing `update_gesture_rating` / `update_chord_rating` methods are not called for `music_layer` submissions (congruency is a separate signal).
- No changes to `FeedbackStore` schema.
- No changes to `GesturePlayer` — multiple instances are used as-is.
