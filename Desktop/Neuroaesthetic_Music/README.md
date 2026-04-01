# Neuroaesthetic Music Research

Real-time biosignal-driven music generation and emotion tracking experiments.

## Project Structure

```
Neuroaesthetic_Music/
├── __main__.py                    # Entry point: home screen dispatcher
├── requirements.txt
│
├── src/                           # All Python source code
│   ├── experiments/               # Emotion tracking experiments
│   │   ├── emotion_gui.py
│   │   ├── experiment_dialogue.py
│   │   ├── video_setup_menu.py
│   │   └── data_saver.py
│   │
│   ├── biosignals/                # Biosignal acquisition
│   │   ├── bitalino_connector.py
│   │   ├── biosignal_recorder.py
│   │   └── biosignal_processor.py
│   │
│   ├── generative_music/          # Compositional tools
│   │   └── gesture_designer/      # Spectral Gesture Designer suite
│   │       ├── gesture_model.py          # NoteEvent / Gesture data model
│   │       ├── gesture_player.py         # SuperCollider OSC playback engine
│   │       ├── gesture_library.py        # JSON library with weighted sampling
│   │       ├── gesture_designer.py       # Full gesture editor UI
│   │       ├── spectral_chord_builder.py # Chord construction tool
│   │       ├── adaptive_texture_ui.py    # Adaptive texture engine
│   │       ├── gesture_sequence_player.py
│   │       ├── chord_predictor.py        # V/A nearest-chord lookup
│   │       ├── co_improviser.py          # Real-time biosignal-driven generation
│   │       ├── session_recorder.py       # JSONL session logging
│   │       ├── feedback_store.py         # Human rating persistence (DBTL)
│   │       ├── library_ranker.py         # Per-participant star scores (DBTL)
│   │       ├── human_feedback.py         # HumanFeedbackWindow UI (DBTL)
│   │       └── generate_music_menu.py    # Tool selection menu
│   │   └── synthesis/
│   │
│   └── utils/                     # Shared utilities
│
├── data/                          # Runtime data (NOT in git)
│   ├── experiment_logs/
│   ├── gesture_library/           # Gesture JSON files (ratings field per-participant)
│   ├── feedback/                  # ratings.jsonl, chord_ratings.jsonl (DBTL)
│   └── videos/
│
├── models/                        # Trained ML models
│   ├── valence_audio_only.pkl     # Random Forest valence regressor
│   └── arousal_audio_only.pkl     # Random Forest arousal regressor
│
├── tests/                         # Pytest test suite
│   └── generative_music/
│       ├── test_human_feedback.py  # FeedbackStore, LibraryRanker, ChordPredictor, GestureLibrary
│       ├── test_biosignal_bridge.py
│       ├── test_co_improviser.py
│       ├── test_performer_listener.py
│       └── test_session_recorder.py
│
├── supercollider/                 # SuperCollider resources
│   └── manual_boot.scd
│
└── notebooks/                     # Jupyter notebooks
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python __main__.py
```

## Features

### Experiment Mode
- Video-based emotion tracking with real-time valence/arousal sliders
- Biosignal integration (Bitalino/PLUX): EDA, PPG, accelerometer
- Automatic data logging to `data/experiment_logs/`

### Generate Music Mode

Five tools accessible from the Compose Music sub-menu:

#### Spectral Gesture Designer
- 16-partial harmonic-series weight control per note
- Quarter-tone and microtonal pitch support
- Pulse bursts, parameter automation, live preview via SuperCollider
- Undo/redo, randomisation, save/load JSON library

#### Spectral Chord Builder
- Construct and preview spectral chords
- ML-predicted valence/arousal for each chord

#### Adaptive Texture Engine
- Real-time biosignal-responsive gesture generation
- Co-improviser responds to EDA, HRV, and performer audio input

#### Gesture Sequence Player
- Chain and play back saved gestures as a sequence

#### Human Feedback (DBTL Learn → Design loop)
Closes the Design-Build-Test-Learn cycle by capturing blind human ratings
and feeding them back into gesture/chord selection and ML retraining.

- **Three evaluation modes** (increasing complexity):
  - *Gesture* — rate a single spectral gesture
  - *Chord Gesture* — rate a single spectral chord
  - *Music* — rate a short phrase of 3–5 chained gestures
- **Blind rating**: ML-predicted valence/arousal is hidden until after submission
  to prevent anchoring bias
- **Dual V/A sliders** (0–100) + **1–5 star rating** per item
- **Participant ID** field correlates ratings with biosignal experiment data
- **Per-participant storage**: ratings are never aggregated across participants,
  preserving individual taste profiles
- **Exploration-weighted sampling**: items with low or no star ratings surface
  more often; 5-star items yield to unexplored items
- **Reveal on submit**: shows ML prediction, user rating, and Δ valence/arousal
- **ML retraining integration**: sessions with ≥ 3 ratings export to pipeline
  with `sample_weight = 0.3` (down-weighted vs. biosignal ground truth)

**Data written:**
- `data/feedback/ratings.jsonl` — one record per submission
- `data/feedback/chord_ratings.jsonl` — chord rating sidecar
- Gesture JSON files gain a `ratings` dict keyed by participant ID

## DBTL Pipeline

```
Design  →  gesture/chord library + ML models (valence_audio_only, arousal_audio_only)
Build   →  SuperCollider synthesis via GesturePlayer OSC
Test    →  Experiment Mode: biosignal + V/A slider capture
Learn   →  Human Feedback: blind V/A + star ratings → FeedbackStore → LibraryRanker
  └─────────────────────────────────────────────────→ weighted_random() draw
                                                     → ChordPredictor rating weight
                                                     → run_pipeline.py retraining
```

## Development

```bash
# Run tests
pytest tests/
```

Source modules:
- `from src.generative_music.gesture_designer.feedback_store import FeedbackStore`
- `from src.generative_music.gesture_designer.library_ranker import LibraryRanker`
- `from src.generative_music.gesture_designer.gesture_library import GestureLibrary`
- `from src.generative_music.gesture_designer.chord_predictor import ChordPredictor`

Data saved to `data/` (excluded from git).
