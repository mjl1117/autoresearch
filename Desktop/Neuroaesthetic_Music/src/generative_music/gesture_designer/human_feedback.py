# src/generative_music/gesture_designer/human_feedback.py
"""
human_feedback.py
HumanFeedbackWindow — blind V/A + star rating for gestures, chord gestures,
and short music sequences. Closes the DBTL Learn → Design loop.

Flow per item:
  1. Load random item (weighted toward under-explored items)
  2. Play via SuperCollider
  3. User rates with dual V/A sliders + 1–5 stars (ML prediction hidden)
  4. Submit → reveal ML prediction, show delta
  5. Save to FeedbackStore + update LibraryRanker
  6. Next Item

MJL Neuroaesthetic Music Research — 2026
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSlider, QTabWidget, QLineEdit, QFrame,
    QSizePolicy, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

from .feedback_store import FeedbackStore
from .library_ranker import LibraryRanker
from .gesture_library import GestureLibrary
from .gesture_player import GesturePlayer
from .chord_predictor import ChordPredictor
from .gesture_model import Gesture, NoteEvent, ChordConfig, PartialWeights

logger = logging.getLogger(__name__)

# ── Palette ───────────────────────────────────────────────────────────────────
WINDOW_BG = '#F8F6EF'
PANEL_BG  = '#EDEADF'
CARD_BG   = '#E3DFD2'
TEXT      = '#1E1A14'
LABEL     = '#7C809B'
ACCENT    = '#A9AFD1'
GREEN     = '#4E845D'
DANGER    = '#C04040'

_BTN = ("QPushButton {{ background:{bg}; color:{col}; border:1px solid {border}; "
        "border-radius:5px; padding:8px 20px; font-size:12pt; }}"
        "QPushButton:hover {{ background:#EDEADF; }}"
        "QPushButton:disabled {{ background:#E3DFD2; color:#B8B5A4; }}")

_SLIDER_STYLE = """
QSlider::groove:horizontal { height:5px; background:#E3DFD2; border-radius:2px; }
QSlider::sub-page:horizontal { background:#A9AFD1; border-radius:2px; }
QSlider::handle:horizontal {
    background:#A9AFD1; width:14px; height:14px;
    margin:-5px 0; border-radius:7px; border:2px solid #FFF;
}
"""

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

    def enterEvent(self, event):
        current = self.text()
        if current == '☆':   # only change color on unfilled stars
            self.setStyleSheet(
                f'color:{ACCENT}; font-size:20pt; background:{PANEL_BG};')
        super().enterEvent(event)

    def leaveEvent(self, event):
        current = self.text()
        if current == '☆':
            self.setStyleSheet(
                f'color:#B8B5A4; font-size:20pt; background:{PANEL_BG};')
        super().leaveEvent(event)


def _btn(text, bg=CARD_BG, col=TEXT, border='#D0CCAC') -> QPushButton:
    b = QPushButton(text)
    b.setStyleSheet(_BTN.format(bg=bg, col=col, border=border))
    return b


def _sep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet('color:#D0CCAC; background:#D0CCAC;')
    return f


# ── Helper functions ──────────────────────────────────────────────────────────

def _gesture_va_prediction(gesture_path: str) -> tuple[float, float]:
    """Predict V/A from partial weights using pre-trained audio RF models.
    Returns (50.0, 50.0) on any failure."""
    try:
        import json, joblib
        import numpy as np

        models_dir = Path(__file__).parent.parent.parent.parent / 'models'
        v_model_path = models_dir / 'valence_audio_only.pkl'
        a_model_path = models_dir / 'arousal_audio_only.pkl'

        if not v_model_path.exists() or not a_model_path.exists():
            return 50.0, 50.0

        with open(gesture_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        events = data.get('events', [])
        if not events:
            return 50.0, 50.0

        feat_rows = []
        for ev in events:
            if ev.get('is_rest', False):
                continue
            freq = _note_to_hz(ev.get('note', 'A'), ev.get('accidental', ''),
                               int(ev.get('octave', 4)))
            weights = np.array([ev.get('partials', {}).get(f'w{i}', 1.0)
                                 for i in range(1, 17)], dtype=float)
            weights = np.clip(weights, 0, None)
            total = weights.sum()
            if total < 1e-9:
                continue
            freqs = np.array([freq * i for i in range(1, 17)])

            rms      = float(np.sqrt(np.mean(weights ** 2)))
            centroid = float(np.dot(weights, freqs) / total)
            bw       = float(np.sqrt(np.dot(weights, (freqs - centroid) ** 2) / total))
            cumsum   = np.cumsum(weights ** 2)
            rolloff  = float(freqs[min(np.searchsorted(cumsum, 0.85 * cumsum[-1]), 15)])
            gm       = float(np.exp(np.mean(np.log(weights + 1e-9))))
            flatness = float(gm / (total / 16 + 1e-9))
            zcr      = 0.0
            feat_rows.append([rms, centroid, rolloff, bw, flatness, zcr])

        if not feat_rows:
            return 50.0, 50.0

        arr  = np.array(feat_rows)
        feat = np.concatenate([arr.mean(axis=0), arr.std(axis=0),
                               arr.min(axis=0), arr.max(axis=0)]).reshape(1, -1)
        v_model = joblib.load(v_model_path)
        a_model = joblib.load(a_model_path)
        return (float(np.clip(v_model.predict(feat)[0], 0, 100)),
                float(np.clip(a_model.predict(feat)[0], 0, 100)))

    except Exception as exc:
        logger.debug(f'gesture V/A prediction failed: {exc}')
        return 50.0, 50.0


def _note_to_hz(note: str, accidental: str, octave: int) -> float:
    import math
    offsets = {'bb': -2, 'b': -1, '3qb': -1.5, 'qb': -0.5,
               '': 0, 'q#': 0.5, '#': 1, '3q#': 1.5, 'x': 2}
    semis   = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    semi    = semis.get(note.upper(), 9) + offsets.get(accidental, 0.0)
    midi    = 60.0 + semi + (octave - 4) * 12.0
    return 440.0 * math.pow(2.0, (midi - 69.0) / 12.0)


def _weighted_random_chord(chords: list[dict], participant_id: str,
                            ranker: LibraryRanker) -> dict:
    weights = []
    for c in chords:
        r = ranker.get_participant_chord_rating(c.get('chord_id', ''), participant_id)
        weights.append(1.0 if r is None else max(0.1, (6.0 - r) / 5.0))
    return random.choices(chords, weights=weights, k=1)[0]


_RHYTHM_MOTIFS = [
    [1, 2, 3, 2],
    [2, 1, 1, 2],
    [1, 1, 2, 1],
    [3, 1, 2],
    [1, 3, 1, 1],
    [2, 2, 1, 1],
    [1, 1, 1, 2],
    [2, 1, 3],
    [1, 2, 1, 1],
    [3, 2, 1],
]


def _build_chord_gesture(chords: list[dict], bpm: float = 72.0) -> 'Gesture':
    """Convert a sequence of chord records into a playable gestural Gesture.

    Each chord becomes one NoteEvent with chord.enabled=True.
    A random rhythmic motif is chosen per gesture and cycled over the events,
    giving each chord gesture a distinct feel.
    The root is always A4 (440 Hz) — matching the existing chord preview behaviour.
    """
    motif = random.choice(_RHYTHM_MOTIFS)
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
            beats=motif[i % len(motif)],
            partials=pw,
            chord=cc,
        )
        events.append(ev)
    return Gesture(name='chord_gesture', bpm=bpm, events=events)


def _select_music_layers(lib: GestureLibrary, predictor: ChordPredictor,
                          ranker: LibraryRanker, pid: str,
                          target_va: tuple[float, float], n: int) -> list[dict]:
    """Pick n layers near target_va from the gesture library and chord predictor.

    Each layer is a dict: {'name': str, 'gesture': Gesture}

    Selection is 50/50 gesture vs chord gesture when both sources are available,
    falling back to gesture-only when the chord predictor has no data.
    GestureLibrary.weighted_random() already applies LibraryRanker weights so
    under-explored items surface more often — no extra weighting needed here.
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


# ── Shared evaluation widget ──────────────────────────────────────────────────

class _EvalWidget(QWidget):
    """Base tab: play → rate → submit → reveal. Subclasses override _load_item()."""

    def __init__(self, store: FeedbackStore, ranker: LibraryRanker,
                 participant_id_fn, item_type: str, parent=None):
        super().__init__(parent)
        self._store = store
        self._ranker = ranker
        self._get_pid = participant_id_fn
        self._item_type = item_type
        self._current_item: Optional[dict] = None
        self._ml_valence: float = 50.0
        self._ml_arousal: float = 50.0
        self._submitted = False

        self.setStyleSheet(f'background:{PANEL_BG}; color:{TEXT};')
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(14)

        # Item name display
        self._name_lbl = QLabel('—')
        self._name_lbl.setStyleSheet(
            f'color:{ACCENT}; font-size:16pt; font-weight:bold;')
        self._name_lbl.setWordWrap(True)
        lay.addWidget(self._name_lbl)

        # Play / Stop row
        btn_row = QHBoxLayout()
        self._play_btn = _btn('▶  Play', bg='#4E845D', col='#FFF', border=GREEN)
        self._play_btn.clicked.connect(self._play)
        self._stop_btn = _btn('■  Stop')
        self._stop_btn.clicked.connect(self._stop)
        btn_row.addWidget(self._play_btn)
        btn_row.addWidget(self._stop_btn)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        lay.addWidget(_sep())

        # V/A sliders
        for attr, label in [('_v_slider', 'Your Valence'),
                             ('_a_slider', 'Your Arousal')]:
            row = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setStyleSheet(f'color:{LABEL}; font-size:11pt;')
            lbl.setMinimumWidth(110)
            s = QSlider(Qt.Orientation.Horizontal)
            s.setRange(0, 100)
            s.setValue(50)
            s.setStyleSheet(_SLIDER_STYLE)
            val_lbl = QLabel('50')
            val_lbl.setStyleSheet(f'color:{TEXT}; font-size:11pt;')
            val_lbl.setMinimumWidth(32)
            s.valueChanged.connect(lambda v, l=val_lbl: l.setText(str(v)))
            row.addWidget(lbl)
            row.addWidget(s, stretch=1)
            row.addWidget(val_lbl)
            lay.addLayout(row)
            setattr(self, attr, s)

        lay.addWidget(_sep())

        # Star rating
        star_row = QHBoxLayout()
        star_lbl = QLabel('How much do you like it?')
        star_lbl.setStyleSheet(f'color:{LABEL}; font-size:11pt;')
        star_lbl.setWordWrap(True)
        star_row.addWidget(star_lbl)
        star_row.addStretch()
        self._star_btns: list[_ClickableLabel] = []
        for i in range(1, 6):
            lbl = _ClickableLabel(i, self._set_stars)
            star_row.addWidget(lbl)
            self._star_btns.append(lbl)
        lay.addLayout(star_row)
        self._stars = 0

        lay.addWidget(_sep())

        # Submit button
        submit_row = QHBoxLayout()
        self._submit_btn = _btn('Submit', bg=ACCENT, col=TEXT, border=ACCENT)
        self._submit_btn.clicked.connect(self._submit)
        submit_row.addWidget(self._submit_btn)
        submit_row.addStretch()
        lay.addLayout(submit_row)

        # Reveal panel (hidden until submit)
        self._reveal_frame = QFrame()
        self._reveal_frame.setStyleSheet(f'background:{CARD_BG}; border-radius:6px;')
        self._reveal_frame.setVisible(False)
        rev_lay = QVBoxLayout(self._reveal_frame)
        rev_lay.setContentsMargins(14, 10, 14, 10)
        self._reveal_lbl = QLabel()
        self._reveal_lbl.setStyleSheet(f'color:{TEXT}; font-size:11pt;')
        self._reveal_lbl.setWordWrap(True)
        rev_lay.addWidget(self._reveal_lbl)
        lay.addWidget(self._reveal_frame)

        lay.addStretch()

        # Next Item button
        self._next_btn = _btn('Next Item →')
        self._next_btn.clicked.connect(self._load_next)
        self._next_btn.setEnabled(False)
        lay.addWidget(self._next_btn)

        QTimer.singleShot(100, self._load_next)

    # ── Subclass API ──────────────────────────────────────────────────────────

    def _load_item(self) -> Optional[dict]:
        raise NotImplementedError

    def _play_item(self, item: dict):
        raise NotImplementedError

    def _stop_item(self):
        raise NotImplementedError

    def _item_id(self, item: dict) -> str:
        raise NotImplementedError

    def _item_display_name(self, item: dict) -> str:
        raise NotImplementedError

    def _after_submit(self, item_id: str, participant_id: str, stars: int):
        pass

    # ── Shared logic ──────────────────────────────────────────────────────────

    def _load_next(self):
        self._stop_item()
        self._submitted = False
        self._stars = 0
        self._set_stars(0)
        self._v_slider.setValue(50)
        self._a_slider.setValue(50)
        self._reveal_frame.setVisible(False)
        self._next_btn.setEnabled(False)
        self._submit_btn.setEnabled(True)

        item = self._load_item()
        if item is None:
            self._name_lbl.setText('No items in library.')
            self._play_btn.setEnabled(False)
            return
        self._current_item = item
        self._name_lbl.setText(self._item_display_name(item))
        self._play_btn.setEnabled(True)

    def _play(self):
        if self._current_item:
            self._play_item(self._current_item)

    def _stop(self):
        self._stop_item()

    def _set_stars(self, n: int):
        self._stars = n
        for lbl, (char, color) in zip(self._star_btns, _compute_star_states(n)):
            lbl.setText(char)
            lbl.setStyleSheet(
                f'color:{color}; font-size:20pt; background:{PANEL_BG};')

    def _submit(self):
        if self._current_item is None:
            return
        if self._stars == 0:
            QMessageBox.information(self, 'Rating required',
                                    'Please select a star rating before submitting.')
            return

        pid = self._get_pid()
        uid = self._item_id(self._current_item)
        uv  = float(self._v_slider.value())
        ua  = float(self._a_slider.value())

        self._store.save_rating(
            participant_id=pid,
            item_type=self._item_type,
            item_id=uid,
            user_valence=uv,
            user_arousal=ua,
            user_stars=self._stars,
            ml_valence=self._ml_valence,
            ml_arousal=self._ml_arousal,
        )
        self._after_submit(uid, pid, self._stars)

        dv = self._ml_valence - uv
        da = self._ml_arousal - ua
        self._reveal_lbl.setText(
            f'<b>ML predicted:</b>  Valence {self._ml_valence:.0f}  |  '
            f'Arousal {self._ml_arousal:.0f}<br>'
            f'<b>Your rating:</b>  Valence {uv:.0f}  |  Arousal {ua:.0f}<br>'
            f'<b>Δ Valence:</b> {dv:+.0f}  &nbsp;  <b>Δ Arousal:</b> {da:+.0f}'
        )
        self._reveal_frame.setVisible(True)
        self._submit_btn.setEnabled(False)
        self._next_btn.setEnabled(True)
        self._submitted = True

        self._stop_item()


# ── Gesture tab ───────────────────────────────────────────────────────────────

class _GestureTab(_EvalWidget):
    def __init__(self, store, ranker, pid_fn, player: GesturePlayer, parent=None):
        self._player = player
        self._lib = GestureLibrary()
        super().__init__(store, ranker, pid_fn, 'gesture', parent)

    def _load_item(self) -> Optional[dict]:
        pid = self._get_pid()
        item = self._lib.weighted_random(participant_id=pid)
        if item is None:
            return None
        self._ml_valence, self._ml_arousal = _gesture_va_prediction(item['path'])
        return item

    def _play_item(self, item: dict):
        gesture = self._lib.load(item['path'])
        if gesture:
            self._player.play_gesture(gesture)

    def _stop_item(self):
        self._player.stop_gesture()

    def _item_id(self, item: dict) -> str:
        return item['name']

    def _item_display_name(self, item: dict) -> str:
        return item['name']

    def _after_submit(self, item_id, participant_id, stars):
        self._ranker.update_gesture_rating(item_id, participant_id, stars)


# ── Chord tab ─────────────────────────────────────────────────────────────────

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
        return {'name': f'Chord Gesture  ·  {chord_ids}',
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


# ── Music (simultaneous layers) tab ──────────────────────────────────────────

class _MusicTab(_EvalWidget):
    """Plays 2–3 simultaneous gesture/chord layers targeted at one V/A point.

    Stores ratings as item_type='music_layer' so the ML can learn combination
    congruency independently of single-gesture quality.
    """

    def __init__(self, store, ranker, pid_fn, player: GesturePlayer,
                 predictor: ChordPredictor, parent=None):
        # player accepted for API compatibility but _MusicTab manages its own per-layer players
        self._predictor = predictor
        self._lib = GestureLibrary()
        self._layers: list[dict] = []
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

        if len(self._layers) < 2:   # need at least 2 layers for meaningful congruency data
            return None

        layer_names = ',  '.join(layer['name'] for layer in self._layers)
        return {'name': f'Layers:  {layer_names}'}

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


# ── Main window ───────────────────────────────────────────────────────────────

class HumanFeedbackWindow(QMainWindow):
    """Top-level window housing the three evaluation tabs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Human Feedback — DBTL Learn')
        self.setMinimumSize(620, 580)
        self.setStyleSheet(f'background:{WINDOW_BG}; color:{TEXT};')

        store     = FeedbackStore()
        ranker    = LibraryRanker()
        player    = GesturePlayer()
        predictor = ChordPredictor()

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        # Title row with participant ID
        title_row = QHBoxLayout()
        title = QLabel('Human Feedback')
        title.setStyleSheet(f'color:{ACCENT}; font-size:20pt; font-weight:bold;')
        title_row.addWidget(title)
        title_row.addStretch()

        pid_lbl = QLabel('Participant ID:')
        pid_lbl.setStyleSheet(f'color:{LABEL}; font-size:11pt;')
        self._pid_edit = QLineEdit()
        self._pid_edit.setPlaceholderText('anonymous')
        self._pid_edit.setFixedWidth(160)
        self._pid_edit.setStyleSheet(
            f'background:{CARD_BG}; color:{TEXT}; border:1px solid #D0CCAC; '
            f'border-radius:4px; padding:4px 8px; font-size:11pt;')
        title_row.addWidget(pid_lbl)
        title_row.addWidget(self._pid_edit)
        root.addLayout(title_row)

        sub = QLabel(
            'Rate gestures, chord gestures, and music phrases. '
            'Your ratings personalise the generator to your preferences.')
        sub.setStyleSheet(f'color:{LABEL}; font-size:10pt;')
        sub.setWordWrap(True)
        root.addWidget(sub)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet('color:#D0CCAC; background:#D0CCAC;')
        root.addWidget(sep)

        # Tab widget
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane { border:none; background:#EDEADF; }
            QTabBar::tab {
                background:#E3DFD2; color:#7C809B;
                padding:8px 24px; border-radius:4px 4px 0 0; font-size:11pt;
                min-width:130px;
            }
            QTabBar::tab:selected { background:#F8F6EF; color:#1E1A14; }
        """)

        pid_fn = lambda: self._pid_edit.text().strip() or 'anonymous'

        tabs.addTab(_GestureTab(store, ranker, pid_fn, player), 'Gesture')
        tabs.addTab(_ChordTab(store, ranker, pid_fn, player, predictor), 'Chord Gesture')
        tabs.addTab(_MusicTab(store, ranker, pid_fn, player, predictor), 'Music')

        root.addWidget(tabs)
        self._player = player

    def closeEvent(self, event):
        self._player.stop_gesture()
        self._player.stop_preview()
        super().closeEvent(event)
