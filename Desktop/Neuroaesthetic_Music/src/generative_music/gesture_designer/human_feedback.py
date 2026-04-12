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
from .gesture_model import (Gesture, NoteEvent, ChordConfig, PartialWeights,
                             CHORD_TYPE_NAMES, CHORD_MAX_VOICES)

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

def _gesture_va_from_event_dicts(events: list,
                                  bpm: float = 80.0) -> tuple[float, float]:
    """Predict V/A from a list of event dicts (the 'events' key of a gesture JSON).

    Uses the 37-feature schema: 24 spectral + 4 temporal + 9 harmonic-transition.
    BPM must be passed explicitly so temporal features are meaningful.
    Returns (50.0, 50.0) on any failure.

    NOTE: feature schema must stay in sync with
          src/machine_learning/gesture_features.py::extract_gesture_features.
    """
    try:
        import joblib
        import numpy as np

        models_dir = Path(__file__).parent.parent.parent.parent / 'models'
        v_model_path = models_dir / 'valence_audio_only.pkl'
        a_model_path = models_dir / 'arousal_audio_only.pkl'

        if not v_model_path.exists() or not a_model_path.exists():
            return 50.0, 50.0
        if not events:
            return 50.0, 50.0

        spectral_rows: list = []
        semitones: list = []
        centroids: list = []
        rmss: list = []
        beat_list: list = []

        for ev in events:
            if ev.get('is_rest', False):
                continue
            note       = ev.get('note', 'A')
            accidental = ev.get('accidental', '')
            octave     = int(ev.get('octave', 4))
            freq       = _note_to_hz(note, accidental, octave)

            weights = np.array([ev.get('partials', {}).get(f'w{i}', 1.0)
                                 for i in range(1, 17)], dtype=float)
            weights = np.clip(weights, 0, None)
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

            # semitone: absolute pitch position (C-1=0, A4=69)
            _semis = {'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}
            _accs  = {'bb':-2,'b':-1,'3qb':-1.5,'qb':-0.5,'':0,
                      'q#':0.5,'#':1,'3q#':1.5,'x':2}
            semi_abs = ((octave + 1) * 12
                        + _semis.get(note.upper(), 9)
                        + _accs.get(accidental, 0.0))

            spectral_rows.append([rms, centroid, rolloff, bw, flatness, 0.0])
            semitones.append(semi_abs)
            centroids.append(centroid)
            rmss.append(rms)
            beat_list.append(float(ev.get('beats', 1)))

        if not spectral_rows:
            return 50.0, 50.0

        arr          = np.array(spectral_rows)
        spectral_vec = np.concatenate([arr.mean(0), arr.std(0),
                                       arr.min(0), arr.max(0)])

        beats_arr    = np.array(beat_list)
        temporal_vec = np.array([
            float(bpm),
            float(beats_arr.mean()),
            float(beats_arr.std()) if len(beats_arr) > 1 else 0.0,
            float(len(spectral_rows)),
        ])

        n = len(spectral_rows)
        if n > 1:
            def _agg(a: np.ndarray) -> list:
                return [float(a.mean()),
                        float(a.std()) if len(a) > 1 else 0.0,
                        float(a.max())]
            semi_d = np.abs(np.diff(np.array(semitones)))
            cent_d = np.abs(np.diff(np.array(centroids)))
            rms_d  = np.abs(np.diff(np.array(rmss)))
            trans_vec = np.array(_agg(semi_d) + _agg(cent_d) + _agg(rms_d))
        else:
            trans_vec = np.zeros(9)

        feat = np.concatenate([spectral_vec, temporal_vec, trans_vec]).reshape(1, -1)
        v_model = joblib.load(v_model_path)
        a_model = joblib.load(a_model_path)
        return (float(np.clip(v_model.predict(feat)[0], 0, 100)),
                float(np.clip(a_model.predict(feat)[0], 0, 100)))

    except Exception as exc:
        logger.debug(f'gesture V/A prediction failed: {exc}')
        return 50.0, 50.0


def _gesture_va_prediction(gesture_path: str) -> tuple[float, float]:
    """Predict V/A from a gesture JSON file. Returns (50.0, 50.0) on failure."""
    try:
        import json
        with open(gesture_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return _gesture_va_from_event_dicts(data.get('events', []),
                                             bpm=float(data.get('bpm', 80.0)))
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


# ── Generative chord gesture constants ───────────────────────────────────────

_NOTES       = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
_ACCIDENTALS = ['', '', '', '', '', '#', 'b', 'q#', 'qb']  # weighted toward natural
_OCTAVES     = [2, 3, 4, 5, 6]
_BPMS        = [60.0, 72.0, 80.0, 90.0, 100.0, 120.0]

# Log path for reviewing all generated chord gestures
_CHORD_GESTURE_LOG = (Path(__file__).parent.parent.parent.parent
                      / 'data' / 'feedback' / 'chord_gesture_log.jsonl')


def _equal_loudness_gain(root_hz: float) -> float:
    """Amplitude correction factor for equal perceived loudness across pitch.

    Applies ~3 dB/octave relative to A4 (440 Hz): lower pitches are boosted,
    higher pitches are attenuated, compensating for the ear's increased
    sensitivity in the 1–4 kHz region.  Result is clamped to [0.25, 4.0].
    """
    import math
    db = -3.0 * math.log2(max(root_hz, 20.0) / 440.0)
    return float(min(4.0, max(0.25, 10.0 ** (db / 20.0))))


def _log_chord_gesture(gesture: 'Gesture') -> None:
    """Append a generated chord gesture to the review log (silent on failure)."""
    try:
        import json
        from datetime import datetime, timezone
        data = gesture.to_dict()
        data['generated_at'] = datetime.now(timezone.utc).isoformat()
        # Annotate each event with a human-readable chord type name
        for ev in data.get('events', []):
            ct = ev.get('chord', {}).get('chord_type', 0)
            ev['chord']['chord_type_name'] = (
                CHORD_TYPE_NAMES[ct]
                if 0 <= ct < len(CHORD_TYPE_NAMES) else 'Unknown')
        _CHORD_GESTURE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(_CHORD_GESTURE_LOG, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data) + '\n')
    except Exception:
        pass


def _build_chord_gesture(n_events: int = None, bpm: float = None) -> 'Gesture':
    """Generate a fully randomised spectral chord gesture.

    Root pitch (note + accidental + octave), chord type, voicing, inversion,
    partial weights, amplitude, and brightness are all drawn independently at
    random for each event — matching the generative conventions of melodic
    gestures.  Beat durations follow a Gaussian (μ=2, σ=0.8) clamped to [1, 4].

    Amplitude is equal-loudness normalised: a 3 dB/octave correction relative
    to A4 compensates for the ear's frequency sensitivity, preventing high-
    octave chords from sounding disproportionately louder.  Partial weights are
    RMS-normalised so different spectral decay profiles produce equal energy.
    """
    import math, numpy as np

    if n_events is None:
        n_events = random.randint(4, 8)
    if bpm is None:
        bpm = float(random.choice(_BPMS))

    events = []
    for _ in range(n_events):
        note       = random.choice(_NOTES)
        accidental = random.choice(_ACCIDENTALS)
        octave     = random.choice(_OCTAVES)

        chord_type = random.randint(0, len(CHORD_TYPE_NAMES) - 1)
        max_v      = CHORD_MAX_VOICES[chord_type]
        num_voices = random.randint(2, max_v)
        inversion  = random.randint(0, min(2, num_voices - 1))
        balance    = round(random.uniform(0.3, 1.0), 3)

        # Spectral profile: exponential decay + noise, then RMS-normalised
        decay = float(np.random.uniform(0.4, 0.95))
        raw   = np.array([decay ** i for i in range(16)], dtype=float)
        raw  += np.random.uniform(0.0, 0.25, 16)
        raw   = np.clip(raw, 0.0, None)
        rms   = float(np.sqrt(np.mean(raw ** 2)))
        if rms > 1e-9:
            raw = raw / rms          # normalise → consistent spectral energy
        pw = PartialWeights()
        for j in range(16):
            pw.set_index(j, float(raw[j]))

        # Equal-loudness amplitude: random base scaled by pitch correction
        root_hz   = _note_to_hz(note, accidental, octave)
        base_amp  = random.uniform(0.15, 0.35)
        amplitude = float(np.clip(base_amp * _equal_loudness_gain(root_hz), 0.05, 0.5))

        beats = int(np.clip(round(float(np.random.normal(2.0, 0.8))), 1, 4))

        events.append(NoteEvent(
            note=note, accidental=accidental, octave=octave,
            amplitude=round(amplitude, 3),
            brightness=round(random.uniform(0.0, 1.0), 3),
            beats=beats,
            partials=pw,
            chord=ChordConfig(
                enabled=True,
                chord_type=chord_type,
                num_voices=num_voices,
                balance=balance,
                inversion=inversion,
            ),
        ))

    gesture = Gesture(name=f'gen_chord_{random.randint(10000, 99999)}',
                      bpm=bpm, events=events)
    _log_chord_gesture(gesture)
    return gesture


def _select_music_layers(lib: GestureLibrary, ranker: LibraryRanker, pid: str,
                          target_va: tuple[float, float], n: int) -> list[dict]:
    """Pick n layers for a music evaluation, mixing melodic and chord gestures.

    Each layer is a dict: {'name': str, 'gesture': Gesture}

    Selection is 50/50 melodic (from library) vs generative chord gesture.
    GestureLibrary.weighted_random() applies LibraryRanker weights so
    under-explored items surface more often.
    """
    layers: list[dict] = []

    for _ in range(n):
        if random.random() < 0.5:
            gesture = _build_chord_gesture()
            layers.append({'name': gesture.name, 'gesture': gesture})
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
    def __init__(self, store, ranker, pid_fn, player: GesturePlayer, parent=None):
        self._player = player
        self._current_gesture = None
        super().__init__(store, ranker, pid_fn, 'chord', parent)

    def _load_item(self) -> Optional[dict]:
        self._current_gesture = _build_chord_gesture()
        self._ml_valence, self._ml_arousal = _gesture_va_from_event_dicts(
            self._current_gesture.to_dict()['events'],
            bpm=self._current_gesture.bpm)
        return {'name': self._current_gesture.name}

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

    def __init__(self, store, ranker, pid_fn, player: GesturePlayer, parent=None):
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
            self._lib, self._ranker, pid, (tv, ta), n_layers)

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

        store  = FeedbackStore()
        ranker = LibraryRanker()
        player = GesturePlayer()

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
        tabs.addTab(_ChordTab(store, ranker, pid_fn, player), 'Chord Gesture')
        tabs.addTab(_MusicTab(store, ranker, pid_fn, player), 'Music')

        root.addWidget(tabs)
        self._player = player

    def closeEvent(self, event):
        self._player.stop_gesture()
        self._player.stop_preview()
        super().closeEvent(event)
