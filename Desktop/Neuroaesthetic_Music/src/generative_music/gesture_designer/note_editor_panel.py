"""
note_editor_panel.py
Detailed editor panel for a single NoteEvent.

Tabs: Pitch & Duration | Timbre | Partial Weights | Pulse Burst | Envelope
Live preview toggle fires events back to the parent designer.

MJL Neuroaesthetic Music Research — 2026
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSlider, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QPushButton, QTabWidget, QSizePolicy, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui  import QFont

from .gesture_model import (
    NoteEvent, NOTE_NAMES, ACCIDENTAL_ITEMS, PartialWeights,
    CHORD_TYPE_NAMES, CHORD_MAX_VOICES,
)

# ── Frequency to Note Conversion ─────────────────────────────────────────────

def hz_to_note_name(frequency: float) -> str:
    """Convert frequency to nearest note name with microtonal accidentals.
    
    Returns note names like 'C4', 'E♭5', 'F♯3', 'Bqb4' (quarter-flat), etc.
    """
    if frequency <= 0:
        return '---'
    
    # A4 = 440 Hz = MIDI 69
    # MIDI = 69 + 12 * log2(freq / 440)
    import math
    midi_float = 69.0 + 12.0 * math.log2(frequency / 440.0)
    
    # Round to nearest quarter-tone (0.5 semitone)
    midi_rounded = round(midi_float * 2) / 2.0  # Round to 0.5 increments
    
    # Get octave
    octave = int(midi_rounded // 12) - 1
    
    # Get semitone within octave (0-11.5 with quarter-tones)
    semitone = midi_rounded % 12
    
    # Note names for natural notes (C=0, D=2, E=4, F=5, G=7, A=9, B=11)
    note_map = [
        ('C', ''),      # 0.0
        ('C', 'q#'),    # 0.5
        ('C', '#'),     # 1.0 / D♭
        ('D', 'qb'),    # 1.5
        ('D', ''),      # 2.0
        ('D', 'q#'),    # 2.5
        ('D', '#'),     # 3.0 / E♭
        ('E', 'qb'),    # 3.5
        ('E', ''),      # 4.0
        ('F', 'qb'),    # 4.5 (E♯ = Fqb)
        ('F', ''),      # 5.0
        ('F', 'q#'),    # 5.5
        ('F', '#'),     # 6.0 / G♭
        ('G', 'qb'),    # 6.5
        ('G', ''),      # 7.0
        ('G', 'q#'),    # 7.5
        ('G', '#'),     # 8.0 / A♭
        ('A', 'qb'),    # 8.5
        ('A', ''),      # 9.0
        ('A', 'q#'),    # 9.5
        ('A', '#'),     # 10.0 / B♭
        ('B', 'qb'),    # 10.5
        ('B', ''),      # 11.0
        ('C', 'qb'),    # 11.5 (B♯ = Cqb next octave)
    ]
    
    # Get note and accidental
    idx = int(semitone * 2) % 24  # *2 for quarter-tones, mod 24 for wraparound
    note, acc = note_map[idx]
    
    # Handle B♯ case (11.5 semitones = Cqb of next octave)
    if idx == 23:  # B♯ case
        octave += 1
    
    # Format accidental symbols
    acc_symbols = {
        '': '',
        'qb': '♭¼',   # quarter-flat
        'q#': '♯¼',   # quarter-sharp
        'b': '♭',
        '#': '♯',
    }
    
    acc_display = acc_symbols.get(acc, acc)
    
    return f'{note}{acc_display}{octave}'

# ── Style helpers ─────────────────────────────────────────────────────────────
PANEL_BG  = '#EDEADF'
CARD_BG   = '#E3DFD2'
LABEL_COL = '#7C809B'
TEXT_COL  = '#1E1A14'
ACCENT    = '#A9AFD1'
GREEN     = '#4E845D'

_SLIDER_STYLE = """
QSlider::groove:horizontal {
    height: 4px; background: #E3DFD2; border-radius: 2px;
}
QSlider::sub-page:horizontal {
    background: #A9AFD1; border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #A9AFD1; border: 2px solid #FFFFFF;
    width: 12px; height: 12px; margin: -4px 0;
    border-radius: 6px;
}
"""
_COMBO_STYLE = """
QComboBox {
    background: #F5F3EC; color: #1E1A14;
    border: 1px solid #D0CCAC; border-radius: 4px;
    padding: 3px 8px; font-size: 12pt;
}
QComboBox QAbstractItemView { background: #F5F3EC; color: #1E1A14; selection-background-color: #D6DCF0; }
"""
_SPIN_STYLE = """
QSpinBox, QDoubleSpinBox {
    background: #F5F3EC;
    color: #1E1A14;
    border: 1px solid #D0CCAC;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 11pt;
    min-height: 28px;
}
QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 22px;
    border-left: 1px solid #D0CCAC;
    background: #E3DFD2;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {
    background: #EDEADF;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    image: none;
    width: 0px;
    height: 0px;
    border-left: 6px solid transparent;
    border-right: 6px solid transparent;
    border-bottom: 8px solid #1E1A14;
    margin: 2px;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 22px;
    border-left: 1px solid #D0CCAC;
    background: #E3DFD2;
}
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background: #EDEADF;
}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    image: none;
    width: 0px;
    height: 0px;
    border-left: 6px solid transparent;
    border-right: 6px solid transparent;
    border-top: 8px solid #1E1A14;
    margin: 2px;
}
"""
_CHECK_STYLE = "QCheckBox { color: #1E1A14; font-size: 12pt; }"
_TAB_STYLE = """
QTabWidget::pane { border: none; background: #EDEADF; }
QTabBar::tab {
    background: #E3DFD2; color: #7C809B;
    padding: 6px 14px; border-radius: 4px 4px 0 0;
    font-size: 11pt;
}
QTabBar::tab:selected { background: #F8F6EF; color: #1E1A14; }
"""


def _label(text: str, small: bool = False) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(f"color: {LABEL_COL}; font-size: {'10' if small else '11'}pt;")
    return lbl


def _val_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(f"color: {TEXT_COL}; font-size: 11pt;")
    lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    lbl.setMinimumWidth(42)
    return lbl


def _make_slider(min_v: int, max_v: int, value: int) -> QSlider:
    s = QSlider(Qt.Orientation.Horizontal)
    s.setRange(min_v, max_v)
    s.setValue(value)
    s.setStyleSheet(_SLIDER_STYLE)
    return s


# ── NoteEditorPanel ───────────────────────────────────────────────────────────

class NoteEditorPanel(QWidget):
    """All controls for editing one NoteEvent.

    Signals:
        event_changed()  — emit whenever any field changes
        preview_toggled(bool) — live preview on/off
    """
    event_changed   = pyqtSignal()
    preview_toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._event: NoteEvent = NoteEvent()
        self._blocking = False   # prevent feedback loops during load

        self.setStyleSheet(f"background: {PANEL_BG}; color: {TEXT_COL};")
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)

        # Header: pitch display + live toggle
        root_layout.addWidget(self._build_header())

        # Tab widget
        tabs = QTabWidget()
        tabs.setStyleSheet(_TAB_STYLE)
        tabs.addTab(self._build_pitch_tab(),    'Pitch')
        tabs.addTab(self._build_timbre_tab(),   'Timbre')
        tabs.addTab(self._build_partials_tab(), 'Partials')
        tabs.addTab(self._build_pulse_tab(),    'Pulse Burst')
        tabs.addTab(self._build_chord_tab(),    'Chord')
        tabs.addTab(self._build_envelope_tab(), 'Envelope')
        root_layout.addWidget(tabs)

    # ── Header ───────────────────────────────────────────────────────────────

    def _build_header(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet(f"background: {CARD_BG}; border-radius: 6px;")
        lay = QHBoxLayout(w)
        lay.setContentsMargins(12, 8, 12, 8)

        self.pitch_display = QLabel('A4')
        self.pitch_display.setStyleSheet(
            f"color: #A9AFD1; font-size: 20pt; font-weight: bold;")
        lay.addWidget(self.pitch_display)
        lay.addStretch()

        self.live_check = QCheckBox('Live Preview')
        self.live_check.setStyleSheet(
            f"QCheckBox {{ color: {ACCENT}; font-size: 11pt; }}")
        self.live_check.toggled.connect(self.preview_toggled.emit)
        lay.addWidget(self.live_check)
        return w

    # ── Pitch & Duration tab ─────────────────────────────────────────────────

    def _build_pitch_tab(self) -> QWidget:
        w = QWidget(); w.setStyleSheet(f"background: {PANEL_BG};")
        grid = QGridLayout(w); grid.setSpacing(10); grid.setContentsMargins(12,12,12,12)

        # Rest toggle
        self.rest_check = QCheckBox('Rest')
        self.rest_check.setStyleSheet(_CHECK_STYLE)
        self.rest_check.toggled.connect(self._on_rest_toggled)
        grid.addWidget(self.rest_check, 0, 0, 1, 2)

        # Note name
        grid.addWidget(_label('Note'), 1, 0)
        self.note_combo = QComboBox()
        self.note_combo.addItems(NOTE_NAMES)
        self.note_combo.setStyleSheet(_COMBO_STYLE)
        self.note_combo.currentTextChanged.connect(self._field_changed)
        grid.addWidget(self.note_combo, 1, 1)

        # Accidental
        grid.addWidget(_label('Accidental'), 2, 0)
        self.acc_combo = QComboBox()
        for key, label in ACCIDENTAL_ITEMS:
            self.acc_combo.addItem(label, key)
        self.acc_combo.setStyleSheet(_COMBO_STYLE)
        self.acc_combo.currentIndexChanged.connect(self._field_changed)
        grid.addWidget(self.acc_combo, 2, 1)

        # Octave
        grid.addWidget(_label('Octave'), 3, 0)
        self.octave_spin = QSpinBox()
        self.octave_spin.setRange(0, 8)
        self.octave_spin.setValue(4)
        self.octave_spin.setStyleSheet(_SPIN_STYLE)
        self.octave_spin.valueChanged.connect(self._field_changed)
        grid.addWidget(self.octave_spin, 3, 1)

        # Hz readout
        grid.addWidget(_label('Frequency'), 4, 0)
        self.hz_label = QLabel('440.0 Hz')
        self.hz_label.setStyleSheet(f"color: {ACCENT}; font-size: 11pt;")
        grid.addWidget(self.hz_label, 4, 1)

        # Beats
        grid.addWidget(_label('Beats'), 5, 0)
        self.beats_spin = QDoubleSpinBox()
        self.beats_spin.setRange(0.25, 16.0)
        self.beats_spin.setSingleStep(0.25)
        self.beats_spin.setDecimals(2)
        self.beats_spin.setValue(1.0)
        self.beats_spin.setStyleSheet(_SPIN_STYLE)
        self.beats_spin.valueChanged.connect(self._field_changed)
        grid.addWidget(self.beats_spin, 5, 1)

        grid.setRowStretch(6, 1)
        return w

    # ── Timbre tab ───────────────────────────────────────────────────────────

    def _build_timbre_tab(self) -> QWidget:
        w = QWidget(); w.setStyleSheet(f"background: {PANEL_BG};")
        lay = QVBoxLayout(w); lay.setContentsMargins(12,12,12,12); lay.setSpacing(14)

        self.brightness_sl, self.brightness_vl = self._add_slider_row(
            lay, 'Brightness', 0, 100, 50)
        self.density_sl, self.density_vl = self._add_slider_row(
            lay, 'Density', 1, 16, 16)
        self.amplitude_sl, self.amplitude_vl = self._add_slider_row(
            lay, 'Amplitude', 0, 100, 25)
        self.stability_sl, self.stability_vl = self._add_slider_row(
            lay, 'Stability', 0, 100, 100)

        self.brightness_sl.valueChanged.connect(self._field_changed)
        self.density_sl.valueChanged.connect(self._field_changed)
        self.amplitude_sl.valueChanged.connect(self._field_changed)
        self.stability_sl.valueChanged.connect(self._field_changed)

        lay.addStretch()
        return w

    def _add_slider_row(self, layout, label, lo, hi, val):
        row = QHBoxLayout()
        lbl = _label(f'{label}:')
        lbl.setMinimumWidth(84)
        row.addWidget(lbl)
        sl = _make_slider(lo, hi, val)
        row.addWidget(sl)
        vl = _val_label(str(val))
        row.addWidget(vl)
        layout.addLayout(row)
        return sl, vl

    # ── Partial weights tab ──────────────────────────────────────────────────

    def _build_partials_tab(self) -> QWidget:
        w = QWidget(); w.setStyleSheet(f"background: {PANEL_BG};")
        grid = QGridLayout(w); grid.setSpacing(6); grid.setContentsMargins(12,12,12,12)

        self._partial_sliders: list = []
        self.partial_freq_labels: list = []
        self._partial_val_labels: list = []

        # Harmonic labels for A3 (root=220) — shown as guide
        PARTIAL_NOTES = [
            'A3','A4','E5','A5','C#6','E6','G6','A6',
            'B6','C#7','D#7','E7','F#7','G7','Ab7','A7'
        ]

        for i in range(16):
            row_i = i // 2
            col_base = (i % 2) * 4

            # Partial label (e.g., "w1")
            harmonic = i + 1
            lbl = _label(f'w{harmonic}', small=True)
            lbl.setMinimumWidth(30)
            grid.addWidget(lbl, row_i, col_base)

            # Slider
            sl = _make_slider(0, 300, 100)  # 0..3.0 mapped as 0..300
            sl.setMinimumWidth(80)
            grid.addWidget(sl, row_i, col_base + 1)

            # Weight value label
            vl = _val_label('1.00')
            grid.addWidget(vl, row_i, col_base + 2)

            # Frequency label (e.g., "440.0 Hz")
            freq_label = QLabel('')
            freq_label.setStyleSheet('color: #7C809B; font-size: 9pt;')
            freq_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            freq_label.setMinimumWidth(65)
            self.partial_freq_labels.append(freq_label)
            grid.addWidget(freq_label, row_i, col_base + 3)

            sl.valueChanged.connect(lambda v, ii=i: self._partial_changed(ii, v))
            self._partial_sliders.append(sl)
            self._partial_val_labels.append(vl)

        # Preset buttons row
        preset_row = QHBoxLayout()
        for name, values in [
            ('Flat',    [1.0]*16),
            ('Dark',    [1/(i+1) for i in range(16)]),
            ('Bright',  [0.3,0.4,0.6,0.8,1.2,1.5,1.8,2.0,2.2,2.0,1.8,1.5,1.2,0.9,0.6,0.4]),
            ('Upper',   [0.2,0.1,0.1,0.1,0.1,0.1,2.0,0.1,2.0,0.1,1.5,0.1,0.1,0.1,0.1,0.1]),
            ('Odd',     [1.0 if i%2==0 else 0.0 for i in range(16)]),
            ('Clear',   [0.0]*16),
        ]:
            btn = QPushButton(name)
            btn.setStyleSheet(self._btn_style())
            btn.clicked.connect(lambda _, v=values: self._apply_preset(v))
            preset_row.addWidget(btn)

        grid.addLayout(preset_row, 8, 0, 1, 8)
        grid.setRowStretch(9, 1)
        return w

    def _partial_changed(self, idx: int, slider_val: int):
        if self._blocking:
            return
        weight = slider_val / 100.0
        self._partial_val_labels[idx].setText(f'{weight:.2f}')
        self._event.partials.set_index(idx, weight)
        self.event_changed.emit()

    def _apply_preset(self, values: list):
        self._blocking = True
        for i, v in enumerate(values):
            self._partial_sliders[i].setValue(int(v * 100))
            self._partial_val_labels[i].setText(f'{v:.2f}')
            self._event.partials.set_index(i, v)
        self._blocking = False
        self.event_changed.emit()

    # ── Pulse burst tab ──────────────────────────────────────────────────────

    def _build_pulse_tab(self) -> QWidget:
        w = QWidget(); w.setStyleSheet(f"background: {PANEL_BG};")
        lay = QVBoxLayout(w); lay.setContentsMargins(12,12,12,12); lay.setSpacing(12)

        self.pulse_check = QCheckBox('Enable Pulse Burst')
        self.pulse_check.setStyleSheet(
            f"QCheckBox {{ color: {ACCENT}; font-size: 12pt; font-weight: bold; }}")
        self.pulse_check.toggled.connect(self._field_changed)
        lay.addWidget(self.pulse_check)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #D0CCAC;"); lay.addWidget(sep)

        self.pulse_count_sl, self.pulse_count_vl = self._add_slider_row(
            lay, 'Pulse Count', 1, 12, 4)
        self.pulse_br_start_sl, self.pulse_br_start_vl = self._add_slider_row(
            lay, 'Bright. Start', 0, 100, 30)
        self.pulse_br_end_sl, self.pulse_br_end_vl = self._add_slider_row(
            lay, 'Bright. End', 0, 100, 90)
        self.pulse_amp_start_sl, self.pulse_amp_start_vl = self._add_slider_row(
            lay, 'Amp Start', 0, 100, 30)
        self.pulse_amp_end_sl, self.pulse_amp_end_vl = self._add_slider_row(
            lay, 'Amp End', 0, 100, 5)
        self.pulse_decay_sl, self.pulse_decay_vl = self._add_slider_row(
            lay, 'Decay (×0.01s)', 2, 50, 12)

        for sl in (self.pulse_count_sl, self.pulse_br_start_sl, self.pulse_br_end_sl,
                   self.pulse_amp_start_sl, self.pulse_amp_end_sl, self.pulse_decay_sl):
            sl.valueChanged.connect(self._field_changed)

        lay.addStretch()
        return w

    # ── Chord tab ────────────────────────────────────────────────────────────

    def _build_chord_tab(self) -> QWidget:
        w = QWidget(); w.setStyleSheet(f"background: {PANEL_BG};")
        lay = QVBoxLayout(w); lay.setContentsMargins(12,12,12,12); lay.setSpacing(12)

        self.chord_check = QCheckBox('Enable Chord')
        self.chord_check.setStyleSheet(
            f"QCheckBox {{ color: #A9AFD1; font-size: 12pt; font-weight: bold; }}")
        self.chord_check.toggled.connect(self._field_changed)
        lay.addWidget(self.chord_check)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #D0CCAC;"); lay.addWidget(sep)

        # Chord type
        type_row = QHBoxLayout()
        lbl = _label('Chord Type:'); lbl.setMinimumWidth(84)
        type_row.addWidget(lbl)
        self.chord_type_combo = QComboBox()
        self.chord_type_combo.addItems(CHORD_TYPE_NAMES)
        self.chord_type_combo.setStyleSheet(_COMBO_STYLE)
        self.chord_type_combo.currentIndexChanged.connect(self._on_chord_type_changed)
        type_row.addWidget(self.chord_type_combo)
        lay.addLayout(type_row)

        self.chord_voices_sl, self.chord_voices_vl = self._add_slider_row(
            lay, 'Voices', 1, 4, 3)
        self.chord_balance_sl, self.chord_balance_vl = self._add_slider_row(
            lay, 'Balance', 10, 100, 70)

        self.chord_voices_sl.valueChanged.connect(self._field_changed)
        self.chord_balance_sl.valueChanged.connect(self._field_changed)

        lay.addStretch()
        return w

    def _on_chord_type_changed(self, idx: int):
        max_v = CHORD_MAX_VOICES[idx]
        cur   = self.chord_voices_sl.value()
        self.chord_voices_sl.setRange(1, max_v)
        self.chord_voices_sl.setValue(min(cur, max_v))
        self._field_changed()

    # ── Envelope tab ────────────────────────────────────────────────────────

    def _build_envelope_tab(self) -> QWidget:
        w = QWidget(); w.setStyleSheet(f"background: {PANEL_BG};")
        lay = QVBoxLayout(w); lay.setContentsMargins(12,12,12,12); lay.setSpacing(14)

        self.attack_sl, self.attack_vl = self._add_slider_row(
            lay, 'Attack (×10ms)', 1, 100, 8)
        self.release_sl, self.release_vl = self._add_slider_row(
            lay, 'Release (×10ms)', 1, 200, 60)

        self.attack_sl.valueChanged.connect(self._field_changed)
        self.release_sl.valueChanged.connect(self._field_changed)

        lay.addStretch()
        return w

    # ── Button style ─────────────────────────────────────────────────────────

    def _btn_style(self):
        return """
        QPushButton {
            background: #A9AFD1; color: #1E1A14;
            border: 1px solid #A9AFD1; border-radius: 4px;
            padding: 4px 10px; font-size: 10pt;
        }
        QPushButton:hover { background: #9AA3C8; color: #1E1A14; }
        QPushButton:pressed { background: #8B95BE; }
        """

    # ── Data binding: load event into UI ─────────────────────────────────────

    def load_event(self, event: NoteEvent):
        """Populate all controls from a NoteEvent."""
        self._event = event
        self._blocking = True

        self.pitch_display.setText(event.pitch_label)
        self.live_check.setChecked(False)

        # Pitch tab
        self.rest_check.setChecked(event.is_rest)
        self._set_note_controls_enabled(not event.is_rest)

        idx = NOTE_NAMES.index(event.note) if event.note in NOTE_NAMES else 5
        self.note_combo.setCurrentIndex(idx)

        acc_keys = [k for k, _ in ACCIDENTAL_ITEMS]
        acc_idx = acc_keys.index(event.accidental) if event.accidental in acc_keys else 4
        self.acc_combo.setCurrentIndex(acc_idx)

        self.octave_spin.setValue(event.octave)
        self.beats_spin.setValue(event.beats)
        self.hz_label.setText(f'{event.frequency:.2f} Hz')

        # Timbre
        self.brightness_sl.setValue(int(event.brightness * 100))
        self.density_sl.setValue(event.density)
        self.amplitude_sl.setValue(int(event.amplitude * 100))
        self.stability_sl.setValue(int(event.stability * 100))

        # Partials
        for i in range(16):
            w_val = event.partials.get_index(i)
            self._partial_sliders[i].setValue(int(w_val * 100))
            self._partial_val_labels[i].setText(f'{w_val:.2f}')

        # Pulse
        self.pulse_check.setChecked(event.pulse.enabled)
        self.pulse_count_sl.setValue(event.pulse.count)
        self.pulse_br_start_sl.setValue(int(event.pulse.brightness_start * 100))
        self.pulse_br_end_sl.setValue(int(event.pulse.brightness_end * 100))
        self.pulse_amp_start_sl.setValue(int(event.pulse.amp_start * 100))
        self.pulse_amp_end_sl.setValue(int(event.pulse.amp_end * 100))
        self.pulse_decay_sl.setValue(int(event.pulse.decay * 100))

        # Chord
        self.chord_check.setChecked(event.chord.enabled)
        self.chord_type_combo.setCurrentIndex(event.chord.chord_type)
        self.chord_voices_sl.setRange(1, CHORD_MAX_VOICES[event.chord.chord_type])
        self.chord_voices_sl.setValue(event.chord.num_voices)
        self.chord_balance_sl.setValue(int(event.chord.balance * 100))

        # Envelope
        self.attack_sl.setValue(int(event.attack * 100))
        self.release_sl.setValue(int(event.release * 100))

        self._blocking = False
        self._update_partial_frequency_labels()

    # ── Read UI → event ──────────────────────────────────────────────────────

    def _on_rest_toggled(self, checked: bool):
        self._set_note_controls_enabled(not checked)
        self._field_changed()

    def _set_note_controls_enabled(self, enabled: bool):
        for w in (self.note_combo, self.acc_combo, self.octave_spin):
            w.setEnabled(enabled)

    def _update_partial_frequency_labels(self):
        """Update frequency labels for each partial based on fundamental."""
        if not hasattr(self, 'partial_freq_labels'):
            return
        
        # Get fundamental frequency
        note = self.note_combo.currentText()
        acc = self.acc_combo.currentData()
        octave = self.octave_spin.value()
        
        from .gesture_model import note_to_hz
        fundamental = note_to_hz(note, acc, octave)
        
        # Update each partial label with note name
        for i, label in enumerate(self.partial_freq_labels):
            harmonic = i + 1
            freq = fundamental * harmonic
            note_name = hz_to_note_name(freq)
            label.setText(note_name)

    def _field_changed(self, *_):
        if self._blocking:
            return
        self._read_into_event()
        self._update_derived_labels()
        self._update_partial_frequency_labels()
        self.event_changed.emit()

    def _read_into_event(self):
        ev = self._event
        ev.is_rest = self.rest_check.isChecked()
        ev.note    = self.note_combo.currentText()
        ev.accidental = self.acc_combo.currentData()
        ev.octave  = self.octave_spin.value()
        ev.beats   = self.beats_spin.value()
        ev.brightness = self.brightness_sl.value() / 100.0
        ev.density    = self.density_sl.value()
        ev.amplitude  = self.amplitude_sl.value() / 100.0
        ev.stability  = self.stability_sl.value() / 100.0
        ev.attack     = self.attack_sl.value() / 100.0
        ev.release    = self.release_sl.value() / 100.0
        ev.pulse.enabled          = self.pulse_check.isChecked()
        ev.pulse.count            = self.pulse_count_sl.value()
        ev.pulse.brightness_start = self.pulse_br_start_sl.value() / 100.0
        ev.pulse.brightness_end   = self.pulse_br_end_sl.value() / 100.0
        ev.pulse.amp_start        = self.pulse_amp_start_sl.value() / 100.0
        ev.pulse.amp_end          = self.pulse_amp_end_sl.value() / 100.0
        ev.pulse.decay            = self.pulse_decay_sl.value() / 100.0
        ev.chord.enabled    = self.chord_check.isChecked()
        ev.chord.chord_type = self.chord_type_combo.currentIndex()
        ev.chord.num_voices = self.chord_voices_sl.value()
        ev.chord.balance    = self.chord_balance_sl.value() / 100.0

    def _update_derived_labels(self):
        ev = self._event
        self.pitch_display.setText(ev.pitch_label)
        self.hz_label.setText(f'{ev.frequency:.2f} Hz')
        self.brightness_vl.setText(f'{ev.brightness:.2f}')
        self.density_vl.setText(str(ev.density))
        self.amplitude_vl.setText(f'{ev.amplitude:.2f}')
        self.stability_vl.setText(f'{ev.stability:.2f}')
        self.attack_vl.setText(f'{ev.attack:.2f}s')
        self.release_vl.setText(f'{ev.release:.2f}s')
        self.pulse_count_vl.setText(str(self.pulse_count_sl.value()))
        self.pulse_br_start_vl.setText(f'{self.pulse_br_start_sl.value()/100:.2f}')
        self.pulse_br_end_vl.setText(f'{self.pulse_br_end_sl.value()/100:.2f}')
        self.pulse_amp_start_vl.setText(f'{self.pulse_amp_start_sl.value()/100:.2f}')
        self.pulse_amp_end_vl.setText(f'{self.pulse_amp_end_sl.value()/100:.2f}')
        self.pulse_decay_vl.setText(f'{self.pulse_decay_sl.value()/100:.2f}s')
        self.chord_voices_vl.setText(str(self.chord_voices_sl.value()))
        self.chord_balance_vl.setText(f'{self.chord_balance_sl.value()/100:.2f}')

    @property
    def current_event(self) -> NoteEvent:
        return self._event
