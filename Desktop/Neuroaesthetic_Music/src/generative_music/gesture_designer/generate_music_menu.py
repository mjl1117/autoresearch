"""
generate_music_menu.py
"Generate Music" entry-point dialog.

Presents a styled dropdown of available compositional tools.
Currently ships: Spectral Gesture Designer.
Additional tools can be added by appending to TOOLS.

MJL Neuroaesthetic Music Research — 2026
"""
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QFrame
)
from PyQt6.QtCore import Qt

# ── Tool registry ─────────────────────────────────────────────────────────────

TOOLS = [
    {
        'label':  'Spectral Gesture Designer',
        'desc':   'Design microtonal harmonic-series gestures with per-partial control, '
                  'pulse bursts, automation, and live SuperCollider preview.',
        'key':    'gesture_designer',
        'ready':  True,
    },
    {
        'label':  'Spectral Chord Builder',
        'desc':   'Construct and preview virtual-root spectral chords — '
                  'Triad, Seventh, Extended, or pure Spectral (harmonic series). '
                  'Save chords to the gesture library for use in compositions.',
        'key':    'chord_builder',
        'ready':  True,
    },
    {
        'label':  'Adaptive Texture Engine',
        'desc':   'Select a valence/arousal target and the engine continuously '
                  'sequences chords from the scored library to navigate toward it.',
        'key':    'texture_engine',
        'ready':  True,
    },
    {
        'label':  'Gesture Sequence Player',
        'desc':   'Fully generative session with biosignal feedback and live co-improviser. '
                  'Trajectory mode navigates toward your target valence/arousal; '
                  'Amplify mode pushes your current emotional state further. '
                  'A performer listener (mic) infers pitch and chord type in real time '
                  'and feeds a probabilistic co-improviser with motif memory. '
                  'All sessions are logged to JSONL for MCLA training.',
        'key':    'sequence_player',
        'ready':  True,
    },
    {
        'label': 'Human Feedback',
        'desc':  'Blind-rate gestures, chord gestures, and short music phrases '
                 'against ML predictions. Your ratings personalise generation '
                 'to your preferences and feed into model retraining.',
        'key':   'human_feedback',
        'ready': True,
    },
    {
        'label': 'Dorico Bridge',
        'desc':  'Bidirectional notation integration with Dorico. Export gestures '
                 'as MusicXML (quarter-tone aware), import your compositional edits '
                 'back as spectral gestures, and train a MIDI encoder for '
                 'notation-based valence/arousal prediction without re-synthesising audio.',
        'key':   'dorico_integration',
        'ready': True,
    },
]


class GenerateMusicMenu(QDialog):
    """Tool-selector dialog for the Generate Music tab."""

    WINDOW_BG  = '#F8F6EF'
    PANEL_BG   = '#EDEADF'
    CARD_BG    = '#E3DFD2'
    TEXT       = '#1E1A14'
    LABEL      = '#7C809B'
    ACCENT     = '#A9AFD1'
    GREEN      = '#4E845D'
    DISABLED   = '#B8B5A4'

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Generate Music')
        self.setMinimumWidth(520)
        self.setStyleSheet(
            f'background: {self.WINDOW_BG}; color: {self.TEXT}; font-size: 13pt;')

        self.selected_tool: str = ''
        self._tool_index: int = 0

        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24)
        lay.setSpacing(18)

        # Title
        title = QLabel('Generate Music')
        title.setStyleSheet(
            f'color: {self.ACCENT}; font-size: 22pt; font-weight: bold;')
        lay.addWidget(title)

        sub = QLabel('Select a compositional tool to open.')
        sub.setStyleSheet(f'color: {self.LABEL}; font-size: 12pt;')
        lay.addWidget(sub)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f'color: #D0CCAC; background: #D0CCAC;')
        lay.addWidget(sep)

        # Tool dropdown
        combo_row = QHBoxLayout()
        combo_row.addWidget(QLabel('Tool:'))
        self.tool_combo = QComboBox()
        self.tool_combo.setStyleSheet(
            f'background: {self.CARD_BG}; color: {self.TEXT};'
            f'border: 1px solid #D0CCAC; border-radius: 5px;'
            f'padding: 6px 12px; font-size: 13pt;')
        for t in TOOLS:
            self.tool_combo.addItem(t['label'])
        self.tool_combo.currentIndexChanged.connect(self._on_tool_changed)
        combo_row.addWidget(self.tool_combo, stretch=1)
        lay.addLayout(combo_row)

        # Description card
        self.desc_label = QLabel()
        self.desc_label.setWordWrap(True)
        self.desc_label.setStyleSheet(
            f'background: {self.CARD_BG}; color: {self.LABEL};'
            f'font-size: 11pt; border-radius: 6px; padding: 12px;')
        lay.addWidget(self.desc_label)

        # Status badge
        self.status_label = QLabel()
        self.status_label.setStyleSheet('font-size: 11pt;')
        lay.addWidget(self.status_label)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton('Cancel')
        cancel_btn.setStyleSheet(self._btn_style(self.CARD_BG, self.LABEL))
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        self.open_btn = QPushButton('Open Tool')
        self.open_btn.setStyleSheet(
            self._btn_style('#A9AFD1', '#1E1A14', border='#A9AFD1'))
        self.open_btn.clicked.connect(self._open_tool)
        btn_row.addWidget(self.open_btn)
        lay.addLayout(btn_row)

        self._on_tool_changed(0)

    def _btn_style(self, bg, col, border='#D0CCAC'):
        return (f'QPushButton {{ background:{bg}; color:{col}; '
                f'border:1px solid {border}; border-radius:5px; '
                f'padding:8px 22px; font-size:12pt; }}'
                f'QPushButton:hover {{ background: {self.PANEL_BG}; }}')

    def _on_tool_changed(self, index: int):
        self._tool_index = index
        t = TOOLS[index]
        self.desc_label.setText(t['desc'])
        if t['ready']:
            self.status_label.setText(f'<span style="color:{self.GREEN}">● Available</span>')
            self.open_btn.setEnabled(True)
        else:
            self.status_label.setText(
                f'<span style="color:{self.DISABLED}">○ Coming soon</span>')
            self.open_btn.setEnabled(False)

    def _open_tool(self):
        self.selected_tool = TOOLS[self._tool_index]['key']
        self.accept()
