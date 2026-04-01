"""
note_box_widget.py
Clickable note tile for the gesture sequence rail.

Draws a spectrum bar-graph of partial weights, shows the pitch label,
beat count, and highlights in green when selected.

MJL Neuroaesthetic Music Research — 2026
"""
from PyQt6.QtWidgets import QWidget, QSizePolicy
from PyQt6.QtCore    import Qt, QRect, QRectF, pyqtSignal
from PyQt6.QtGui     import QPainter, QColor, QPen, QFont, QFontMetrics, QBrush

from .gesture_model import NoteEvent

# ── Palette (Deep Resonance dark theme) ──────────────────────────────────────
BG_NORMAL   = QColor('#2E2418')   # warm dark brown card — clearly not black
BG_HOVER    = QColor('#3A3028')   # lifted warm brown on hover
BG_REST     = QColor('#201810')   # rest tile — slightly recessed
BORDER_NORM = QColor('#4A3A2C')   # warm brown border
BORDER_SEL  = QColor('#A8B4E0')   # luminous periwinkle selection
TEXT_MAIN   = QColor('#EDE8DF')   # warm white
TEXT_DIM    = QColor('#8A8478')   # muted sand
BAR_COLOR   = QColor('#5A6898')   # mid-periwinkle bar fill
BAR_HIGH    = QColor('#A8B4E0')   # luminous periwinkle for boosted partials
PULSE_DOT   = QColor('#C8905C')   # warm gold pulse indicator
CHORD_COLOR = QColor('#7EC8E8')   # sky signal — chord lines

BOX_W, BOX_H = 118, 100


class NoteBoxWidget(QWidget):
    """A single tile in the gesture sequence rail.

    Signals:
        clicked(index)   — emitted when the user clicks this tile
    """
    clicked = pyqtSignal(int)

    def __init__(self, event: NoteEvent, index: int, parent=None):
        super().__init__(parent)
        self.event     = event
        self.index     = index
        self.selected  = False
        self._hovered  = False

        self.setFixedSize(BOX_W, BOX_H)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)

    def update_event(self, event: NoteEvent):
        self.event = event
        self.update()

    def set_selected(self, sel: bool):
        self.selected = sel
        self.update()

    # ── Events ───────────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.index)

    def enterEvent(self, event):
        self._hovered = True
        self.update()

    def leaveEvent(self, event):
        self._hovered = False
        self.update()

    # ── Paint ────────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        ev = self.event
        w, h = self.width(), self.height()

        # Background
        bg = BG_REST if ev.is_rest else (BG_HOVER if self._hovered else BG_NORMAL)
        p.fillRect(0, 0, w, h, bg)

        # Border
        pen = QPen(BORDER_SEL if self.selected else BORDER_NORM,
                   2.5 if self.selected else 1.0)
        p.setPen(pen)
        p.drawRoundedRect(1, 1, w - 2, h - 2, 6, 6)

        if ev.is_rest:
            self._draw_rest(p, w, h)
        else:
            self._draw_spectrum(p, w, h, ev)
            self._draw_pitch(p, w, h, ev)

        if ev.pulse.enabled:
            self._draw_pulse_indicator(p, w, h)

        if ev.chord.enabled:
            self._draw_chord_indicator(p, w, h)

        p.end()

    def _draw_spectrum(self, p: QPainter, w: int, h: int, ev: NoteEvent):
        """Draw a 16-bar partial weight spectrum in the lower half."""
        weights = ev.partials.as_list()
        n       = 16
        bar_area_top  = 36
        bar_area_bot  = h - 8
        bar_area_h    = bar_area_bot - bar_area_top
        bar_w         = (w - 16) / n
        max_w         = max(weights) if any(weights) else 1.0

        for i, wt in enumerate(weights):
            norm    = wt / max_w if max_w > 0 else 0
            bar_h   = max(1, int(norm * bar_area_h))
            bx      = int(8 + i * bar_w)
            bw      = max(2, int(bar_w) - 1)
            by      = bar_area_bot - bar_h

            color = BAR_HIGH if wt > 1.5 else BAR_COLOR
            fade  = QColor(color)
            fade.setAlpha(80 + int(150 * norm))
            p.fillRect(bx, by, bw, bar_h, fade)

    def _draw_pitch(self, p: QPainter, w: int, h: int, ev: NoteEvent):
        """Draw pitch label and beat count."""
        # Pitch label — Baskerville for editorial character
        font = QFont('Baskerville', 12, QFont.Weight.Bold)
        p.setFont(font)
        p.setPen(QPen(TEXT_MAIN))
        fm = p.fontMetrics()
        elided = fm.elidedText(ev.pitch_label, Qt.TextElideMode.ElideRight, w - 4)
        p.drawText(QRect(0, 5, w, 22), Qt.AlignmentFlag.AlignHCenter, elided)

        # Beats / brightness mini-info — Menlo for numerical data
        font2 = QFont('Menlo', 7)
        p.setFont(font2)
        p.setPen(QPen(TEXT_DIM))
        info = f'×{ev.beats}  br={ev.brightness:.1f}'
        p.drawText(QRect(0, 20, w, 14), Qt.AlignmentFlag.AlignHCenter, info)

    def _draw_rest(self, p: QPainter, w: int, h: int):
        """Draw a rest symbol."""
        font = QFont('Baskerville', 18)
        p.setFont(font)
        p.setPen(QPen(TEXT_DIM))
        p.drawText(QRect(0, 0, w, h - 16), Qt.AlignmentFlag.AlignCenter, '—')
        font2 = QFont('Menlo', 8)
        p.setFont(font2)
        beats_text = f'REST ×{self.event.beats}'
        p.drawText(QRect(0, h - 20, w, 16), Qt.AlignmentFlag.AlignHCenter, beats_text)

    def _draw_pulse_indicator(self, p: QPainter, w: int, h: int):
        """Small dot cluster in top-right indicating pulse burst."""
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(PULSE_DOT))
        n = min(self.event.pulse.count, 6)
        for i in range(n):
            p.drawEllipse(w - 8 - i * 6, 6, 4, 4)

    def _draw_chord_indicator(self, p: QPainter, w: int, h: int):
        """Three stacked horizontal lines in top-left indicating chord mode."""
        p.setPen(QPen(CHORD_COLOR, 2))
        for i in range(3):
            y = 7 + i * 5
            p.drawLine(5, y, 16, y)
