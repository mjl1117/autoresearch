"""
Neuroaesthetic Music Research — Main Entry Point
"""
import os
import sys
import subprocess
from pathlib import Path

# ── Qt Multimedia backend selection ──────────────────────────────────
# The default FFmpeg backend in Qt 6 has a known bug where the player
# freezes or stops (~5 s) when transitioning between clips that have
# different audio properties (e.g. 48 kHz mono intro → 44.1 kHz stereo
# clips).  The position stays at 0 despite PlayingState firing.
#
# On macOS the native AVFoundation ("darwin") backend handles sequential
# clip playback reliably and avoids the FFmpeg issue entirely.
# Must be set BEFORE QApplication is created.
import platform
if platform.system() == "Darwin":
    os.environ["QT_MEDIA_BACKEND"] = "darwin"
else:
    # On non-macOS, disable HW-accelerated decoding to reduce FFmpeg issues.
    os.environ["QT_FFMPEG_DECODING_HW_DEVICE_TYPES"] = ","

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from theme import (
    apply_theme,
    BG_PRIMARY, BG_SECONDARY, BG_TERTIARY, BG_ELEVATED,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_ACCENT, TEXT_DISABLED,
    ACCENT, ACCENT_HOVER, ACCENT_PRESS, ACCENT_GLOW, ACCENT_LIGHT,
    BORDER, BORDER_STRONG, BORDER_FOCUS,
    GOLD, SUCCESS, DANGER,
    BTN_PRIMARY, BTN_SECONDARY, BTN_GHOST
)

from PyQt6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QGridLayout, QSizePolicy
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPainter, QColor, QLinearGradient, QPen, QFont

from src.experiments.emotion_gui import EmotionGUI
from src.experiments.experiment_dialogue import ListenerIDDialog, ExperimentEndDialog
from src.experiments.video_setup_menu import VideoSetupMenu
from src.utils.save_metadata import save_experiment_metadata
from src.biosignals.bitalino_setup_dialog import BitalinoSetupDialog


_sc_process = None

def cleanup_supercollider():
    """Kill SuperCollider when app exits."""
    import subprocess
    try:
        subprocess.run(['killall', 'sclang'], stderr=subprocess.DEVNULL, check=False)
        subprocess.run(['killall', 'scsynth'], stderr=subprocess.DEVNULL, check=False)
        print("\n✓ SuperCollider stopped")
    except:
        pass


class _ModuleCard(QFrame):
    """A clickable module card with icon, name, and description."""

    def __init__(self, symbol: str, name: str, desc: str, choice: str,
                 accent_color: str = ACCENT, parent=None):
        super().__init__(parent)
        self._choice = choice
        self._accent = accent_color
        self._hovered = False

        self.setFixedHeight(138)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._apply_style(False)

        inner = QVBoxLayout(self)
        inner.setContentsMargins(20, 16, 20, 14)
        inner.setSpacing(5)

        # Top row: symbol + name
        top = QHBoxLayout()
        top.setSpacing(12)

        sym = QLabel(symbol)
        sym.setStyleSheet(f"""
            font-size: 22pt;
            color: {accent_color};
            background: transparent;
            min-width: 34px;
        """)

        nm = QLabel(name)
        nm.setStyleSheet(f"""
            font-family: "Baskerville", "Didot", "Palatino", serif;
            font-size: 15pt;
            font-weight: bold;
            color: {TEXT_PRIMARY};
            background: transparent;
        """)

        top.addWidget(sym)
        top.addWidget(nm)
        top.addStretch()

        # Description
        ds = QLabel(desc)
        ds.setStyleSheet(f"""
            font-family: "Optima", "Gill Sans", "Candara", sans-serif;
            font-size: 10pt;
            color: {TEXT_SECONDARY};
            background: transparent;
        """)
        ds.setWordWrap(True)

        inner.addLayout(top)
        inner.addWidget(ds)
        inner.addStretch()

    def _apply_style(self, hovered: bool):
        border_color = self._accent if hovered else BORDER_STRONG
        bg = BG_TERTIARY if hovered else BG_SECONDARY
        border_width = "2px" if hovered else "1px"
        self.setStyleSheet(f"""
            QFrame {{
                background: {bg};
                border: {border_width} solid {border_color};
                border-radius: 14px;
            }}
        """)

    def enterEvent(self, event):
        self._hovered = True
        self._apply_style(True)

    def leaveEvent(self, event):
        self._hovered = False
        self._apply_style(False)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # Find HomeScreen parent and trigger selection
            parent = self.parent()
            while parent and not isinstance(parent, HomeScreen):
                parent = parent.parent()
            if parent:
                parent._select(self._choice)


class HomeScreen(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Neuroaesthetic Music Research')
        self.setMinimumWidth(700)
        self.setMinimumHeight(580)
        self.setStyleSheet(f'background: {BG_PRIMARY}; color: {TEXT_PRIMARY};')
        self.choice = None

        root = QVBoxLayout(self)
        root.setContentsMargins(52, 44, 52, 36)
        root.setSpacing(0)

        # ── Brand mark ──────────────────────────────────────────────────────
        brand_row = QHBoxLayout()
        brand_row.setSpacing(12)

        mark = QLabel('◈')
        mark.setStyleSheet(f"""
            font-size: 26pt;
            color: {ACCENT};
            background: transparent;
        """)

        brand_text = QLabel('NEUROAESTHETIC')
        brand_text.setStyleSheet(f"""
            font-family: "Optima", "Gill Sans", "Candara", sans-serif;
            font-size: 10pt;
            font-weight: 700;
            color: {ACCENT};
            background: transparent;
            letter-spacing: 4px;
        """)
        brand_text.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        brand_row.addStretch()
        brand_row.addWidget(mark)
        brand_row.addWidget(brand_text)
        brand_row.addStretch()
        root.addLayout(brand_row)

        root.addSpacing(10)

        # ── Title ────────────────────────────────────────────────────────────
        title = QLabel('Music Research')
        title.setStyleSheet(f"""
            font-family: "Baskerville", "Didot", "Palatino", serif;
            font-size: 36pt;
            font-weight: bold;
            color: {TEXT_PRIMARY};
            background: transparent;
            letter-spacing: -1px;
        """)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(title)

        root.addSpacing(10)

        # ── Tagline ───────────────────────────────────────────────────────────
        tagline = QLabel('Where biosignals meet musical meaning')
        tagline.setStyleSheet(f"""
            font-family: "Optima", "Gill Sans", "Candara", sans-serif;
            font-size: 12pt;
            color: {TEXT_SECONDARY};
            background: transparent;
            letter-spacing: 0.5px;
        """)
        tagline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(tagline)

        root.addSpacing(8)

        # ── Accent rule ────────────────────────────────────────────────────
        rule = QFrame()
        rule.setFrameShape(QFrame.Shape.HLine)
        rule.setFixedHeight(1)
        rule.setStyleSheet(f"background: {ACCENT_GLOW}; border: none; margin: 0 80px;")
        root.addWidget(rule)

        root.addSpacing(28)

        # ── Module cards — 2×2 grid ───────────────────────────────────────
        grid = QGridLayout()
        grid.setSpacing(12)

        modules = [
            ('experiment', '◉', 'Run Experiment',
             'Capture valence & arousal in sync with biosignal recording',
             ACCENT),
            ('music', '♬', 'Compose Music',
             'Spectral gesture designer with SuperCollider synthesis',
             ACCENT),
            ('analyze', '∿', 'Analyze Audio',
             'Extract spectral and temporal features from recordings',
             GOLD),
            ('transfer', '◈', 'Transfer Learning',
             'Train the PhysioEncoder neural model on your biosignal data',
             GOLD),
        ]

        for i, (choice, sym, name, desc, color) in enumerate(modules):
            card = _ModuleCard(sym, name, desc, choice, accent_color=color)
            grid.addWidget(card, i // 2, i % 2)

        root.addLayout(grid)

        root.addSpacing(24)

        # ── Quit ─────────────────────────────────────────────────────────────
        quit_btn = QPushButton('Exit')
        quit_btn.setStyleSheet(BTN_GHOST)
        quit_btn.setFixedWidth(120)
        quit_btn.clicked.connect(self.reject)
        root.addWidget(quit_btn, alignment=Qt.AlignmentFlag.AlignCenter)

    def _select(self, choice: str):
        self.choice = choice
        self.accept()


def run_experiment_mode(app):
    while True:
        # Step 1: Admin picks intro video + clips
        setup = VideoSetupMenu()
        if setup.exec() != QDialog.DialogCode.Accepted:
            break
        intro_video, clips = setup.selected_intro_video, setup.selected_files
        verbal_on = setup.verbal_responses_enabled

        # Step 2: Configure Bitalino — only shown if biosignals were enabled
        # in the setup menu. If not, skip the dialog entirely.
        if setup.biosignals_enabled:
            bitalino_dlg = BitalinoSetupDialog()
            if bitalino_dlg.exec() != QDialog.DialogCode.Accepted:
                break
            biosignals_enabled = bitalino_dlg.enable_biosignals
            bitalino_mac = bitalino_dlg.mac_address
        else:
            biosignals_enabled = False
            bitalino_mac = None

        while True:
            # Step 3: Enter participant ID
            id_dlg = ListenerIDDialog()
            if id_dlg.exec() != QDialog.DialogCode.Accepted:
                break

            listener_id = id_dlg.listener_id
            save_experiment_metadata(listener_id, intro_video, clips)

            # Step 4: Run experiment with biosignal config
            window = EmotionGUI(
                participant_id=listener_id,
                clip_paths=clips,
                intro_video_path=intro_video,
                biosignals_enabled=biosignals_enabled,
                bitalino_mac=bitalino_mac,
                enable_verbal_responses=verbal_on,
            )
            window.showFullScreen()
            app.exec()

            # Brief pause between sessions so the Qt multimedia backend
            # fully releases decoder resources before the next window.
            import time
            time.sleep(1.0)

            end = ExperimentEndDialog()
            if end.exec() != 1:
                break


def launch_supercollider():
    """Launch SuperCollider using the shell script."""
    import socket
    import time
    
    # Check if already running
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(0.3)
            s.sendto(b'\x00', ('127.0.0.1', 57110))
            s.recvfrom(256)
            print("✓ SuperCollider already running")
            return
    except:
        pass
    
    print("Starting SuperCollider...")
    
    script_path = Path('start_supercollider.sh')
    if script_path.exists():
        try:
            script_path.chmod(0o755)
            subprocess.Popen(
                ['osascript', '-e', f'do shell script "{script_path.absolute()} > /dev/null 2>&1 &"'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print("✓ SuperCollider launching...")
            print("  Waiting for server to start (5 seconds)...")
            for i in range(10):
                time.sleep(0.5)
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                        s.settimeout(0.3)
                        s.sendto(b'\x00', ('127.0.0.1', 57110))
                        s.recvfrom(256)
                        print(f"✓ SuperCollider ready after {(i+1)*0.5:.1f}s")
                        return
                except:
                    pass
            print("⚠ Still waiting for SuperCollider...")
            print("  It should be ready soon - continue using the app")
        except Exception as e:
            print(f"⚠ Could not launch: {e}")
    else:
        print("⚠ start_supercollider.sh not found")
        print("  Please start SuperCollider manually")


def run_generate_music_mode(app):
    try:
        from src.generative_music.gesture_designer.generate_music_menu import GenerateMusicMenu
        from src.generative_music.gesture_designer.gesture_designer import GestureDesigner
    except ImportError as e:
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(None, 'Not Installed', f'Gesture Designer not found:\n{e}')
        return

    while True:
        # Verify SC is alive before each tool open — also warms the UDP socket
        # so the first connection check inside GesturePlayer succeeds reliably.
        launch_supercollider()

        menu = GenerateMusicMenu()
        if menu.exec() != QDialog.DialogCode.Accepted:
            break

        if menu.selected_tool == 'gesture_designer':
            designer = GestureDesigner()
            designer.showFullScreen()
            designer.raise_()
            designer.activateWindow()
            app.exec()
        elif menu.selected_tool == 'chord_builder':
            from src.generative_music.gesture_designer.spectral_chord_builder import SpectralChordBuilder
            builder = SpectralChordBuilder()
            builder.show()
            builder.raise_()
            builder.activateWindow()
            app.exec()
        elif menu.selected_tool == 'texture_engine':
            from src.generative_music.gesture_designer.adaptive_texture_ui import AdaptiveTextureUI
            dlg = AdaptiveTextureUI()
            dlg.exec()
        elif menu.selected_tool == 'sequence_player':
            from src.generative_music.gesture_designer.gesture_sequence_player import GestureSequencePlayer
            dlg = GestureSequencePlayer()
            dlg.exec()
        elif menu.selected_tool == 'human_feedback':
            from src.generative_music.gesture_designer.human_feedback import HumanFeedbackWindow
            win = HumanFeedbackWindow()
            win.show()
            win.raise_()
            win.activateWindow()
            app.exec()
        elif menu.selected_tool == 'dorico_integration':
            from src.dorico_bridge.dorico_bridge_ui import DoricoBridgeWindow
            win = DoricoBridgeWindow()
            win.show()
            win.raise_()
            win.activateWindow()
            app.exec()

    # SC is only needed inside the generate-music tab — shut it down on exit.
    cleanup_supercollider()


def run_audio_analysis_mode(app):
    """Audio analysis: analyze, compare, and batch process recordings."""
    try:
        from src.audio_analysis.audio_analysis_menu import AudioAnalysisMenu
    except ImportError as e:
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(None, 'Not Installed', f'Audio Analysis not found:\n{e}')
        return

    menu = AudioAnalysisMenu()
    menu.exec()


def run_transfer_learning_mode(app):
    """Transfer learning: pretrain, fine-tune, and evaluate the PhysioEncoder."""
    try:
        from src.transfer_learning.transfer_learning_menu import TransferLearningMenu
    except ImportError as e:
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(None, 'Not Installed', f'Transfer Learning not found:\n{e}')
        return

    menu = TransferLearningMenu()
    menu.exec()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    apply_theme(app)
    while True:
        home = HomeScreen()
        if home.exec() != QDialog.DialogCode.Accepted:
            break
        if home.choice == 'experiment':
            run_experiment_mode(app)
        elif home.choice == 'music':
            run_generate_music_mode(app)
        elif home.choice == 'analyze':
            run_audio_analysis_mode(app)
        elif home.choice == 'transfer':
            run_transfer_learning_mode(app)
    sys.exit()
