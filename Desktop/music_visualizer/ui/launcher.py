from __future__ import annotations
from enum import Enum, auto
from typing import Callable

_CLEAN_FONTS = ["Helvetica Neue", "Helvetica", "Arial", "Trebuchet MS", None]

def _font(size: int):
    import pygame
    for name in _CLEAN_FONTS:
        f = pygame.font.SysFont(name, size)
        if f is not None:
            return f
    return pygame.font.Font(None, size)


class LauncherState(Enum):
    EXPANDED = auto()
    COLLAPSED = auto()
    TRANSITIONING = auto()


_EXPANDED_H = 112
_COLLAPSED_H = 36
_TRANSITION_S = 0.2


class LauncherUI:
    """
    Top-bar state machine. Pygame rendering is in draw().
    All state transitions are pure Python so they can be unit-tested
    without a display.
    """

    def __init__(self, width: int, height: int) -> None:
        self._w = width
        self._h = height
        self.state = LauncherState.EXPANDED
        self.mode: str = "prerecorded"
        self.file_path: str | None = None
        self.selected_device_index: int = 0
        self.device_list: list[dict] = []

        self.expanded_height: int = _EXPANDED_H
        self.collapsed_height: int = _COLLAPSED_H
        self.bar_height: float = float(_EXPANDED_H)

        self._transition_elapsed: float = 0.0
        self._transition_target: float = float(_EXPANDED_H)

        self.status_text: str = ""
        self.is_recording: bool = False

        # Callbacks set by main.py
        self.on_export_requested: Callable[[], None] = lambda: None
        self.on_play_requested: Callable[[], None] = lambda: None
        self.on_stop_requested: Callable[[], None] = lambda: None
        self.on_go_live_requested: Callable[[], None] = lambda: None

    def toggle_mode(self) -> None:
        self.mode = "live" if self.mode == "prerecorded" else "prerecorded"

    def set_file_path(self, path: str) -> None:
        self.file_path = path

    def set_device_list(self, devices: list[dict]) -> None:
        self.device_list = devices

    def on_play(self) -> None:
        self._start_transition(_COLLAPSED_H)
        self.state = LauncherState.COLLAPSED

    def on_stop(self) -> None:
        self._start_transition(_EXPANDED_H)
        self.state = LauncherState.EXPANDED

    def request_export(self) -> None:
        self.on_export_requested()

    def tick(self, dt: float) -> None:
        # Check if there's an active transition (transition_elapsed is being tracked)
        if self._transition_elapsed < _TRANSITION_S:
            self._transition_elapsed += dt
            t = min(self._transition_elapsed / _TRANSITION_S, 1.0)
            t = t * t * (3.0 - 2.0 * t)  # ease-in-out cubic
            start = (
                float(_EXPANDED_H)
                if self._transition_target == _COLLAPSED_H
                else float(_COLLAPSED_H)
            )
            self.bar_height = start + (self._transition_target - start) * t
            if self._transition_elapsed >= _TRANSITION_S:
                self.bar_height = self._transition_target

    def _start_transition(self, target_h: float) -> None:
        self._transition_target = float(target_h)
        self._transition_elapsed = 0.0

    def handle_click(self, x: int, y: int) -> None:
        """Process a mouse click at (x, y). Call from the event loop on MOUSEBUTTONDOWN."""
        bar_h = int(self.bar_height)
        if y > bar_h:
            return  # click outside the bar

        if bar_h <= _COLLAPSED_H + 4:
            # Clicking collapsed strip expands it
            self._start_transition(_EXPANDED_H)
            self.state = LauncherState.EXPANDED
            return

        btn_w, btn_h = 140, 34
        mode_x = 14
        mode_y = (bar_h - btn_h) // 2
        rx = self._w - 280

        # Mode toggles
        if mode_y <= y <= mode_y + btn_h:
            if mode_x <= x <= mode_x + btn_w:
                self.mode = "prerecorded"
                return
            if mode_x + btn_w + 6 <= x <= mode_x + 2 * btn_w + 6:
                self.mode = "live"
                return

        # Action buttons
        if mode_y <= y <= mode_y + btn_h:
            if self.mode == "prerecorded":
                if rx <= x <= rx + 110:
                    self.on_play_requested()
                    return
                if rx + 118 <= x <= rx + 236:
                    self.on_export_requested()
                    return
            else:
                if rx <= x <= rx + 148:
                    self.on_go_live_requested()
                    return

    def draw(self, surface) -> None:
        """Render the top bar onto a pygame Surface. Called once per frame."""
        import pygame

        bar_h = int(self.bar_height)
        bar_surf = pygame.Surface((self._w, bar_h), pygame.SRCALPHA)
        bar_surf.fill((19, 19, 31, 230))
        pygame.draw.line(bar_surf, (42, 42, 58), (0, bar_h - 1), (self._w, bar_h - 1), 1)

        if bar_h <= _COLLAPSED_H + 4:
            self._draw_collapsed_strip(bar_surf, bar_h)
        else:
            self._draw_expanded_panel(bar_surf, bar_h)

        surface.blit(bar_surf, (0, 0))

    def _draw_collapsed_strip(self, surf, h: int) -> None:
        import pygame
        font = _font(24)
        dot_color = (78, 195, 161) if self.mode == "prerecorded" else (225, 29, 72)
        pygame.draw.circle(surf, dot_color, (18, h // 2), 7)
        label = self.status_text or ("LIVE" if self.mode == "live" else "READY")
        text = font.render(label, True, dot_color)
        surf.blit(text, (34, h // 2 - text.get_height() // 2))

    def _draw_expanded_panel(self, surf, h: int) -> None:
        import pygame
        font_md = _font(24)
        font_sm = _font(22)

        mode_x = 14
        btn_w, btn_h = 140, 34
        mode_y = (h - btn_h) // 2

        pre_col = (139, 92, 246) if self.mode == "prerecorded" else (38, 38, 56)
        live_col = (225, 29, 72) if self.mode == "live" else (38, 38, 56)

        pygame.draw.rect(surf, pre_col, (mode_x, mode_y, btn_w, btn_h), border_radius=5)
        pygame.draw.rect(surf, live_col, (mode_x + btn_w + 6, mode_y, btn_w, btn_h), border_radius=5)

        def _center_text(text_surf, rect_x, rect_w, rect_y, rect_h):
            surf.blit(text_surf, (
                rect_x + (rect_w - text_surf.get_width()) // 2,
                rect_y + (rect_h - text_surf.get_height()) // 2,
            ))

        _center_text(font_sm.render("Pre-recorded", True, (255, 255, 255)), mode_x, btn_w, mode_y, btn_h)
        _center_text(font_sm.render("Live Input", True, (255, 255, 255)), mode_x + btn_w + 6, btn_w, mode_y, btn_h)

        # Status text center
        if self.status_text:
            st = font_sm.render(self.status_text, True, (160, 160, 180))
            surf.blit(st, (self._w // 2 - st.get_width() // 2, mode_y + (btn_h - st.get_height()) // 2))

        rx = self._w - 280
        if self.mode == "prerecorded":
            play_col = (78, 195, 161)
            pygame.draw.rect(surf, play_col, (rx, mode_y, 110, btn_h), border_radius=6)
            _center_text(font_md.render("▶  Play", True, (10, 10, 20)), rx, 110, mode_y, btn_h)
            pygame.draw.rect(surf, (38, 38, 56), (rx + 118, mode_y, 118, btn_h), border_radius=6)
            pygame.draw.rect(surf, (60, 60, 80), (rx + 118, mode_y, 118, btn_h), 1, border_radius=6)
            _center_text(font_md.render("Export", True, (200, 200, 220)), rx + 118, 118, mode_y, btn_h)
        else:
            pygame.draw.rect(surf, (225, 29, 72), (rx, mode_y, 148, btn_h), border_radius=6)
            _center_text(font_md.render("⏺  Go Live", True, (255, 255, 255)), rx, 148, mode_y, btn_h)
