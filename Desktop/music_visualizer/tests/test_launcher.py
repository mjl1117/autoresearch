import pytest
from ui.launcher import LauncherState, LauncherUI


def test_initial_state_is_expanded():
    ui = LauncherUI(width=1920, height=1080)
    assert ui.state == LauncherState.EXPANDED


def test_play_collapses_bar():
    ui = LauncherUI(width=1920, height=1080)
    ui.on_play()
    assert ui.state == LauncherState.COLLAPSED


def test_stop_expands_bar():
    ui = LauncherUI(width=1920, height=1080)
    ui.on_play()
    ui.on_stop()
    assert ui.state == LauncherState.EXPANDED


def test_mode_toggle():
    ui = LauncherUI(width=1920, height=1080)
    assert ui.mode == "prerecorded"
    ui.toggle_mode()
    assert ui.mode == "live"
    ui.toggle_mode()
    assert ui.mode == "prerecorded"


def test_set_file_path():
    ui = LauncherUI(width=1920, height=1080)
    ui.set_file_path("/path/to/track.mp3")
    assert ui.file_path == "/path/to/track.mp3"


def test_bar_height_collapses():
    ui = LauncherUI(width=1920, height=1080)
    ui.on_play()
    for _ in range(60):
        ui.tick(dt=1.0 / 60)
    assert ui.bar_height < ui.expanded_height


def test_export_requested_fires():
    ui = LauncherUI(width=1920, height=1080)
    ui.set_file_path("/path/to/track.mp3")
    fired = []
    ui.on_export_requested = lambda: fired.append(True)
    ui.request_export()
    assert len(fired) == 1
