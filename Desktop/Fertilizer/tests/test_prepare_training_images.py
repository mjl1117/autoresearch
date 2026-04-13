"""Tests for prepare_training_images.py"""
import importlib
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import tifffile
from prepare_training_images import find_tiff_files, select_frame_indices, compute_auto_params, preprocess_frame, save_preprocessed_tiff, detect_python_executable, FALLBACK_PYTHON


def test_module_imports():
    """Script must import without errors."""
    mod = importlib.import_module("prepare_training_images")
    assert hasattr(mod, "CPSAM_MODEL_PATH")
    assert hasattr(mod, "FALLBACK_PYTHON")


def test_find_tiff_files_flat(tmp_path):
    """Finds .tif and .tiff files in the root directory."""
    (tmp_path / "a.tif").touch()
    (tmp_path / "b.tiff").touch()
    (tmp_path / "ignore.png").touch()
    result = find_tiff_files(tmp_path)
    names = {p.name for p in result}
    assert names == {"a.tif", "b.tiff"}


def test_find_tiff_files_recursive(tmp_path):
    """Finds TIFFs in nested subdirectories."""
    sub = tmp_path / "sub" / "deep"
    sub.mkdir(parents=True)
    (tmp_path / "root.tif").touch()
    (sub / "nested.tif").touch()
    result = find_tiff_files(tmp_path)
    names = {p.name for p in result}
    assert names == {"root.tif", "nested.tif"}


def test_find_tiff_files_empty(tmp_path):
    """Returns empty list when no TIFFs exist."""
    assert find_tiff_files(tmp_path) == []


def test_select_frame_indices_single():
    """Single-frame TIFF returns [0]."""
    assert select_frame_indices(1) == [0]


def test_select_frame_indices_two():
    """Two-frame TIFF returns [0, 1]."""
    assert select_frame_indices(2) == [0, 1]


def test_select_frame_indices_timelapse():
    """Multi-frame TIFF returns first and last only."""
    assert select_frame_indices(20) == [0, 19]


def test_select_frame_indices_three():
    """Three frames: first and last, not middle."""
    assert select_frame_indices(3) == [0, 2]


def test_compute_auto_params_with_pixel_size():
    """Computes radius and sigma from physical units when pixel size is known."""
    params = compute_auto_params(pixel_size_um=0.1, max_bead_diameter_um=50.0)
    assert params["rb_radius"] == pytest.approx(500.0)
    assert params["dog_sigma_high"] == pytest.approx(125.0)


def test_compute_auto_params_fallback():
    """Returns fallback values when pixel size is None."""
    params = compute_auto_params(pixel_size_um=None, max_bead_diameter_um=50.0)
    assert params["rb_radius"] == pytest.approx(50.0)
    assert params["dog_sigma_high"] == pytest.approx(10.0)


def test_compute_auto_params_zero_pixel_size():
    """pixel_size_um=0 is treated as unknown — falls back to defaults."""
    params = compute_auto_params(pixel_size_um=0, max_bead_diameter_um=50.0)
    assert params["rb_radius"] == pytest.approx(50.0)
    assert params["dog_sigma_high"] == pytest.approx(10.0)


def _synthetic_frame(h=64, w=64):
    rng = np.random.default_rng(42)
    return rng.integers(100, 4000, size=(h, w), dtype=np.uint16)


def test_preprocess_frame_output_range():
    """Output is float32 normalised to [0, 1]."""
    frame = _synthetic_frame()
    result = preprocess_frame(
        frame,
        rolling_ball=True, rb_radius=20.0,
        dog=True, dog_sigma_low=1.0, dog_sigma_high=5.0,
        clahe=True, clahe_clip_limit=0.01,
    )
    assert result.dtype == np.float32
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0


def test_preprocess_frame_shape_preserved():
    """Output shape matches input shape."""
    frame = _synthetic_frame(48, 96)
    result = preprocess_frame(
        frame,
        rolling_ball=False, rb_radius=20.0,
        dog=False, dog_sigma_low=1.0, dog_sigma_high=5.0,
        clahe=False, clahe_clip_limit=0.01,
    )
    assert result.shape == (48, 96)


def test_preprocess_frame_all_disabled():
    """With all steps disabled, output is still float32 [0,1] normalised."""
    frame = _synthetic_frame()
    result = preprocess_frame(
        frame,
        rolling_ball=False, rb_radius=20.0,
        dog=False, dog_sigma_low=1.0, dog_sigma_high=5.0,
        clahe=False, clahe_clip_limit=0.01,
    )
    assert result.dtype == np.float32
    assert float(result.max()) == pytest.approx(1.0, abs=1e-5)


def test_save_preprocessed_tiff_creates_file(tmp_path):
    """Saves a uint16 TIFF at the given path."""
    frame = np.random.default_rng(0).random((32, 32)).astype(np.float32)
    out = tmp_path / "out.tif"
    save_preprocessed_tiff(frame, out)
    assert out.exists()


def test_save_preprocessed_tiff_dtype(tmp_path):
    """Saved TIFF is uint16."""
    frame = np.linspace(0, 1, 32 * 32, dtype=np.float32).reshape(32, 32)
    out = tmp_path / "out.tif"
    save_preprocessed_tiff(frame, out)
    loaded = tifffile.imread(str(out))
    assert loaded.dtype == np.uint16


def test_save_preprocessed_tiff_creates_parents(tmp_path):
    """Creates intermediate directories if they don't exist."""
    frame = np.zeros((16, 16), dtype=np.float32)
    out = tmp_path / "sub" / "deep" / "frame.tif"
    save_preprocessed_tiff(frame, out)
    assert out.exists()


def test_detect_python_executable_conda_env():
    """Returns the current Python when running inside a conda env."""
    fake_exe = "/Users/matthew/miniforge3/envs/membrane-image/bin/python"
    with patch("prepare_training_images.sys") as mock_sys:
        mock_sys.executable = fake_exe
        result = detect_python_executable()
    assert result == fake_exe


def test_detect_python_executable_system_python():
    """Falls back to membrane-image Python when not in a conda env."""
    with patch("prepare_training_images.sys") as mock_sys:
        mock_sys.executable = "/opt/homebrew/opt/python@3.14/bin/python3.14"
        result = detect_python_executable()
    assert result == FALLBACK_PYTHON


def test_detect_python_executable_returns_string():
    """Always returns a string."""
    result = detect_python_executable()
    assert isinstance(result, str)
    assert len(result) > 0
