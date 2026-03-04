"""
tests/test_plot_log.py
======================

Unit tests for plot_log.py.

All tests are mne-free and run without any neural-data dependencies.
"""

import re
import sys
import textwrap
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from plot_log import (
    _EPOCH_RE,
    _KV_RE,
    _NPI_RE,
    _BASE_RE,
    find_log,
    list_runs,
    parse_log,
    plot_metrics,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_log(tmp_path: Path, content: str) -> Path:
    """Write *content* to a training.log file inside a run directory."""
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()
    log = run_dir / "training.log"
    log.write_text(textwrap.dedent(content), encoding="utf-8")
    return log


# ---------------------------------------------------------------------------
# TestRegexPatterns — fast unit tests for the compiled regexes
# ---------------------------------------------------------------------------


class TestRegexPatterns:
    """Verify the regex patterns match/don't-match expected log lines."""

    def test_epoch_re_matches_standard_line(self):
        line = "2026-02-15 02:06:03 - twinbrain_v5 - INFO - ✓ Epoch 5/100: train_loss=0.1234"
        m = _EPOCH_RE.search(line)
        assert m is not None
        assert m.group(1) == "5"

    def test_epoch_re_matches_without_checkmark(self):
        line = "INFO - Epoch 10/50: train_loss=0.23"
        m = _EPOCH_RE.search(line)
        assert m is not None
        assert m.group(1) == "10"

    def test_epoch_re_no_match_on_other_lines(self):
        for line in [
            "Starting training...",
            "Step 3/100 complete",
            "epoch_counter=5",  # 'epoch' inside a key=value, not the pattern
        ]:
            assert _EPOCH_RE.search(line) is None, f"Should not match: {line!r}"

    def test_kv_re_captures_floats(self):
        line = "train_loss=0.1234, val_loss=0.5678, pred_r2_eeg=0.100"
        pairs = dict(_KV_RE.findall(line))
        assert pairs["train_loss"] == "0.1234"
        assert pairs["val_loss"] == "0.5678"
        assert pairs["pred_r2_eeg"] == "0.100"

    def test_kv_re_does_not_capture_time_with_unit(self):
        # "time=47.2s" — the 's' suffix creates no word boundary after the float.
        line = "time=47.2s, ETA=1.2 hours"
        pairs = dict(_KV_RE.findall(line))
        assert "time" not in pairs

    def test_kv_re_captures_time_without_unit(self):
        # "time=47.2," — comma is a non-word char, so word boundary IS present.
        # In practice the log writes "time=47.2s" (with 's'), so this tests the
        # regex boundary logic rather than actual log content.
        line = "foo=1.234, time=47.2, bar=5.678"
        pairs = dict(_KV_RE.findall(line))
        assert "foo" in pairs
        assert "bar" in pairs
        # time=47.2 (no unit suffix) would match — that's acceptable behaviour
        # since "time" is not a metric we rely on.

    def test_kv_re_captures_negative(self):
        line = "ar1_r2_eeg=-0.842"
        pairs = dict(_KV_RE.findall(line))
        assert float(pairs["ar1_r2_eeg"]) == pytest.approx(-0.842)

    def test_npi_re_matches_debug_npi_line(self):
        line = "  📐 超NPI指标(h=1): decorr_h1_eeg=0.12  ar1_r2_h1_eeg=0.45"
        m = _NPI_RE.search(line)
        assert m is not None
        assert "decorr_h1_eeg" in m.group(1)

    def test_base_re_matches_multistep_line(self):
        line = "  📐 多步基线: decorr_eeg=0.34  ar1_r2_eeg=-0.12"
        m = _BASE_RE.search(line)
        assert m is not None
        assert "decorr_eeg" in m.group(1)


# ---------------------------------------------------------------------------
# TestParseLog — parse_log() with synthetic log content
# ---------------------------------------------------------------------------


class TestParseLog:
    """Tests for parse_log() covering normal usage, edge cases, and DEBUG metrics."""

    # ── Helpers ──────────────────────────────────────────────────────────────

    _INFO_ONLY_LOG = """\
        2026-02-15 02:06:03 - twinbrain_v5 - INFO - TwinBrain V5 starting
        2026-02-15 02:06:03 - twinbrain_v5 - INFO - ✓ Epoch 1/10: train_loss=0.5000, time=30.0s, ETA=5 分钟
        2026-02-15 02:06:03 - twinbrain_v5 - INFO - ✓ Epoch 2/10: train_loss=0.4800, time=29.0s, ETA=4 分钟
        2026-02-15 02:06:03 - twinbrain_v5 - INFO - ✓ Epoch 5/10: train_loss=0.4500, val_loss=0.4700, pred_r2_eeg=0.100  pred_r2_fmri=0.180  r2_eeg=0.810  r2_fmri=0.860, time=31.0s, ETA=3 分钟
        2026-02-15 02:06:03 - twinbrain_v5 - INFO - ✓ Epoch 10/10: train_loss=0.4100, val_loss=0.4300, pred_r2_eeg=0.150  pred_r2_fmri=0.240  r2_eeg=0.880  r2_fmri=0.900, time=30.0s, ETA=0 分钟
    """

    _DEBUG_LOG = """\
        2026-02-15 02:06:03 - twinbrain_v5 - INFO - ✓ Epoch 5/10: train_loss=0.4500, val_loss=0.4700, pred_r2_eeg=0.100  r2_eeg=0.810, time=31.0s, ETA=3 分钟
        2026-02-15 02:06:03 - twinbrain_v5 - DEBUG -   📐 超NPI指标(h=1): decorr_h1_eeg=-0.200  ar1_r2_h1_eeg=0.820  pred_r2_h1_eeg=0.050
        2026-02-15 02:06:03 - twinbrain_v5 - DEBUG -   📐 多步基线: decorr_eeg=0.460  ar1_r2_eeg=-0.840
    """

    def test_basic_metrics_extracted(self, tmp_path):
        log = _make_log(tmp_path, self._INFO_ONLY_LOG)
        m = parse_log(log)
        assert "train_loss" in m
        assert "val_loss" in m
        assert "pred_r2_eeg" in m
        assert "r2_eeg" in m

    def test_train_loss_count_matches_all_epochs(self, tmp_path):
        log = _make_log(tmp_path, self._INFO_ONLY_LOG)
        m = parse_log(log)
        # 4 epoch lines → 4 train_loss values
        assert len(m["train_loss"]) == 4

    def test_val_loss_count_matches_validation_epochs(self, tmp_path):
        log = _make_log(tmp_path, self._INFO_ONLY_LOG)
        m = parse_log(log)
        # Only epochs 5 and 10 have val_loss
        assert len(m["val_loss"]) == 2

    def test_val_epoch_key_tracks_validation_epoch_numbers(self, tmp_path):
        log = _make_log(tmp_path, self._INFO_ONLY_LOG)
        m = parse_log(log)
        assert m["val_epoch"] == [5, 10]

    def test_epoch_key_tracks_all_epoch_numbers(self, tmp_path):
        log = _make_log(tmp_path, self._INFO_ONLY_LOG)
        m = parse_log(log)
        assert m["epoch"] == [1, 2, 5, 10]

    def test_metric_values_are_correct(self, tmp_path):
        log = _make_log(tmp_path, self._INFO_ONLY_LOG)
        m = parse_log(log)
        assert m["val_loss"][0] == pytest.approx(0.4700)
        assert m["pred_r2_fmri"][1] == pytest.approx(0.240)
        assert m["r2_eeg"][0] == pytest.approx(0.810)

    def test_debug_skill_scores_attached_to_correct_epoch(self, tmp_path):
        log = _make_log(tmp_path, self._DEBUG_LOG)
        m = parse_log(log)
        assert "decorr_h1_eeg" in m
        assert "ar1_r2_h1_eeg" in m
        assert "decorr_eeg" in m
        assert "ar1_r2_eeg" in m
        assert m["decorr_h1_eeg"][0] == pytest.approx(-0.200)
        assert m["ar1_r2_eeg"][0] == pytest.approx(-0.840)

    def test_empty_log_returns_empty_dict(self, tmp_path):
        log = _make_log(tmp_path, "No epoch lines here\n")
        assert parse_log(log) == {}

    def test_partial_log_no_validation(self, tmp_path):
        content = """\
            INFO - ✓ Epoch 1/5: train_loss=0.50, time=10.0s, ETA=1 分钟
            INFO - ✓ Epoch 2/5: train_loss=0.48, time=10.0s, ETA=1 分钟
        """
        log = _make_log(tmp_path, content)
        m = parse_log(log)
        assert len(m["train_loss"]) == 2
        assert "val_loss" not in m
        assert m["val_epoch"] == []

    def test_negative_r2_handled(self, tmp_path):
        content = "INFO - ✓ Epoch 3/10: train_loss=0.60, val_loss=0.80, pred_r2_eeg=-0.050  r2_eeg=0.700, time=30.0s, ETA=1 分钟\n"
        log = _make_log(tmp_path, content)
        m = parse_log(log)
        assert m["pred_r2_eeg"][0] == pytest.approx(-0.050)


# ---------------------------------------------------------------------------
# TestFindLog — find_log() with temporary directory trees
# ---------------------------------------------------------------------------


class TestFindLog:
    """Tests for the log file discovery logic."""

    def test_direct_log_path(self, tmp_path):
        log = tmp_path / "training.log"
        log.write_text("INFO - ✓ Epoch 1/5: train_loss=0.5\n")
        assert find_log(str(log)) == log

    def test_run_directory_with_log(self, tmp_path):
        run_dir = tmp_path / "run_A"
        run_dir.mkdir()
        log = run_dir / "training.log"
        log.write_text("dummy\n")
        assert find_log(str(run_dir)) == log

    def test_missing_log_in_directory_raises(self, tmp_path):
        run_dir = tmp_path / "empty_run"
        run_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="No training.log"):
            find_log(str(run_dir))

    def test_nonexistent_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            find_log(str(tmp_path / "ghost"))

    def test_auto_discover_latest_run(self, tmp_path):
        # Create two run directories; the second is "newer" (modified later)
        (tmp_path / "run_old").mkdir()
        (tmp_path / "run_old" / "training.log").write_text("old\n")
        import time
        time.sleep(0.01)
        (tmp_path / "run_new").mkdir()
        (tmp_path / "run_new" / "training.log").write_text("new\n")

        # Patch _OUTPUTS_DIR via module-level attribute
        import plot_log as pl
        original = pl._OUTPUTS_DIR
        pl._OUTPUTS_DIR = tmp_path
        try:
            log = find_log(None)
            assert log.parent.name == "run_new"
        finally:
            pl._OUTPUTS_DIR = original

    def test_auto_discover_empty_outputs_raises(self, tmp_path):
        import plot_log as pl
        original = pl._OUTPUTS_DIR
        pl._OUTPUTS_DIR = tmp_path  # empty temp dir
        try:
            with pytest.raises(FileNotFoundError):
                find_log(None)
        finally:
            pl._OUTPUTS_DIR = original


# ---------------------------------------------------------------------------
# TestPlotMetrics — plot_metrics() output file creation
# ---------------------------------------------------------------------------


class TestPlotMetrics:
    """Tests for plot_metrics() — verify PNG file creation and graceful degradation."""

    _SAMPLE_METRICS = {
        "epoch":      [1, 2, 5, 10],
        "train_loss": [0.50, 0.48, 0.45, 0.41],
        "val_epoch":  [5, 10],
        "val_loss":   [0.47, 0.43],
        "pred_r2_eeg":  [0.10, 0.15],
        "pred_r2_fmri": [0.18, 0.24],
        "r2_eeg":        [0.81, 0.88],
        "r2_fmri":       [0.86, 0.90],
    }

    _SAMPLE_METRICS_WITH_SKILL = {
        **_SAMPLE_METRICS,
        "decorr_eeg":    [0.46, 0.55],
        "ar1_r2_eeg":    [-0.84, -0.80],
        "pred_r2_h1_eeg": [0.05, 0.08],
    }

    def test_creates_png_file(self, tmp_path):
        pytest.importorskip("matplotlib")
        out = tmp_path / "curves.png"
        plot_metrics(self._SAMPLE_METRICS, out, title_prefix="Test")
        assert out.exists()
        assert out.stat().st_size > 1000  # non-trivial PNG

    def test_creates_png_with_skill_scores(self, tmp_path):
        pytest.importorskip("matplotlib")
        out = tmp_path / "curves_skill.png"
        plot_metrics(self._SAMPLE_METRICS_WITH_SKILL, out)
        assert out.exists()

    def test_empty_metrics_does_not_raise(self, tmp_path):
        pytest.importorskip("matplotlib")
        out = tmp_path / "empty.png"
        # Should run without error even with minimal/empty metrics
        plot_metrics({}, out)
        # File may or may not be written; no exception is the only requirement

    def test_train_loss_only_no_crash(self, tmp_path):
        pytest.importorskip("matplotlib")
        out = tmp_path / "train_only.png"
        metrics = {
            "epoch": [1, 2, 3],
            "train_loss": [0.5, 0.48, 0.46],
            "val_epoch": [],
        }
        plot_metrics(metrics, out)
        assert out.exists()


# ---------------------------------------------------------------------------
# TestListRuns — list_runs() smoke test
# ---------------------------------------------------------------------------


class TestListRuns:
    def test_list_runs_prints_run_names(self, tmp_path, capsys):
        run = tmp_path / "my_run"
        run.mkdir()
        (run / "training.log").write_text(
            "INFO - ✓ Epoch 5/10: train_loss=0.50, val_loss=0.47, time=10.0s, ETA=1 分钟\n"
        )
        list_runs(tmp_path)
        captured = capsys.readouterr()
        assert "my_run" in captured.out

    def test_list_runs_empty_dir(self, tmp_path, capsys):
        list_runs(tmp_path)
        captured = capsys.readouterr()
        assert "No run" in captured.out
