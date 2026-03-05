"""
tests/test_fmri_autocorr_audit.py
===================================

Pure-NumPy tests for diagnostics/fmri_autocorr_audit.py.

Tests require only numpy (no PyTorch, nilearn, etc.) and validate:
1. compute_lag_autocorr  — known synthetic timeseries
2. theoretical_bandpass_ar1 — closed-form numerical check
3. autocorr_all_lags — monotone decay for AR(1) process
4. audit_fmri_timeseries — output schema and key cross-validation identity
5. explain_low_autocorr / print_audit_report — smoke tests
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Make sure the repo root is on the path (mirrors conftest.py)
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from diagnostics.fmri_autocorr_audit import (
    audit_fmri_timeseries,
    autocorr_all_lags,
    compute_lag_autocorr,
    explain_low_autocorr,
    print_audit_report,
    theoretical_bandpass_ar1,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _ar1_process(rho: float, T: int, n_rois: int = 1) -> np.ndarray:
    """Generate an AR(1) process  y_t = rho*y_{t-1} + eps_t."""
    if abs(rho) >= 1.0:
        raise ValueError(f"rho must satisfy |rho| < 1 for a stationary process, got {rho}")
    ts = np.zeros((n_rois, T))
    noise = RNG.standard_normal((n_rois, T))
    ts[:, 0] = noise[:, 0]
    for t in range(1, T):
        ts[:, t] = rho * ts[:, t - 1] + np.sqrt(1 - rho**2) * noise[:, t]
    return ts


# ---------------------------------------------------------------------------
# compute_lag_autocorr
# ---------------------------------------------------------------------------


def test_lag_autocorr_perfect_correlation():
    """A linear ramp has lag-1 autocorrelation = 1 (y[t+1] = y[t] + c)."""
    n_rois, T = 3, 200
    ramp = np.arange(T, dtype=float)
    ts = np.tile(ramp, (n_rois, 1))  # [3, 200], same linear ramp
    rho = compute_lag_autocorr(ts, lag=1)
    assert rho.shape == (n_rois,)
    np.testing.assert_allclose(rho, 1.0, atol=1e-6)


def test_lag_autocorr_ar1_process():
    """AR(1) with ρ=0.9 should yield empirical lag-1 autocorr near 0.9."""
    ts = _ar1_process(rho=0.9, T=5000, n_rois=10)
    rho = compute_lag_autocorr(ts, lag=1)
    mean_rho = np.mean(rho)
    # Allow 0.05 tolerance for finite-sample noise
    assert abs(mean_rho - 0.9) < 0.05, f"Expected ≈0.9, got {mean_rho:.4f}"


def test_lag_autocorr_low_rho():
    """AR(1) with ρ=0.23 should yield empirical lag-1 autocorr near 0.23."""
    ts = _ar1_process(rho=0.23, T=5000, n_rois=20)
    rho = compute_lag_autocorr(ts, lag=1)
    mean_rho = float(np.mean(rho))
    assert abs(mean_rho - 0.23) < 0.05, f"Expected ≈0.23, got {mean_rho:.4f}"


def test_lag_autocorr_1d_input():
    """1-D input should be handled as a single ROI."""
    ts_1d = _ar1_process(rho=0.5, T=500, n_rois=1).squeeze()
    rho = compute_lag_autocorr(ts_1d, lag=1)
    assert rho.shape == (1,)


def test_lag_autocorr_lag_too_large():
    """lag >= T should raise ValueError."""
    ts = _ar1_process(rho=0.5, T=10, n_rois=2)
    with pytest.raises(ValueError):
        compute_lag_autocorr(ts, lag=10)


def test_lag_autocorr_zero_variance_roi():
    """All-constant ROI should return NaN (not crash)."""
    ts = np.ones((3, 100))
    ts[0] = RNG.standard_normal(100)  # non-constant first ROI
    rho = compute_lag_autocorr(ts, lag=1)
    assert np.isfinite(rho[0])
    assert np.isnan(rho[1])
    assert np.isnan(rho[2])


# ---------------------------------------------------------------------------
# theoretical_bandpass_ar1
# ---------------------------------------------------------------------------


def test_theoretical_bandpass_ar1_value():
    """Numerical check for the default TwinBrain parameters."""
    import math

    tr, hp, lp = 2.0, 0.01, 0.1
    rho = theoretical_bandpass_ar1(tr=tr, high_pass=hp, low_pass=lp)
    # Manually computed reference:
    expected = (
        math.sin(2 * math.pi * lp * tr) - math.sin(2 * math.pi * hp * tr)
    ) / (2 * math.pi * (lp - hp) * tr)
    np.testing.assert_allclose(rho, expected, rtol=1e-10)


def test_theoretical_bandpass_ar1_range():
    """Result should be in (0, 1) for reasonable fMRI parameters."""
    rho = theoretical_bandpass_ar1(tr=2.0, high_pass=0.01, low_pass=0.1)
    assert 0.0 < rho < 1.0, f"Expected (0,1), got {rho:.4f}"


def test_theoretical_bandpass_ar1_invalid():
    """low_pass <= high_pass should raise ValueError."""
    with pytest.raises(ValueError):
        theoretical_bandpass_ar1(tr=2.0, high_pass=0.1, low_pass=0.01)


def test_theoretical_bandpass_ar1_exceeds_observed():
    """For ds006040 default params the theory gives a higher ρ than observed."""
    rho_theory = theoretical_bandpass_ar1(tr=2.0, high_pass=0.01, low_pass=0.1)
    rho_observed = 0.23  # from training logs
    assert rho_theory > rho_observed, (
        f"Expected theory ({rho_theory:.4f}) > observed ({rho_observed:.4f})"
    )


# ---------------------------------------------------------------------------
# autocorr_all_lags
# ---------------------------------------------------------------------------


def test_autocorr_all_lags_shape():
    """Output length must equal max_lag."""
    ts = _ar1_process(rho=0.5, T=500, n_rois=5)
    acf = autocorr_all_lags(ts, max_lag=15)
    assert acf.shape == (15,)


def test_autocorr_all_lags_decay():
    """AR(1) process should show monotonically decaying ACF."""
    ts = _ar1_process(rho=0.9, T=10000, n_rois=20)
    acf = autocorr_all_lags(ts, max_lag=10)
    # ACF should decrease from lag 1 to lag 10 (allow small non-monotone blips)
    assert acf[0] > acf[-1], f"ACF should decay: lag1={acf[0]:.4f}, lag10={acf[-1]:.4f}"
    # AR(1) with ρ=0.9 → ACF(k) ≈ 0.9^k
    for k in range(1, 11):
        np.testing.assert_allclose(acf[k - 1], 0.9 ** k, atol=0.05)


def test_autocorr_all_lags_max_lag_capped():
    """max_lag is capped to T-1 when it exceeds the timeseries length."""
    ts = _ar1_process(rho=0.5, T=5, n_rois=2)
    acf = autocorr_all_lags(ts, max_lag=100)
    assert acf.shape == (4,)  # capped to T-1 = 4


# ---------------------------------------------------------------------------
# audit_fmri_timeseries
# ---------------------------------------------------------------------------


def test_audit_output_schema():
    """audit_fmri_timeseries should return a dict with required keys."""
    ts = _ar1_process(rho=0.7, T=300, n_rois=50)
    report = audit_fmri_timeseries(ts, tr=2.0, high_pass=0.01, low_pass=0.1, label="test")
    required = {
        "label", "n_rois", "T", "tr", "high_pass", "low_pass",
        "lag1_rho_mean", "lag1_rho_std", "lag1_rho_min", "lag1_rho_max",
        "pct_rois_above_0p5", "pct_rois_above_0p7",
        "theoretical_rho", "implied_ar1_r2_h1",
        "literature_restingstate_rho", "acf_mean", "evidence_lines",
    }
    assert required.issubset(report.keys()), (
        f"Missing keys: {required - report.keys()}"
    )


def test_audit_cross_validation_identity():
    """implied_ar1_r2_h1 must equal 2*lag1_rho_mean − 1 (z-score formula)."""
    ts = _ar1_process(rho=0.23, T=1000, n_rois=50)
    report = audit_fmri_timeseries(ts, tr=2.0)
    expected_r2 = 2 * report["lag1_rho_mean"] - 1.0
    np.testing.assert_allclose(
        report["implied_ar1_r2_h1"], expected_r2, rtol=1e-9
    )


def test_audit_low_rho_ds006040_params():
    """With ds006040-like ρ=0.23, implied_ar1_r2_h1 should be near −0.54."""
    ts = _ar1_process(rho=0.23, T=2000, n_rois=100)
    report = audit_fmri_timeseries(ts, tr=2.0, high_pass=0.01, low_pass=0.1)
    # 2*0.23 - 1 = -0.54
    assert -0.60 < report["implied_ar1_r2_h1"] < -0.40, (
        f"Expected near -0.54, got {report['implied_ar1_r2_h1']:.4f}"
    )


def test_audit_high_rho_restingstate():
    """With ρ=0.90 (resting state), implied_ar1_r2_h1 should be near 0.80."""
    ts = _ar1_process(rho=0.90, T=5000, n_rois=50)
    report = audit_fmri_timeseries(ts, tr=2.0)
    assert 0.70 < report["implied_ar1_r2_h1"] < 0.95, (
        f"Expected near 0.80, got {report['implied_ar1_r2_h1']:.4f}"
    )


def test_audit_evidence_lines_nonempty():
    """evidence_lines must contain at least 5 non-empty strings."""
    ts = _ar1_process(rho=0.23, T=500, n_rois=20)
    report = audit_fmri_timeseries(ts, tr=2.0)
    lines = report["evidence_lines"]
    assert len(lines) >= 5
    for line in lines:
        assert len(line) > 20


def test_audit_acf_length():
    """acf_mean should have length == max_lag (default 10)."""
    ts = _ar1_process(rho=0.5, T=300, n_rois=10)
    report = audit_fmri_timeseries(ts, tr=2.0, max_lag=10)
    assert len(report["acf_mean"]) == 10


# ---------------------------------------------------------------------------
# explain_low_autocorr
# ---------------------------------------------------------------------------


def test_explain_low_autocorr_returns_list():
    """explain_low_autocorr should return the same evidence_lines list."""
    ts = _ar1_process(rho=0.23, T=500, n_rois=20)
    report = audit_fmri_timeseries(ts, tr=2.0)
    lines = explain_low_autocorr(report)
    assert isinstance(lines, list)
    assert lines == report["evidence_lines"]


# ---------------------------------------------------------------------------
# print_audit_report — smoke test
# ---------------------------------------------------------------------------


def test_print_audit_report_smoke(capsys):
    """print_audit_report should produce non-empty stdout without exceptions."""
    ts = _ar1_process(rho=0.23, T=300, n_rois=50)
    report = audit_fmri_timeseries(ts, tr=2.0, label="sub-029")
    print_audit_report(report)
    captured = capsys.readouterr()
    assert len(captured.out) > 200
    assert "sub-029" in captured.out
    assert "证据" in captured.out
