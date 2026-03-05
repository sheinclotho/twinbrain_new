"""
diagnostics — TwinBrain Diagnostic Utilities
============================================

A collection of standalone, NumPy-only diagnostic functions for auditing
data properties before and after preprocessing.  No PyTorch or graph
dependencies are required so the tools can be run independently of the
training pipeline.

Current modules
---------------
fmri_autocorr_audit
    Functions to measure, interpret and explain the lag-1 (and higher-lag)
    autocorrelation of fMRI BOLD timeseries.  The module provides both an
    *empirical* path (compute autocorrelation from actual data) and a
    *theoretical* path (derive the expected autocorrelation for white noise
    passed through a given bandpass filter).  Together these form the
    scientific evidence that low observed ρ in a dataset such as ds006040
    is a genuine data property, NOT an artifact of preprocessing.
"""

from .fmri_autocorr_audit import (
    compute_lag_autocorr,
    theoretical_bandpass_ar1,
    autocorr_all_lags,
    audit_fmri_timeseries,
    print_audit_report,
    explain_low_autocorr,
)

__all__ = [
    "compute_lag_autocorr",
    "theoretical_bandpass_ar1",
    "autocorr_all_lags",
    "audit_fmri_timeseries",
    "print_audit_report",
    "explain_low_autocorr",
]
