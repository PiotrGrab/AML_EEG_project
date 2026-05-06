# =============================================================
# preprocessing.py
# Shared data loading and normalisation utilities.
# Every other script imports from here so preprocessing
# stays consistent across EDA, grid search, and final training.
# =============================================================

import numpy as np
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery
from sklearn.preprocessing import LabelEncoder

# ─── Band-pass and resampling parameters ─────────────────────
FMIN     = 8      # Hz  (low mu / low beta)
FMAX     = 30     # Hz  (high beta)
RESAMPLE = 128    # Hz  — target sampling rate after resampling


def load_subject(subject_id, fmin=FMIN, fmax=FMAX, resample=RESAMPLE):
    """
    Download (or load from cache) BNCI2014_001 for one subject.

    The LeftRightImagery paradigm applies:
      • band-pass filter  [fmin, fmax] Hz
      • resampling to `resample` Hz
      • epoch extraction around motor-imagery cues

    Returns
    -------
    X    : ndarray, shape (n_trials, n_channels, n_times)  — raw float64
    y    : ndarray, shape (n_trials,)                      — string labels
    meta : DataFrame with columns: subject, session, run
    """
    dataset  = BNCI2014_001()
    paradigm = LeftRightImagery(fmin=fmin, fmax=fmax, resample=resample)
    X, y, meta = paradigm.get_data(dataset=dataset, subjects=[subject_id])
    return X, y, meta


def session_split(X, y, meta):
    """
    Split trials by session following the standard BNCI2014_001 protocol:
        session 0 (session_T in the dataset) → training set
        session 1 (session_E in the dataset) → held-out test set

    This mirrors the evaluation protocol used in the EEG-Conformer paper
    (Song et al., 2023) and avoids any data leakage between splits.

    Returns
    -------
    X_train, X_test : ndarray  (float64, unnormalised)
    y_train, y_test : ndarray  (int, label-encoded)
    le              : fitted LabelEncoder  (inverse_transform for class names)
    """
    le    = LabelEncoder()
    y_enc = le.fit_transform(y)

    sessions       = meta["session"].values
    unique_sessions = sorted(meta["session"].unique())

    if len(unique_sessions) < 2:
        raise ValueError(
            f"Subject has only {len(unique_sessions)} session(s); "
            "need at least 2 for a session-based train/test split."
        )

    train_mask = sessions == unique_sessions[0]
    test_mask  = sessions == unique_sessions[1]

    return (
        X[train_mask].copy(),
        X[test_mask].copy(),
        y_enc[train_mask],
        y_enc[test_mask],
        le,
    )


def normalize(X):
    """
    Per-trial, per-channel z-score normalisation followed by hard clipping
    to [-5, 5] standard deviations.

    Computed independently for every trial so there is no information
    leakage from the test set into the training set.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_channels, n_times)

    Returns
    -------
    X_norm : ndarray, float32, same shape
    """
    mean = X.mean(axis=2, keepdims=True)
    std  = X.std(axis=2, keepdims=True) + 1e-8
    return np.clip((X - mean) / std, -5, 5).astype(np.float32)
