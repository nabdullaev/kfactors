"""Small shared helpers."""
from __future__ import annotations

import numpy as np


def check_random_state(random_state=None) -> np.random.Generator:
    """Return a numpy Generator from None | int | Generator."""
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def as_2d_array(X) -> np.ndarray:
    """Convert DataFrame/array-like to a float 2-D ndarray (n x p); raise on NaN/inf or ndim != 2."""
    A = np.asarray(getattr(X, "values", X), dtype=float)
    if A.ndim != 2:
        raise ValueError(f"expected a 2-D array (n x p), got shape {A.shape}")
    if A.shape[0] < 3 or A.shape[1] < 2:
        raise ValueError(f"need at least 3 rows and 2 columns, got shape {A.shape}")
    if not np.all(np.isfinite(A)):
        raise ValueError("X contains NaN or inf; pass a complete window")
    return A


def preprocess(X, standardize: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Column-centre (and standardise with ddof=0 if requested) X.

    Returns (Z, mean, scale); scale is all ones when standardize=False.
    Zero-variance columns raise ValueError.
    """
    A = as_2d_array(X)
    mean = A.mean(axis=0)
    Z = A - mean
    if standardize:
        scale = Z.std(axis=0)
        bad = scale <= 1e-12 * max(1.0, float(np.abs(A).max()))
        if np.any(bad):
            raise ValueError(f"zero-variance columns at positions {np.flatnonzero(bad)[:10].tolist()}")
        Z = Z / scale
    else:
        scale = np.ones(A.shape[1])
    return Z, mean, scale


def eigvals_desc(Z: np.ndarray) -> np.ndarray:
    """Eigenvalues of Z'Z/n sorted descending, length p.

    Uses the smaller Gram matrix when p > n; trailing values are exactly 0 in that case.
    """
    n, p = Z.shape
    if p <= n:
        ev = np.linalg.eigvalsh(Z.T @ Z / n)
    else:
        ev = np.concatenate([np.linalg.eigvalsh(Z @ Z.T / n), np.zeros(p - n)])
    ev = np.clip(ev, 0.0, None)
    return np.sort(ev)[::-1]
