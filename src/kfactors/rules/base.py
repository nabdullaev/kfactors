"""Rule registry, :class:`KResult` and the public :func:`select_k` entry point.

Every rule is a function ``rule(Z, *, kmax, **kwargs) -> KResult`` registered in :data:`RULES`
with :func:`register`. ``Z`` is the already pre-processed ``n x p`` matrix (column-centred and,
if requested, column-standardised with ``ddof=0``), so the sample covariance used by every rule is

    S = Z'Z / n,

with eigenvalues ``lambda_1 >= ... >= lambda_p``. Because ``Z`` is column-centred its rank is at
most ``m = min(n - 1, p)``; only the first ``m`` eigenvalues are *non-trivial* and the remaining
ones are (numerically) zero. Rules must only use the non-trivial part where it matters (medians,
ratios, regressions).
"""
from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from kfactors.utils import eigvals_desc, preprocess


class RuleWarning(UserWarning):
    """Issued when a rule cannot be applied as designed (panel too small, no convergence)."""


@dataclass(frozen=True)
class KResult:
    """Outcome of a number-of-factors rule.

    Attributes
    ----------
    k : int
        Selected number of factors (``0 <= k <= kmax_effective``).
    method : str
        Registry name of the rule, e.g. ``"onatski"``.
    eigenvalues : ndarray, shape (p,)
        Eigenvalues of ``S = Z'Z / n`` in descending order (trailing values are 0 when
        ``p >= n``).
    criterion : ndarray or None
        Rule-specific path, over ``k = 0..kmax`` (information criteria, ratios) or per
        eigenvalue rank (threshold rules); see the rule's docstring.
    threshold : float, ndarray or None
        Rule-specific threshold. In eigenvalue units of ``S`` for ``kaiser``, ``mp``,
        ``mp_market``, ``gavish_donoho`` and ``parallel`` (an array over ranks; with
        ``devolatilize=True`` in units of the de-volatilised panel); the target share for
        ``variance_share``; Onatski's ``delta`` (an eigenvalue-gap size) for ``onatski``;
        ``None`` for ``bai_ng``, ``ahn_horenstein`` and degenerate panels.
    info : dict
        Diagnostics. Always contains ``kmax_effective``; rules add their own keys.
    """

    k: int
    method: str
    eigenvalues: np.ndarray
    criterion: np.ndarray | None = None
    threshold: float | np.ndarray | None = None
    info: dict = field(default_factory=dict)


RULES: dict[str, Callable[..., KResult]] = {}


def register(name: str):
    """Decorator registering a rule under ``name`` in :data:`RULES`."""
    def deco(fn):
        RULES[name] = fn
        return fn
    return deco


def effective_kmax(n: int, p: int, kmax: int) -> int:
    """Clip ``kmax`` to what an ``n x p`` column-centred panel can support.

    Returns ``max(1, min(kmax, min(n - 1, p) - 2))``. Column-centring removes one degree of
    freedom, so only ``m = min(n - 1, p)`` eigenvalues are non-trivial; the ``- 2`` keeps
    ``lambda_{kmax+1} > 0`` and ``sum_{j > kmax+1} lambda_j > 0`` so that ratio and growth-ratio
    criteria (Ahn & Horenstein 2013) and the Bai & Ng (2002) residual variance ``V(kmax)`` stay
    finite. The result is never below 1 (for tiny panels rules must cope with ``kmax = 1``).

    Parameters
    ----------
    n, p : int
        Number of observations (rows) and series (columns).
    kmax : int
        Requested maximum number of factors, ``>= 1``.

    Raises
    ------
    TypeError
        If ``kmax`` is not an integer (bools are rejected).
    ValueError
        If ``kmax < 1``.
    """
    if isinstance(kmax, bool) or not isinstance(kmax, (int, np.integer)):
        raise TypeError(f"kmax must be a positive integer, got {kmax!r}")
    if kmax < 1:
        raise ValueError(f"kmax must be >= 1, got {kmax}")
    return int(max(1, min(int(kmax), min(int(n) - 1, int(p)) - 2)))


def _spectrum(Z: np.ndarray, kmax: int) -> tuple[np.ndarray, int, int, int, int]:
    """Shared rule prologue: ``(eigenvalues of Z'Z/n, n, p, m = min(n-1, p), kmax_effective)``."""
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2:
        raise ValueError(f"expected a 2-D array (n x p), got shape {Z.shape}")
    n, p = Z.shape
    if n < 3 or p < 2:
        raise ValueError(f"need at least 3 rows and 2 columns, got shape {Z.shape}")
    if not np.all(np.isfinite(Z)):
        raise ValueError("Z contains NaN or inf")
    ev = eigvals_desc(Z)
    if not ev[0] > 0:
        raise ValueError("the panel has zero total variance; no factors can be estimated")
    m = min(n - 1, p)
    return ev, n, p, m, effective_kmax(n, p, kmax)


def _accepts(fn: Callable, name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return False
    return name in params


def select_k(X, method: str = "onatski", *, kmax: int = 20, standardize: bool = True,
             random_state=None, **kwargs) -> KResult:
    """Choose the number of factors of ``X`` with rule ``method``.

    Parameters
    ----------
    X : array-like or DataFrame, shape (n, p)
        Complete panel (no NaN), rows = time, columns = series; ``n >= 3``, ``p >= 2``.
    method : str
        A key of :data:`RULES`: ``variance_share``, ``kaiser``, ``mp``, ``mp_market``,
        ``gavish_donoho``, ``bai_ng``, ``ahn_horenstein``, ``onatski``, ``parallel``.
    kmax : int
        Maximum number of factors considered; clipped with :func:`effective_kmax` to
        ``max(1, min(kmax, min(n - 1, p) - 2))``. Every rule returns ``0 <= k <= kmax_effective``.
    standardize : bool
        Standardise columns (correlation matrix) after centring (``ddof=0``).
    random_state : None, int or numpy Generator
        Forwarded only to rules that accept it (stochastic rules such as ``parallel``).
    **kwargs
        Rule-specific options (e.g. ``criterion="IC_p2"`` for ``bai_ng``).

    Returns
    -------
    KResult
        ``info`` additionally contains ``n``, ``p``, ``kmax_requested`` and ``standardize``.

    Raises
    ------
    ValueError
        Unknown ``method`` (the message lists the available rules), ``kmax < 1``, NaN/inf in
        ``X``, too small panel or zero-variance columns when ``standardize=True``.
    TypeError
        Non-integer ``kmax`` or a keyword argument the rule does not accept.

    Warnings
    --------
    Warnings raised inside the rule (e.g. :class:`RuleWarning`) are re-issued so that they point
    at the code that called ``select_k`` rather than at this module.
    """
    if method not in RULES:
        raise ValueError(f"unknown method {method!r}; available rules: {sorted(RULES)}")
    Z, _, _ = preprocess(X, standardize=standardize)
    n, p = Z.shape
    k_eff = effective_kmax(n, p, kmax)
    fn = RULES[method]
    if _accepts(fn, "random_state"):
        kwargs["random_state"] = random_state
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = fn(Z, kmax=k_eff, **kwargs)
    for w in caught:
        warnings.warn(w.message, w.category, stacklevel=2)
    res.info.update({"n": n, "p": p, "kmax_requested": int(kmax), "standardize": bool(standardize)})
    return res
