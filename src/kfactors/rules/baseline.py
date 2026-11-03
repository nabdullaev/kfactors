"""Textbook baselines: cumulative variance share and the Kaiser (1960) rule.

Both rules are deliberately naive references; neither has a consistency guarantee for the number
of factors in an approximate factor model (they are included because practitioners use them).
"""
from __future__ import annotations

import numpy as np

from kfactors.rules.base import KResult, _spectrum, register


@register("variance_share")
def variance_share(Z, *, kmax: int, share: float = 0.9) -> KResult:
    """Smallest ``k`` whose leading components explain at least ``share`` of total variance.

    ``k = min{k >= 0 : sum_{j<=k} lambda_j / sum_j lambda_j >= share}`` (Jolliffe 2002,
    *Principal Component Analysis*, Sec. 6.1.1), then capped at ``kmax``. This is the rule behind
    ``sklearn.decomposition.PCA(n_components=share)``.

    Parameters
    ----------
    Z : ndarray, shape (n, p)
        Pre-processed (centred) panel.
    kmax : int
        Maximum number of factors (clipped with ``effective_kmax``).
    share : float in (0, 1]
        Target cumulative explained-variance share.

    Returns
    -------
    KResult
        ``criterion[k]`` = cumulative share explained by ``k`` components, ``k = 0..kmax``;
        ``threshold = share``; ``info["k_uncapped"]`` is the value before capping at ``kmax``.
    """
    if not 0.0 < share <= 1.0:
        raise ValueError(f"share must be in (0, 1], got {share}")
    ev, _n, _p, _m, kmax = _spectrum(Z, kmax)
    total = ev.sum()
    cum = np.concatenate(([0.0], np.cumsum(ev))) / total  # cum[k] = share of the first k
    # tolerance guards share = 1 against round-off in the cumulative sum
    k_unc = int(np.argmax(cum >= share - 1e-12))
    k = min(k_unc, kmax)
    return KResult(k=k, method="variance_share", eigenvalues=ev, criterion=cum[: kmax + 1],
                   threshold=float(share),
                   info={"kmax_effective": kmax, "k_uncapped": k_unc, "capped": k_unc > kmax,
                         "share": float(share)})


@register("kaiser")
def kaiser(Z, *, kmax: int) -> KResult:
    """Kaiser (1960) / Guttman rule: number of eigenvalues above their average.

    The average is ``trace(S) / p``, which equals 1 for a correlation matrix, giving the classic
    "eigenvalue greater than one" rule (Kaiser, H. F. 1960, *Educ. Psychol. Meas.* 20, 141-151).
    The count is capped at ``kmax``.

    Returns
    -------
    KResult
        ``criterion`` = ``lambda_j / mean(lambda)`` for ranks ``j = 1..kmax+1``;
        ``threshold`` = ``mean(lambda)``; ``info["k_uncapped"]`` before capping.
    """
    ev, _n, p, _m, kmax = _spectrum(Z, kmax)
    mean = ev.sum() / p
    k_unc = int(np.sum(ev > mean))
    k = min(k_unc, kmax)
    return KResult(k=k, method="kaiser", eigenvalues=ev, criterion=ev[: kmax + 1] / mean,
                   threshold=float(mean),
                   info={"kmax_effective": kmax, "k_uncapped": k_unc, "capped": k_unc > kmax})
