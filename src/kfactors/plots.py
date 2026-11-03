"""Standard figures (matplotlib; work with the non-interactive Agg backend).

Every function draws into ``ax`` (or a new figure when ``ax is None``) and returns the
``matplotlib.axes.Axes``; the figure is ``ax.figure``. Nothing calls ``plt.show``.

* :func:`plot_spectrum` -- scree plot with Marchenko-Pastur noise edges;
* :func:`plot_heatmap` -- rule x design table of a benchmark score;
* :func:`plot_k_curve` -- economic metric vs k with bootstrap band, oracle set and rule picks.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

__all__ = ["mp_median", "plot_heatmap", "plot_k_curve", "plot_spectrum", "spectrum_edges"]


# --------------------------------------------------------------------------------------------
# Marchenko-Pastur helpers
# --------------------------------------------------------------------------------------------

@lru_cache(maxsize=256)
def _mp_median_cached(y: float) -> float:
    from scipy import integrate, optimize

    a, b = (1.0 - np.sqrt(y)) ** 2, (1.0 + np.sqrt(y)) ** 2

    def density(x):
        return np.sqrt(max((b - x) * (x - a), 0.0)) / (2.0 * np.pi * y * x)

    def cdf_minus_half(x):
        return integrate.quad(density, a, x, limit=200)[0] - 0.5

    return float(optimize.brentq(cdf_minus_half, a + 1e-12, b - 1e-12, xtol=1e-12))


def mp_median(y: float) -> float:
    """Median of the Marchenko-Pastur law with aspect ratio ``y in (0, 1]`` and unit variance.

    The MP density (Marchenko & Pastur, 1967, *Math. USSR-Sbornik* 1(4)) is
    ``f(x) = sqrt((b - x)(x - a)) / (2 pi y x)`` on ``[a, b] = [(1 - sqrt y)^2, (1 + sqrt y)^2]``;
    the median solves ``F(x) = 1/2`` (numerical quadrature + Brent). This is the ``mu_beta`` of
    Gavish & Donoho (2014, *IEEE Trans. Inf. Theory* 60(8), eq. (4)); e.g. ``mp_median(1)``
    is about 0.6528.
    """
    y = float(y)
    if not (0.0 < y <= 1.0):
        raise ValueError(f"MP aspect ratio must lie in (0, 1], got {y}")
    return _mp_median_cached(round(y, 12))


def spectrum_edges(eigenvalues, n: int, p: int) -> dict:
    """Noise-level estimates and Marchenko-Pastur upper edges used by :func:`plot_spectrum`.

    Parameters
    ----------
    eigenvalues : array-like
        The *full* spectrum of ``S = Xc' Xc / n`` (any order; at least ``min(n - 1, p)``
        values, trailing zeros allowed), as returned by :func:`kfactors.utils.eigvals_desc`.
    n, p : int
        Sample size and dimension that produced the spectrum; ``gamma = p / n``.

    Returns
    -------
    dict
        ``gamma``; ``sigma2_median`` and ``edge_mp`` (naive edge
        ``sigma2 (1 + sqrt(gamma))^2`` with ``sigma2`` fitted by matching the median of the
        non-trivial eigenvalues to the MP median); ``sigma2_laloux`` and ``edge_laloux``
        (Laloux, Cizeau, Bouchaud & Potters, 1999, *PRL* 83(7): ``sigma2 = 1 - lambda_1 / p``
        for a correlation matrix, generalised to ``(trace - lambda_1) / p``);
        ``n_above_mp`` and ``n_above_laloux`` (number of eigenvalues strictly above each edge).

    Notes
    -----
    For ``p <= n`` the non-trivial eigenvalues follow ``sigma2 * MP(p/n)``; for ``p > n`` the
    ``n`` non-zero eigenvalues are ``(p/n) sigma2 * MP(n/p)`` (they coincide with those of
    ``Xc Xc' / n``). After column-centring the rank is ``min(n - 1, p)``; only those values
    (and only strictly positive ones) enter the median. Spikes inflate the median slightly,
    which is why the MP edge is called "naive".
    """
    ev = np.sort(np.asarray(eigenvalues, dtype=float).ravel())[::-1]
    n, p = int(n), int(p)
    if n < 2 or p < 2:
        raise ValueError(f"need n >= 2 and p >= 2, got n={n}, p={p}")
    if ev.size == 0 or not np.all(np.isfinite(ev)):
        raise ValueError("eigenvalues must be a non-empty finite array")
    if np.any(ev < -1e-8 * max(1.0, abs(ev[0]))):
        raise ValueError("eigenvalues of a covariance matrix must be >= 0")
    m = min(n - 1, p)
    if ev.size < m:
        raise ValueError(f"need the full spectrum: got {ev.size} eigenvalues, expected at least "
                         f"min(n - 1, p) = {m} (the noise level is estimated from the bulk)")
    gamma = p / n
    bulk = ev[:m]
    bulk = bulk[bulk > 1e-12 * max(ev[0], 1e-300)]
    if bulk.size == 0:
        raise ValueError("all eigenvalues are zero")
    if p <= n:
        y, c = p / n, 1.0
    else:
        y, c = n / p, p / n
    sigma2_med = float(np.median(bulk) / (c * mp_median(y)))
    edge_factor = (1.0 + np.sqrt(gamma)) ** 2
    edge_mp = sigma2_med * edge_factor
    sigma2_lal = float((ev.sum() - ev[0]) / p)
    edge_lal = sigma2_lal * edge_factor
    return {
        "gamma": gamma,
        "sigma2_median": sigma2_med,
        "edge_mp": edge_mp,
        "n_above_mp": int(np.sum(ev > edge_mp)),
        "sigma2_laloux": sigma2_lal,
        "edge_laloux": edge_lal,
        "n_above_laloux": int(np.sum(ev > edge_lal)),
    }


def _new_ax(figsize):
    import matplotlib.pyplot as plt

    _, ax = plt.subplots(figsize=figsize)
    return ax


# --------------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------------

def plot_spectrum(eigenvalues, n: int, p: int, ax=None, *, log: bool = True, title: str | None = None,
                  market_adjusted: bool = True, top: int | None = 50):
    """Scree of eigenvalues with naive MP edge and (optionally) Laloux market-adjusted edge.

    Parameters
    ----------
    eigenvalues : array-like
        Full spectrum of ``S = Xc' Xc / n`` (see :func:`spectrum_edges`).
    n, p : int
        Sample size and dimension.
    ax : matplotlib Axes, optional
        Target axes; a new figure is created when ``None``.
    log : bool
        Logarithmic y axis (numerically zero eigenvalues, ``<= 1e-10 lambda_1``, are
        not drawn).
    title : str, optional
        Defaults to ``"Spectrum (n=..., p=..., p/n=...)"``.
    market_adjusted : bool
        Also draw the Laloux et al. (1999) edge, whose noise level excludes ``lambda_1``.
    top : int or None
        Number of leading eigenvalues drawn (``None`` = all). Edges always use the full
        spectrum.

    Returns
    -------
    matplotlib.axes.Axes
        The legend reports each edge value and the count of eigenvalues above it.
    """
    edges = spectrum_edges(eigenvalues, n, p)
    ev = np.sort(np.asarray(eigenvalues, dtype=float).ravel())[::-1]
    if top is not None:
        if int(top) < 1:
            raise ValueError(f"top must be >= 1, got {top}")
        ev = ev[: int(top)]
    ranks = np.arange(1, ev.size + 1)
    if ax is None:
        ax = _new_ax((7, 4.5))
    # on a log axis, numerically-zero eigenvalues (rank deficiency, p > n - 1) are not drawn
    tol = 1e-10 * max(float(ev[0]), 1e-300)
    show = ev > tol if log else np.ones(ev.size, dtype=bool)
    ax.plot(ranks[show], ev[show], "o-", ms=3.5, lw=1, color="C0", label="eigenvalues")
    ax.axhline(edges["edge_mp"], color="C3", ls="--", lw=1.4,
               label=(f"MP edge (bulk median, $\\sigma^2$={edges['sigma2_median']:.3g}) = "
                      f"{edges['edge_mp']:.3g}: {edges['n_above_mp']} above"))
    if market_adjusted:
        ax.axhline(edges["edge_laloux"], color="C2", ls=":", lw=1.6,
                   label=(f"Laloux market-adjusted edge ($\\sigma^2$={edges['sigma2_laloux']:.3g})"
                          f" = {edges['edge_laloux']:.3g}: {edges['n_above_laloux']} above"))
    if log:
        ax.set_yscale("log")
    ax.set_xlabel("eigenvalue rank")
    ax.set_ylabel("eigenvalue of S")
    if title is None:
        title = f"Spectrum (n={n}, p={p}, p/n={edges['gamma']:.2f})"
    ax.set_title(title)
    ax.legend(fontsize="small", loc="upper right")
    ax.grid(True, which="major", alpha=0.3)
    return ax


def plot_heatmap(df, ax=None, *, value: str = "accuracy", title: str | None = None,
                 annotate: bool | None = None, fmt: str = ".2f", cmap: str = "viridis"):
    """Heatmap rule x design from a long DataFrame with columns [rule, design, value].

    Parameters
    ----------
    df : pandas.DataFrame
        Long format with columns ``"rule"``, ``"design"`` and ``value``. Duplicate
        (rule, design) pairs are averaged. Row/column order follows first appearance.
        Missing combinations are drawn in light grey.
    ax : matplotlib Axes, optional
    value : str
        Name of the value column (also the colorbar label).
    title : str, optional
    annotate : bool, optional
        Write the value in every cell; default: only when the table has <= 600 cells.
    fmt : str
        Format spec of the annotations.
    cmap : str
        Matplotlib colormap name.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib as mpl
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}")
    missing = [c for c in ("rule", "design", value) if c not in df.columns]
    if missing:
        raise ValueError(f"df is missing columns {missing}; has {list(df.columns)}")
    if df.empty:
        raise ValueError("df is empty")
    vals = pd.to_numeric(df[value], errors="coerce")
    rules = list(pd.unique(df["rule"]))
    designs = list(pd.unique(df["design"]))
    table = (df.assign(**{value: vals})
               .pivot_table(index="rule", columns="design", values=value, aggfunc="mean",
                            dropna=False)
               .reindex(index=rules, columns=designs))
    M = table.to_numpy(dtype=float)
    nr, nc = M.shape
    if ax is None:
        ax = _new_ax((min(4 + 0.45 * nc, 40), min(2 + 0.4 * nr, 30)))
    cm = mpl.colormaps[cmap].with_extremes(bad="lightgrey")
    finite = M[np.isfinite(M)]
    vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    if vmin == vmax:
        vmin, vmax = vmin - 0.5, vmax + 0.5
    im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=cm, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks(np.arange(nc), labels=[str(d) for d in designs], rotation=90,
                  fontsize="small")
    ax.set_yticks(np.arange(nr), labels=[str(r) for r in rules], fontsize="small")
    ax.set_xlabel("design")
    ax.set_ylabel("rule")
    if annotate is None:
        annotate = nr * nc <= 600
    if annotate:
        size = "small" if nr * nc <= 150 else "x-small"
        for i in range(nr):
            for j in range(nc):
                v = M[i, j]
                if not np.isfinite(v):
                    continue
                r, g, b, _ = cm(im.norm(v))
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=size,
                        color="black" if lum > 0.5 else "white")
    ax.figure.colorbar(im, ax=ax, label=value)
    ax.set_title(title if title is not None else f"{value} by rule and design")
    return ax


def _contiguous_runs(ks) -> list[tuple[int, int]]:
    ks = sorted({int(k) for k in ks})
    runs: list[tuple[int, int]] = []
    for k in ks:
        if runs and k == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], k)
        else:
            runs.append((k, k))
    return runs


def plot_k_curve(curve, ax=None, *, title: str | None = None):
    """Metric vs k with CI band, oracle confidence set shaded, rule picks marked (KCurve).

    Parameters
    ----------
    curve : kfactors.evaluate.KCurve
        Uses ``curve.curve`` (index k; column ``metric`` and optionally ``low``/``high``),
        ``curve.rules`` (index rule; column ``k_median`` and optionally ``in_set``),
        ``curve.oracle_k``, ``curve.oracle_set`` and ``curve.metric`` (axis label). Any object
        with these attributes is accepted.
    ax : matplotlib Axes, optional
    title : str, optional
        Defaults to ``"<metric> vs number of factors k"``.

    Returns
    -------
    matplotlib.axes.Axes

    Notes
    -----
    The band is the bootstrap interval ``[low, high]`` stored in the curve; the oracle set
    (smallest set of k covering >= 1 - alpha of bootstrap argmax draws) is shaded; each rule's
    median selected k is a vertical line, solid if it lies in the oracle set and dashed
    otherwise, labelled with the rule name (rules with the same median k share one line and
    one label). Rules with a missing ``k_median`` are skipped.
    """
    import matplotlib.transforms as mtransforms
    import pandas as pd

    for attr in ("curve", "rules", "oracle_k", "oracle_set", "metric"):
        if not hasattr(curve, attr):
            raise TypeError(f"plot_k_curve expects a KCurve-like object; missing {attr!r}")
    c = curve.curve
    if not isinstance(c, pd.DataFrame) or "metric" not in c.columns:
        raise ValueError("curve.curve must be a DataFrame indexed by k with a 'metric' column")
    if c.empty:
        raise ValueError("curve.curve is empty")
    c = c.sort_index()
    k = np.asarray(c.index, dtype=float)
    y = c["metric"].to_numpy(dtype=float)
    metric_name = str(curve.metric)

    if ax is None:
        ax = _new_ax((8, 4.5))
    if {"low", "high"} <= set(c.columns):
        lo, hi = c["low"].to_numpy(dtype=float), c["high"].to_numpy(dtype=float)
        if np.any(np.isfinite(lo) & np.isfinite(hi)):
            ax.fill_between(k, lo, hi, color="C0", alpha=0.2, lw=0, label="bootstrap CI")
    ax.plot(k, y, "o-", color="C0", ms=4, lw=1.5, label=metric_name)

    oracle_set = list(curve.oracle_set or [])
    for i, (a, b) in enumerate(_contiguous_runs(oracle_set)):
        ax.axvspan(a - 0.5, b + 0.5, color="gold", alpha=0.25, lw=0,
                   label="oracle set" if i == 0 else None)
    ok = curve.oracle_k
    if ok is not None and ok in c.index:
        ax.plot([ok], [c.loc[ok, "metric"]], marker="*", ms=15, color="goldenrod",
                mec="black", ls="none", zorder=5, label=f"oracle k = {ok}")

    rules = curve.rules
    if isinstance(rules, pd.DataFrame) and not rules.empty and "k_median" in rules.columns:
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        set_ks = {int(v) for v in oracle_set}
        groups: dict[float, list[tuple[str, bool]]] = {}
        for rule, row in rules.iterrows():
            km = float(pd.to_numeric(row["k_median"], errors="coerce"))
            if not np.isfinite(km):
                continue
            if "in_set" in rules.columns and pd.notna(row["in_set"]):
                inside = bool(row["in_set"])
            else:
                inside = km.is_integer() and int(km) in set_ks
            groups.setdefault(km, []).append((str(rule), inside))
        # rules with the same median k share one marker and one label
        for i, km in enumerate(sorted(groups)):
            names = ", ".join(r for r, _ in groups[km])
            inside = any(f for _, f in groups[km])
            color = f"C{(i % 9) + 1}"
            ax.axvline(km, color=color, lw=1.3, ls="-" if inside else "--", alpha=0.9)
            ax.text(km, 0.97 - 0.07 * (i % 6), f" {names} (k={km:g})", transform=trans,
                    color=color, fontsize="small", ha="left", va="top",
                    bbox={"facecolor": "white", "alpha": 0.7, "lw": 0, "pad": 0.5})

    ax.set_xlabel("number of factors k")
    ax.set_ylabel(metric_name)
    ax.set_title(title if title is not None else f"{metric_name} vs number of factors k")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize="small", loc="lower right")
    return ax
