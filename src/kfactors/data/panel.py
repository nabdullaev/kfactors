"""Panel: returns + point-in-time tradability mask."""
from __future__ import annotations

import operator
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_DEFAULT_PPY = {"D": 252, "M": 12}


@dataclass(eq=False)
class Panel:
    """A return panel with a point-in-time investable-universe mask.

    Parameters
    ----------
    returns : pandas.DataFrame
        ``n x p`` simple returns in decimals (0.01 = 1%), indexed by a sorted, unique
        :class:`~pandas.DatetimeIndex`; columns are unique asset ids; ``NaN`` = no data.
    tradable : pandas.DataFrame
        Boolean frame with exactly the same index and columns. ``tradable.iloc[t, j]`` is True iff
        asset ``j`` is in the investable universe at date ``t`` using only information available
        at ``t`` (no look-ahead). Must be False wherever ``returns`` is NaN.
    name : str
    freq : str
        ``"D"`` (daily) or ``"M"`` (monthly); other values require ``meta['periods_per_year']``.
    meta : dict
        Provenance (source url, sha256, download date, units, licence, ...).

    Notes
    -----
    Validation happens in ``__post_init__`` and raises (never silently repairs) on: non-DataFrame
    inputs, non-datetime / unsorted / duplicated index, duplicated columns, label or shape mismatch
    between ``returns`` and ``tradable``, non-boolean ``tradable``, non-numeric or infinite
    returns, and ``tradable`` True where ``returns`` is NaN. Integer/bool returns are cast to
    float64.
    """

    returns: pd.DataFrame
    tradable: pd.DataFrame
    name: str = "panel"
    freq: str = "D"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate: sorted unique DatetimeIndex, identical shape/labels, bool tradable;
        tradable is False wherever returns is NaN."""
        R, T = self.returns, self.tradable
        if not isinstance(R, pd.DataFrame) or not isinstance(T, pd.DataFrame):
            raise TypeError("returns and tradable must be pandas DataFrames")
        if not isinstance(R.index, pd.DatetimeIndex):
            raise TypeError(f"returns.index must be a DatetimeIndex, got {type(R.index).__name__}")
        if R.index.has_duplicates:
            dup = R.index[R.index.duplicated()][:3].tolist()
            raise ValueError(f"returns.index has duplicated dates, e.g. {dup}")
        if not R.index.is_monotonic_increasing:
            raise ValueError("returns.index must be sorted in increasing order")
        if R.columns.has_duplicates:
            dup = R.columns[R.columns.duplicated()][:5].tolist()
            raise ValueError(f"returns has duplicated columns, e.g. {dup}")
        if T.shape != R.shape:
            raise ValueError(f"tradable shape {T.shape} != returns shape {R.shape}")
        if not T.index.equals(R.index) or not T.columns.equals(R.columns):
            raise ValueError("tradable must have exactly the same index and columns as returns")
        if R.shape[1] and not all(pd.api.types.is_bool_dtype(d) for d in T.dtypes):
            raise TypeError("tradable must be boolean (dtype bool) in every column")
        if R.shape[1]:
            non_num = [c for c, d in R.dtypes.items()
                       if not (pd.api.types.is_numeric_dtype(d) or pd.api.types.is_bool_dtype(d))]
            if non_num:
                raise TypeError(f"returns must be numeric; non-numeric columns: {non_num[:5]}")
            if any(d != np.float64 for d in R.dtypes):
                R = R.astype(np.float64)
                self.returns = R
            vals = R.to_numpy()
            if np.isinf(vals).any():
                raise ValueError("returns contain +/-inf")
            tv = T.to_numpy(dtype=bool)
            bad = tv & np.isnan(vals)
            if bad.any():
                i, j = np.argwhere(bad)[0]
                raise ValueError(
                    f"tradable is True where returns is NaN ({int(bad.sum())} cells), e.g. "
                    f"date={R.index[i].date()} asset={R.columns[j]!r}")
        if not isinstance(self.freq, str) or not self.freq:
            raise ValueError("freq must be a non-empty string ('D' or 'M')")
        if self.meta is None:
            self.meta = {}
        if not isinstance(self.meta, dict):
            raise TypeError("meta must be a dict")

    def __repr__(self) -> str:
        n, p = self.shape
        span = (f"{self.returns.index[0].date()}..{self.returns.index[-1].date()}" if n else "empty")
        return f"Panel(name={self.name!r}, freq={self.freq!r}, n={n}, p={p}, {span})"

    @property
    def shape(self) -> tuple[int, int]:
        return self.returns.shape

    def periods_per_year(self) -> int:
        """meta['periods_per_year'] if set, else 252 for 'D', 12 for 'M'."""
        ppy = self.meta.get("periods_per_year")
        if ppy is not None:
            ppy = int(ppy)
            if ppy <= 0:
                raise ValueError(f"meta['periods_per_year'] must be positive, got {ppy}")
            return ppy
        try:
            return _DEFAULT_PPY[self.freq]
        except KeyError:
            raise ValueError(f"unknown freq {self.freq!r}; set meta['periods_per_year']") from None

    def _replace(self, returns: pd.DataFrame, tradable: pd.DataFrame) -> Panel:
        return Panel(returns, tradable, name=self.name, freq=self.freq, meta=dict(self.meta))

    def window(self, end: int, length: int, *, complete: bool = True) -> Panel:
        """Rows end-length+1 .. end (integer positions, inclusive). If complete, keep only
        columns tradable at row ``end`` with no NaN in the window. Raises IndexError if the
        window starts before row 0.

        Negative ``end`` counts from the last row (``-1`` = last), as in Python indexing; it is
        normalised before the start check. ``length`` must be a positive integer. The tradability
        mask is evaluated at row ``end`` only (the decision date), so no information after
        ``end`` is used.
        """
        end = operator.index(end)
        length = operator.index(length)
        n = self.shape[0]
        if length < 1:
            raise ValueError(f"length must be >= 1, got {length}")
        if end < 0:
            end += n
        if not 0 <= end < n:
            raise IndexError(f"end={end} out of range for a panel with {n} rows")
        start = end - length + 1
        if start < 0:
            raise IndexError(f"window of length {length} ending at row {end} starts before row 0")
        R = self.returns.iloc[start:end + 1]
        T = self.tradable.iloc[start:end + 1]
        if complete:
            keep = T.to_numpy(dtype=bool)[-1] & ~np.isnan(R.to_numpy()).any(axis=0)
            R, T = R.loc[:, keep], T.loc[:, keep]
        return self._replace(R, T)

    def subset(self, start=None, end=None, assets=None) -> Panel:
        """Label-based subset: dates in ``[start, end]`` (inclusive) and the listed ``assets``.

        Parameters
        ----------
        start, end : date-like or None
            Inclusive bounds (anything :func:`pandas.Timestamp` accepts); None = open.
        assets : list-like or None
            Asset ids to keep, in the given order. Unknown ids raise ``KeyError``.
        """
        idx = self.returns.index

        def ts(x):
            t = pd.Timestamp(x)
            if idx.tz is not None and t.tz is None:
                t = t.tz_localize(idx.tz)
            return t

        if start is not None and end is not None and ts(start) > ts(end):
            raise ValueError(f"start {start} is after end {end}")
        lo = 0 if start is None else idx.searchsorted(ts(start), side="left")
        hi = len(idx) if end is None else idx.searchsorted(ts(end), side="right")
        R = self.returns.iloc[lo:hi]
        T = self.tradable.iloc[lo:hi]
        if assets is not None:
            assets = list(assets)
            missing = [a for a in assets if a not in self.returns.columns]
            if missing:
                raise KeyError(f"unknown assets: {missing[:10]}")
            R, T = R.loc[:, assets], T.loc[:, assets]
        return self._replace(R, T)
