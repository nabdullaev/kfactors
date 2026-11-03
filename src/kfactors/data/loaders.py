"""Public-data loaders returning Panel objects (simple returns in decimals).

load_jkp: JKP Global Factor Data, URL pattern
  https://jkpfactors-data.s3.amazonaws.com/public/[{region}]_[all_factors]_[{freq}]_[{weighting}].zip
  (brackets URL-encoded). Long CSV (location, name, freq, weighting, direction, n_stocks,
  n_stocks_min, date, ret) -> pivot name x date. Units already decimal. tradable = ~NaN.
  meta includes licence "CC BY-NC 4.0" and citation (Jensen, Kelly & Pedersen 2023).

load_french: Ken French data library zips (e.g. 100_Portfolios_10x10_Daily_CSV.zip,
  49_Industry_Portfolios_daily_CSV.zip and monthly variants). Parse the requested block
  (value-weighted by default) from the multi-block CSV; -99.99 and -999 -> NaN; percent -> decimal.
  tradable = ~NaN.

load_binance: Binance Vision spot daily klines for {quote} pairs from
  https://data.binance.vision/data/spot/monthly/klines/{SYMBOL}/1d/{SYMBOL}-1d-{YYYY}-{MM}.zip
  Symbol list from the S3 listing API (prefix data/spot/monthly/klines/). Excludes stablecoins,
  leveraged tokens (UP/DOWN/BULL/BEAR suffixes) and, by default, wrapped/staked duplicates
  (WBTC, WBETH, BETH, BNSOL, ...). Returns from close-to-close. Point-in-time universe: at each
  date, tradable = top ``top`` symbols by trailing ``volume_window``-day median quote volume among
  symbols with a full trailing window and a full 24 h bar that day (listing date respected;
  delisted symbols kept until their last full bar). Ticker reuse is split into separate assets
  (suffix "_2") after a gap > ``max_gap`` days or a gap of any length across which the price
  moves by more than a factor ``max_jump`` (LUNA -> LUNA 2.0: 17-day gap, 177,400x).
  Timestamps auto-detected (ms vs us). meta periods_per_year = 365.

Verified formats (downloaded 2026-11-03)
----------------------------------------
* JKP ``[usa]_[all_factors]_[monthly]_[vw_cap].zip``: one CSV member of the same name, header
  ``location,name,freq,weighting,direction,n_stocks,n_stocks_min,date,ret``; ``date`` is the
  calendar month end (``YYYY-MM-DD``), ``ret`` a decimal return; 153 factors, 1926-01..2025-12, no
  NaN rows in the long file (missing = absent row); balanced from 1971-11-30.
* French CSVs: CRLF, latin-1, free-text preamble, then blocks ``<title line>`` /
  ``,<col1>,<col2>,...`` / ``YYYYMMDD|YYYYMM|YYYY, v1, v2, ...``; values in percent; missing coded
  -99.99 or -999. Column labels may carry trailing blanks (``"Food "``) and are stripped.
* Binance monthly kline zips: one CSV member, 12 columns (see :data:`KLINE_COLUMNS`), no header in
  the files checked (the parser accepts an optional header row); ``open_time``/``close_time`` in
  milliseconds up to 2024-12 and microseconds from 2025-01.
"""
from __future__ import annotations

import io
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from urllib.parse import quote as urlquote
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from kfactors.data import cache
from kfactors.data.panel import Panel

__all__ = [
    "DERIVATIVE_BASES",
    "FRENCH_DATASETS",
    "KLINE_COLUMNS",
    "LEVERAGED_SUFFIXES",
    "STABLE_BASES",
    "binance_candidates",
    "load_binance",
    "load_french",
    "load_jkp",
    "parse_french_csv",
    "parse_kline_csv",
]

# ============================================================================================
# JKP
# ============================================================================================
JKP_BASE = "https://jkpfactors-data.s3.amazonaws.com/public/"
JKP_COLUMNS = ("location", "name", "freq", "weighting", "direction", "n_stocks", "n_stocks_min",
               "date", "ret")
JKP_CITATION = ("Jensen, T. I., Kelly, B., & Pedersen, L. H. (2023). Is There a Replication "
                "Crisis in Finance? Journal of Finance, 78(5), 2465-2518. "
                "Data: https://jkpfactors.com")
_FREQ_CODE = {"monthly": "M", "daily": "D"}


def _jkp_filename(region: str, freq: str, weighting: str) -> str:
    return f"[{region}]_[all_factors]_[{freq}]_[{weighting}]"


def _read_single_member(path, suffix: str = ".csv") -> tuple[str, bytes]:
    with zipfile.ZipFile(path) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(suffix) and not
                   m.startswith("__MACOSX")]
        if len(members) != 1:
            raise ValueError(f"{path}: expected exactly one {suffix} member, found {members}")
        return members[0], zf.read(members[0])


def _provenance(url: str, relpath: str) -> dict:
    e = cache.manifest_entry(relpath) or {}
    return {"url": url, "relpath": relpath, "sha256": e.get("sha256"), "bytes": e.get("bytes"),
            "downloaded_at": e.get("downloaded_at")}


def load_jkp(region: str = "usa", freq: str = "monthly", weighting: str = "vw_cap", *,
             start=None, end=None, refresh: bool = False) -> Panel:
    """Load the JKP Global Factor Data factor-return panel (Jensen, Kelly & Pedersen 2023).

    Parameters
    ----------
    region : str
        JKP region/country code, e.g. ``"usa"``, ``"world"``, ``"developed"``.
    freq : {"monthly", "daily"}
    weighting : {"vw_cap", "vw", "ew"}
        Portfolio weighting of the long-short factors (capped value weights by default).
    start, end : date-like, optional
        Inclusive date bounds applied after loading.
    refresh : bool
        Re-download the snapshot.

    Returns
    -------
    Panel
        ``returns``: dates (calendar period ends) x factor names, decimal long-short returns
        (``ret`` column, already in decimals); ``tradable = returns.notna()``. ``freq`` is
        ``"M"`` or ``"D"``. ``meta`` holds url, sha256, download time, licence (CC BY-NC 4.0),
        citation, the factor ``direction`` signs and ``first_complete_date`` (first date from which
        every factor is observed; 1971-11-30 for usa/monthly/vw_cap in the 2026 snapshot).
        ``ret`` is already signed per the JKP ``direction`` (long the leg predicted to earn more),
        so ``meta["direction"]`` is informational only; do not multiply returns by it again
        (from 1972, 91% of direction=+1 and 86% of direction=-1 usa/monthly/vw_cap factors have
        a positive mean).

    Notes
    -----
    The JKP files are long-format; missing observations are absent rows, so the pivoted panel has
    NaN before a factor's first observation. Pre-1970 rows are sparse (most factors need
    Compustat). Use ``start="1972-01-01"`` for a balanced panel.
    """
    region, freq, weighting = str(region).lower(), str(freq).lower(), str(weighting).lower()
    if freq not in _FREQ_CODE:
        raise ValueError(f"freq must be one of {sorted(_FREQ_CODE)}, got {freq!r}")
    if weighting not in ("vw_cap", "vw", "ew"):
        raise ValueError(f"weighting must be 'vw_cap', 'vw' or 'ew', got {weighting!r}")
    if not re.fullmatch(r"[a-z0-9_]+", region):
        raise ValueError(f"invalid region {region!r}")
    fname = _jkp_filename(region, freq, weighting)
    url = JKP_BASE + urlquote(fname + ".zip", safe="_.")
    relpath = f"jkp/{region}_all_factors_{freq}_{weighting}.zip"
    path = cache.fetch(url, relpath, refresh=refresh)
    member, raw = _read_single_member(path)
    df = pd.read_csv(io.BytesIO(raw))
    missing = [c for c in JKP_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{member}: missing JKP columns {missing}; got {list(df.columns)}")
    for col, want in (("location", region), ("freq", freq), ("weighting", weighting)):
        got = set(df[col].astype(str).str.lower().unique())
        if got != {want}:
            raise ValueError(f"{member}: column {col!r} has values {sorted(got)}, expected {want!r}")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    if df.duplicated(["name", "date"]).any():
        raise ValueError(f"{member}: duplicated (name, date) rows")
    R = df.pivot(index="date", columns="name", values="ret").sort_index()
    R.columns = R.columns.astype(str)
    R.columns.name = None
    R.index.name = "date"
    R = R.astype(np.float64)
    complete = R.notna().all(axis=1).to_numpy()
    first_complete = None
    if complete.any():
        i0 = int(np.argmax(complete))
        first_complete = str(R.index[i0].date())
    direction = df.drop_duplicates("name").set_index("name")["direction"].astype(int).to_dict()
    meta = {
        "source": "JKP Global Factor Data", **_provenance(url, relpath),
        "member": member, "region": region, "jkp_freq": freq, "weighting": weighting,
        "units": "simple returns, decimal (0.01 = 1%)", "licence": "CC BY-NC 4.0",
        "citation": JKP_CITATION, "first_complete_date": first_complete,
        "direction": {str(k): v for k, v in direction.items()},
    }
    panel = Panel(R, R.notna(), name=f"jkp_{region}_{freq}_{weighting}",
                  freq=_FREQ_CODE[freq], meta=meta)
    if start is not None or end is not None:
        panel = panel.subset(start, end)
    return panel


# ============================================================================================
# Ken French data library
# ============================================================================================
FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
#: dataset -> (daily zip, monthly zip, citation)
_FF1993 = ("Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks "
           "and bonds. Journal of Financial Economics, 33(1), 3-56.")
_FF1997 = ("Fama, E. F., & French, K. R. (1997). Industry costs of equity. Journal of Financial "
           "Economics, 43(2), 153-193.")
FRENCH_DATASETS = {
    "100_size_bm": ("100_Portfolios_10x10_Daily_CSV.zip", "100_Portfolios_10x10_CSV.zip", _FF1993),
    "25_size_bm": ("25_Portfolios_5x5_Daily_CSV.zip", "25_Portfolios_5x5_CSV.zip", _FF1993),
    "49_industries": ("49_Industry_Portfolios_daily_CSV.zip", "49_Industry_Portfolios_CSV.zip",
                      _FF1997),
}
_FRENCH_MISSING = (-99.99, -999.0)
_DATA_ROW = re.compile(r"^\s*(\d{4}|\d{6}|\d{8})\s*,")


def parse_french_csv(text: str) -> list[dict]:
    """Split a Ken French multi-block CSV into its data blocks.

    A block is ``title line`` (the last non-empty, non-data text line before the header),
    a header line starting with ``,`` and the consecutive data rows whose first field is a
    4-, 6- or 8-digit date (``YYYY``, ``YYYYMM``, ``YYYYMMDD``).

    Returns
    -------
    list of dict
        ``{"title": str, "columns": list[str], "dates": list[str], "values": ndarray}`` in file
        order; values are raw floats (percent; sentinels not yet replaced).
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks, title, i = [], "", 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(","):
            cols = [c.strip() for c in line.split(",")[1:]]
            while cols and cols[-1] == "":
                cols.pop()
            j = i + 1
            while j < len(lines) and _DATA_ROW.match(lines[j]):
                j += 1
            rows = lines[i + 1:j]
            if rows:
                df = pd.read_csv(io.StringIO("\n".join(rows)), header=None, dtype={0: str},
                                 skipinitialspace=True)
                vals = df.iloc[:, 1:].to_numpy(dtype=np.float64)
                if vals.shape[1] > len(cols):  # trailing empty fields
                    extra = vals[:, len(cols):]
                    if not np.all(np.isnan(extra)):
                        raise ValueError(f"block {title!r}: more values than column labels")
                    vals = vals[:, :len(cols)]
                if vals.shape[1] != len(cols):
                    raise ValueError(f"block {title!r}: {vals.shape[1]} values vs {len(cols)} labels")
                blocks.append({"title": title, "columns": cols,
                               "dates": df.iloc[:, 0].str.strip().tolist(), "values": vals})
            title = ""
            i = j
            continue
        if line.strip() and not _DATA_ROW.match(line):
            title = line.strip()
        i += 1
    return blocks


def _french_block(blocks: list[dict], weighting: str, freq: str) -> dict:
    wname = {"vw": "value", "ew": "equal"}[weighting]
    fname = {"daily": "daily", "monthly": "monthly"}[freq]
    pat = re.compile(rf"average\s+{wname}\s+weighted\s+returns\s*-+\s*{fname}\b", re.IGNORECASE)
    hits = [b for b in blocks if pat.search(b["title"])]
    if len(hits) != 1:
        titles = [b["title"] for b in blocks]
        raise ValueError(f"expected one block matching {pat.pattern!r}, found {len(hits)}; "
                         f"blocks: {titles}")
    return hits[0]


def load_french(dataset: str = "100_size_bm", freq: str = "daily", weighting: str = "vw", *,
                start=None, end=None, refresh: bool = False) -> Panel:
    """dataset in {"100_size_bm", "49_industries", "25_size_bm"}.

    Load portfolio returns from the Kenneth R. French data library.

    Parameters
    ----------
    dataset : {"100_size_bm", "49_industries", "25_size_bm"}
        10x10 size/book-to-market (Fama & French 1993), 49 industries (Fama & French 1997),
        5x5 size/book-to-market.
    freq : {"daily", "monthly"}
    weighting : {"vw", "ew"}
        Value- ("Average Value Weighted Returns") or equal-weighted block.
    start, end : date-like, optional
        Inclusive date bounds.
    refresh : bool
        Re-download the snapshot.

    Returns
    -------
    Panel
        Decimal simple returns (file values / 100); the sentinels -99.99 and -999 become NaN;
        ``tradable = returns.notna()``. Daily index = trading dates; monthly index = calendar
        month ends (``YYYYMM`` -> last day of the month, same convention as JKP).

    Notes
    -----
    No outlier filtering. Before 1963 the thin small-cap portfolios have artefacts in the daily
    files (100_size_bm: -100% for ME1 BM2 on 1927-11-10, +196% on 1932-07-20; 49_industries:
    Paper +150% on 1932-08-11); from 1963-07 the daily 100_size_bm extremes are -38% / +46%.
    Use ``start="1963-07-01"`` for these panels (as the default benchmark config does).

    Raises
    ------
    ValueError
        Unknown dataset/freq/weighting, a missing/ambiguous block, dates inconsistent with
        ``freq``, or a value below -100% that is not a missing-data sentinel.
    """
    if dataset not in FRENCH_DATASETS:
        raise ValueError(f"dataset must be one of {sorted(FRENCH_DATASETS)}, got {dataset!r}")
    freq, weighting = str(freq).lower(), str(weighting).lower()
    if freq not in ("daily", "monthly"):
        raise ValueError(f"freq must be 'daily' or 'monthly', got {freq!r}")
    if weighting not in ("vw", "ew"):
        raise ValueError(f"weighting must be 'vw' or 'ew', got {weighting!r}")
    daily_zip, monthly_zip, citation = FRENCH_DATASETS[dataset]
    zname = daily_zip if freq == "daily" else monthly_zip
    url = FRENCH_BASE + zname
    relpath = f"french/{zname}"
    path = cache.fetch(url, relpath, refresh=refresh)
    member, raw = _read_single_member(path)
    blocks = parse_french_csv(raw.decode("latin-1"))
    blk = _french_block(blocks, weighting, freq)
    width = 8 if freq == "daily" else 6
    bad = [d for d in blk["dates"] if len(d) != width]
    if bad:
        raise ValueError(f"{member}: dates {bad[:3]} inconsistent with freq={freq!r}")
    if freq == "daily":
        idx = pd.to_datetime(blk["dates"], format="%Y%m%d")
    else:
        idx = pd.to_datetime(blk["dates"], format="%Y%m") + pd.offsets.MonthEnd(0)
    vals = blk["values"].copy()
    sentinel = np.zeros(vals.shape, dtype=bool)
    for s in _FRENCH_MISSING:
        sentinel |= np.isclose(vals, s, rtol=0.0, atol=1e-6)
    impossible = (vals < -100.0) & ~sentinel
    if impossible.any():
        raise ValueError(f"{member}: {int(impossible.sum())} values below -100% that are not "
                         f"missing-data sentinels")
    vals[sentinel] = np.nan
    R = pd.DataFrame(vals / 100.0, index=pd.DatetimeIndex(idx, name="date"),
                     columns=pd.Index(blk["columns"]))
    meta = {
        "source": "Kenneth R. French Data Library", **_provenance(url, relpath),
        "member": member, "block": blk["title"], "dataset": dataset, "french_freq": freq,
        "weighting": weighting, "units": "simple returns, decimal (file is percent; divided by 100)",
        "missing_sentinels": list(_FRENCH_MISSING), "n_missing_cells": int(sentinel.sum()),
        "citation": citation,
        "licence": "Kenneth R. French Data Library; free for research use, cite the source",
    }
    panel = Panel(R, R.notna(), name=f"french_{dataset}_{freq}_{weighting}",
                  freq="D" if freq == "daily" else "M", meta=meta)
    if start is not None or end is not None:
        panel = panel.subset(start, end)
    return panel


# ============================================================================================
# Binance Vision
# ============================================================================================
BINANCE_BASE = "https://data.binance.vision/"
BINANCE_LISTING = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
KLINE_PREFIX = "data/spot/monthly/klines/"
KLINE_COLUMNS = ("open_time", "open", "high", "low", "close", "volume", "close_time",
                 "quote_volume", "count", "taker_buy_base", "taker_buy_quote", "ignore")
#: Bases excluded as fiat-pegged stablecoins, fiat currencies or commodity-pegged tokens.
#: USD stablecoins: USDC, BUSD, TUSD, FDUSD, USDP, PAX (old USDP ticker), DAI, UST/USTC (TerraUSD
#: and its post-depeg ticker), SUSD, USDS, USDSB, XUSD, USD1, BFUSD, RLUSD, USDE, PYUSD, USDJ,
#: GUSD, LUSD, CUSD, USDD, USDB; EUR stablecoins: EURI, AEUR, EURC; fiat legs: EUR, GBP,
#: AUD, TRY, BRL, RUB, UAH, NGN, ZAR, ARS, PLN, RON, JPY, MXN, COP, CZK, KZT, IDR; fiat-pegged
#: tokens: BIDR, IDRT, BVND, BKRW, TRYB, VAI; gold-pegged: PAXG, XAUT.
#: USTC is kept in the list although it has floated freely since the May 2022 TerraUSD depeg
#: (it is excluded as a failed stablecoin, not because it is pegged today). FRAX is *not* in the
#: list: Binance's ``FRAXUSDT`` (archives from 2026-01) is the renamed FXS governance token, a
#: volatile asset; Frax's USD stablecoin (frxUSD) is caught by the ``"USD"`` substring rule.
STABLE_BASES = frozenset({
    "USDC", "BUSD", "TUSD", "FDUSD", "USDP", "PAX", "DAI", "UST", "USTC", "SUSD", "USDS", "USDSB",
    "XUSD", "USD1", "BFUSD", "RLUSD", "USDE", "PYUSD", "USDJ", "GUSD", "LUSD", "CUSD",
    "USDD", "USDB", "EURI", "AEUR", "EURC", "EUR", "GBP", "AUD", "TRY", "BRL", "RUB", "UAH", "NGN",
    "ZAR", "ARS", "PLN", "RON", "JPY", "MXN", "COP", "CZK", "KZT", "IDR", "BIDR", "IDRT", "BVND",
    "BKRW", "TRYB", "VAI", "PAXG", "XAUT",
})
#: Bases excluded (by default) as wrapped or liquid-staking claims on another listed coin. They are
#: near-copies of the underlying (daily-return correlation > 0.997 for WBTC/BTC, WBETH/ETH,
#: BETH/ETH and BNSOL/SOL in 2022-2025) and would add a spurious near-zero eigenvalue and a
#: tiny-variance "spread" for residual and minimum-variance strategies. Binance USDT pairs: WBTC,
#: WBETH, BETH, BNSOL; the others are listed so that future listings are caught as well.
#: Names that merely start with W/ST or contain BTC/ETH/SOL (WNXM, BTCST, ETHFI, SOLV, RESOLV)
#: are independent tokens and are not listed.
DERIVATIVE_BASES = frozenset({
    "WBTC", "WBETH", "BETH", "BNSOL", "WETH", "STETH", "WSTETH", "CBETH", "RETH", "WBNB",
})
#: Binance leveraged tokens are <BASE><SUFFIX> (BTCUP, ETHDOWN, BNBBULL, ...) or the bare BULL/BEAR.
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")
_LISTING_TTL = timedelta(days=1)
_PUBLICATION_LAG = timedelta(days=7)
_S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def _s3_list(prefix: str, *, delimiter: str | None = "/", timeout: float = 60.0
             ) -> tuple[list[str], list[str]]:
    """List the Binance Vision S3 bucket (ListObjects v1) with marker pagination.

    Returns (common_prefixes, keys).
    """
    prefixes, keys, marker = [], [], ""
    for _ in range(10_000):
        params = {"prefix": prefix}
        if delimiter:
            params["delimiter"] = delimiter
        if marker:
            params["marker"] = marker
        body = cache.http_get(BINANCE_LISTING + "?" + urlencode(params), timeout=timeout)
        root = ET.fromstring(body)
        ns = _S3_NS if root.tag.startswith(_S3_NS) else ""
        # <Prefix> also appears at top level (= the query prefix); keep CommonPrefixes only
        page_p = [p.text for cp in root.iter(f"{ns}CommonPrefixes")
                  for p in cp.iter(f"{ns}Prefix") if p.text]
        page_k = [k.text for c in root.iter(f"{ns}Contents") for k in c.iter(f"{ns}Key")]
        prefixes += page_p
        keys += page_k
        truncated = (root.findtext(f"{ns}IsTruncated") or "false").strip().lower() == "true"
        if not truncated:
            return prefixes, keys
        nxt = root.findtext(f"{ns}NextMarker") or ""
        if not nxt:
            cands = page_p + page_k
            if not cands:
                raise RuntimeError(f"S3 listing for {prefix!r} truncated without a marker")
            nxt = max(cands)
        if nxt <= marker:
            raise RuntimeError(f"S3 listing for {prefix!r} did not advance (marker {nxt!r})")
        marker = nxt
    raise RuntimeError(f"S3 listing for {prefix!r} exceeded the page limit")


def _now() -> datetime:
    return datetime.now(UTC)


def _listing_fresh(info: dict, needed_until: pd.Timestamp) -> bool:
    try:
        fetched = datetime.fromisoformat(info["fetched_at"])
    except (KeyError, TypeError, ValueError):
        return False
    final = needed_until.to_pydatetime().replace(tzinfo=UTC) + _PUBLICATION_LAG
    return fetched >= final or (_now() - fetched) < _LISTING_TTL


def _cached_listing(relpath: str, needed_until: pd.Timestamp, refresh: bool, compute):
    """JSON listing cached under the cache root (re-listed when stale or on refresh)."""
    path = cache.cache_root() / relpath
    if path.exists() and not refresh:
        try:
            info = json.loads(path.read_text(encoding="utf-8"))
            if _listing_fresh(info, needed_until):
                return info["items"]
        except (json.JSONDecodeError, KeyError):
            pass
    items = compute()
    payload = {"fetched_at": _now().isoformat(timespec="seconds"), "items": items}
    cache._atomic_write_bytes(path, json.dumps(payload).encode("utf-8"))
    return items


def _is_leveraged(base: str, bases: set[str]) -> bool:
    for suf in LEVERAGED_SUFFIXES:
        if base.endswith(suf):
            stem = base[: -len(suf)]
            if stem == "" or stem in bases:
                return True
    return False


def binance_candidates(symbols, quote: str = "USDT", *, exclude_derivatives: bool = True
                       ) -> tuple[list[str], dict]:
    """Filter exchange symbols to the ``quote`` pairs eligible for the universe.

    Keeps ``<BASE><quote>`` symbols whose base is not in :data:`STABLE_BASES`, does not contain
    ``"USD"`` (catches newer USD-pegged tokens such as ``RWUSD``/``USDEB`` not yet in the list),
    is not a leveraged token and (when ``exclude_derivatives``) is not a wrapped/staked
    duplicate in :data:`DERIVATIVE_BASES`. A base is treated as a leveraged token iff it is
    ``<STEM><SUFFIX>`` with ``SUFFIX`` in :data:`LEVERAGED_SUFFIXES` and ``STEM`` either empty
    (``BULL``, ``BEAR``) or itself a base quoted in ``quote`` (``BTCUP`` -> ``BTC``). This avoids
    false positives such as ``JUP`` or ``SUPER``, which a plain suffix test would drop.

    Parameters
    ----------
    exclude_derivatives : bool, default True
        Drop wrapped / liquid-staking duplicates (WBTC, WBETH, BETH, BNSOL, ...). Pass ``False``
        to keep them (the pre-2026-11 behaviour).

    Returns
    -------
    (sorted candidate symbols, {"stable": [...], "leveraged": [...], "derivative": [...]}
    excluded symbols; ``"derivative"`` is empty when ``exclude_derivatives=False``)
    """
    quote = quote.upper()
    pairs = sorted({s.upper() for s in symbols if s.upper().endswith(quote) and len(s) > len(quote)})
    bases = {s[: -len(quote)] for s in pairs}
    keep, stable, lev, deriv = [], [], [], []
    for s in pairs:
        b = s[: -len(quote)]
        if b in STABLE_BASES or "USD" in b:
            stable.append(s)
        elif _is_leveraged(b, bases):
            lev.append(s)
        elif exclude_derivatives and b in DERIVATIVE_BASES:
            deriv.append(s)
        else:
            keep.append(s)
    return keep, {"stable": stable, "leveraged": lev, "derivative": deriv}


def _ts_per_second(t: int) -> int:
    """Ticks per second of an epoch timestamp of unknown unit (detected from magnitude).

    >= 1e17 ns, >= 1e14 us, >= 1e11 ms, else s. Binance spot klines use ms until 2024-12-31 and
    us from 2025-01-01.
    """
    if t >= 10 ** 17:
        return 10 ** 9
    if t >= 10 ** 14:
        return 10 ** 6
    if t >= 10 ** 11:
        return 10 ** 3
    return 1


def _ts_to_day(t: int) -> int:
    """Epoch timestamp of unknown unit -> days since 1970-01-01 (UTC); see :func:`_ts_per_second`."""
    return t // (86_400 * _ts_per_second(t))


def parse_kline_csv(raw: bytes, *, with_hours: bool = False):
    """Parse a Binance kline CSV (with or without header) -> (day, close, quote_volume[, hours]).

    ``day`` is int64 days since the Unix epoch of the bar's ``open_time`` (unit auto-detected per
    row, see :func:`_ts_per_second`). With ``with_hours=True`` a fourth float64 array gives the bar
    length in hours, ``(close_time - open_time + 1 tick) / 3600 s`` in the unit of ``open_time``
    (24.0 for a full daily bar; e.g. 3.0 for a bar cut short by a delisting). It is NaN when
    ``close_time`` is not an integer or not after ``open_time``.
    """
    days, close, qv, hours = [], [], [], []
    for n, line in enumerate(raw.splitlines()):
        line = line.strip()
        if not line:
            continue
        f = line.split(b",")
        if len(f) < 8:
            raise ValueError(f"kline row {n} has {len(f)} fields, expected 12: {line[:80]!r}")
        try:
            t = int(f[0])
        except ValueError:
            if n == 0 and f[0].strip().lower().strip(b'"') in (b"open_time", b"opentime",
                                                                 b"open time"):
                continue  # header row
            raise ValueError(f"kline row {n}: bad open_time {f[0]!r}") from None
        days.append(_ts_to_day(t))
        close.append(float(f[4]))
        qv.append(float(f[7]))
        if with_hours:
            try:
                ct = int(f[6])
            except ValueError:
                ct = None
            hours.append((ct - t + 1) / _ts_per_second(t) / 3600.0
                         if ct is not None and ct > t else np.nan)
    out = (np.asarray(days, dtype=np.int64), np.asarray(close, dtype=np.float64),
           np.asarray(qv, dtype=np.float64))
    if with_hours:
        return (*out, np.asarray(hours, dtype=np.float64))
    return out


def _parse_kline_zip(path, ym: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _, raw = _read_single_member(path)
    d, c, v, h = parse_kline_csv(raw, with_hours=True)
    if len(d):
        p = pd.Period(ym, freq="M")
        lo = (p.start_time - pd.Timestamp("1970-01-01")).days
        hi = (p.end_time.normalize() - pd.Timestamp("1970-01-01")).days
        if d.min() < lo or d.max() > hi:
            raise ValueError(f"{path}: bar dates outside {ym}; timestamp unit misdetected?")
    return d, c, v, h


_PARTIAL_BAR_HOURS = 24.0 - 1e-6
BINANCE_LICENCE = "CC BY-NC-SA 4.0 (Binance Vision Dataset Terms v1.0)"
BINANCE_CITATION = ("Binance Vision public market data, spot monthly 1d klines, "
                    "https://data.binance.vision")


def load_binance(top: int = 100, start: str = "2022-01-01", end: str = "2025-12-31", *,
                 quote: str = "USDT", volume_window: int = 30, history_start: str = "2021-06-01",
                 max_symbols: int | None = None, max_gap: int = 30, refresh: bool = False,
                 n_jobs: int = 8, symbols: list[str] | None = None,
                 max_jump: float | None = 5.0, exclude_derivatives: bool = True) -> Panel:
    """Binance spot daily close-to-close returns with a point-in-time liquidity universe.

    Parameters
    ----------
    top : int
        Universe size: at each date the ``top`` eligible assets by trailing median quote volume.
    start : str
        First date with a non-empty universe. Rows from ``history_start`` to ``start`` are kept in
        the panel (estimation history for walk-forward windows) but ``tradable`` is all False there.
    end : str
        Last date (inclusive). Only complete-month archives exist, so the panel ends at
        ``min(end, last available bar)``.
    quote : str
        Quote asset (``"USDT"``).
    volume_window : int
        Trailing window (calendar days, including ``t``) of the quote-volume median. An asset is
        eligible at ``t`` only if it has a bar on each of those days and an observed return at ``t``.
    history_start : str
        First month downloaded.
    max_symbols : int, optional
        Keep only the first ``max_symbols`` candidates in alphabetical order *among those with at
        least one archive month in [history_start, end]* (deterministic; for tests/demos).
    max_gap : int
        Ticker-reuse split: if more than ``max_gap`` calendar days without a bar separate two bars
        of the same symbol, the later bars form a new asset ``<BASE>_2`` (``_3``, ...).
    refresh : bool
        Re-list and re-download everything.
    n_jobs : int
        Download threads (the per-symbol S3 listings use ``min(32, 4 * n_jobs)`` threads).
    symbols : list of str, optional
        Explicit symbols (e.g. ``["BTCUSDT", "ETHUSDT"]``) instead of the exchange listing; the
        stablecoin/leveraged/derivative filters still apply. (Additive keyword; not in the
        original stub.)
    max_jump : float or None, default 5.0
        Ticker-reuse split on price: a gap of at least one missing day across which the close
        changes by more than a factor ``max_jump`` (``close_after / close_before > max_jump`` or
        ``< 1 / max_jump``) also starts a new asset, whatever the gap length. This separates
        LUNA classic (last bar 2022-05-13) from LUNA 2.0 (first bar 2022-05-31, a 17-day gap
        with a 177,400x price ratio), which ``max_gap=30`` alone does not, and redenominations
        such as QUICK, SUN, BNX, STRAX (1:1000 .. 1:10). Bars on consecutive days are never split
        on price. ``None`` disables the rule (the pre-2026-11 behaviour). Must be > 1.
    exclude_derivatives : bool, default True
        Drop wrapped / liquid-staking duplicates of other listed coins (:data:`DERIVATIVE_BASES`:
        WBTC, WBETH, BETH, BNSOL, ...), reported in ``meta['excluded']['derivative']``. ``False``
        keeps them (the pre-2026-11 behaviour).

    Returns
    -------
    Panel
        Calendar-day index ``history_start .. end``; columns = asset ids (base asset, plus ``_k``
        suffix for re-used tickers). ``returns[t] = close[t] / close[t-1] - 1`` when both days
        have bars, else NaN (no multi-day returns across gaps). ``tradable`` as described above.
        ``meta['periods_per_year'] = 365``; ``meta['symbol']`` maps asset id -> exchange symbol;
        ``meta['ticker_splits']`` lists every split (``reason`` ``"gap"`` or ``"price_jump"``,
        ``gap_days``, ``price_ratio``); ``meta['partial_bars']`` lists the bars shorter than 24 h
        inside the panel (``asset``, ``date``, ``hours``).

    Notes
    -----
    * Survivorship: the S3 listing contains delisted symbols, so the universe is point-in-time;
      a delisted asset stays tradable up to its last full bar (then its returns are NaN).
    * Partial bars: a daily bar shorter than 24 h (``close_time - open_time``; typically the 3 h
      last bar of a delisting or token migration, e.g. LUNA 2022-05-13 at 0.7 h) keeps its
      return, which is a genuine close-to-close price change, but the asset is not tradable on
      that day: it cannot be held to the next close. 142 such bars exist in the 2021-06..2025-12
      USDT archives.
    * Look-ahead: ``tradable[t]`` uses quote volumes of days ``t-volume_window+1 .. t``, the
      return at ``t`` and the length of bar ``t`` only; a strategy deciding at ``t`` and holding
      from ``t+1`` uses no future data. Execution caveat: with the engine's ``lag=0`` the
      position is entered at the very close (UTC 00:00) whose bar defines the universe and the
      signal. That is the standard close-to-close convention, but optimistic for a 24/7 market;
      ``lag=1`` is the conservative choice.
    * Ticker renames and migrations (MATIC -> POL, RNDR -> RENDER, FTM -> S, EOS -> A,
      MKR -> SKY at 1:24000, BTT -> BTTC, TOMO -> VIC) are *not* chained: old and new tickers
      are separate assets, and the successor only becomes tradable once it has a full
      ``volume_window`` of bars (e.g. MATIC last tradable 2024-09-10, POL first 2024-10-12 in the
      top-100 panel); an engine window additionally needs ``window`` complete returns.
    * No outlier filtering: extreme but genuine returns (e.g. UNFI +452% on 2022-06-07, LUNA
      -90% in May 2022) are kept. Robustify estimation windows in the strategy if needed, never
      the realised returns.
    * Ties in median volume are broken alphabetically by asset id.
    * Bar date = UTC date of ``open_time``; returns are UTC close-to-close.
    * Available months are taken from a per-symbol S3 listing (one request per symbol), so
      non-existent months are not requested; a listed month that nevertheless answers 404 is
      treated as unavailable.
    """
    top, volume_window, max_gap = int(top), int(volume_window), int(max_gap)
    if top < 1 or volume_window < 1 or max_gap < 0:
        raise ValueError("top and volume_window must be >= 1 and max_gap >= 0")
    if max_jump is not None:
        max_jump = float(max_jump)
        if not max_jump > 1.0:
            raise ValueError(f"max_jump must be > 1 or None, got {max_jump!r}")
    quote = quote.upper()
    t_hist, t_start, t_end = (pd.Timestamp(history_start).normalize(),
                              pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize())
    if not t_hist <= t_start <= t_end:
        raise ValueError("need history_start <= start <= end")
    months = [str(p) for p in pd.period_range(t_hist.to_period("M"), t_end.to_period("M"),
                                              freq="M")]
    needed_until = t_end.to_period("M").end_time.normalize()

    # ---- 1. candidate symbols ------------------------------------------------------------
    if symbols is None:
        all_syms = _cached_listing(
            "binance/listings/spot_monthly_klines.json", needed_until, refresh,
            lambda: sorted(p[len(KLINE_PREFIX):].strip("/")
                           for p in _s3_list(KLINE_PREFIX)[0]))
    else:
        all_syms = [str(s).upper() for s in symbols]
    cands, excluded = binance_candidates(all_syms, quote, exclude_derivatives=exclude_derivatives)
    if symbols is not None and not cands:
        raise ValueError(f"none of {symbols} is an eligible {quote} pair")

    # ---- 2. available months per symbol (per-symbol S3 listing, cached) -----------------
    def list_months(sym):
        def compute():
            keys = _s3_list(f"{KLINE_PREFIX}{sym}/1d/")[1]
            pat = re.compile(rf"/{re.escape(sym)}-1d-(\d{{4}}-\d{{2}})\.zip$")
            return sorted(m.group(1) for k in keys if (m := pat.search(k)))
        return _cached_listing(f"binance/listings/1d/{sym}.json", needed_until, refresh, compute)

    # listing requests are latency-bound (~0.5-1 s each to the ap-northeast-1 bucket), so they
    # use more threads than the downloads
    with ThreadPoolExecutor(max_workers=max(1, min(32, 4 * int(n_jobs)))) as ex:
        avail = dict(zip(cands, ex.map(list_months, cands)))
    wanted = set(months)
    in_range = [s for s in cands if wanted.intersection(avail[s])]
    if max_symbols is not None:
        in_range = in_range[: int(max_symbols)]
    if not in_range:
        raise ValueError(f"no {quote} symbol has data between {history_start} and {end}")

    # ---- 3. download month archives (only those not cached) -----------------------------
    avail_sets = {s: set(avail[s]) for s in in_range}
    jobs = [(s, m) for s in in_range for m in months if m in avail_sets[s]]
    items = [(f"{BINANCE_BASE}{KLINE_PREFIX}{s}/1d/{s}-1d-{m}.zip",
              f"binance/{KLINE_PREFIX}{s}/1d/{s}-1d-{m}.zip") for s, m in jobs]
    n_cached = sum((cache.cache_root() / rel).exists() for _, rel in items) if not refresh else 0
    paths = cache.fetch_many(items, refresh=refresh, n_jobs=n_jobs, missing_ok=True)
    n_404 = sum(p is None for p in paths)

    # ---- 4. parse into per-symbol bar series --------------------------------------------
    per_sym: dict[str, list] = {}
    for (s, m), p in zip(jobs, paths):
        if p is not None:
            per_sym.setdefault(s, []).append(_parse_kline_zip(p, m))
    epoch = pd.Timestamp("1970-01-01")

    def day_str(day) -> str:
        return str((epoch + pd.Timedelta(days=int(day))).date())

    d_lo, d_hi = (t_hist - epoch).days, (t_end - epoch).days
    series, sym_of, splits, n_dupes = {}, {}, [], 0
    for s in in_range:
        if s not in per_sym:
            continue
        d, c, v, h = (np.concatenate([x[i] for x in per_sym[s]]) for i in range(4))
        order = np.argsort(d, kind="stable")
        d, c, v, h = d[order], c[order], v[order], h[order]
        keep = np.ones(len(d), dtype=bool)
        keep[:-1] = d[:-1] != d[1:]  # duplicated day: keep the last occurrence
        n_dupes += int((~keep).sum())
        d, c, v, h = d[keep], c[keep], v[keep], h[keep]
        sel = (d >= d_lo) & (d <= d_hi)
        d, c, v, h = d[sel], c[sel], v[sel], h[sel]
        c = np.where(c > 0, c, np.nan)  # a non-positive close is a data error
        if not len(d):
            continue
        gap = np.diff(d) - 1
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = c[1:] / c[:-1]
        by_gap = gap > max_gap
        by_jump = np.zeros_like(by_gap)
        if max_jump is not None:  # NaN ratios compare False -> no split
            by_jump = (gap >= 1) & ((ratio > max_jump) | (ratio < 1.0 / max_jump))
        brk = np.flatnonzero(by_gap | by_jump) + 1
        base = s[: -len(quote)]
        for k, seg in enumerate(np.split(np.arange(len(d)), brk)):
            aid = base if k == 0 else f"{base}_{k + 1}"
            series[aid] = (d[seg], c[seg], v[seg], h[seg])
            sym_of[aid] = s
            if k:
                i = seg[0] - 1  # index into gap/ratio of the break before this segment
                pr = float(ratio[i])
                splits.append({"asset": aid, "symbol": s, "first_bar": day_str(d[seg][0]),
                               "gap_days": int(gap[i]),
                               "reason": "gap" if by_gap[i] else "price_jump",
                               "price_ratio": float(f"{pr:.6g}") if np.isfinite(pr) else None})
    if not series:
        raise ValueError("no kline data in the requested range")

    # ---- 5. calendar-day panel ------------------------------------------------------------
    last_day = min(d_hi, max(int(x[0].max()) for x in series.values()))
    ndays = last_day - d_lo + 1
    ids = sorted(series)
    C = np.full((ndays, len(ids)), np.nan)
    V = np.full((ndays, len(ids)), np.nan)
    H = np.full((ndays, len(ids)), np.nan)
    for j, aid in enumerate(ids):
        d, c, v, h = series[aid]
        m = d <= last_day
        C[d[m] - d_lo, j] = c[m]
        V[d[m] - d_lo, j] = v[m]
        H[d[m] - d_lo, j] = h[m]
    ret = np.full_like(C, np.nan)
    ret[1:] = C[1:] / C[:-1] - 1.0  # NaN unless both consecutive days have a bar
    has_ret = ~np.isnan(ret).all(axis=0)
    ids = [a for a, hr in zip(ids, has_ret) if hr]
    ret, V, H = ret[:, has_ret], V[:, has_ret], H[:, has_ret]
    index = pd.date_range(t_hist, periods=ndays, freq="D", name="date")
    with np.errstate(invalid="ignore"):
        partial = H < _PARTIAL_BAR_HOURS  # NaN (no bar / unknown length) -> False
    partial_bars = [{"asset": ids[j], "date": str(index[i].date()),
                     "hours": round(float(H[i, j]), 3)}
                    for i, j in sorted(zip(*np.nonzero(partial)), key=lambda x: (ids[x[1]], x[0]))]

    # ---- 6. point-in-time universe ---------------------------------------------------------
    med = (pd.DataFrame(V).rolling(volume_window, min_periods=volume_window).median()
           .to_numpy())
    eligible = (~np.isnan(med) & ~np.isnan(ret) & ~partial
                & np.asarray(index >= t_start)[:, None])
    score = np.where(eligible, med, -np.inf)
    order = np.argsort(-score, axis=1, kind="stable")  # ids sorted -> alphabetical tie-break
    rank = np.empty_like(order)
    rows = np.arange(order.shape[0])[:, None]
    rank[rows, order] = np.arange(order.shape[1])[None, :]
    trad = eligible & (rank < top)

    R = pd.DataFrame(ret, index=index, columns=pd.Index(ids))
    T = pd.DataFrame(trad, index=index, columns=pd.Index(ids))
    meta = {
        "source": "Binance Vision spot monthly 1d klines",
        "url_pattern": BINANCE_BASE + KLINE_PREFIX + "{SYMBOL}/1d/{SYMBOL}-1d-{YYYY}-{MM}.zip",
        "listing": BINANCE_LISTING + "?prefix=" + KLINE_PREFIX,
        "quote": quote, "top": top, "volume_window": volume_window, "max_gap": max_gap,
        "max_jump": max_jump, "exclude_derivatives": bool(exclude_derivatives),
        "history_start": str(t_hist.date()), "start": str(t_start.date()),
        "end": str(index[-1].date()), "requested_end": str(t_end.date()),
        "periods_per_year": 365,
        "units": "simple close-to-close returns, decimal; UTC calendar days",
        "universe_rule": (f"top {top} by trailing {volume_window}-day median quote volume among "
                          f"assets with a bar on each of the last {volume_window} days, an "
                          f"observed return at t and a full 24 h bar at t; t >= start"),
        "n_listed_symbols": len(all_syms), "n_candidates": len(cands),
        "n_symbols_in_range": len(in_range), "n_assets": len(ids),
        "excluded": excluded, "stable_bases": sorted(STABLE_BASES),
        "derivative_bases": sorted(DERIVATIVE_BASES) if exclude_derivatives else [],
        "symbol": {a: sym_of[a] for a in ids}, "ticker_splits": splits,
        "partial_bars": partial_bars,
        "n_files": len(items), "n_files_cached_before": int(n_cached), "n_files_404": int(n_404),
        "n_duplicate_bars_dropped": n_dupes,
        "licence": BINANCE_LICENCE, "licence_url": BINANCE_BASE,
        "citation": BINANCE_CITATION,
    }
    return Panel(R, T, name=f"binance_spot_{quote.lower()}_top{top}", freq="D", meta=meta)
