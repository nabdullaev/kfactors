"""Tests for kfactors.data.loaders.

Offline tests serve the tiny fixtures in tests/fixtures/ (regenerate with
tests/fixtures/make_fixtures.py) through a fake HTTP layer (``cache._http_get`` is monkeypatched),
so the real cache/manifest code path is exercised. Tests marked ``network`` hit the real
endpoints (run with ``-m network``).
"""
from __future__ import annotations

import hashlib
import io
import threading
import zipfile
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import numpy as np
import pandas as pd
import pytest

from kfactors.data import cache, loaders
from kfactors.data.loaders import (
    binance_candidates,
    load_binance,
    load_french,
    load_jkp,
    parse_french_csv,
    parse_kline_csv,
)
from kfactors.data.panel import Panel

FIX = Path(__file__).resolve().parent / "fixtures"
JKP_URL = ("https://jkpfactors-data.s3.amazonaws.com/public/"
           "%5Busa%5D_%5Ball_factors%5D_%5Bmonthly%5D_%5Bvw_cap%5D.zip")


class FixtureServer:
    """Routes real loader URLs to fixture files; everything else is a 404."""

    def __init__(self):
        self.calls = []
        self.lock = threading.Lock()
        self.overrides = {}

    def __call__(self, url, timeout):
        with self.lock:
            self.calls.append(url)
        if url in self.overrides:
            return 200, self.overrides[url], {}
        path = self.route(url)
        if path is None or not path.exists():
            return 404, b"", {}
        return 200, path.read_bytes(), {}

    @staticmethod
    def route(url):
        if url == JKP_URL:
            return FIX / "jkp" / "usa_all_factors_monthly_vw_cap.zip"
        if url.startswith(loaders.FRENCH_BASE):
            return FIX / "french" / url[len(loaders.FRENCH_BASE):]
        if url.startswith(loaders.BINANCE_LISTING):
            q = parse_qs(urlparse(url).query)
            prefix = q["prefix"][0]
            if prefix == loaders.KLINE_PREFIX:
                return FIX / "binance" / "listing" / ("root_p2.xml" if "marker" in q
                                                      else "root_p1.xml")
            sym = prefix[len(loaders.KLINE_PREFIX):].split("/")[0]
            return FIX / "binance" / "listing" / f"{sym}.xml"
        if url.startswith(loaders.BINANCE_BASE):
            return FIX / "binance" / "klines" / url.rsplit("/", 1)[1]
        return None

    def data_calls(self):
        return [u for u in self.calls if u.startswith(loaders.BINANCE_BASE)]


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("KFACTORS_DATA", str(tmp_path / "cache"))
    monkeypatch.setattr(cache, "_sleep", lambda s: None)
    srv = FixtureServer()
    monkeypatch.setattr(cache, "_http_get", srv)
    return srv


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ============================================================================ JKP
def test_jkp_known_answer(server):
    import sys
    sys.path.insert(0, str(FIX))
    try:
        from make_fixtures import JKP_DATES, JKP_FACTORS, jkp_value
    finally:
        sys.path.pop(0)
    P = load_jkp()
    assert isinstance(P, Panel) and P.freq == "M" and P.periods_per_year() == 12
    assert P.shape == (6, 3)
    assert list(P.returns.columns) == sorted(JKP_FACTORS)
    assert list(P.returns.index) == list(pd.to_datetime(JKP_DATES))
    for j, name in enumerate(JKP_FACTORS):
        for i in range(6):
            v = P.returns[name].iloc[i]
            if name == "late_factor" and i < 2:
                assert np.isnan(v)
            else:
                assert v == jkp_value(j, i)  # exact: decimals are passed through unchanged
    pd.testing.assert_frame_equal(P.tradable, P.returns.notna())
    m = P.meta
    assert m["licence"] == "CC BY-NC 4.0" and "Jensen" in m["citation"] and "2023" in m["citation"]
    assert m["url"] == JKP_URL
    assert m["sha256"] == _sha(FIX / "jkp" / "usa_all_factors_monthly_vw_cap.zip")
    assert m["first_complete_date"] == "2020-03-31"
    assert m["direction"]["age"] == -1 and m["direction"]["be_me"] == 1
    assert "decimal" in m["units"]


def test_jkp_start_end_and_cache_reuse(server):
    P = load_jkp(start="2020-03-01", end="2020-05-31")
    assert P.shape == (3, 3) and not P.returns.isna().any().any()
    n = len(server.calls)
    load_jkp()
    assert len(server.calls) == n  # served from cache


def test_jkp_argument_validation(server):
    with pytest.raises(ValueError, match="freq"):
        load_jkp(freq="weekly")
    with pytest.raises(ValueError, match="weighting"):
        load_jkp(weighting="cap")
    with pytest.raises(ValueError, match="region"):
        load_jkp(region="../usa")


def test_jkp_rejects_mismatched_file(server):
    # a file whose 'weighting' column disagrees with the request must not load silently
    raw = zipfile.ZipFile(FIX / "jkp" / "usa_all_factors_monthly_vw_cap.zip")
    txt = raw.read(raw.namelist()[0]).decode().replace(",vw_cap,", ",ew,")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[usa]_[all_factors]_[monthly]_[vw_cap].csv", txt)
    server.overrides[JKP_URL] = buf.getvalue()
    with pytest.raises(ValueError, match="weighting"):
        load_jkp()


# ============================================================================ French
def test_parse_french_blocks_structure():
    raw = zipfile.ZipFile(FIX / "french" / "49_Industry_Portfolios_CSV.zip")
    text = raw.read(raw.namelist()[0]).decode("latin-1")
    blocks = parse_french_csv(text)
    titles = [b["title"] for b in blocks]
    assert titles == ["Average Value Weighted Returns -- Monthly",
                      "Average Equal Weighted Returns -- Monthly",
                      "Average Value Weighted Returns -- Annual",
                      "Number of Firms in Portfolios"]
    assert blocks[0]["columns"] == ["Agric", "Food", "Soda", "Beer"]   # trailing blanks stripped
    assert blocks[2]["dates"] == ["1927", "2025"]
    assert blocks[0]["values"].shape == (5, 4)


def test_french_daily_known_answer_and_sentinels(server):
    import sys
    sys.path.insert(0, str(FIX))
    try:
        from make_fixtures import FR_COLS_100, FR_DAILY_DATES, FR_DAILY_EW, FR_DAILY_VW
    finally:
        sys.path.pop(0)
    P = load_french("100_size_bm", "daily")
    assert P.freq == "D" and P.periods_per_year() == 252
    assert list(P.returns.columns) == FR_COLS_100
    assert list(P.returns.index) == list(pd.to_datetime(FR_DAILY_DATES, format="%Y%m%d"))
    exp = np.array(FR_DAILY_VW, dtype=float)
    miss = np.isclose(exp, -99.99) | np.isclose(exp, -999)
    exp = np.where(miss, np.nan, exp / 100)
    np.testing.assert_allclose(P.returns.values, exp, rtol=0, atol=1e-15)
    assert P.returns.iloc[5, 3] == -1.0              # a genuine -100% is kept
    assert P.meta["n_missing_cells"] == 3
    pd.testing.assert_frame_equal(P.tradable, P.returns.notna())
    assert P.meta["block"] == "Average Value Weighted Returns -- Daily"
    assert P.meta["sha256"] == _sha(FIX / "french" / "100_Portfolios_10x10_Daily_CSV.zip")
    E = load_french("100_size_bm", "daily", "ew")
    exp_ew = np.array(FR_DAILY_EW, dtype=float)
    exp_ew = np.where(np.isclose(exp_ew, -99.99) | np.isclose(exp_ew, -999), np.nan, exp_ew / 100)
    np.testing.assert_allclose(E.returns.values, exp_ew, rtol=0, atol=1e-15)


def test_french_monthly_cross_check_independent_parser(server):
    """Cross-check against a plain pandas read of the block located by line numbers."""
    P = load_french("49_industries", "monthly")
    assert P.freq == "M"
    assert list(P.returns.index) == list(pd.to_datetime(
        ["1926-07-31", "1926-08-31", "1926-09-30", "2026-08-31", "2026-09-30"]))
    raw = zipfile.ZipFile(FIX / "french" / "49_Industry_Portfolios_CSV.zip")
    lines = raw.read(raw.namelist()[0]).decode("latin-1").splitlines()
    h = lines.index("  Average Value Weighted Returns -- Monthly") + 1
    ref = pd.read_csv(io.StringIO("\n".join(lines[h:h + 6])), index_col=0)
    ref.columns = ref.columns.str.strip()
    ref = ref.replace([-99.99, -999], np.nan) / 100
    np.testing.assert_allclose(P.returns.values, ref.values, rtol=0, atol=1e-15)
    assert list(P.returns.columns) == list(ref.columns)
    assert P.returns["Soda"].isna().sum() == 3
    S = load_french("49_industries", "monthly", start="2026-01-01")
    assert S.shape == (2, 4)


def test_french_errors(server, tmp_path):
    with pytest.raises(ValueError, match="dataset"):
        load_french("6_size_bm")
    with pytest.raises(ValueError, match="freq"):
        load_french(freq="weekly")
    with pytest.raises(ValueError, match="weighting"):
        load_french(weighting="xx")
    # 25_size_bm daily is not in the fixtures -> the fake server answers 404
    with pytest.raises(cache.DownloadNotFound):
        load_french("25_size_bm", "daily")
    # impossible value (< -100% and not a sentinel) must raise
    raw = zipfile.ZipFile(FIX / "french" / "49_Industry_Portfolios_CSV.zip")
    txt = raw.read(raw.namelist()[0]).decode("latin-1").replace("   2.36,", "-150.00,", 1)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("49_Industry_Portfolios.csv", txt)
    server.overrides[loaders.FRENCH_BASE + "49_Industry_Portfolios_CSV.zip"] = buf.getvalue()
    with pytest.raises(ValueError, match="below -100%"):
        load_french("49_industries", "monthly", refresh=True)


def test_french_missing_block_raises():
    with pytest.raises(ValueError, match="expected one block"):
        loaders._french_block([{"title": "Average Equal Weighted Returns -- Daily"}], "vw", "daily")


# ============================================================================ Binance helpers
def test_binance_candidates_filters():
    syms = ["BTCUSDT", "BTCUPUSDT", "BTCDOWNUSDT", "BULLUSDT", "BEARUSDT", "ETHBULLUSDT",
            "ETHUSDT", "JUPUSDT", "SYRUPUSDT", "USDCUSDT", "FDUSDUSDT", "EURUSDT", "PAXGUSDT",
            "RWUSDUSDT", "ETHBTC", "USDT", "SUPERUSDT"]
    keep, ex = binance_candidates(syms, "USDT")
    assert keep == ["BTCUSDT", "ETHUSDT", "JUPUSDT", "SUPERUSDT", "SYRUPUSDT"]
    assert ex["leveraged"] == ["BEARUSDT", "BTCDOWNUSDT", "BTCUPUSDT", "BULLUSDT", "ETHBULLUSDT"]
    assert ex["stable"] == ["EURUSDT", "FDUSDUSDT", "PAXGUSDT", "RWUSDUSDT", "USDCUSDT"]
    assert ex["derivative"] == []


def test_binance_candidates_derivatives_and_frax():
    """Regression RD-2 / RD-5: wrapped/staked duplicates dropped by default (opt-out), FRAX kept."""
    syms = ["BTCUSDT", "WBTCUSDT", "ETHUSDT", "WBETHUSDT", "BETHUSDT", "SOLUSDT", "BNSOLUSDT",
            "BTCSTUSDT", "ETHFIUSDT", "SOLVUSDT", "WNXMUSDT", "FRAXUSDT", "FXSUSDT", "FRXUSDUSDT",
            "USTCUSDT"]
    keep, ex = binance_candidates(syms, "USDT")
    assert ex["derivative"] == ["BETHUSDT", "BNSOLUSDT", "WBETHUSDT", "WBTCUSDT"]
    assert keep == ["BTCSTUSDT", "BTCUSDT", "ETHFIUSDT", "ETHUSDT", "FRAXUSDT", "FXSUSDT",
                    "SOLUSDT", "SOLVUSDT", "WNXMUSDT"]
    assert ex["stable"] == ["FRXUSDUSDT", "USTCUSDT"]
    keep2, ex2 = binance_candidates(syms, "USDT", exclude_derivatives=False)
    assert ex2["derivative"] == [] and set(keep2) == set(keep) | set(ex["derivative"])
    assert "FRAX" not in loaders.STABLE_BASES
    assert loaders.DERIVATIVE_BASES.isdisjoint(loaders.STABLE_BASES)


def test_parse_kline_csv_units_and_header():
    day = 20089  # 2025-01-01
    rows = [f"{day * 86400 * 10 ** 6},1,1,1,2.5,1,0,10.5,1,1,1,0",           # microseconds
            f"{(day + 1) * 86400 * 10 ** 3},1,1,1,3.5,1,0,11.5,1,1,1,0",     # milliseconds
            f"{(day + 2) * 86400},1,1,1,4.5,1,0,12.5,1,1,1,0"]               # seconds
    hdr = ",".join(loaders.KLINE_COLUMNS)
    for text in ("\n".join(rows), hdr + "\n" + "\n".join(rows) + "\n\n"):
        d, c, v = parse_kline_csv(text.encode())
        np.testing.assert_array_equal(d, [day, day + 1, day + 2])
        np.testing.assert_array_equal(c, [2.5, 3.5, 4.5])
        np.testing.assert_array_equal(v, [10.5, 11.5, 12.5])
    with pytest.raises(ValueError, match="open_time"):
        parse_kline_csv(b"1,2,3,4,5,6,7,8,9,10,11,12\nfoo,1,1,1,1,1,1,1,1,1,1,1")
    with pytest.raises(ValueError, match="fields"):
        parse_kline_csv(b"1,2,3")


def test_parse_kline_csv_bar_hours():
    """Regression RD-4: bar length from close_time (ms and us), NaN when close_time is unusable."""
    day = 20089  # 2025-01-01
    ms, us = day * 86_400_000, day * 86_400_000_000
    rows = [f"{ms},1,1,1,1,1,{ms + 86_400_000 - 1},1,1,1,1,0",                # full day, ms
            f"{us},1,1,1,1,1,{us + 3 * 3_600_000_000 - 1},1,1,1,1,0",        # 3 h, us
            f"{ms},1,1,1,1,1,0,1,1,1,1,0",                                    # close <= open
            f"{ms},1,1,1,1,1,x,1,1,1,1,0"]                                    # not an integer
    d, c, v, h = parse_kline_csv("\n".join(rows).encode(), with_hours=True)
    np.testing.assert_allclose(h[:2], [24.0, 3.0])
    assert np.isnan(h[2:]).all()
    assert len(parse_kline_csv("\n".join(rows).encode())) == 3   # default: 3-tuple as before


def test_parse_kline_zip_month_check(tmp_path):
    p = tmp_path / "X-1d-2025-01.zip"
    with zipfile.ZipFile(p, "w") as zf:  # bar of 2024-12-31 in the January file -> error
        zf.writestr("X-1d-2025-01.csv", f"{20088 * 86400000},1,1,1,1,1,0,1,1,1,1,0\n")
    with pytest.raises(ValueError, match="outside 2025-01"):
        loaders._parse_kline_zip(p, "2025-01")


def test_s3_listing_pagination(server):
    prefixes, keys = loaders._s3_list(loaders.KLINE_PREFIX)
    syms = [p.split("/")[-2] for p in prefixes]
    assert syms == sorted(syms) and len(syms) == 11 and keys == []
    listing_calls = [u for u in server.calls if u.startswith(loaders.BINANCE_LISTING)]
    assert len(listing_calls) == 2 and "marker=" in listing_calls[1]
    _, keys = loaders._s3_list(loaders.KLINE_PREFIX + "AAAUSDT/1d/")
    assert len(keys) == 6  # 3 zips + 3 CHECKSUM files


# ============================================================================ Binance loader
KW = {"start": "2024-12-01", "end": "2025-01-31", "history_start": "2024-11-01", "n_jobs": 4}


def test_binance_panel_known_answer(server):
    P = load_binance(top=2, **KW)
    assert P.freq == "D" and P.periods_per_year() == 365
    idx = pd.date_range("2024-11-01", "2025-01-31", freq="D")
    assert P.returns.index.equals(idx)
    assert list(P.returns.columns) == ["AAA", "BBB", "CCC", "DDD", "DDD_2", "EEE"]
    R = P.returns
    # AAA: +1% every day, including across the 2024-12-31 -> 2025-01-01 ms -> us switch
    np.testing.assert_allclose(R["AAA"].iloc[1:], 0.01, atol=1e-9)
    assert np.isnan(R["AAA"].iloc[0])  # no previous close inside the panel
    # BBB alternates 50 / 55
    np.testing.assert_allclose(R["BBB"].iloc[1:3], [0.1, 50 / 55 - 1], atol=1e-12)
    # CCC listed 2024-12-10: no returns before 2024-12-11
    assert R["CCC"].first_valid_index() == pd.Timestamp("2024-12-11")
    # EEE misses 2024-12-15: returns of 12-15 and 12-16 are NaN (no multi-day return)
    assert R.loc["2024-12-14":"2024-12-17", "EEE"].isna().tolist() == [False, True, True, False]
    # DDD split by the 39-day gap; no return spans the gap
    assert R["DDD"].last_valid_index() == pd.Timestamp("2024-11-10")
    assert R["DDD_2"].first_valid_index() == pd.Timestamp("2024-12-21")
    assert R["DDD"].loc["2024-11-11":].isna().all()
    m = P.meta
    assert m["ticker_splits"] == [{"asset": "DDD_2", "symbol": "DDDUSDT",
                                   "first_bar": "2024-12-20", "gap_days": 39, "reason": "gap",
                                   "price_ratio": 0.1}]
    assert m["symbol"]["DDD_2"] == "DDDUSDT" and m["symbol"]["AAA"] == "AAAUSDT"
    assert m["excluded"] == {"stable": ["USDCUSDT"], "leveraged": ["BTCUPUSDT", "BULLUSDT"],
                             "derivative": []}
    assert m["partial_bars"] == []
    assert m["licence"].startswith("CC BY-NC-SA 4.0") and "binance.vision" in m["citation"]
    assert m["n_listed_symbols"] == 11 and m["n_candidates"] == 7    # incl. BTC, JUP
    assert m["n_symbols_in_range"] == 6                              # BTC has only 2019 data
    assert m["n_files_404"] == 1                                     # JUP listed but missing
    assert m["n_files"] == 15


def test_binance_point_in_time_universe(server):
    P = load_binance(top=2, **KW)
    T = P.tradable
    assert not T.loc[:"2024-11-30"].any().any()                     # nothing before start

    def members(day):
        row = T.loc[day]
        return sorted(row.index[row.values])

    assert members("2024-12-01") == ["AAA", "BBB"]     # CCC/DDD_2 not listed yet
    assert members("2025-01-07") == ["AAA", "BBB"]     # CCC has 29 days of volume only
    assert members("2025-01-08") == ["BBB", "CCC"]     # CCC: full 30-day window (12-10..01-08)
    assert members("2025-01-17") == ["BBB", "CCC"]
    assert members("2025-01-18") == ["CCC", "DDD_2"]   # DDD_2 window 12-20..01-18 complete
    assert (T.sum(axis=1).loc["2024-12-01":] == 2).all()
    # tradable implies an observed return
    assert not (T & P.returns.isna()).any().any()
    # EEE (smallest volume) is never top-2; with top=10 it is excluded while its 30-day window
    # contains the missing 2024-12-15 bar (12-15 .. 01-13)
    Q = load_binance(top=10, **KW)
    e = Q.tradable["EEE"]
    assert e.loc["2024-12-01":"2024-12-14"].all()
    assert not e.loc["2024-12-15":"2025-01-13"].any()
    assert e.loc["2025-01-14":].all()


def test_binance_no_look_ahead(server):
    """tradable/returns up to t must not depend on data after t (truncation + mutation)."""
    full = load_binance(top=2, **KW)
    short = load_binance(top=2, start="2024-12-01", end="2025-01-12",
                         history_start="2024-11-01")
    pd.testing.assert_frame_equal(full.tradable.loc[:"2025-01-12"], short.tradable)
    pd.testing.assert_frame_equal(full.returns.loc[:"2025-01-12"], short.returns)
    # mutate every January bar of BBB (huge volume) -> December universe unchanged
    zp = FIX / "binance" / "klines" / "BBBUSDT-1d-2025-01.zip"
    zf = zipfile.ZipFile(zp)
    rows = zf.read(zf.namelist()[0]).decode().splitlines()
    rows = [",".join(f[:7] + ["1e12"] + f[8:]) for f in (r.split(",") for r in rows)]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as out:
        out.writestr("BBBUSDT-1d-2025-01.csv", "\n".join(rows))
    server.overrides[loaders.BINANCE_BASE + loaders.KLINE_PREFIX
                     + "BBBUSDT/1d/BBBUSDT-1d-2025-01.zip"] = buf.getvalue()
    mutated = load_binance(top=2, refresh=True, **KW)
    pd.testing.assert_frame_equal(mutated.tradable.loc[:"2024-12-31"],
                                  full.tradable.loc[:"2024-12-31"])
    assert mutated.tradable.loc["2025-01-18", "BBB"]  # the change does act from January on


def test_binance_caching_only_downloads_missing_months(server):
    load_binance(top=2, **KW)
    n_data, n_all = len(server.data_calls()), len(server.calls)
    assert n_data == 15
    P = load_binance(top=2, **KW)
    assert len(server.calls) == n_all + 1     # only JUP's 404 month is re-requested
    assert P.meta["n_files_cached_before"] == 14
    # extending the history fetches nothing new (no months before 2024-11 are listed)
    load_binance(top=2, start="2024-12-01", end="2025-01-31", history_start="2024-10-01")
    assert len(server.data_calls()) == n_data + 2


def test_binance_max_symbols_and_explicit_symbols(server):
    P = load_binance(top=5, max_symbols=2, **KW)
    assert list(P.returns.columns) == ["AAA", "BBB"]   # alphabetical among in-range symbols
    Q = load_binance(top=5, symbols=["cccusdt", "USDCUSDT", "AAAUSDT"], **KW)
    assert list(Q.returns.columns) == ["AAA", "CCC"]
    assert Q.meta["excluded"]["stable"] == ["USDCUSDT"]
    with pytest.raises(ValueError, match="eligible"):
        load_binance(symbols=["USDCUSDT"], **KW)
    with pytest.raises(ValueError, match="no USDT symbol"):
        load_binance(symbols=["BTCUSDT"], **KW)


def test_binance_argument_validation(server):
    with pytest.raises(ValueError, match="history_start"):
        load_binance(start="2024-10-01", history_start="2024-11-01", end="2025-01-31")
    with pytest.raises(ValueError):
        load_binance(top=0, **KW)


def test_binance_end_clipped_to_last_bar(server):
    P = load_binance(top=2, start="2024-12-01", end="2025-03-31", history_start="2024-11-01")
    assert P.returns.index[-1] == pd.Timestamp("2025-01-31")
    assert P.meta["requested_end"] == "2025-03-31"


def _serve_synthetic(server, sym, bars):
    """Serve a synthetic symbol: ``bars`` = [(day, close, quote_volume, bar_hours), ...]."""
    by_month = {}
    for day, c, qv, hrs in bars:
        ms = int((day - pd.Timestamp("1970-01-01")).total_seconds() * 1000)
        ct = ms + int(round(hrs * 3_600_000)) - 1
        by_month.setdefault(str(day.to_period("M")), []).append(
            f"{ms},{c},{c},{c},{c},1,{ct},{qv},1,1,1,0")
    keys = "".join(f"<Contents><Key>{loaders.KLINE_PREFIX}{sym}/1d/{sym}-1d-{m}.zip</Key>"
                   f"</Contents>" for m in sorted(by_month))
    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n<ListBucketResult '
           'xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><IsTruncated>false</IsTruncated>'
           f"{keys}</ListBucketResult>")
    q = urlencode({"prefix": f"{loaders.KLINE_PREFIX}{sym}/1d/", "delimiter": "/"})
    server.overrides[f"{loaders.BINANCE_LISTING}?{q}"] = xml.encode()
    for m, rows in by_month.items():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(f"{sym}-1d-{m}.csv", "\n".join(rows) + "\n")
        server.overrides[f"{loaders.BINANCE_BASE}{loaders.KLINE_PREFIX}{sym}/1d/{sym}-1d-{m}.zip"
                         ] = buf.getvalue()


@pytest.fixture
def luna_like(server):
    """LLL: classic chain to 2024-11-20 (last bar 0.7 h, -80%), 17-day gap, relaunch at 500x
    (the LUNA -> LUNA 2.0 pattern). NNN: 19-day gap with a 0.86x price ratio (NBT-like, same
    asset) and a -98.8% one-day crash on consecutive bars. WBTC: wrapped duplicate."""
    days = pd.date_range("2024-11-01", "2025-01-31", freq="D")
    first = days[days <= "2024-11-20"]
    lll = [(d, 10.0, 100.0, 24.0) for d in first[:-1]] + [(first[-1], 2.0, 100.0, 0.7)]
    lll += [(d, 1000.0, 100.0, 24.0) for d in days[days >= "2024-12-08"]]
    nnn = [(d, 10.0, 200.0, 24.0) for d in first]
    nnn += [(d, 8.6 if d < pd.Timestamp("2025-01-10") else 0.1, 200.0, 24.0)
            for d in days[days >= "2024-12-10"]]
    wbtc = [(d, 50.0 + (d.day % 2), 300.0, 24.0) for d in days]
    for sym, bars in (("LLLUSDT", lll), ("NNNUSDT", nnn), ("WBTCUSDT", wbtc)):
        _serve_synthetic(server, sym, bars)
    return {"top": 10, "start": "2024-11-10", "end": "2025-01-31", "history_start": "2024-11-01",
            "volume_window": 5, "symbols": ["LLLUSDT", "NNNUSDT", "WBTCUSDT"], "n_jobs": 2}


def test_binance_price_jump_split_partial_bar_and_derivatives(luna_like):
    """Regression RD-1 (LUNA/LUNA 2.0 not split by max_gap=30), RD-2 (WBTC kept), RD-4 (partial
    last bar tradable)."""
    P = load_binance(**luna_like)
    R, T, m = P.returns, P.tradable, P.meta
    assert list(R.columns) == ["LLL", "LLL_2", "NNN"]
    assert m["ticker_splits"] == [{"asset": "LLL_2", "symbol": "LLLUSDT",
                                   "first_bar": "2024-12-08", "gap_days": 17,
                                   "reason": "price_jump", "price_ratio": 500.0}]
    assert m["max_gap"] == 30 and m["max_jump"] == 5.0
    assert R["LLL"].last_valid_index() == pd.Timestamp("2024-11-20")
    assert R["LLL_2"].first_valid_index() == pd.Timestamp("2024-12-09")
    # NNN: 19-day gap at 0.86x is not split; the consecutive-day crash is a return, not a split
    assert R.loc["2025-01-10", "NNN"] == pytest.approx(0.1 / 8.6 - 1)
    # partial last bar: return kept, asset not tradable that day (cannot be held to t+1)
    assert R.loc["2024-11-20", "LLL"] == pytest.approx(-0.8)
    assert T.loc["2024-11-19", "LLL"] and not T.loc["2024-11-20", "LLL"]
    assert m["partial_bars"] == [{"asset": "LLL", "date": "2024-11-20", "hours": 0.7}]
    assert not (T & R.isna()).any().any()
    # wrapped duplicate excluded by default and reported
    assert m["excluded"]["derivative"] == ["WBTCUSDT"] and m["exclude_derivatives"] is True
    # opt-outs restore the previous behaviour
    Q = load_binance(max_jump=None, exclude_derivatives=False, **luna_like)
    assert list(Q.returns.columns) == ["LLL", "NNN", "WBTC"]
    assert Q.meta["ticker_splits"] == [] and Q.meta["excluded"]["derivative"] == []
    # a short max_gap still splits on the gap, reported as such
    G = load_binance(max_gap=10, **luna_like)
    assert [(s["asset"], s["reason"]) for s in G.meta["ticker_splits"]] == [
        ("LLL_2", "gap"), ("NNN_2", "gap")]
    with pytest.raises(ValueError, match="max_jump"):
        load_binance(max_jump=1.0, **luna_like)


# ============================================================================ network
@pytest.fixture(scope="module")
def net_cache(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setenv("KFACTORS_DATA", str(tmp_path_factory.mktemp("kf_net_cache")))
    yield
    mp.undo()


@pytest.mark.network
def test_network_jkp_usa_monthly(net_cache):
    P = load_jkp("usa", "monthly", "vw_cap")
    assert P.shape[1] >= 150 and P.shape[0] >= 1200
    assert P.returns.index[0] == pd.Timestamp("1926-01-31")
    assert P.meta["first_complete_date"] == "1971-11-30"
    assert not P.returns.loc["1971-11-30":].isna().any().any()
    assert P.returns.abs().max().max() < 1.0  # decimals, not percent
    assert len(P.meta["sha256"]) == 64 and P.meta["bytes"] > 1_000_000


@pytest.mark.network
def test_network_french_49_monthly(net_cache):
    P = load_french("49_industries", "monthly")
    assert P.shape[1] == 49 and P.returns.index[0] == pd.Timestamp("1926-07-31")
    assert not P.returns.loc["1970":].isna().any().any()
    assert 0.02 < P.returns.loc["1970":].std().median() < 0.15  # monthly decimal vol
    E = load_french("49_industries", "monthly", "ew")
    assert not np.allclose(E.returns.fillna(0).values, P.returns.fillna(0).values)


@pytest.mark.network
def test_network_binance_small(net_cache):
    P = load_binance(top=2, symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"], history_start="2024-12-01",
                     start="2024-12-31", end="2025-01-31", n_jobs=4)
    assert list(P.returns.columns) == ["BTC", "ETH", "SOL"]
    assert P.returns.index[0] == pd.Timestamp("2024-12-01")
    assert P.returns.index[-1] == pd.Timestamp("2025-01-31")
    assert P.returns.iloc[1:].notna().all().all()              # continuous across ms -> us
    assert P.returns.abs().max().max() < 0.5
    assert (P.tradable.loc["2024-12-31":].sum(axis=1) == 2).all()
    assert P.tradable.loc["2025-01-15", "BTC"]                  # BTC is the most liquid
    assert P.meta["n_files"] == 6 and P.meta["n_files_404"] == 0


def test_binance_listing_cache_freshness(server, tmp_path, monkeypatch):
    import json
    from datetime import UTC, datetime

    load_binance(top=2, **KW)

    def listing_calls():
        return sum(u.startswith(loaders.BINANCE_LISTING) for u in server.calls)

    n = listing_calls()
    load_binance(top=2, **KW)                       # listings fetched < 1 day ago: reused
    assert listing_calls() == n
    root = Path(cache.cache_root()) / "binance" / "listings"
    files = [root / "spot_monthly_klines.json", *sorted((root / "1d").glob("*.json"))]
    # pretend every listing was taken on 2025-01-15, before Jan 2025 could be complete
    for f in files:
        info = json.loads(f.read_text())
        info["fetched_at"] = "2025-01-15T00:00:00+00:00"
        f.write_text(json.dumps(info))
    load_binance(top=2, **KW)                       # stale -> re-listed (root: 2 pages)
    assert listing_calls() == n + 2 + (len(files) - 1)
    m = listing_calls()
    # a listing taken after end-of-month + publication lag is final: never re-listed
    monkeypatch.setattr(loaders, "_now", lambda: datetime(2030, 1, 1, tzinfo=UTC))
    load_binance(top=2, **KW)                       # last listing (today) > 2025-01-31 + 7 d
    assert listing_calls() == m
    # corrupt listing json is rebuilt instead of crashing
    files[0].write_text("{oops")
    load_binance(top=2, **KW)
    assert listing_calls() == m + 2
