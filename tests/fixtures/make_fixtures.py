"""Regenerate the tiny offline fixtures used by tests/test_loaders.py.

Run from the repo root:  .venv/bin/python tests/fixtures/make_fixtures.py

The files mimic the real formats verified on 2026-11-03:

* JKP: zip with one CSV ``[usa]_[all_factors]_[monthly]_[vw_cap].csv``; long format, header
  ``location,name,freq,weighting,direction,n_stocks,n_stocks_min,date,ret``; decimal returns;
  missing observations are absent rows.
* Ken French: zip with one CSV, CRLF line endings, a free-text preamble, blocks introduced by a
  title line and a header starting with ``,``; dates YYYYMMDD (daily) / YYYYMM (monthly) / YYYY
  (annual); percent values right-aligned with leading blanks; missing = -99.99 or -999.
* Binance Vision: per-month zips with one CSV ``{SYM}-1d-{YYYY}-{MM}.csv`` (12 columns, usually no
  header), open_time in ms until 2024-12 and in us from 2025-01; S3 ListBucketResult XML listings
  (CommonPrefixes for the symbol list with marker pagination; Contents/Key per symbol, including
  ``.CHECKSUM`` keys).

All numbers are synthetic and chosen so that tests can check exact answers (see SPEC below).
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FIXED_ZIP_DATE = (2026, 1, 1, 0, 0, 0)  # deterministic zip bytes


def _zip(path: Path, member: str, text: str | bytes) -> None:
    data = text.encode("latin-1") if isinstance(text, str) else text
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        info = zipfile.ZipInfo(member, date_time=FIXED_ZIP_DATE)
        info.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(info, data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.getvalue())


# ------------------------------------------------------------------------------------------
# JKP: 3 factors x 6 month ends; "late_factor" only from the 3rd month.
# ret(name, i) = (j + 1) / 1000 * (-1) ** i with j the factor position (exact decimals).
# ------------------------------------------------------------------------------------------
JKP_FACTORS = ["age", "be_me", "late_factor"]
JKP_DATES = ["2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30", "2020-05-31", "2020-06-30"]


def jkp_value(j: int, i: int) -> float:
    return (j + 1) / 1000 * (-1) ** i


def make_jkp() -> None:
    lines = ["location,name,freq,weighting,direction,n_stocks,n_stocks_min,date,ret"]
    for j, name in enumerate(JKP_FACTORS):
        for i, d in enumerate(JKP_DATES):
            if name == "late_factor" and i < 2:
                continue
            direction = -1 if name == "age" else 1
            lines.append(f"usa,{name},monthly,vw_cap,{direction},{500 + i},{8 + i},{d},"
                         f"{jkp_value(j, i)!r}")
    _zip(HERE / "jkp" / "usa_all_factors_monthly_vw_cap.zip",
         "[usa]_[all_factors]_[monthly]_[vw_cap].csv", "\n".join(lines) + "\n")


# ------------------------------------------------------------------------------------------
# Ken French
# ------------------------------------------------------------------------------------------
FR_COLS_100 = ["SMALL LoBM", "ME1 BM2", "ME2 BM1", "BIG HiBM"]
FR_DAILY_DATES = ["19260701", "19260702", "19260706", "19260707", "20260929", "20260930"]
# value-weighted daily, percent; -99.99 and -999 are missing
FR_DAILY_VW = [
    [-99.99, 0.00, 1.59, -3.08],
    [-99.99, -0.27, 0.00, -0.97],
    [0.50, 1.01, -999.0, 2.35],
    [-1.25, -1.67, 4.92, 0.51],
    [0.45, -0.06, -0.56, -0.23],
    [0.62, 0.06, 0.43, -100.00],  # a genuine -100% return is NOT a sentinel
]
FR_DAILY_EW = [[v + 0.01 if v > -99 else v for v in row] for row in FR_DAILY_VW]


def _fr_row(date: str, vals) -> str:
    return date + "," + ",".join(f"{v:7.2f}" for v in vals)


def make_french() -> None:
    hdr = "," + ",".join(FR_COLS_100)
    pre = [
        "This file was created using the 202609 CRSP database.",
        "It contains value-weighted returns for the intersections of 10 ME portfolios",
        "and 10 BE/ME portfolios.",
        "",
        "Missing data are indicated by -99.99 or -999.",
        "",
        "",
    ]
    lines = list(pre)
    lines += ["  Average Value Weighted Returns -- Daily", hdr]
    lines += [_fr_row(d, v) for d, v in zip(FR_DAILY_DATES, FR_DAILY_VW)]
    lines += ["", "", "  Average Equal Weighted Returns -- Daily", hdr]
    lines += [_fr_row(d, v) for d, v in zip(FR_DAILY_DATES, FR_DAILY_EW)]
    lines += ["", "", "  Number of Firms in Portfolios", hdr]
    lines += [d + "," + ",".join(f"{k:7d}" for k in (3, 5, 7, 9)) for d in FR_DAILY_DATES]
    lines += ["", "Copyright 2026 Eugene F. Fama and Kenneth R. French", ""]
    _zip(HERE / "french" / "100_Portfolios_10x10_Daily_CSV.zip", "100_Portfolios_10x10_Daily.csv",
         "\r\n".join(lines))

    # monthly 49-industry style: labels padded with blanks, plus annual + firm-count blocks
    cols = ["Agric", "Food ", "Soda ", "Beer "]
    hdr = "," + ",".join(cols)
    months = ["192607", "192608", "192609", "202608", "202609"]
    vw = [[2.36, 0.09, -99.99, -5.19], [2.23, 2.71, -99.99, 27.03], [-0.57, 1.59, -99.99, 4.02],
          [1.00, -1.00, 0.50, 0.25], [0.10, 0.20, 0.30, 0.40]]
    ew = [[v * 2 if v > -99 else v for v in row] for row in vw]
    lines = ["This file was created using the 202609 CRSP database.",
             "It contains value- and equal-weighted returns for 49 industry portfolios.", "",
             "Missing data are indicated by -99.99 or -999.", "", "",
             "  Average Value Weighted Returns -- Monthly", hdr]
    lines += [_fr_row(d, v) for d, v in zip(months, vw)]
    lines += ["", "", "  Average Equal Weighted Returns -- Monthly", hdr]
    lines += [_fr_row(d, v) for d, v in zip(months, ew)]
    lines += ["", "", "  Average Value Weighted Returns -- Annual", hdr]
    lines += [_fr_row(y, v) for y, v in zip(["1927", "2025"], vw[:2])]
    lines += ["", "", "  Number of Firms in Portfolios", hdr]
    lines += [d + "," + ",".join(f"{k:7d}" for k in (1, 2, 0, 4)) for d in months]
    lines += ["", "Copyright 2026 Eugene F. Fama and Kenneth R. French", ""]
    _zip(HERE / "french" / "49_Industry_Portfolios_CSV.zip", "49_Industry_Portfolios.csv",
         "\r\n".join(lines))


# ------------------------------------------------------------------------------------------
# Binance Vision
# ------------------------------------------------------------------------------------------
# SPEC (history 2024-11-01 .. 2025-01-31, quote volumes constant per symbol):
#   AAAUSDT  2024-11-01..2025-01-31, close 100 * 1.01**k   (returns exactly +1%), qv 1000
#            2024-12 file has a header row; 2025-01 file uses microseconds.
#   BBBUSDT  2024-11-01..2025-01-31, close alternates 50 / 55 ,                    qv 2000
#   CCCUSDT  listed 2024-12-10 ..2025-01-31, close 10,                             qv 5000
#   DDDUSDT  2024-11-01..2024-11-10, then 2024-12-20..2025-01-31 (gap 39 d -> DDD_2), qv 3000
#   EEEUSDT  2024-11-01..2025-01-31 but no bar on 2024-12-15 (1-day gap, no split),  qv 500
#   JUPUSDT  listed month 2024-12 whose archive answers 404                       (kept symbol)
#   BTCUSDT  only 2019 archives (no data in range)
#   USDCUSDT stablecoin, BTCUPUSDT / BULLUSDT leveraged, ETHBTC other quote: excluded
BN_MONTHS = ["2024-11", "2024-12", "2025-01"]
BN_SYMBOLS_ROOT = ["AAAUSDT", "BBBUSDT", "BTCUPUSDT", "BTCUSDT", "BULLUSDT", "CCCUSDT",
                   "DDDUSDT", "EEEUSDT", "ETHBTC", "JUPUSDT", "USDCUSDT"]
KLINE_HEADER = ("open_time,open,high,low,close,volume,close_time,quote_volume,count,"
                "taker_buy_volume,taker_buy_quote_volume,ignore")


def bn_days(sym: str) -> list[pd.Timestamp]:
    all_days = pd.date_range("2024-11-01", "2025-01-31", freq="D")
    if sym == "CCCUSDT":
        return [d for d in all_days if d >= pd.Timestamp("2024-12-10")]
    if sym == "DDDUSDT":
        return [d for d in all_days if d <= pd.Timestamp("2024-11-10")
                or d >= pd.Timestamp("2024-12-20")]
    if sym == "EEEUSDT":
        return [d for d in all_days if d != pd.Timestamp("2024-12-15")]
    return list(all_days)


BN_QV = {"AAAUSDT": 1000.0, "BBBUSDT": 2000.0, "CCCUSDT": 5000.0, "DDDUSDT": 3000.0,
         "EEEUSDT": 500.0}


def bn_close(sym: str, day: pd.Timestamp) -> float:
    k = (day - pd.Timestamp("2024-11-01")).days
    if sym == "AAAUSDT":
        return 100.0 * 1.01 ** k
    if sym == "BBBUSDT":
        return 50.0 if k % 2 == 0 else 55.0
    if sym == "DDDUSDT":
        return 7.0 if day < pd.Timestamp("2024-12-01") else 0.7
    if sym == "EEEUSDT":
        return 2.0 + 0.01 * k
    return 10.0


def _kline_line(sym: str, day: pd.Timestamp) -> str:
    ms = int((day - pd.Timestamp("1970-01-01")).total_seconds() * 1000)
    micro = day >= pd.Timestamp("2025-01-01")
    scale = 1000 if micro else 1
    ot = ms * scale
    ct = (ms + 86_400_000) * scale - 1
    c = bn_close(sym, day)
    qv = BN_QV[sym]
    return (f"{ot},{c:.8f},{c * 1.02:.8f},{c * 0.98:.8f},{c:.8f},{qv / c:.8f},{ct},"
            f"{qv:.8f},100,{qv / c / 2:.8f},{qv / 2:.8f},0")


def _s3_root_page(prefixes, next_marker=None, marker="") -> str:
    trunc = "true" if next_marker else "false"
    nm = f"<NextMarker>{next_marker}</NextMarker>" if next_marker else ""
    cps = "".join(f"<CommonPrefixes><Prefix>data/spot/monthly/klines/{p}/</Prefix>"
                  f"</CommonPrefixes>" for p in prefixes)
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            "<Name>data.binance.vision</Name><Prefix>data/spot/monthly/klines/</Prefix>"
            f"<Marker>{marker}</Marker>{nm}<MaxKeys>1000</MaxKeys><Delimiter>/</Delimiter>"
            f"<IsTruncated>{trunc}</IsTruncated>{cps}</ListBucketResult>")


def _s3_symbol_page(sym: str, months) -> str:
    contents = ""
    for m in months:
        for suffix in ("", ".CHECKSUM"):
            key = f"data/spot/monthly/klines/{sym}/1d/{sym}-1d-{m}.zip{suffix}"
            contents += (f"<Contents><Key>{key}</Key><LastModified>2025-02-01T00:00:00.000Z"
                         "</LastModified><ETag>&quot;0&quot;</ETag><Size>1</Size>"
                         "<StorageClass>STANDARD</StorageClass></Contents>")
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
            f"<Name>data.binance.vision</Name><Prefix>data/spot/monthly/klines/{sym}/1d/</Prefix>"
            "<Marker></Marker><MaxKeys>1000</MaxKeys><Delimiter>/</Delimiter>"
            f"<IsTruncated>false</IsTruncated>{contents}</ListBucketResult>")


def make_binance() -> None:
    root = HERE / "binance"
    (root / "listing").mkdir(parents=True, exist_ok=True)
    half = len(BN_SYMBOLS_ROOT) // 2
    p1, p2 = BN_SYMBOLS_ROOT[:half], BN_SYMBOLS_ROOT[half:]
    (root / "listing" / "root_p1.xml").write_text(
        _s3_root_page(p1, next_marker=f"data/spot/monthly/klines/{p1[-1]}/"))
    (root / "listing" / "root_p2.xml").write_text(
        _s3_root_page(p2, marker=f"data/spot/monthly/klines/{p1[-1]}/"))
    months_of = {s: sorted({str(d.to_period("M")) for d in bn_days(s)}) for s in BN_QV}
    months_of["JUPUSDT"] = ["2024-12"]
    months_of["BTCUSDT"] = ["2019-01", "2019-02"]
    for sym, months in months_of.items():
        (root / "listing" / f"{sym}.xml").write_text(_s3_symbol_page(sym, months))
    for sym in BN_QV:
        days = bn_days(sym)
        for m in BN_MONTHS:
            p = pd.Period(m, freq="M")
            rows = [_kline_line(sym, d) for d in days if d.to_period("M") == p]
            if not rows:
                continue
            if sym in ("AAAUSDT", "EEEUSDT") and m == "2024-12":
                rows = [KLINE_HEADER] + rows
            _zip(root / "klines" / f"{sym}-1d-{m}.zip", f"{sym}-1d-{m}.csv",
                 "\n".join(rows) + "\n")


if __name__ == "__main__":
    np.seterr(all="raise")
    make_jkp()
    make_french()
    make_binance()
    print("fixtures written to", HERE)
