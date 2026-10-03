"""Download every dataset the study uses into data/cache/ (git-ignored).

Sources (all free, cited in the quant note):
  * Yahoo Finance via yfinance: daily OHLCV for US-listed ETFs, total-return adjusted
    (splits and distributions folded into OHLC with auto_adjust=True), plus Cboe
    volatility indices used only as signals.
  * FRED (St. Louis Fed): 3-month T-bill (DTB3) for the cash rate.
  * Kenneth French Data Library: daily Fama-French 5 factors and momentum.
  * U.S. Treasury Fiscal Data API: every Treasury auction (dates, terms, sizes).

Usage:  python data/download.py            (idempotent; re-downloads everything)
"""
from __future__ import annotations

import io
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402

ETFS = [
    # US and international equity
    "SPY", "QQQ", "IWM", "DIA", "MDY", "VTI", "EFA", "EEM", "VGK", "EWJ",
    # US sectors
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "VNQ",
    # Treasuries, credit, inflation-linked
    "BIL", "SHY", "IEI", "IEF", "TLH", "TLT", "AGG", "BND", "LQD", "HYG", "TIP", "EMB",
    # commodities and currencies
    "GLD", "SLV", "USO", "DBC", "UUP", "FXE", "FXY",
]
INDICES = ["^VIX", "^VIX3M", "^VIX9D", "^GSPC"]
DOWNLOAD_START = "2003-01-01"   # extra warm-up before HISTORY_START
UA = {"User-Agent": "Mozilla/5.0 (research; gator-quant-hacks)"}


def _yf_long(tickers: list[str]) -> pd.DataFrame:
    import yfinance as yf

    frames = []
    for i in range(0, len(tickers), 10):
        chunk = tickers[i : i + 10]
        for attempt in range(4):
            try:
                raw = yf.download(
                    chunk, start=DOWNLOAD_START, auto_adjust=True, group_by="ticker",
                    progress=False, threads=True, actions=False,
                )
                break
            except Exception as exc:  # network hiccup / rate limit
                print(f"  retry {chunk} ({exc})")
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"yfinance failed for {chunk}")
        for t in chunk:
            if t not in raw.columns.get_level_values(0):
                print(f"  WARNING: {t} missing from yfinance response")
                continue
            sub = raw[t].dropna(how="all").copy()
            sub.columns = [c.lower() for c in sub.columns]
            sub["ticker"] = t
            sub.index.name = "date"
            frames.append(sub.reset_index())
        time.sleep(1)
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    return out[["date", "ticker", "open", "high", "low", "close", "volume"]]


def download_prices() -> None:
    etf = _yf_long(ETFS)
    etf = etf.dropna(subset=["open", "close"])
    etf = etf[(etf["open"] > 0) & (etf["close"] > 0)]
    etf.to_parquet(C.DATA_DIR / "etf_daily.parquet", index=False)
    idx = _yf_long(INDICES)
    idx.to_parquet(C.DATA_DIR / "index_daily.parquet", index=False)
    print(f"etf_daily: {etf['ticker'].nunique()} tickers, {len(etf)} rows, "
          f"{etf['date'].min().date()} .. {etf['date'].max().date()}")


def download_fred(series: list[str]) -> None:
    frames = []
    for s in series:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}"
        r = requests.get(url, headers=UA, timeout=60)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = ["date", s]
        df["date"] = pd.to_datetime(df["date"])
        df[s] = pd.to_numeric(df[s], errors="coerce")
        frames.append(df.set_index("date"))
    fred = pd.concat(frames, axis=1).sort_index()
    fred.reset_index().to_parquet(C.DATA_DIR / "fred_daily.parquet", index=False)
    # daily cash return from the 3-month T-bill (discount yield, % p.a.), forward-filled over holidays
    rf = (fred["DTB3"].ffill() / 100 / C.TRADING_DAYS).rename("rf").to_frame()
    rf.reset_index().to_parquet(C.DATA_DIR / "rf_daily.parquet", index=False)
    print(f"fred: {list(fred.columns)} {fred.index.min().date()} .. {fred.index.max().date()}")


def _french_zip(name: str) -> pd.DataFrame:
    url = f"https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{name}_CSV.zip"
    r = requests.get(url, headers=UA, timeout=120)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    text = z.read(z.namelist()[0]).decode("latin-1")
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip()[:8].isdigit()) - 1
    rows = []
    header = [h.strip() for h in lines[start].split(",")]
    for l in lines[start + 1 :]:
        parts = [p.strip() for p in l.split(",")]
        if not parts[0].isdigit() or len(parts[0]) != 8:
            if rows:
                break
            continue
        rows.append(parts)
    df = pd.DataFrame(rows, columns=["date"] + header[1:])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date").apply(pd.to_numeric, errors="coerce") / 100.0
    return df


def download_french() -> None:
    ff5 = _french_zip("F-F_Research_Data_5_Factors_2x3_daily")
    mom = _french_zip("F-F_Momentum_Factor_daily")
    mom.columns = ["Mom"]
    ff = ff5.join(mom, how="inner")
    ff.columns = [c.replace("-", "_") for c in ff.columns]   # Mkt_RF, SMB, HML, RMW, CMA, RF, Mom
    ff.reset_index().to_parquet(C.DATA_DIR / "ff_daily.parquet", index=False)
    print(f"french: {list(ff.columns)} {ff.index.min().date()} .. {ff.index.max().date()}")


def download_auctions() -> None:
    base = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query"
    fields = ",".join([
        "record_date", "cusip", "security_type", "security_term", "auction_date", "issue_date",
        "maturity_date", "offering_amt", "total_accepted", "bid_to_cover_ratio", "high_yield",
        "reopening", "announcemt_date", "original_security_term",
        "primary_dealer_accepted", "direct_bidder_accepted", "indirect_bidder_accepted",
    ])
    rows, page = [], 1
    while True:
        params = {"fields": fields, "filter": "auction_date:gte:2002-01-01", "page[size]": 1000, "page[number]": page,
                  "sort": "auction_date"}
        r = requests.get(base, params=params, headers=UA, timeout=120)
        r.raise_for_status()
        js = r.json()
        rows += js["data"]
        if page >= js["meta"]["total-pages"]:
            break
        page += 1
    df = pd.DataFrame(rows)
    for c in ("auction_date", "issue_date", "maturity_date", "announcemt_date", "record_date"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ("offering_amt", "total_accepted", "bid_to_cover_ratio", "high_yield",
              "primary_dealer_accepted", "direct_bidder_accepted", "indirect_bidder_accepted"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.to_parquet(C.DATA_DIR / "treasury_auctions.parquet", index=False)
    print(f"auctions: {len(df)} rows {df['auction_date'].min().date()} .. {df['auction_date'].max().date()}")


def main() -> None:
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    download_prices()
    download_fred(["DTB3", "DGS10", "DGS2", "DGS30", "T10Y2Y", "BAMLH0A0HYM2", "VIXCLS"])
    download_french()
    download_auctions()


if __name__ == "__main__":
    which = sys.argv[1:] or ["all"]
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if "all" in which:
        main()
    else:
        for w in which:
            {"prices": download_prices, "french": download_french, "auctions": download_auctions,
             "fred": lambda: download_fred(["DTB3", "DGS10", "DGS2", "DGS30", "T10Y2Y", "BAMLH0A0HYM2", "VIXCLS"])}[w]()
