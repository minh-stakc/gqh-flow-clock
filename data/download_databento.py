"""Build CME futures total-return series from Databento GLBX.MDP3 (CME Globex).

Requires DATABENTO_API_KEY (environment or a git-ignored .env file). Raw vendor responses are
cached under GQH_DATA_DIR/databento_raw/ and never re-requested if present (no repeat billing);
licensed raw data is never committed.

Pulls (about $5 of usage at Databento list prices, checked with metadata.get_cost first):
  * ohlcv-1d, continuous symbology, volume-ranked front (.v.0) and second (.v.1) contracts, for
    the 20 roots of the PCT-F replication, 2010-06-06 .. 2026-10-03;
  * ohlcv-1h for ES and ZN (.v.0/.v.1), sampled at the bar ending 16:00 ET for IFC-F.

Construction:
  * Each day's return is computed WITHIN one contract: close_t(c) / close_{t-1}(c) - 1, where c is
    the front contract on day t and close_{t-1}(c) comes from whichever rank c had the day before.
    So the roll never creates a fake price jump. A roll day (front contract changes) is flagged and
    the engine charges one extra round trip of the position on it.
  * Futures excess returns are compounded over all CME sessions between consecutive NYSE trading
    days, then the T-bill return of the NYSE day is added: the series is a fully collateralised
    total-return index on the NYSE calendar, so the engine treats it like an ETF (no borrow fee).
  * volume column = front-contract dollar volume / index level, so close * volume = dollar volume.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402

ROOTS_DAILY = ["ES", "NQ", "RTY", "YM", "ZT", "ZF", "ZN", "ZB", "CL", "NG", "GC", "SI", "HG",
               "6E", "6J", "6B", "6A", "6C", "ZC", "ZS"]
ROOTS_HOURLY = ["ES", "ZN"]
START = "2010-06-06"
END = os.environ.get("GQH_DATA_END", "2026-10-03")   # later dates for the forward test (FORWARD_TEST.md)
# contract multipliers (USD per 1.0 of quoted price; grains quoted in cents per bushel)
MULT = {"ES": 50, "NQ": 20, "RTY": 50, "YM": 5, "ZT": 2000, "ZF": 1000, "ZN": 1000, "ZB": 1000,
        "CL": 1000, "NG": 10000, "GC": 100, "SI": 5000, "HG": 25000, "6E": 125000, "6J": 12_500_000,
        "6B": 62500, "6A": 100000, "6C": 100000, "ZC": 50, "ZS": 50}
RAW = C.DATA_DIR / "databento_raw"


def _key() -> str:
    if os.environ.get("DATABENTO_API_KEY"):
        return os.environ["DATABENTO_API_KEY"]
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("DATABENTO_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("DATABENTO_API_KEY not set (environment or .env)")


def _pull(schema: str, roots: list[str], name: str) -> pd.DataFrame:
    path = RAW / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    import time

    import databento as db

    client = db.Historical(_key())
    RAW.mkdir(parents=True, exist_ok=True)
    years = list(range(int(START[:4]), int(END[:4]) + 1))
    # one request per root (and per 4-year block for hourly bars) keeps each stream short
    blocks = [(START, END)] if schema == "ohlcv-1d" else [
        (max(START, f"{y}-01-01"), min(END, f"{y + 4}-01-01")) for y in years[::4]]
    parts = []
    for r in roots:
        whole = RAW / f"{name}__{r}__batch.parquet"     # written by a Databento batch job, if used
        if whole.exists():
            parts.append(pd.read_parquet(whole))
            continue
        for a, b in blocks:
            piece = RAW / f"{name}__{r}__{a[:4]}.parquet"
            if not piece.exists():
                q = dict(dataset="GLBX.MDP3", schema=schema, stype_in="continuous",
                         symbols=[f"{r}.v.0", f"{r}.v.1"], start=a, end=b)
                for attempt in range(5):
                    try:
                        d = client.timeseries.get_range(**q).to_df().reset_index()
                        break
                    except Exception as exc:          # gateway timeouts: back off and retry
                        print(f"  retry {r} {a} ({str(exc)[:60]})")
                        time.sleep(10 * (attempt + 1))
                else:
                    raise RuntimeError(f"Databento pull failed for {r} {a}")
                d.to_parquet(piece, index=False)
            parts.append(pd.read_parquet(piece))
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(path, index=False)
    return df


def _within_contract_returns(px: pd.DataFrame, root: str) -> pd.DataFrame:
    """px: rows (date, rank, instrument_id, close, volume) for one root, one observation per date
    and rank. Returns a frame indexed by date with ret, roll, front_id, close0, volume0."""
    p0 = px[px["rank"] == 0].set_index("date").sort_index()
    by_day_id = px.set_index(["date", "instrument_id"])["close"]
    by_day_id = by_day_id[~by_day_id.index.duplicated(keep="last")]
    dates = p0.index
    out = pd.DataFrame(index=dates)
    out["front_id"] = p0["instrument_id"]
    out["close0"] = p0["close"]
    out["volume0"] = p0["volume"]
    prev_dates = pd.Series(dates).shift(1).values
    rets, missing = [], 0
    for d, pdte, cid, c in zip(dates, prev_dates, out["front_id"], out["close0"]):
        if pd.isna(pdte):
            rets.append(np.nan)
            continue
        prev = by_day_id.get((pdte, cid), np.nan)
        if not (np.isfinite(prev) and prev > 0 and c > 0):
            missing += 1
            rets.append(np.nan)
        else:
            rets.append(c / prev - 1.0)
    out["ret"] = rets
    out["roll"] = (out["front_id"] != out["front_id"].shift(1)).astype(float)
    out.iloc[0, out.columns.get_loc("roll")] = 0.0
    if missing:
        print(f"  {root}: {missing} days without a same-contract previous close (return set to 0)")
    return out


def _to_nyse(per_root: pd.DataFrame, ticker: str, root: str, rf: pd.Series, cal: pd.DatetimeIndex) -> pd.DataFrame:
    r = per_root["ret"].fillna(0.0)
    # compound CME-session returns between NYSE days; map each CME date to the next NYSE date >= it
    nyse_pos = cal.searchsorted(r.index, side="left")
    ok = nyse_pos < len(cal)
    grp = pd.Series(cal[nyse_pos[ok]], index=r.index[ok])
    comp = (1 + r[ok]).groupby(grp.values).prod() - 1
    roll = per_root["roll"][ok].groupby(grp.values).max()
    vol_usd = (per_root["volume0"] * per_root["close0"] * MULT[root])[ok].groupby(grp.values).sum()
    first = per_root.index.min()
    cal_r = cal[cal >= first]
    comp = comp.reindex(cal_r).fillna(0.0)
    tr = comp + rf.reindex(cal_r).ffill().fillna(0.0)
    level = 100 * (1 + tr).cumprod()
    out = pd.DataFrame({"date": cal_r, "ticker": ticker, "close": level.values})
    out["open"] = out["high"] = out["low"] = out["close"]
    out["volume"] = (vol_usd.reindex(cal_r).fillna(0.0) / level).values
    out["roll"] = roll.reindex(cal_r).fillna(0.0).values
    return out[["date", "ticker", "open", "high", "low", "close", "volume", "roll"]]


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["root"] = df["symbol"].str.split(".").str[0]
    df["rank"] = df["symbol"].str.split(".").str[2].astype(int)
    return df


def build_daily(rf: pd.Series, cal: pd.DatetimeIndex) -> None:
    raw = _prep(_pull("ohlcv-1d", ROOTS_DAILY, "glbx_ohlcv1d_v01"))
    raw["date"] = pd.to_datetime(raw["ts_event"]).dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    raw = raw[raw["date"].dt.dayofweek < 5]          # Sunday-evening session bars are dropped
    frames = []
    for root in ROOTS_DAILY:
        px = raw[raw["root"] == root][["date", "rank", "instrument_id", "close", "volume"]]
        px = px.drop_duplicates(["date", "rank"], keep="last")
        per = _within_contract_returns(px, root)
        frames.append(_to_nyse(per, f"F_{root}", root, rf, cal))
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(C.DATA_DIR / "futures_daily.parquet", index=False)
    print(f"futures_daily: {out['ticker'].nunique()} tickers {out['date'].min().date()} .. {out['date'].max().date()}")


def build_hourly_1600(rf: pd.Series, cal: pd.DatetimeIndex) -> None:
    raw = _prep(_pull("ohlcv-1h", ROOTS_HOURLY, "glbx_ohlcv1h_v01_es_zn"))
    daily = _prep(_pull("ohlcv-1d", ROOTS_DAILY, "glbx_ohlcv1d_v01"))
    ts = pd.to_datetime(raw["ts_event"]).dt.tz_convert("America/New_York")
    raw["date"] = ts.dt.tz_localize(None).dt.normalize()
    raw["start_min"] = ts.dt.hour * 60 + ts.dt.minute
    # last bar that STARTS between 09:00 and 15:00 ET -> close is the last trade before 16:00 ET
    raw = raw[(raw["start_min"] >= 9 * 60) & (raw["start_min"] <= 15 * 60)]
    raw = raw.sort_values("start_min").drop_duplicates(["date", "symbol"], keep="last")
    daily["date"] = pd.to_datetime(daily["ts_event"]).dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    frames = []
    for root in ROOTS_HOURLY:
        px = raw[raw["root"] == root][["date", "rank", "instrument_id", "close"]].copy()
        vol = daily[(daily["root"] == root) & (daily["rank"] == 0)].set_index("date")["volume"]
        px["volume"] = px["date"].map(vol).fillna(0.0)
        per = _within_contract_returns(px, root)
        frames.append(_to_nyse(per, f"{root}16", root, rf, cal))
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(C.DATA_DIR / "futures_1600.parquet", index=False)
    print(f"futures_1600: {sorted(out['ticker'].unique())} {out['date'].min().date()} .. {out['date'].max().date()}")


def main() -> None:
    rf = pd.read_parquet(C.DATA_DIR / "rf_daily.parquet").set_index("date")["rf"]
    rf.index = pd.to_datetime(rf.index)
    etf = pd.read_parquet(C.DATA_DIR / "etf_daily.parquet")
    cal = pd.DatetimeIndex(sorted(etf.loc[etf["ticker"] == "SPY", "date"].unique()))
    build_daily(rf, cal)
    build_hourly_1600(rf, cal)


if __name__ == "__main__":
    main()
