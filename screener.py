"""
NSE breakout screener — top ~1,250 stocks by turnover, full technical suite.

Universe: every EQ-series stock in the official NSE bhavcopy, ranked by turnover,
top N taken (self-updating, no stale index list). ETF-like symbols excluded.

Breakout signals (latest completed daily bar, all require close > 200 EMA):
  D55_BREAKOUT   close above prior 55-day Donchian upper (strongest)
  D20_BREAKOUT   close above prior 20-day high
  W52_BREAKOUT   close above prior 52-week high
  EMA50_RECLAIM  close crosses back above the 50 EMA
  GOLDEN_CROSS   50 EMA crossed above 200 EMA within last 5 sessions
  MACD_CROSS     MACD line crossed above signal within last 3 sessions
  NEAR_BREAKOUT  close within 1.5% below the 20-day Donchian upper (watchlist)

Chart patterns (heuristic detection):
  VCP            volatility contraction: successive tightening ranges near highs
  BULL_FLAG      sharp run-up then tight sideways drift, breaking flag high
  FLAT_BASE      tight multi-week base near 52w high, breaking base high
  DOUBLE_BOTTOM  two matched lows, neckline broken
  COIL           NR7 + inside bar near highs (pre-breakout tension)
  BB_SQZ         Bollinger band width in the tightest quintile of 6 months

Context columns: RS rating (1-99 percentile vs universe), RSI(14), ADX(14)/+DI,
volume ratio, delivery % (from bhavcopy), ATR%, F&O availability.

Usage:  python screener.py
Output: dashboard.html + data/results.json
"""

import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
BHAVCOPY = DATA_DIR / "bhavcopy.csv"
INDUSTRY_CSVS = [DATA_DIR / "totalmarket.csv", DATA_DIR / "nifty500.csv"]
FO_SYMBOLS = DATA_DIR / "fo_symbols.txt"
RESULTS_JSON = DATA_DIR / "results.json"
DASHBOARD = HERE / "dashboard.html"

UNIVERSE_SIZE = 1250
HISTORY_PERIOD = "15mo"
CHUNK = 100
MIN_TURNOVER_CR = 3.0        # 20-day avg turnover floor (crores); universe is broader now
NEAR_PCT = 1.5
ATR_STOP_MULT = 2.0
REWARD_R = 2.0
ETF_LIST = DATA_DIR / "etf_list.csv"   # official NSE ETF list — exact exclusion
ETF_PATTERN = r"BEES|ETF$|IETF|^NIFTY|^SENSEX|CASE$|^LIQUID"  # fallback for new ETFs


# ---------------------------------------------------------------- universe

def load_universe():
    bhav = pd.read_csv(BHAVCOPY, skipinitialspace=True)
    bhav.columns = [c.strip() for c in bhav.columns]
    eq = bhav[bhav["SERIES"].str.strip() == "EQ"].copy()
    eq["SYMBOL"] = eq["SYMBOL"].str.strip()
    eq = eq[~eq["SYMBOL"].str.contains(ETF_PATTERN, case=False, regex=True)]
    if ETF_LIST.exists():
        etfs = set(pd.read_csv(ETF_LIST, encoding="latin-1")["Symbol"].str.strip())
        eq = eq[~eq["SYMBOL"].isin(etfs)]
    eq["TURNOVER_LACS"] = pd.to_numeric(eq["TURNOVER_LACS"], errors="coerce").fillna(0)
    eq["DELIV_PER"] = pd.to_numeric(eq["DELIV_PER"], errors="coerce")
    eq = eq.sort_values("TURNOVER_LACS", ascending=False).head(UNIVERSE_SIZE)

    industry = {}
    for csv in INDUSTRY_CSVS:
        if csv.exists():
            df = pd.read_csv(csv)
            for _, r in df.iterrows():
                industry.setdefault(str(r["Symbol"]).strip(), str(r["Industry"]).strip())

    fo = set()
    if FO_SYMBOLS.exists():
        fo = {s.strip() for s in FO_SYMBOLS.read_text().splitlines() if s.strip()}

    uni = {}
    for _, r in eq.iterrows():
        sym = r["SYMBOL"]
        uni[sym] = {
            "industry": industry.get(sym, "—"),
            "deliv_per": None if pd.isna(r["DELIV_PER"]) else float(r["DELIV_PER"]),
            "fo": sym in fo,
        }
    return uni


def download_history(symbols):
    out = {}
    tickers = {s: f"{s}.NS" for s in symbols}
    names = list(tickers.values())
    rev = {v: k for k, v in tickers.items()}

    def fetch(batch):
        got = []
        data = yf.download(batch, period=HISTORY_PERIOD, interval="1d", group_by="ticker",
                           auto_adjust=True, threads=True, progress=False)
        for tkr in batch:
            try:
                df = data[tkr].dropna(how="all")
            except KeyError:
                continue
            if len(df) >= 130:
                out[rev[tkr]] = df
                got.append(tkr)
        return got

    for i in range(0, len(names), CHUNK):
        chunk = names[i : i + CHUNK]
        print(f"  downloading {i + 1}-{min(i + len(chunk), len(names))} of {len(names)} ...")
        fetch(chunk)

    # one retry pass for network flakes (transient DNS/timeout failures)
    missing = [t for t in names if rev[t] not in out]
    if missing and len(missing) < len(names) // 2:
        print(f"  retrying {len(missing)} failed tickers ...")
        time.sleep(15)
        for i in range(0, len(missing), CHUNK):
            fetch(missing[i : i + CHUNK])
    return out


# ---------------------------------------------------------------- indicators

def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()


def wilder(s, n):
    return s.ewm(alpha=1 / n, adjust=False).mean()


def true_range(df):
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)


def rsi(c, n=14):
    d = c.diff()
    gain = wilder(d.clip(lower=0), n)
    loss = wilder(-d.clip(upper=0), n)
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def adx_di(df, n=14):
    up, dn = df["High"].diff(), -df["Low"].diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    atr_ = wilder(true_range(df), n)
    pdi = 100 * wilder(pdm, n) / atr_
    mdi = 100 * wilder(mdm, n) / atr_
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return wilder(dx, n), pdi, mdi


# ---------------------------------------------------------------- patterns

def detect_patterns(df, close, don20, hi52):
    """Heuristic chart-pattern tags on the latest bar."""
    tags = []
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    n = len(df)
    near_highs = close >= 0.85 * float(hi52.iloc[-1])

    # VCP: three successive 15-bar ranges contracting, near highs
    if n >= 60 and near_highs:
        r3 = (h.iloc[-15:].max() - l.iloc[-15:].min())
        r2 = (h.iloc[-30:-15].max() - l.iloc[-30:-15].min())
        r1 = (h.iloc[-45:-30].max() - l.iloc[-45:-30].min())
        if r3 < r2 < r1 and r3 <= 0.6 * r1:
            tags.append("VCP")

    # BULL_FLAG: >=15% run-up into a tight <=7% drift of 5-10 bars, breaking flag high
    if n >= 45:
        flag = df.iloc[-9:-1]
        runup_start = c.iloc[-35:-9].min()
        flag_hi, flag_lo = flag["High"].max(), flag["Low"].min()
        if (runup_start > 0 and flag["Close"].iloc[0] / runup_start >= 1.15
                and (flag_hi - flag_lo) / close <= 0.07
                and close > flag_hi):
            tags.append("BULL_FLAG")

    # FLAT_BASE: 4-week tight base (<=10%) near 52w high, breaking base high
    if n >= 45 and close >= 0.88 * float(hi52.iloc[-1]):
        base = df.iloc[-21:-1]
        base_hi, base_lo = base["High"].max(), base["Low"].min()
        if (base_hi - base_lo) / close <= 0.10 and close > base_hi:
            tags.append("FLAT_BASE")

    # DOUBLE_BOTTOM: two matched swing lows (within 3.5%), neckline broken recently
    if n >= 120:
        lows = l.iloc[-120:]
        is_swing = (lows == lows.rolling(9, center=True, min_periods=5).min())
        swings = lows[is_swing]
        if len(swings) >= 2:
            idx = list(swings.index)
            for a in range(len(idx) - 1):
                for b in range(a + 1, len(idx)):
                    la, lb = float(swings.loc[idx[a]]), float(swings.loc[idx[b]])
                    gap = df.index.get_loc(idx[b]) - df.index.get_loc(idx[a])
                    if gap >= 20 and abs(la - lb) / max(la, lb) <= 0.035:
                        between = h.loc[idx[a]:idx[b]]
                        neckline = float(between.max())
                        depth = (neckline - min(la, lb)) / neckline
                        prev_c = float(c.iloc[-2])
                        if depth >= 0.08 and close > neckline and \
                                (prev_c <= neckline or float(c.iloc[-4]) <= neckline):
                            tags.append("DOUBLE_BOTTOM")
                            break
                if "DOUBLE_BOTTOM" in tags:
                    break

    # COIL: NR7 + inside bar near highs (pre-breakout tension, watch tag)
    if n >= 10 and near_highs:
        rng = (h - l).iloc[-7:]
        inside = h.iloc[-1] <= h.iloc[-2] and l.iloc[-1] >= l.iloc[-2]
        if rng.iloc[-1] == rng.min() and inside:
            tags.append("COIL")

    return tags


# ---------------------------------------------------------------- scan

def scan_symbol(sym, meta, df):
    c, h, l, o, v = df["Close"], df["High"], df["Low"], df["Open"], df["Volume"]

    ema20_s, ema100_s = ema(c, 20), ema(c, 100)
    ema50, ema200 = ema(c, 50), ema(c, 200)
    don20 = h.rolling(20).max().shift(1)
    don55 = h.rolling(55).max().shift(1)
    hi52 = h.rolling(250, min_periods=120).max()
    hi52_prev = h.rolling(250, min_periods=120).max().shift(1)
    vol20 = v.rolling(20).mean()
    atr14 = wilder(true_range(df), 14)
    rsi14 = rsi(c)
    adx14, pdi, mdi = adx_di(df)
    macd = ema(c, 12) - ema(c, 26)
    macd_sig = ema(macd, 9)

    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    bw = (4 * std20) / sma20
    bw_pct = bw.rolling(120, min_periods=60).rank(pct=True)

    close, prev_close = float(c.iloc[-1]), float(c.iloc[-2])
    last = {
        "ema50": float(ema50.iloc[-1]), "ema50_prev": float(ema50.iloc[-2]),
        "ema200": float(ema200.iloc[-1]),
        "don20": float(don20.iloc[-1]), "don55": float(don55.iloc[-1]),
        "atr": float(atr14.iloc[-1]), "vol20": float(vol20.iloc[-1]),
        "rsi": float(rsi14.iloc[-1]), "adx": float(adx14.iloc[-1]),
        "pdi": float(pdi.iloc[-1]), "mdi": float(mdi.iloc[-1]),
        "bwp": float(bw_pct.iloc[-1]) if not math.isnan(bw_pct.iloc[-1]) else 1.0,
    }
    if any(math.isnan(last[k]) for k in ("don55", "ema200", "atr")):
        return None

    vol_ratio = float(v.iloc[-1]) / last["vol20"] if last["vol20"] > 0 else 0.0
    turnover_cr = close * last["vol20"] / 1e7
    if turnover_cr < MIN_TURNOVER_CR or close <= last["ema200"]:
        return None

    signals = []
    if close > last["don55"]:
        signals.append("D55_BREAKOUT")
    if close > last["don20"]:
        signals.append("D20_BREAKOUT")
    if close > float(hi52_prev.iloc[-1]):
        signals.append("W52_BREAKOUT")
    if prev_close <= last["ema50_prev"] and close > last["ema50"]:
        signals.append("EMA50_RECLAIM")
    gcross = (ema50 > ema200) & (ema50.shift(1) <= ema200.shift(1))
    if gcross.iloc[-5:].any():
        signals.append("GOLDEN_CROSS")
    mcross = (macd > macd_sig) & (macd.shift(1) <= macd_sig.shift(1))
    if mcross.iloc[-3:].any():
        signals.append("MACD_CROSS")

    breakout = any(s in signals for s in ("D20_BREAKOUT", "D55_BREAKOUT", "W52_BREAKOUT"))
    # stocks holding above ALL DMAs get a wider pre-breakout net: these are the
    # coiled setups worth a buy-stop order at the trigger
    above_all_dmas = all(close > float(s.iloc[-1])
                         for s in (ema20_s, ema50, ema100_s, ema200))
    near_band = 3.0 if above_all_dmas else NEAR_PCT
    if not breakout and close <= last["don20"] and \
            (last["don20"] - close) / close * 100 <= near_band:
        signals.append("NEAR_BREAKOUT")

    patterns = detect_patterns(df, close, don20, hi52)
    if last["bwp"] <= 0.20:
        patterns.append("BB_SQZ")

    # inclusion rule: a real signal, or a pattern coiling within 5% of the D20 trigger
    has_signal = bool(signals)
    coiling = bool(patterns) and close >= 0.95 * last["don20"]
    if not has_signal and not coiling:
        return None
    if not has_signal:
        signals.append("SETUP_WATCH")

    stop = round(close - ATR_STOP_MULT * last["atr"], 2)
    risk = close - stop
    target = round(close + REWARD_R * risk, 2)
    tgt1 = round(close + 1.5 * risk, 2)      # house convention: two targets
    tgt2 = round(close + 2.5 * risk, 2)
    dema = [close > float(s.iloc[-1]) for s in (ema20_s, ema50, ema100_s, ema200)]
    dist_52w = (float(hi52.iloc[-1]) - close) / close * 100

    # weekly timeframe confirmation
    wc = c.resample("W").last().dropna()
    wh = h.resample("W").max().dropna()
    wk = None
    if len(wc) >= 25:
        w20 = ema(wc, 20)
        wrsi = rsi(wc)
        wk = {
            "above20w": bool(wc.iloc[-1] > w20.iloc[-1]),
            "rsi": round(float(wrsi.iloc[-1]), 1) if not math.isnan(wrsi.iloc[-1]) else None,
            "hh10w": bool(wc.iloc[-1] > wh.iloc[:-1].tail(10).max()),
        }

    # nearest overhead resistance from prior swing highs (5-bar pivots, last 250 bars)
    hh = h.iloc[-250:]
    piv = hh[(hh == hh.rolling(11, center=True, min_periods=6).max())]
    piv = piv[piv.index <= h.index[-6]]        # exclude the breakout bars themselves
    overhead = sorted(float(p) for p in piv if p > close * 1.005)
    resistance = round(overhead[0], 2) if overhead else None

    # accumulation/distribution: up-day vs down-day volume, last 25 sessions
    chg = c.diff().iloc[-25:]
    vv = v.iloc[-25:]
    upv = float(vv[chg > 0].sum())
    dnv = float(vv[chg < 0].sum())
    ad_ratio = upv / dnv if dnv > 0 else 3.0
    ad = ("A" if ad_ratio >= 1.5 else "B" if ad_ratio >= 1.2 else
          "C" if ad_ratio >= 0.9 else "D" if ad_ratio >= 0.7 else "E")

    # returns for RS rating (percentile assigned later, universe-wide)
    def ret(bars):
        return float(c.iloc[-1] / c.iloc[-bars - 1] - 1) if len(c) > bars else 0.0
    rs_raw = 0.5 * ret(63) + 0.3 * ret(126) + 0.2 * ret(21)

    spark = [round(x, 2) for x in c.iloc[-60:].tolist()]
    k0 = max(0, len(df) - 75)
    candles = [[round(float(o.iloc[j]), 2), round(float(h.iloc[j]), 2),
                round(float(l.iloc[j]), 2), round(float(c.iloc[j]), 2),
                int(v.iloc[j])] for j in range(k0, len(df))]
    return {
        "symbol": sym,
        "industry": meta["industry"],
        "fo": meta["fo"],
        "deliv_per": meta["deliv_per"],
        "date": str(df.index[-1].date()),
        "close": round(close, 2),
        "chg_pct": round((close / prev_close - 1) * 100, 2),
        "signals": signals,
        "patterns": patterns,
        "rsi": round(last["rsi"], 1),
        "adx": round(last["adx"], 1),
        "di_bull": last["pdi"] > last["mdi"],
        "vol_ratio": round(vol_ratio, 2),
        "turnover_cr": round(turnover_cr, 1),
        "atr_pct": round(last["atr"] / close * 100, 2),
        "don20": round(last["don20"], 2),
        "don55": round(last["don55"], 2),
        "ema50": round(last["ema50"], 2),
        "entry": round(close, 2),
        "stop": stop,
        "target": target,
        "tgt1": tgt1,
        "tgt2": tgt2,
        "dema": dema,
        "wk": wk,
        "resistance": resistance,
        "ad": ad,
        "dist_52w_pct": round(dist_52w, 2),
        "rs_raw": rs_raw,
        "spark": spark,
        "candles": candles,
    }


def assign_rs_and_score(rows, all_rs):
    """RS rating = percentile of blended momentum vs the whole scanned universe."""
    arr = np.sort(np.array(all_rs))
    for r in rows:
        pct = np.searchsorted(arr, r["rs_raw"]) / max(len(arr), 1)
        r["rs"] = int(round(1 + 98 * pct))
        del r["rs_raw"]

        strength = (3 * ("D55_BREAKOUT" in r["signals"])
                    + 2 * ("D20_BREAKOUT" in r["signals"])
                    + 2 * ("W52_BREAKOUT" in r["signals"])
                    + ("EMA50_RECLAIM" in r["signals"])
                    + ("GOLDEN_CROSS" in r["signals"])
                    + ("MACD_CROSS" in r["signals"]))
        pat_bonus = (8 * ("VCP" in r["patterns"])
                     + 8 * ("BULL_FLAG" in r["patterns"])
                     + 6 * ("FLAT_BASE" in r["patterns"])
                     + 6 * ("DOUBLE_BOTTOM" in r["patterns"])
                     + 3 * ("COIL" in r["patterns"])
                     + 3 * ("BB_SQZ" in r["patterns"]))
        rsi_bonus = 3 if 55 <= r["rsi"] <= 75 else (-4 if r["rsi"] > 80 else 0)
        adx_bonus = 5 if (r["adx"] > 25 and r["di_bull"]) else 0
        dl = r["deliv_per"]
        deliv_bonus = 5 if (dl and dl >= 60) else (2 if (dl and dl >= 40) else 0)

        r["score"] = round(6 * strength + pat_bonus + 0.18 * r["rs"]
                           + 4 * min(r["vol_ratio"], 5) + adx_bonus + rsi_bonus
                           + deliv_bonus - 0.3 * min(r["dist_52w_pct"], 30), 1)


def market_regime(retries=3):
    """Nifty vs its 50/200 EMAs — context for how aggressively to take signals."""
    try:
        nifty = pd.Series(dtype=float)
        for attempt in range(retries):
            nifty = yf.download("^NSEI", period="2y", interval="1d",
                                auto_adjust=True, progress=False)["Close"].dropna()
            if hasattr(nifty, "columns"):
                nifty = nifty.iloc[:, 0]
            if len(nifty) > 250:
                break
            print(f"  regime fetch attempt {attempt + 1} empty, retrying ...")
            time.sleep(10)
        if len(nifty) < 250:
            raise RuntimeError("could not fetch ^NSEI history")
        e50, e200 = ema(nifty, 50), ema(nifty, 200)
        above50 = float(nifty.iloc[-1]) > float(e50.iloc[-1])
        above200 = float(nifty.iloc[-1]) > float(e200.iloc[-1])
        return {
            "nifty": round(float(nifty.iloc[-1]), 1),
            "above50": above50, "above200": above200,
            "label": ("RISK-ON" if above50 and above200 else
                      "RECOVERY" if above50 else
                      "CAUTION" if above200 else "RISK-OFF"),
            "spark": [round(x, 1) for x in nifty.iloc[-120:].tolist()],
        }
    except Exception as e:
        print(f"  ! regime check failed: {e}")
        return None


def compute_breadth(hist):
    """Market internals across the whole scanned universe."""
    b = {"above200": 0, "above50": 0, "adv": 0, "hi52": 0, "lo52": 0, "total": 0}
    for df in hist.values():
        c, h, l = df["Close"], df["High"], df["Low"]
        if len(c) < 120:
            continue
        b["total"] += 1
        close = float(c.iloc[-1])
        if close > float(ema(c, 200).iloc[-1]):
            b["above200"] += 1
        if close > float(ema(c, 50).iloc[-1]):
            b["above50"] += 1
        if close > float(c.iloc[-2]):
            b["adv"] += 1
        if float(h.iloc[-1]) >= float(h.rolling(250, min_periods=120).max().iloc[-1]):
            b["hi52"] += 1
        if float(l.iloc[-1]) <= float(l.rolling(250, min_periods=120).min().iloc[-1]):
            b["lo52"] += 1

    # distribution days: universe average down >0.2% on higher aggregate volume,
    # counted over the last 25 sessions
    closes = pd.DataFrame({s: df["Close"] for s, df in hist.items()})
    vols = pd.DataFrame({s: df["Volume"] for s, df in hist.items()})
    avg_ret = closes.pct_change().mean(axis=1)
    aggvol = vols.sum(axis=1)
    dist = (avg_ret < -0.002) & (aggvol > aggvol.shift(1))
    b["dist25"] = int(dist.iloc[-25:].sum())
    return b


def compute_rrg(hist, universe):
    """Sector rotation: x = 63d return vs Nifty, y = 21d momentum vs Nifty."""
    try:
        nifty = yf.download("^NSEI", period="1y", interval="1d",
                            auto_adjust=True, progress=False)["Close"].dropna()
        if hasattr(nifty, "columns"):
            nifty = nifty.iloc[:, 0]
        n63 = float(nifty.iloc[-1] / nifty.iloc[-64] - 1)
        n21 = float(nifty.iloc[-1] / nifty.iloc[-22] - 1)
    except Exception as e:
        print(f"  ! rrg nifty fetch failed: {e}")
        return None

    sectors = {}
    for sym, df in hist.items():
        ind = universe.get(sym, {}).get("industry", "—")
        if not ind or ind == "—":
            continue
        c = df["Close"]
        if len(c) < 70:
            continue
        r63 = float(c.iloc[-1] / c.iloc[-64] - 1)
        r21 = float(c.iloc[-1] / c.iloc[-22] - 1)
        sectors.setdefault(ind, []).append((r63, r21))

    out = []
    for ind, vals in sectors.items():
        if len(vals) < 5:
            continue
        r63s = sorted(v[0] for v in vals)
        r21s = sorted(v[1] for v in vals)
        med63 = r63s[len(r63s) // 2]
        med21 = r21s[len(r21s) // 2]
        out.append({"sector": ind, "n": len(vals),
                    "x": round((med63 - n63) * 100, 2),
                    "y": round((med21 - n21) * 100, 2)})
    return sorted(out, key=lambda s: -s["x"])


PORTFOLIO_JSON = DATA_DIR / "paper_portfolio.json"
FRESH_SIGNALS = ("D20_BREAKOUT", "D55_BREAKOUT", "W52_BREAKOUT")


def is_agrade(r):
    """Same rule the backtest validated: D55 breakout + RS>=80 + Vol>=1.5x + RSI<78."""
    return ("D55_BREAKOUT" in r["signals"] and r["rs"] >= 80
            and r["vol_ratio"] >= 1.5 and r["rsi"] < 78)


def trade_reason(r):
    if "W52_BREAKOUT" in r["signals"]:
        headline = "fresh 52-week high"
    elif "D55_BREAKOUT" in r["signals"]:
        headline = "55-day channel breakout"
    else:
        headline = "20-day high breakout"
    grade_tag = "A-grade (RS>=80, Vol>=1.5x, RSI<78)" if is_agrade(r) else "raw signal"
    dlv = f"{r['deliv_per']:.0f}" if r["deliv_per"] is not None else "—"
    return (f"{headline} — RS {r['rs']}, Vol {r['vol_ratio']}x, RSI {r['rsi']}, "
            f"Dlv {dlv}% — {grade_tag}")


def compute_nav_row(trades, as_of, nifty_close, nifty_base):
    """Equal-weight average-return index (rebased to 100) per grade bucket,
    plus a blended total across both buckets pooled together."""
    def avg_return(pred):
        vals = []
        for t in trades:
            if not pred(t) or t["entry_date"] > as_of:
                continue
            r = t["return_pct"] if t["status"] != "OPEN" else \
                (t["last_price"] / t["entry_price"] - 1) * 100
            vals.append(r)
        idx = round(100 * (1 + (sum(vals) / len(vals)) / 100), 3) if vals else 100.0
        return idx, len(vals)

    a_nav, a_n = avg_return(lambda t: t["grade"] == "A")
    raw_nav, raw_n = avg_return(lambda t: t["grade"] == "RAW")
    tot_nav, tot_n = avg_return(lambda t: True)
    nifty_idx = round(100 * nifty_close / nifty_base, 3) if nifty_base else None
    return {"date": as_of, "a_nav": a_nav, "a_trades": a_n,
            "raw_nav": raw_nav, "raw_trades": raw_n,
            "tot_nav": tot_nav, "tot_trades": tot_n, "nifty_idx": nifty_idx}


def update_paper_portfolio(rows, hist, regime, as_of):
    """Forward-tracked virtual portfolio: 1 position per fresh breakout signal,
    entered at the close it fired, held to the existing fixed stop / 2R target.
    Persists in data/paper_portfolio.json across daily runs."""
    data = json.loads(PORTFOLIO_JSON.read_text(encoding="utf-8")) \
        if PORTFOLIO_JSON.exists() else {"trades": [], "nav_history": [], "nifty_base": None}
    trades = data["trades"]
    open_syms = {t["symbol"] for t in trades if t["status"] == "OPEN"}

    for r in rows:
        if r["symbol"] in open_syms:
            continue
        fired = [s for s in FRESH_SIGNALS if s in r["signals"]]
        if not fired:
            continue
        trades.append({
            "symbol": r["symbol"], "grade": "A" if is_agrade(r) else "RAW",
            "signal": fired[0], "reason": trade_reason(r),
            "industry": r["industry"],
            "entry_date": as_of, "entry_price": r["close"],
            "stop": r["stop"], "target": r["target"],
            "status": "OPEN", "exit_date": None, "exit_price": None,
            "return_pct": None, "last_price": r["close"], "last_date": as_of,
        })

    for t in trades:
        if t["status"] != "OPEN" or t["entry_date"] == as_of:
            continue          # monitoring starts the session after entry
        df = hist.get(t["symbol"])
        if df is None or str(df.index[-1].date()) != as_of:
            continue          # stock fell out of today's universe — leave stale
        hi = float(df["High"].iloc[-1])
        lo = float(df["Low"].iloc[-1])
        cl = float(df["Close"].iloc[-1])
        if lo <= t["stop"]:                 # conservative: stop wins a same-day overlap
            t["status"], t["exit_date"], t["exit_price"] = "STOP", as_of, t["stop"]
            t["return_pct"] = round((t["stop"] / t["entry_price"] - 1) * 100, 2)
        elif hi >= t["target"]:
            t["status"], t["exit_date"], t["exit_price"] = "TARGET", as_of, t["target"]
            t["return_pct"] = round((t["target"] / t["entry_price"] - 1) * 100, 2)
        else:
            t["last_price"], t["last_date"] = round(cl, 2), as_of

    nifty_close = regime.get("nifty") if regime else None
    if nifty_close is not None:
        if data["nifty_base"] is None:
            data["nifty_base"] = nifty_close
        row = compute_nav_row(trades, as_of, nifty_close, data["nifty_base"])
        if data["nav_history"] and data["nav_history"][-1]["date"] == as_of:
            data["nav_history"][-1] = row
        else:
            data["nav_history"].append(row)

    PORTFOLIO_JSON.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return data


MOMO_JSON = DATA_DIR / "momo_core.json"
MOMO_N = 15


def compute_momo_core(hist, universe, regime, as_of):
    """Momentum Core: top-15 by 12-1 momentum, rebalanced on month change.
    Holdings persist in data/momo_core.json between runs."""
    prev = json.loads(MOMO_JSON.read_text(encoding="utf-8")) \
        if MOMO_JSON.exists() else None
    month = str(as_of)[:7]

    # score every eligible stock: 12m return skipping the most recent month
    scores = {}
    for sym, df in hist.items():
        c, v = df["Close"], df["Volume"]
        if len(c) < 260:
            continue
        close = float(c.iloc[-1])
        vol20 = float(v.rolling(20).mean().iloc[-1])
        if close * vol20 / 1e7 < MIN_TURNOVER_CR:
            continue
        if close <= float(ema(c, 200).iloc[-1]):
            continue
        mom = float(c.iloc[-22] / c.iloc[-251] - 1)
        scores[sym] = mom
    ranked = sorted(scores, key=scores.get, reverse=True)[:MOMO_N]

    if prev and prev.get("month") == month:
        holdings = [h["sym"] for h in prev["holdings"]]
        rebalanced = False
    else:
        holdings = ranked
        rebalanced = True
    prev_syms = {h["sym"] for h in (prev or {}).get("holdings", [])}

    out = {
        "month": month,
        "rebalanced_today": rebalanced,
        "exposure": 1.0 if (regime and regime.get("label") == "RISK-ON") else 0.5,
        "holdings": [],
        "exits": sorted(prev_syms - set(holdings)) if rebalanced else [],
    }
    for sym in holdings:
        df = hist.get(sym)
        if df is None:
            continue
        c = df["Close"]
        out["holdings"].append({
            "sym": sym,
            "industry": universe.get(sym, {}).get("industry", "—"),
            "close": round(float(c.iloc[-1]), 2),
            "chg_pct": round(float(c.iloc[-1] / c.iloc[-2] - 1) * 100, 2),
            "mom_pct": round(scores.get(sym, 0) * 100, 1),
            "status": ("ADD" if rebalanced and sym not in prev_syms else "HOLD"),
        })
    MOMO_JSON.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def build_dashboard(results, scanned, as_of, regime=None, breadth=None, rrg=None,
                    momo=None, portfolio=None):
    payload = json.dumps({"asOf": as_of, "scanned": scanned, "rows": results,
                          "regime": regime, "breadth": breadth, "rrg": rrg,
                          "momo": momo, "portfolio": portfolio})
    template = (HERE / "template.html").read_text(encoding="utf-8")
    DASHBOARD.write_text(template.replace("/*__DATA__*/null", payload), encoding="utf-8")


def main():
    universe = load_universe()
    print(f"Universe: {len(universe)} symbols (top {UNIVERSE_SIZE} by NSE turnover)")

    print("Downloading daily history via yfinance ...")
    hist = download_history(list(universe))
    print(f"Got usable history for {len(hist)} symbols")

    regime = market_regime()
    nifty_ret21 = None
    if regime and len(regime.get("spark", [])) > 22:
        sp = regime["spark"]
        nifty_ret21 = sp[-1] / sp[-22] - 1

    rows, all_rs = [], []
    for sym, df in hist.items():
        try:
            # RS pool should reflect the whole universe, not only matches
            c = df["Close"]
            def ret(b):
                return float(c.iloc[-1] / c.iloc[-b - 1] - 1) if len(c) > b else 0.0
            all_rs.append(0.5 * ret(63) + 0.3 * ret(126) + 0.2 * ret(21))
            row = scan_symbol(sym, universe[sym], df)
            if row:
                if nifty_ret21 is not None and len(c) > 22:
                    row["rs_up"] = bool(ret(21) > nifty_ret21)
                else:
                    row["rs_up"] = None
                rows.append(row)
        except Exception as e:
            print(f"  ! {sym}: {e}")

    assign_rs_and_score(rows, all_rs)
    rows.sort(key=lambda r: r["score"], reverse=True)
    as_of = max((r["date"] for r in rows), default=str(datetime.now().date()))

    breadth = compute_breadth(hist)
    rrg = compute_rrg(hist, universe)
    momo = compute_momo_core(hist, universe, regime, as_of)
    portfolio = update_paper_portfolio(rows, hist, regime, as_of)
    RESULTS_JSON.write_text(json.dumps({"asOf": as_of, "rows": rows,
                                        "regime": regime, "breadth": breadth,
                                        "rrg": rrg, "momo": momo,
                                        "portfolio": portfolio},
                                       indent=1),
                            encoding="utf-8")
    build_dashboard(rows, len(hist), as_of, regime, breadth, rrg, momo, portfolio)

    fresh = [r for r in rows
             if any(s in r["signals"] for s in ("D20_BREAKOUT", "D55_BREAKOUT",
                                                "W52_BREAKOUT"))]
    open_trades = [t for t in portfolio["trades"] if t["status"] == "OPEN"]
    print(f"\nDone. {len(fresh)} breakouts, {len(rows) - len(fresh)} watchlist "
          f"(data as of {as_of})")
    print(f"Virtual portfolio: {len(portfolio['trades'])} trades total, "
          f"{len(open_trades)} open")
    print(f"Dashboard: {DASHBOARD}")


if __name__ == "__main__":
    main()
