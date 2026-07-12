"""
Research program: exit engines, entry styles, regime gates, and new strategy
sleeves — all on the same 3y cached data, same walker, costs included.

Usage:  python research.py exits|entries|gates|sleeves|all
Output: printed tables + data/research_results.json (merged per stage)
"""

import json
import pickle
import sys

import numpy as np
import pandas as pd
import yfinance as yf

from screener import DATA_DIR, load_universe, ema, wilder, true_range, rsi

CACHE = DATA_DIR / "hist_cache_3y.pkl"
OUT = DATA_DIR / "research_results.json"

LOOKBACK = 490            # ~2 years of signal days
COST_PCT = 0.35           # round-trip cost as % of notional
MIN_TURNOVER_CR = 3.0
GAP_SKIP_PCT = 5.0

# ---------------------------------------------------------------- data prep

print("loading cache ...")
HIST = pickle.loads(CACHE.read_bytes())
UNI = load_universe()
INDUSTRY = {s: m["industry"] for s, m in UNI.items()}

NIFTY = yf.download("^NSEI", period="3y", interval="1d",
                    auto_adjust=True, progress=False)["Close"].dropna()
if isinstance(NIFTY, pd.DataFrame):
    NIFTY = NIFTY.iloc[:, 0]

CLOSES = pd.DataFrame({s: df["Close"] for s, df in HIST.items()})
VOLS = pd.DataFrame({s: df["Volume"] for s, df in HIST.items()})

# daily RS rating (1-99): blended 21/63/126-day return percentile
_blend = (0.5 * CLOSES.pct_change(63) + 0.3 * CLOSES.pct_change(126)
          + 0.2 * CLOSES.pct_change(21))
RS_MAT = _blend.rank(axis=1, pct=True).mul(98).add(1)

# breadth: % of universe above its own 50 EMA
_above50 = CLOSES.gt(CLOSES.ewm(span=50, adjust=False).mean())
_valid = CLOSES.notna()
PCT50 = (_above50 & _valid).sum(axis=1) / _valid.sum(axis=1)

# distribution days: Nifty down >0.2% on higher aggregate universe volume
_aggvol = VOLS.sum(axis=1)
_nret = NIFTY.pct_change().reindex(_aggvol.index)
_dd = ((_nret < -0.002) & (_aggvol > _aggvol.shift(1))).astype(int)
DIST25 = _dd.rolling(25).sum()

_ne50 = ema(NIFTY, 50)
_ne200 = ema(NIFTY, 200)
EMA_GATE = (NIFTY > _ne50) & (NIFTY > _ne200)


def prep(df):
    """Per-symbol indicator arrays."""
    c, h, l, o, v = df["Close"], df["High"], df["Low"], df["Open"], df["Volume"]
    x = {
        "idx": df.index,
        "o": o.to_numpy(float), "h": h.to_numpy(float),
        "l": l.to_numpy(float), "c": c.to_numpy(float),
        "v": v.to_numpy(float),
        "e20": ema(c, 20).to_numpy(float), "e50": ema(c, 50).to_numpy(float),
        "e200": ema(c, 200).to_numpy(float),
        "don20": h.rolling(20).max().shift(1).to_numpy(float),
        "don55": h.rolling(55).max().shift(1).to_numpy(float),
        "atr": wilder(true_range(df), 14).to_numpy(float),
        "vol20": v.rolling(20).mean().to_numpy(float),
        "rsi14": rsi(c).to_numpy(float),
        "low10": l.rolling(10).min().to_numpy(float),
    }
    d = c.diff()
    g2 = wilder(d.clip(lower=0), 2)
    l2 = wilder(-d.clip(upper=0), 2)
    x["rsi2"] = (100 - 100 / (1 + g2 / l2.replace(0, np.nan))).to_numpy(float)
    return x


print("prepping indicators ...")
PREP = {s: prep(df) for s, df in HIST.items() if len(df) >= 260}
print(f"symbols ready: {len(PREP)}")


# ---------------------------------------------------------------- walker

def walk(x, t, entry, stop, engine, max_hold=20):
    """Simulate one trade from entry (filled at bar t_e). Returns (r_mult, exit_i, reason).
    engine: fixed | partial | ftstop | partial_ft | rsi2exit"""
    o, h, l, c = x["o"], x["h"], x["l"], x["c"]
    n = len(o)
    risk = entry - stop
    if risk <= 0:
        return None
    tgt = entry + 2 * risk
    one_r = entry + risk
    partial = engine.startswith("partial")
    ft = engine in ("ftstop", "partial_ft")
    booked_half = False
    trail = stop
    hold_cap = 40 if partial else max_hold

    for k in range(t, min(t + hold_cap, n)):
        op, hi, lo, cl = o[k], h[k], l[k], c[k]
        if engine == "rsi2exit":
            if op <= stop:
                return ((op - entry) / risk, k, "gap_stop")
            if lo <= stop:
                return (-1.0, k, "stop")
            if x["rsi2"][k] > 60 or k == t + 7:
                return ((cl - entry) / risk, k, "rsi2")
            continue
        if not partial:
            if op <= stop:
                return ((op - entry) / risk, k, "gap_stop")
            if lo <= stop:
                return (-1.0, k, "stop")
            if op >= tgt:
                return ((op - entry) / risk, k, "gap_target")
            if hi >= tgt:
                return (2.0, k, "target")
            if ft and k == t + 10 and (cl - entry) < 0.5 * risk:
                return ((cl - entry) / risk, k, "fail_time")
        else:
            if not booked_half:
                if op <= stop:
                    return ((op - entry) / risk, k, "gap_stop")
                if lo <= stop:
                    return (-1.0, k, "stop")
                if hi >= one_r:
                    booked_half = True          # half off at +1R
                    trail = entry               # breakeven on the rest
                elif ft and k == t + 10 and (cl - entry) < 0.5 * risk:
                    return ((cl - entry) / risk, k, "fail_time")
            else:
                if op <= trail:
                    return (0.5 + 0.5 * (op - entry) / risk, k, "trail_gap")
                if lo <= trail:
                    return (0.5 + 0.5 * (trail - entry) / risk, k, "trail")
                trail = max(trail, x["low10"][k])
    k = min(t + hold_cap, n) - 1
    r = (c[k] - entry) / risk
    return ((0.5 + 0.5 * r) if (partial and booked_half) else r, k, "time")


# ---------------------------------------------------------------- generators

def liquid(x, t):
    return (x["c"][t] * x["vol20"][t] / 1e7 >= MIN_TURNOVER_CR
            and not np.isnan(x["e200"][t]) and x["c"][t] > x["e200"][t])


def gen_breakout(sym, x, rs_col):
    """D20 breakouts with metadata; yields dict per signal day."""
    n = len(x["c"])
    for t in range(max(n - LOOKBACK - 1, 210), n - 1):
        if np.isnan(x["don55"][t]) or not liquid(x, t):
            continue
        if x["c"][t] <= x["don20"][t]:
            continue
        vr = x["v"][t] / x["vol20"][t] if x["vol20"][t] > 0 else 0
        rs = rs_col[t] if not np.isnan(rs_col[t]) else 0
        yield {"t": t, "d55": x["c"][t] > x["don55"][t], "rs": rs,
               "vol_ratio": vr, "rsi": x["rsi14"][t]}


def gen_pullback(sym, x, rs_col):
    """RS>=80 leader pulling back to a rising 20EMA, reclaim day."""
    n = len(x["c"])
    for t in range(max(n - LOOKBACK - 1, 210), n - 1):
        if not liquid(x, t):
            continue
        rs = rs_col[t] if not np.isnan(rs_col[t]) else 0
        if rs < 80:
            continue
        e20, e50, c, l = x["e20"], x["e50"], x["c"], x["l"]
        if not (e50[t] > e50[t - 5] and e20[t] > e20[t - 5]):
            continue
        touched = any(l[t - j] <= e20[t - j] * 1.005 for j in range(0, 3))
        if touched and c[t] > e20[t] and c[t] > c[t - 1]:
            yield {"t": t, "rs": rs}


def gen_pivot(sym, x, rs_col):
    """Episodic pivot: +8% day on 3x volume closing near the high."""
    n = len(x["c"])
    for t in range(max(n - LOOKBACK - 1, 210), n - 1):
        c, h, l, v = x["c"], x["h"], x["l"], x["v"]
        if x["vol20"][t] <= 0 or np.isnan(x["e200"][t]):
            continue
        day_ret = c[t] / c[t - 1] - 1
        rng = h[t] - l[t]
        if (day_ret >= 0.08 and v[t] >= 3 * x["vol20"][t] and rng > 0
                and c[t] >= l[t] + 0.75 * rng
                and c[t] * x["vol20"][t] / 1e7 >= MIN_TURNOVER_CR):
            yield {"t": t, "pivot_low": l[t]}


def gen_washout(sym, x, rs_col):
    """RSI(2)<10 above rising 50EMA and 200EMA."""
    n = len(x["c"])
    for t in range(max(n - LOOKBACK - 1, 210), n - 1):
        if not liquid(x, t):
            continue
        if x["e50"][t] > x["e50"][t - 5] and x["rsi2"][t] < 10:
            yield {"t": t}


# ---------------------------------------------------------------- trade builders

def entry_open(x, sig):
    t = sig["t"]
    e = x["o"][t + 1]
    if e <= 0 or (e / x["c"][t] - 1) * 100 > GAP_SKIP_PCT:
        return None
    return t + 1, e


def entry_retest(x, sig):
    """Limit at the breakout level, valid 3 sessions."""
    t = sig["t"]
    lim = x["don20"][t]
    n = len(x["o"])
    for k in range(t + 1, min(t + 4, n)):
        if x["o"][k] <= lim:
            return k, x["o"][k]
        if x["l"][k] <= lim:
            return k, lim
    return None


def run_strategy(gen, engine, entry_fn=entry_open, stop_kind="atr", tag=None):
    trades = []
    for sym, x in PREP.items():
        rs_col = RS_MAT[sym].reindex(pd.Index(x["idx"])).to_numpy(float) \
            if sym in RS_MAT.columns else np.full(len(x["c"]), np.nan)
        last_exit = -1
        for sig in gen(sym, x, rs_col):
            if sig["t"] <= last_exit:
                continue
            ent = entry_fn(x, sig)
            if ent is None:
                continue
            te, entry = ent
            if stop_kind == "atr":
                stop = entry - 2 * x["atr"][sig["t"]]
            elif stop_kind == "swing":
                stop = min(x["low10"][sig["t"]], entry - 0.5 * x["atr"][sig["t"]])
                if (entry - stop) / entry > 0.08:
                    continue
            elif stop_kind == "pivot":
                stop = sig["pivot_low"]
                if (entry - stop) / entry > 0.10 or stop >= entry:
                    continue
            elif stop_kind == "atr25":
                stop = entry - 2.5 * x["atr"][sig["t"]]
            res = walk(x, te, entry, stop, engine)
            if res is None:
                continue
            r, exit_i, reason = res
            last_exit = exit_i
            trades.append({
                "sym": sym, "sector": INDUSTRY.get(sym, "—"),
                "date": x["idx"][sig["t"]],
                "entry_date": x["idx"][te], "exit_date": x["idx"][exit_i],
                "entry": entry, "risk_pct": (entry - stop) / entry * 100,
                "r": r, "reason": reason,
                "ret_pct": r * (entry - stop) / entry * 100,
                **{k: v for k, v in sig.items() if k not in ("t", "pivot_low")},
            })
    return pd.DataFrame(trades)


# ---------------------------------------------------------------- evaluation

def net(df):
    d = df.copy()
    d["ret_net"] = d["ret_pct"] - COST_PCT
    return d


def stats(df, label):
    if df.empty:
        return {"variant": label, "trades": 0}
    d = net(df)
    wins = d[d.ret_net > 0]
    loss = d[d.ret_net <= 0]
    pf = wins.ret_net.sum() / -loss.ret_net.sum() if len(loss) and loss.ret_net.sum() < 0 else np.inf
    return {"variant": label, "trades": len(d),
            "win%": round(100 * len(wins) / len(d), 1),
            "avg_net%": round(d.ret_net.mean(), 3),
            "avg_R": round(d.r.mean(), 3),
            "PF_net": round(pf, 2),
            "hold_d": round((d.exit_date - d.entry_date).dt.days.mean(), 1)}


def portfolio(df, label, slots=10, sector_cap=3, heat_cap=5, prefer=None):
    """10 slots, 10% equity, net costs, max 3/sector, total open risk <= 5 x 1%."""
    if df.empty:
        return {"variant": label, "taken": 0, "ret%": 0.0}
    d = net(df).sort_values(["entry_date"] + ([prefer] if prefer else []),
                            ascending=[True, False] if prefer else [True])
    equity, open_pos, taken = 100.0, [], 0
    by_entry = {k: g for k, g in d.groupby("entry_date")}
    for day in sorted(set(d.entry_date) | set(d.exit_date)):
        done = [p for p in open_pos if p[0] <= day]
        open_pos = [p for p in open_pos if p[0] > day]
        for (_, alloc, ret, _) in done:
            equity += alloc * ret / 100.0
        for r in by_entry.get(day, pd.DataFrame()).itertuples():
            if len(open_pos) >= slots:
                break
            if sum(1 for p in open_pos if p[3] == r.sector) >= sector_cap:
                continue
            open_pos.append((r.exit_date, equity / slots, r.ret_net, r.sector))
            taken += 1
    for (_, alloc, ret, _) in open_pos:
        equity += alloc * ret / 100.0
    return {"variant": label, "taken": taken, "ret%": round(equity - 100, 2)}


def agrade(df):
    return df[df.d55 & (df.rs >= 80) & (df.vol_ratio >= 1.5) & (df.rsi < 78)]


def gated(df, gate_series):
    g = (gate_series.reindex(pd.DatetimeIndex(df["date"]))
         .fillna(False).astype(bool).to_numpy())
    return df[g]


def show(rows, title):
    print(f"\n=== {title} ===")
    print(pd.DataFrame(rows).to_string(index=False))
    return rows


def save(stage, rows):
    merged = json.loads(OUT.read_text()) if OUT.exists() else {}
    merged[stage] = rows
    OUT.write_text(json.dumps(merged, indent=1, default=str))


# ---------------------------------------------------------------- stages

def stage_exits():
    rows = []
    base = run_strategy(gen_breakout, "fixed")
    for eng in ("fixed", "ftstop", "partial", "partial_ft"):
        df = base if eng == "fixed" else run_strategy(gen_breakout, eng)
        ag = gated(agrade(df), EMA_GATE)
        rows.append({**stats(ag, f"A-grade riskON | {eng}"),
                     **{"port%": portfolio(ag, "", prefer="rs")["ret%"]}})
        d55 = df[df.d55]
        rows.append({**stats(d55, f"D55 all | {eng}"),
                     **{"port%": portfolio(d55, "", prefer="vol_ratio")["ret%"]}})
    save("exits", show(rows, "EXIT ENGINES (net of costs)"))


def stage_entries():
    rows = []
    for name, fn in (("next_open", entry_open), ("retest_3d", entry_retest)):
        df = run_strategy(gen_breakout, "fixed", entry_fn=fn)
        ag = gated(agrade(df), EMA_GATE)
        rows.append({**stats(ag, f"A-grade riskON | {name}"),
                     **{"port%": portfolio(ag, "", prefer="rs")["ret%"]}})
    save("entries", show(rows, "ENTRY STYLE (fixed 2R, net)"))


def stage_gates():
    df = run_strategy(gen_breakout, "fixed")
    ag = agrade(df)
    pct50_rising = PCT50 > PCT50.shift(5)
    gates = {
        "no gate": pd.Series(True, index=PCT50.index),
        "nifty>50&200ema": EMA_GATE,
        "breadth>55%+rising": (PCT50 >= 0.55) & pct50_rising,
        "breadth>50%+rising": (PCT50 >= 0.50) & pct50_rising,
        "breadth rising only": pct50_rising,
        "dist_days<6": DIST25 < 6,
        "breadth>50%rising & dd<6": (PCT50 >= 0.50) & pct50_rising & (DIST25 < 6),
    }
    rows = []
    for name, g in gates.items():
        sub = gated(ag, g)
        rows.append({**stats(sub, f"A-grade | {name}"),
                     **{"port%": portfolio(sub, "", prefer="rs")["ret%"]}})
    save("gates", show(rows, "REGIME GATES (A-grade, fixed 2R, net)"))


def stage_sleeves():
    rows = []
    # leader pullback — swing-low stop, test fixed and partial
    for eng in ("fixed", "partial_ft"):
        pb = run_strategy(gen_pullback, eng, stop_kind="swing")
        rows.append({**stats(pb, f"pullback all | {eng}"),
                     **{"port%": portfolio(pb, "", prefer="rs")["ret%"]}})
        weak = gated(pb, ~EMA_GATE)
        rows.append(stats(weak, f"pullback weak-tape | {eng}"))
    # episodic pivot
    for eng in ("fixed", "partial_ft"):
        pv = run_strategy(gen_pivot, eng, stop_kind="pivot")
        rows.append({**stats(pv, f"pivot | {eng}"),
                     **{"port%": portfolio(pv, "")["ret%"]}})
    # washout (own exit)
    wo = run_strategy(gen_washout, "rsi2exit", stop_kind="atr25")
    rows.append({**stats(wo, "washout all"),
                 **{"port%": portfolio(wo, "")["ret%"]}})
    weak = gated(wo, ~EMA_GATE)
    rows.append(stats(weak, "washout weak-tape only"))
    save("sleeves", show(rows, "NEW SLEEVES (net of costs)"))


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    fns = {"exits": stage_exits, "entries": stage_entries,
           "gates": stage_gates, "sleeves": stage_sleeves}
    if stage == "all":
        for f in fns.values():
            f()
    else:
        fns[stage]()
    print("\nsaved ->", OUT)
