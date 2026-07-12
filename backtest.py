"""
Backtest of the breakout screener rules over the last N trading sessions.

Replays signals bar-by-bar on the same top-turnover universe:
  signal on day T (close > prior Donchian high, close > 200 EMA, turnover filter)
  -> enter next day's OPEN
  -> stop  = entry - 2*ATR(14 @ signal day)
  -> target = entry + 2 * risk (2R)
  -> exit at stop/target intraday (both hit same day => stop first, conservative)
  -> time exit at close after MAX_HOLD sessions
  -> one open position per symbol; re-entry allowed after exit
  -> entries gapping >5% above signal close are skipped (no chasing)

Strategies compared:
  D20      all 20-day Donchian breakouts
  D55      all 55-day Donchian breakouts
  A-GRADE  D55 + RS rating >= 80 + vol_ratio >= 1.5 + RSI < 78

Usage:  python backtest.py
Output: console stats + data/backtest_trades.csv + backtest.html
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from screener import (DATA_DIR, HERE, load_universe, ema, wilder, true_range,
                      rsi, adx_di, CHUNK)

HIST_PERIOD = "3y"           # signal window + ~1y indicator warm-up
CACHE = DATA_DIR / f"hist_cache_{HIST_PERIOD}.pkl"
TRADES_CSV = DATA_DIR / "backtest_trades.csv"
REPORT = HERE / "backtest.html"

# sessions to backtest: `python backtest.py 490` = ~2 years; default ~3 months
LOOKBACK_SESSIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 63
MAX_HOLD = 20               # fixed-target exit
MAX_HOLD_TRAIL = 40         # trailing exit: winners need room to run
ATR_STOP_MULT = 2.0
TRAIL_ATR_MULT = 2.5        # chandelier: highest high since entry - 2.5*ATR
REWARD_R = 2.0
MIN_TURNOVER_CR = 3.0
GAP_SKIP_PCT = 5.0


def download_all(symbols):
    if CACHE.exists():
        print("Using cached history:", CACHE.name)
        return pickle.loads(CACHE.read_bytes())
    out = {}
    names = [f"{s}.NS" for s in symbols]
    rev = {f"{s}.NS": s for s in symbols}
    for i in range(0, len(names), CHUNK):
        chunk = names[i:i + CHUNK]
        print(f"  downloading {i + 1}-{min(i + len(chunk), len(names))} of {len(names)} ...")
        data = yf.download(chunk, period=HIST_PERIOD, interval="1d", group_by="ticker",
                           auto_adjust=True, threads=True, progress=False)
        for tkr in chunk:
            try:
                df = data[tkr].dropna(how="all")
            except KeyError:
                continue
            if len(df) >= 260:
                out[rev[tkr]] = df
    CACHE.write_bytes(pickle.dumps(out))
    return out


def rs_rank_matrix(hist):
    """Daily cross-sectional RS rating (1-99) from blended 21/63/126-day returns."""
    closes = pd.DataFrame({s: df["Close"] for s, df in hist.items()})
    blend = (0.5 * closes.pct_change(63) + 0.3 * closes.pct_change(126)
             + 0.2 * closes.pct_change(21))
    return blend.rank(axis=1, pct=True).mul(98).add(1)


def simulate(hist, rs_mat, exit_mode="fixed"):
    """exit_mode 'fixed': 2R target / 2xATR stop. 'trail': chandelier stop, no target."""
    max_hold = MAX_HOLD if exit_mode == "fixed" else MAX_HOLD_TRAIL
    trades = []
    for sym, df in hist.items():
        c, h, l, o, v = df["Close"], df["High"], df["Low"], df["Open"], df["Volume"]
        n = len(df)
        if n < 260:
            continue
        ema200 = ema(c, 200)
        don20 = h.rolling(20).max().shift(1)
        don55 = h.rolling(55).max().shift(1)
        vol20 = v.rolling(20).mean()
        atr14 = wilder(true_range(df), 14)
        rsi14 = rsi(c)

        start = max(n - LOOKBACK_SESSIONS - 1, 210)
        in_pos_until = -1
        for t in range(start, n - 1):          # need t+1 for entry
            if t <= in_pos_until:
                continue
            close = float(c.iloc[t])
            if np.isnan(don55.iloc[t]) or np.isnan(ema200.iloc[t]):
                continue
            turn = close * float(vol20.iloc[t]) / 1e7
            if turn < MIN_TURNOVER_CR or close <= float(ema200.iloc[t]):
                continue
            d20 = close > float(don20.iloc[t])
            d55 = close > float(don55.iloc[t])
            if not d20:
                continue

            entry = float(o.iloc[t + 1])
            if entry <= 0 or (entry / close - 1) * 100 > GAP_SKIP_PCT:
                continue
            atr = float(atr14.iloc[t])
            stop = entry - ATR_STOP_MULT * atr
            risk = entry - stop
            if risk <= 0:
                continue
            target = entry + REWARD_R * risk

            # walk forward
            exit_px, exit_i, reason = None, None, None
            trail, hh = stop, entry
            for k in range(t + 1, min(t + 1 + max_hold, n)):
                op, hi, lo, cl = (float(o.iloc[k]), float(h.iloc[k]),
                                  float(l.iloc[k]), float(c.iloc[k]))
                if exit_mode == "fixed":
                    if op <= stop:
                        exit_px, exit_i, reason = op, k, "gap_stop"; break
                    if lo <= stop:                 # stop first if both hit
                        exit_px, exit_i, reason = stop, k, "stop"; break
                    if op >= target:
                        exit_px, exit_i, reason = op, k, "gap_target"; break
                    if hi >= target:
                        exit_px, exit_i, reason = target, k, "target"; break
                else:                              # trailing chandelier
                    if op <= trail:
                        exit_px, exit_i, reason = op, k, "gap_stop"; break
                    if lo <= trail:
                        exit_px, exit_i, reason = trail, k, "trail"; break
                    hh = max(hh, hi)               # ratchet after today survives
                    trail = max(trail, hh - TRAIL_ATR_MULT * atr)
            if exit_px is None:
                k = min(t + max_hold, n - 1)
                exit_px, exit_i = float(c.iloc[k]), k
                reason = "time" if k == t + max_hold else "open"

            vol_ratio = float(v.iloc[t]) / float(vol20.iloc[t]) if vol20.iloc[t] > 0 else 0
            try:
                rs = float(rs_mat.loc[df.index[t], sym])
            except KeyError:
                rs = np.nan
            trades.append({
                "symbol": sym,
                "signal_date": str(df.index[t].date()),
                "entry_date": str(df.index[t + 1].date()),
                "exit_date": str(df.index[exit_i].date()),
                "d55": d55,
                "rs": None if np.isnan(rs) else round(rs),
                "rsi": round(float(rsi14.iloc[t]), 1),
                "vol_ratio": round(vol_ratio, 2),
                "entry": round(entry, 2),
                "stop": round(stop, 2),
                "target": round(target, 2),
                "exit": round(exit_px, 2),
                "reason": reason,
                "hold_days": exit_i - t - 1,
                "ret_pct": round((exit_px / entry - 1) * 100, 2),
                "r_mult": round((exit_px - entry) / risk, 2),
            })
            in_pos_until = exit_i
    return pd.DataFrame(trades)


def portfolio_sim(df, label, slots=10, prefer=None):
    """10-slot, 10%-of-equity-per-trade chronological simulation."""
    if df.empty:
        return {"strategy": label, "taken": 0, "return_pct": 0.0}
    df = df.copy()
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df = df.sort_values(["entry_date"] + ([prefer] if prefer else []),
                        ascending=[True, False] if prefer else [True])
    equity, open_pos, taken = 100.0, [], 0
    by_entry = {d: g for d, g in df.groupby("entry_date")}
    for d in sorted(set(df["entry_date"]) | set(df["exit_date"])):
        open_pos, done = [p for p in open_pos if p[0] > d], \
                         [p for p in open_pos if p[0] <= d]
        for (_, alloc, ret) in done:
            equity += alloc * ret / 100.0
        for r in by_entry.get(d, pd.DataFrame()).itertuples():
            if len(open_pos) >= slots:
                break
            open_pos.append((r.exit_date, equity / slots, r.ret_pct))
            taken += 1
    for (_, alloc, ret) in open_pos:
        equity += alloc * ret / 100.0
    return {"strategy": label, "taken": taken, "return_pct": round(equity - 100, 2)}


def stats(df, label):
    if df.empty:
        return {"strategy": label, "trades": 0}
    closed = df[df["reason"] != "open"]
    wins = df[df["r_mult"] > 0]
    losses = df[df["r_mult"] <= 0]
    gross_win = wins["ret_pct"].sum()
    gross_loss = -losses["ret_pct"].sum()
    return {
        "strategy": label,
        "trades": len(df),
        "still_open": int((df["reason"] == "open").sum()),
        "win_rate": round(100 * len(wins) / len(df), 1),
        "avg_win_pct": round(wins["ret_pct"].mean(), 2) if len(wins) else 0,
        "avg_loss_pct": round(losses["ret_pct"].mean(), 2) if len(losses) else 0,
        "expectancy_pct": round(df["ret_pct"].mean(), 2),
        "expectancy_r": round(df["r_mult"].mean(), 3),
        "total_r": round(df["r_mult"].sum(), 1),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else float("inf"),
        "avg_hold_days": round(df["hold_days"].mean(), 1),
        "portfolio_ret_1pct_risk": round(df["r_mult"].sum() * 1.0, 1),
        "_closed": len(closed),
    }


def benchmark():
    """Nifty return over the window + daily risk-on flag (close > 50 & 200 EMA)."""
    nifty = yf.download("^NSEI", period=HIST_PERIOD, interval="1d",
                        auto_adjust=True, progress=False)["Close"].dropna()
    if isinstance(nifty, pd.DataFrame):
        nifty = nifty.iloc[:, 0]
    riskon = (nifty > ema(nifty, 50)) & (nifty > ema(nifty, 200))
    window = nifty.iloc[-LOOKBACK_SESSIONS:]
    ret = round(float(window.iloc[-1] / window.iloc[0] - 1) * 100, 2)
    return ret, str(window.index[0].date()), str(window.index[-1].date()), riskon


def subsets(trades, riskon):
    t = trades.copy()
    t["_sd"] = pd.to_datetime(t["signal_date"])
    t["riskon"] = t["_sd"].map(riskon).fillna(False)
    d55 = t[t["d55"]]
    ag = t[t["d55"] & (t["rs"].fillna(0) >= 80)
           & (t["vol_ratio"] >= 1.5) & (t["rsi"] < 78)]
    return [("D20 (all)", t, "vol_ratio"), ("D55", d55, "vol_ratio"),
            ("A-GRADE", ag, "rs"), ("A-GRADE risk-on", ag[ag["riskon"]], "rs")]


def build_report(sections, trades, bench, span):
    """sections: list of (title, all_stats, port_stats) per exit mode."""
    def port_table(port_stats):
        return "".join(
            f"<tr><td>{p['strategy']}</td><td>{p['taken']}</td>"
            f"<td class='{'pos' if p['return_pct'] >= 0 else 'neg'}'>"
            f"{p['return_pct']:+.2f}%</td></tr>" for p in port_stats)

    def stats_table(all_stats):
        return "".join(
            "<tr>" + "".join(
                f"<td>{s.get(k, '—')}</td>" for k in
                ("strategy", "trades", "win_rate", "expectancy_pct", "expectancy_r",
                 "total_r", "profit_factor", "avg_win_pct", "avg_loss_pct",
                 "avg_hold_days", "portfolio_ret_1pct_risk", "still_open")
            ) + "</tr>" for s in all_stats)

    section_html = ""
    for title, all_stats, port_stats in sections:
        section_html += f"""
<h2>{title} — realistic portfolio (10 slots, 10% of equity per trade)</h2>
<table><tr><th>Strategy</th><th>Trades taken</th><th>{LOOKBACK_SESSIONS}-session return</th></tr>
{port_table(port_stats)}</table>
<h2>{title} — per-trade stats (all signals, unconstrained)</h2>
<table><tr><th>Strategy</th><th>Trades</th><th>Win%</th><th>Avg trade %</th>
<th>Avg R</th><th>Total R</th><th>Profit factor</th><th>Avg win %</th><th>Avg loss %</th>
<th>Hold (d)</th><th>Sum R %</th><th>Open</th></tr>{stats_table(all_stats)}</table>"""

    top = trades.sort_values("ret_pct", ascending=False).head(15)
    bot = trades.sort_values("ret_pct").head(10)

    def trow(r):
        return (f"<tr><td class='l'>{r.symbol}</td><td>{r.entry_date}</td>"
                f"<td>{r.exit_date}</td><td>{r.entry}</td><td>{r.exit}</td>"
                f"<td class='{'pos' if r.ret_pct >= 0 else 'neg'}'>{r.ret_pct}%</td>"
                f"<td>{r.r_mult}R</td><td>{r.reason}</td></tr>")

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Breakout backtest — {span[0]} to {span[1]}</title><style>
body{{margin:0;padding:24px;background:#0e1117;color:#dbe2ef;font:14px/1.5 "Segoe UI",sans-serif}}
h1{{font-size:20px}} h2{{font-size:16px;margin-top:28px}}
table{{border-collapse:collapse;margin-top:10px}}
th,td{{padding:6px 12px;border-bottom:1px solid #262d3b;text-align:right;white-space:nowrap}}
th{{color:#7d8799;font-size:11.5px}} td.l{{text-align:left;font-weight:700;color:#4f9cf9}}
.pos{{color:#2ecc8f}}.neg{{color:#ff5d73}}
.note{{color:#7d8799;font-size:12.5px;max-width:900px}}</style></head><body>
<h1>Breakout screener backtest — {span[0]} to {span[1]}</h1>
<p class="note">Entry next-day open · stop 2×ATR(14) · target 2R · max hold {MAX_HOLD}
sessions · both-hit-same-day counted as stop (conservative) · entries gapping &gt;5% skipped ·
one position per symbol. Fixed exit: 2R target / 2×ATR stop, max hold {MAX_HOLD}.
Trailing exit: chandelier {TRAIL_ATR_MULT}×ATR from highest high, no target, max hold
{MAX_HOLD_TRAIL}. "Risk-on" = signal day with Nifty above its 50 and 200 EMA.
<b>Nifty 50 over the same window: {bench:+.2f}%</b>.</p>
{section_html}
<h2>Best 15 trades (fixed exit, all D20)</h2>
<table><tr><th>Symbol</th><th>Entry</th><th>Exit</th><th>In</th><th>Out</th><th>Ret</th>
<th>R</th><th>Exit type</th></tr>{"".join(trow(r) for r in top.itertuples())}</table>
<h2>Worst 10 trades (all D20)</h2>
<table><tr><th>Symbol</th><th>Entry</th><th>Exit</th><th>In</th><th>Out</th><th>Ret</th>
<th>R</th><th>Exit type</th></tr>{"".join(trow(r) for r in bot.itertuples())}</table>
<p class="note">Caveats: universe is today's top-1250 by turnover (mild survivorship /
lookahead in universe selection); no brokerage/slippage/STT deducted; delivery% not used
historically. Treat relative performance between strategies as the signal, absolute
numbers as optimistic by ~0.3-0.5% per round trip.</p></body></html>"""
    REPORT.write_text(html, encoding="utf-8")


def main():
    universe = load_universe()
    print(f"Universe: {len(universe)} symbols")
    hist = download_all(list(universe))
    print(f"History for {len(hist)} symbols; computing RS matrix ...")
    rs_mat = rs_rank_matrix(hist)

    bench, b0, b1, riskon = benchmark()
    sections, fixed_trades = [], None
    for mode, title in (("fixed", "Fixed 2R exit"), ("trail", "Trailing chandelier exit")):
        print(f"Simulating trades ({mode}) ...")
        trades = simulate(hist, rs_mat, mode)
        if trades.empty:
            print("No trades generated."); return
        csv_path = TRADES_CSV if mode == "fixed" else \
            TRADES_CSV.with_name("backtest_trades_trail.csv")
        trades.to_csv(csv_path, index=False)
        if mode == "fixed":
            fixed_trades = trades

        subs = subsets(trades, riskon)
        all_stats = [stats(df, label) for label, df, _ in subs]
        port_stats = [portfolio_sim(df, label, prefer=pref) for label, df, pref in subs]
        sections.append((title, all_stats, port_stats))

        print(f"\n{title} — portfolio (10 slots, 10% equity per trade):")
        for p in port_stats:
            print(f"  {p['strategy']:<18} taken {p['taken']:>4}  return {p['return_pct']:+.2f}%")
        cols = ("strategy", "trades", "win_rate", "expectancy_pct", "expectancy_r",
                "total_r", "profit_factor", "avg_hold_days")
        print(pd.DataFrame(all_stats)[list(cols)].to_string(index=False))
        print()

    build_report(sections, fixed_trades, bench, (b0, b1))
    print(f"Nifty 50 same window ({b0} to {b1}): {bench:+.2f}%")
    print(f"\nTrades CSV: {TRADES_CSV} (+ _trail variant)\nReport:     {REPORT}")


if __name__ == "__main__":
    main()
