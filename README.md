# NSE Breakout Screener (top ~1,250 by turnover)

Daily EOD breakout screener for Indian equity cash market positions.

## Run

Automated: the Windows scheduled task **NSE-Breakout-Screener** runs `run_daily.bat`
weekdays at 18:30 — it fetches the latest bhavcopy + F&O list, rescans everything, and
regenerates `dashboard.html`. Log: `data/update_log.txt`.

Manual: `python daily_update.py` (fetch fresh data + scan) or `python screener.py`
(scan with existing data). ~8-12 min. Then open `dashboard.html` in any browser.

## Dashboard (CBSL house style)

Three tabs: **Call Cards** (report page-1 format: Action/CMP/TGT1-TGT2/SL strip, Key Data,
Daily Oscillator Direction, Weekly Timeframe box, auto-drafted TECHNICAL RATIONALE,
copy-report-text button), **Screener Table**, **Desk Track Record** (394 published calls
crawled from canmoney.in, 64% resolved hit rate).

Expert context per call: weekly trend vs 20W EMA + weekly RSI + 10-week-high flag,
nearest swing-high resistance (drawn on chart; rationale flags if it blocks TGT1),
A/D rating A-E (25-session up/down volume balance), RS trend arrow (vs Nifty, 21 sessions),
and a sector-rotation strip (breakout count + median RS per industry).

## Legacy notes

- **Regime banner** (Nifty vs 50/200 EMA) + **breadth gauges** (% above 200/50 EMA,
  advancers, new 52w highs) — how aggressive to be today
- **★ A-grade spotlight** cards with mini candle charts (the proven subset)
- Click any row → **full candlestick chart** with EMA20/50, D20 trigger and
  entry/stop/target drawn, plus indicator grid
- **Position sizing**: set capital + %-risk once (saved in browser) → live Qty column
- **Star** stocks to build a watchlist (saved in browser); filters, search, CSV export

## Backtest

`python backtest.py [sessions]` — 63 = 3 months (default), 490 = 2 years. Compares
fixed-2R vs trailing exits across D20/D55/A-grade/A-grade-risk-on. Key findings (2y):
raw breakouts lose money; **A-grade + risk-on regime +20%** vs flat Nifty; fixed 2R
target beats trailing stops. Report: `backtest.html`.

## Universe

Every EQ-series stock in the official NSE bhavcopy, ranked by turnover, top 1,250 taken.
Self-updating — no stale index list. ETFs excluded via the official NSE ETF list.
Refresh `data/bhavcopy.csv` for the latest session to update the universe, delivery %
and turnover ranking:
`https://archives.nseindia.com/products/content/sec_bhavdata_full_DDMMYYYY.csv`

## Signals (all require close > 200 EMA + 20-day avg turnover >= Rs 3 cr)

| Tag   | Setup |
|-------|-------|
| D55   | Close above prior 55-day Donchian upper channel (strongest) |
| D20   | Close above prior 20-day high |
| 52WH  | Close above prior 52-week high |
| EMA50 | Close crossed back above 50 EMA |
| GC    | 50 EMA crossed above 200 EMA within last 5 sessions |
| MACD  | MACD crossed above signal within last 3 sessions |
| NEAR  | Within 1.5% below the D20 channel — tomorrow's watchlist |
| WATCH | No signal yet, but a pattern is coiling within 5% of the trigger |

## Chart patterns (heuristic — always confirm on your own chart)

| Tag    | Pattern |
|--------|---------|
| VCP    | Volatility contraction: three successively tighter 15-bar ranges near highs |
| FLAG   | >=15% run-up, then tight <=7% drift, breaking the flag high |
| BASE   | 4-week flat base (<=10%) near 52w high, breaking the base high |
| DBLBOT | Two matched swing lows (>=8% deep), neckline broken |
| COIL   | NR7 + inside bar near highs |
| SQZ    | Bollinger band width in tightest quintile of 6 months |

## Context columns

- **RS** — blended 1/3/6-month momentum percentile vs the whole universe (>=80 = leader)
- **RSI(14)**, **ADX(14)** (down-arrow = -DI above +DI), **Vol×** (vs 20-day avg)
- **Dlv%** — delivery percentage last session, from NSE bhavcopy (>=50 highlighted:
  breakout + high delivery + high volume = genuine accumulation)
- **ATR%** — for position sizing
- **F&O** badge — stock has futures/options (hedgeable); from `data/fo_symbols.txt`

## Trade levels

Entry = CMP, Stop = close - 2xATR(14), Target = 2R.

## Files

- `screener.py` — scan logic; tunables at the top
- `template.html` — dashboard shell; `dashboard.html` regenerated every run
- `data/bhavcopy.csv` — NSE full bhavcopy (universe + delivery %)
- `data/etf_list.csv` — official NSE ETF list (excluded)
- `data/fo_symbols.txt` — F&O stock list (from F&O bhavcopy)
- `data/totalmarket.csv`, `data/nifty500.csv` — industry name mapping
- `data/results.json` — raw scan output

Screener output, not investment advice — confirm on your own charts before entry.
