"""
Daily automation: fetch the latest NSE bhavcopy (universe + delivery %),
refresh the F&O list, then run the full breakout scan.

Run manually with `python daily_update.py`, or via the scheduled task
"NSE-Breakout-Screener" (weekdays 18:30, created by run_daily.bat).
"""

import io
import zipfile
from datetime import date, timedelta

import requests

import screener

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
BHAV_URL = "https://archives.nseindia.com/products/content/sec_bhavdata_full_{d}.csv"
FO_URL = ("https://archives.nseindia.com/content/fo/"
          "BhavCopy_NSE_FO_0_0_0_{d}_F_0000.csv.zip")
ETF_URL = "https://archives.nseindia.com/content/equities/eq_etfseclist.csv"


def latest_trading_day_files():
    """Try today backwards up to 7 days for a published bhavcopy."""
    for back in range(0, 8):
        d = date.today() - timedelta(days=back)
        if d.weekday() >= 5:          # weekend
            continue
        url = BHAV_URL.format(d=d.strftime("%d%m%Y"))
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
        except requests.RequestException as e:
            print(f"  {d}: request failed ({e})")
            continue
        if r.status_code == 200 and r.text.lstrip().startswith("SYMBOL"):
            screener.BHAVCOPY.write_bytes(r.content)
            print(f"bhavcopy: {d} ({len(r.content) // 1024} KB)")
            return d
        print(f"  {d}: not published (HTTP {r.status_code})")
    raise RuntimeError("no bhavcopy found in the last 7 days")


def refresh_fo_symbols(d):
    """Best effort — keep the old list if the F&O bhavcopy isn't available."""
    try:
        r = requests.get(FO_URL.format(d=d.strftime("%Y%m%d")),
                         headers=HEADERS, timeout=60)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            csv_bytes = z.read(z.namelist()[0])
        import pandas as pd
        fo = pd.read_csv(io.BytesIO(csv_bytes))
        syms = sorted(fo.loc[fo["FinInstrmTp"] == "STF", "TckrSymb"].unique())
        if len(syms) > 100:
            screener.FO_SYMBOLS.write_text("\n".join(syms), encoding="utf-8")
            print(f"F&O list: {len(syms)} symbols")
    except Exception as e:
        print(f"F&O list refresh skipped ({e}) — keeping previous list")


def refresh_etf_list():
    """Best effort weekly-ish refresh of the ETF exclusion list."""
    try:
        r = requests.get(ETF_URL, headers=HEADERS, timeout=30)
        if r.status_code == 200 and r.text.lstrip().startswith("Symbol"):
            (screener.DATA_DIR / "etf_list.csv").write_bytes(r.content)
            print("ETF list refreshed")
    except Exception as e:
        print(f"ETF list refresh skipped ({e})")


def main():
    print(f"=== daily update {date.today()} ===")
    try:
        d = latest_trading_day_files()
        refresh_fo_symbols(d)
        refresh_etf_list()
    except Exception as e:
        # NSE archives sometimes block cloud IPs — scan with the last
        # committed bhavcopy rather than failing the whole run.
        print(f"bhavcopy refresh failed ({e}) — using existing data files")
    screener.main()


if __name__ == "__main__":
    main()
