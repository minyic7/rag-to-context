"""
fetch_financials.py — Pull CBA.AX financial data via yfinance
Saves to demo/data/:
  balance_sheet.csv
  income_statement.csv
  cash_flow.csv
  key_metrics.csv   (summary of key ratios/stats)

Usage:
  uv run python crawler/fetch_financials.py --out data
"""

import argparse
from pathlib import Path
import pandas as pd
import yfinance as yf

TICKER = "CBA.AX"


def fetch_and_save(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ticker = yf.Ticker(TICKER)

    tables = {
        "balance_sheet":     ticker.balance_sheet,
        "income_statement":  ticker.financials,
        "cash_flow":         ticker.cashflow,
    }

    for name, df in tables.items():
        if df is None or df.empty:
            print(f"  [skip] {name} — no data returned")
            continue

        # Columns are timestamps → convert to "FY YYYY" labels
        df = df.copy()
        df.columns = [
            f"FY{c.year}" if hasattr(c, "year") else str(c)
            for c in df.columns
        ]
        df.index.name = "metric"

        path = out_dir / f"{name}.csv"
        df.to_csv(path)
        print(f"  [saved] {path}  ({len(df)} rows × {len(df.columns)} years)")

    # ── Key metrics snapshot ──────────────────────────────────────────────
    info = ticker.info or {}
    metrics = {
        "ticker":                TICKER,
        "company_name":          info.get("longName", "Commonwealth Bank of Australia"),
        "market_cap_aud":        info.get("marketCap"),
        "pe_ratio":              info.get("trailingPE"),
        "eps":                   info.get("trailingEps"),
        "dividend_yield":        info.get("dividendYield"),
        "52w_high":              info.get("fiftyTwoWeekHigh"),
        "52w_low":               info.get("fiftyTwoWeekLow"),
        "book_value_per_share":  info.get("bookValue"),
        "price_to_book":         info.get("priceToBook"),
        "return_on_equity":      info.get("returnOnEquity"),
        "return_on_assets":      info.get("returnOnAssets"),
        "sector":                info.get("sector"),
        "industry":              info.get("industry"),
    }
    metrics_df = pd.DataFrame([metrics]).T
    metrics_df.columns = ["value"]
    metrics_df.index.name = "metric"
    path = out_dir / "key_metrics.csv"
    metrics_df.to_csv(path)
    print(f"  [saved] {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data", help="Output directory")
    args = parser.parse_args()

    print(f"Fetching {TICKER} financials from Yahoo Finance...")
    fetch_and_save(Path(args.out))
    print("Done.")


if __name__ == "__main__":
    main()
