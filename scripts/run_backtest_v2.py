"""提案A（ルールベース戦略高度化）バックテスト実行エントリスクリプト

使用方法:
    python scripts/run_backtest_v2.py --ticker 7203
    python scripts/run_backtest_v2.py --all
"""

import sys
import argparse
from pathlib import Path

# ルートディレクトリをPYTHONPATHに追加
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src_v2.config import load_config, DATA_DIR
from src_v2.backtester import (
    BacktestEngine,
    load_ticker_data,
    print_backtest_report,
)
from src_v2.backtester.data_handler import get_available_tickers


def run_single_ticker(ticker: str, config: dict):
    """単一銘柄のバックテスト実行"""
    print(f"\n--- Backtesting Ticker: {ticker} ---")
    try:
        df = load_ticker_data(ticker, data_dir=str(DATA_DIR))
        if df.empty:
            print(f"Skipping {ticker}: No market data found.")
            return None

        engine = BacktestEngine(config)
        result = engine.run(ticker, df)
        print_backtest_report(ticker, result["metrics"])
        return result
    except Exception as e:
        print(f"Error testing {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Run Proposal A Backtest (v2)")
    parser.add_argument("--ticker", type=str, default="7203", help="Ticker symbol (e.g., 7203)")
    parser.add_argument("--all", action="store_true", help="Run backtest on all available tickers in market_data")
    parser.add_argument("--config", type=str, default=None, help="Custom config json path")

    args = parser.parse_args()

    config = load_config() if args.config is None else load_config(Path(args.config))

    if args.all:
        tickers = get_available_tickers(data_dir=str(DATA_DIR))
        if not tickers:
            print(f"No market data tickers found in {DATA_DIR}")
            return

        summary = []
        for ticker in sorted(tickers):
            res = run_single_ticker(ticker, config)
            if res and res.get("metrics"):
                metrics = res["metrics"]
                metrics["ticker"] = ticker
                summary.append(metrics)

        print("\n" + "=" * 80)
        print("                        ALL TICKERS SUMMARY REPORT (Proposal A)")
        print("=" * 80)
        for metrics in summary:
            print(
                f"Ticker: {metrics.get('ticker', 'N/A'):<6} | "
                f"Trades: {metrics['total_trades']:<3} | "
                f"WinRate: {metrics['win_rate_pct']:>5.1f}% | "
                f"PF: {metrics['profit_factor']:>4.2f} | "
                f"PnL: {metrics['total_pnl_yen']:>9,.0f} JPY | "
                f"Sharpe: {metrics['sharpe_ratio']:>4.2f}"
            )
        print("=" * 80)
    else:
        run_single_ticker(args.ticker, config)


if __name__ == "__main__":
    main()
