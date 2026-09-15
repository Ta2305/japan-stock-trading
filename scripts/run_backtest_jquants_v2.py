"""J-Quants 1分足データ対応 高速バックテスト＆モデル評価スクリプト

market_data/jquants/bars_1m/ に保存された過去2年分の1分足Parquetファイル、
および tickers.csv に登録された高流動性銘柄を読み込み、現行の Meta-labeling 機械学習モデル
(saved_models/) のパフォーマンス（勝率, PF, 損益, ドローダウン）を評価・計測します。

使用例:
    python scripts/run_backtest_jquants_v2.py --tickers tickers.csv --max-stocks 50
"""

import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src_v2.strategy.trend_follow import AdvancedTrendFollowStrategy
from src_v2.backtester.engine import BacktestEngine

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("run_backtest_jquants_v2")


def load_jquants_bars_for_ticker(ticker_symbol: str, bars_dir: Path, max_days: int = 60) -> pd.DataFrame:
    """J-Quants 1分足 Parquet ディレクトリから特定銘柄の時系列1分足を抽出・変換する"""
    raw_code = ticker_symbol.replace(".T", "")
    code_5digit = raw_code + "0" if len(raw_code) <= 4 else raw_code
    parquet_files = sorted(list(bars_dir.glob("*.parquet")))[-max_days:]

    if not parquet_files:
        return pd.DataFrame()

    dfs = []
    for p_file in parquet_files:
        try:
            day_df = pd.read_parquet(p_file)
            day_df["Code"] = day_df["Code"].astype(str)
            ticker_df = day_df[day_df["Code"] == code_5digit]
            if not ticker_df.empty:
                dfs.append(ticker_df)
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    full_df = pd.concat(dfs, ignore_index=True)
    full_df["datetime"] = pd.to_datetime(full_df["Date"] + " " + full_df["Time"])
    full_df = full_df.rename(columns={
        "O": "open", "H": "high", "L": "low", "C": "close", "Vo": "volume"
    })
    
    clean_df = full_df[["datetime", "open", "high", "low", "close", "volume"]].sort_values("datetime").reset_index(drop=True)
    return clean_df


def main():
    parser = argparse.ArgumentParser(
        description="Run Meta-labeling Model Evaluation on J-Quants 1-minute bars dataset."
    )
    parser.add_argument(
        "--bars-dir",
        type=str,
        default="market_data/jquants/bars_1m",
        help="Path to J-Quants 1m bars directory",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default="tickers.csv",
        help="Path to screened tickers CSV file",
    )
    parser.add_argument(
        "--max-stocks",
        type=int,
        default=50,
        help="Number of stocks to evaluate in backtest (Default: 50)",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=60,
        help="Number of recent trading days to test per stock (Default: 60)",
    )

    args = parser.parse_args()
    bars_dir = Path(args.bars_dir)
    tickers_file = Path(args.tickers)

    if not tickers_file.exists():
        logger.error(f"Tickers file {tickers_file} not found!")
        return

    df_tickers = pd.read_csv(tickers_file)
    ticker_list = df_tickers["ticker"].tolist()[:args.max_stocks]

    logger.info("==============================================================")
    logger.info("   J-QUANTS 1M BARS BASELINE BACKTEST & EVALUATION START")
    logger.info(f" Target Stocks Count: {len(ticker_list)}")
    logger.info(f" Recent Days per Stock: {args.max_days}")
    logger.info("==============================================================")

    config = {
        "strategy": {
            "short_window": 5,
            "long_window": 20,
            "atr_period": 14,
        },
        "risk_management": {
            "initial_cash": 1_000_000.0,
            "stop_loss_atr_multiplier": 1.5,
            "take_profit_atr_multiplier": 2.5,
            "trailing_stop_enabled": True,
            "trailing_stop_atr_multiplier": 1.5,
            "risk_per_trade_pct": 1.0,
        },
        "execution": {
            "use_slippage": True,
        }
    }

    strategy = AdvancedTrendFollowStrategy(config=config["strategy"])
    engine = BacktestEngine(config=config, model_dir="saved_models")
    engine.strategy = strategy

    stock_results = []
    total_trades = 0
    winning_trades = 0
    total_pnl = 0.0
    gross_profit = 0.0
    gross_loss = 0.0

    pbar = tqdm(ticker_list, desc="Backtesting Stocks")
    for ticker in pbar:
        df_bars = load_jquants_bars_for_ticker(ticker, bars_dir, max_days=args.max_days)
        if df_bars.empty or len(df_bars) < 50:
            continue

        try:
            result = engine.run(ticker, df_bars)
            metrics = result.get("metrics", {})
            trades = result.get("trades", [])

            n_tr = metrics.get("total_trades", len(trades))
            pnl = metrics.get("total_pnl", sum(t.pnl for t in trades if hasattr(t, "pnl")))

            if n_tr > 0:
                total_trades += n_tr
                wins = sum(1 for t in trades if getattr(t, "pnl", 0) > 0)
                winning_trades += wins
                total_pnl += pnl

                g_prof = sum(getattr(t, "pnl", 0) for t in trades if getattr(t, "pnl", 0) > 0)
                g_loss = abs(sum(getattr(t, "pnl", 0) for t in trades if getattr(t, "pnl", 0) < 0))
                gross_profit += g_prof
                gross_loss += g_loss

                stock_results.append({
                    "ticker": ticker,
                    "trades": n_tr,
                    "win_rate": (wins / n_tr) * 100 if n_tr > 0 else 0.0,
                    "pnl": pnl,
                    "pf": (g_prof / g_loss) if g_loss > 0 else np.nan,
                })
        except Exception as e:
            logger.warning(f"Failed backtest for {ticker}: {e}")
            continue

    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
    overall_pf = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)

    logger.info("\n==============================================================")
    logger.info("   BASELINE EVALUATION SUMMARY RESULTS (J-QUANTS 1M BARS)")
    logger.info("==============================================================")
    logger.info(f" Total Evaluated Stocks : {len(stock_results)} / {len(ticker_list)}")
    logger.info(f" Total Trades Executed  : {total_trades}")
    logger.info(f" Overall Win Rate       : {win_rate:.2f}% ({winning_trades} / {total_trades})")
    logger.info(f" Gross Profit           : +¥{gross_profit:,.0f}")
    logger.info(f" Gross Loss             : -¥{gross_loss:,.0f}")
    logger.info(f" Total Net PnL          : ¥{total_pnl:+,.0f}")
    logger.info(f" Profit Factor (PF)     : {overall_pf:.2f}")
    logger.info("==============================================================")

    res_df = pd.DataFrame(stock_results)
    if not res_df.empty:
        out_summary = Path("docs/baseline_evaluation_jquants_results.csv")
        res_df.to_csv(out_summary, index=False)
        logger.info(f"Saved baseline evaluation summary to {out_summary.resolve()}")


if __name__ == "__main__":
    main()
