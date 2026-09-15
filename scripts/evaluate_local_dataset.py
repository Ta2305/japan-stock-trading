"""ローカル保存済み J-Quants 1分足 ＆ 日足データセット完全対応 バックテスト・評価スクリプト

ローカルの market_data/jquants/ に保存された全データセットから、
銘柄コード (例: 7203, 9984, 8035, 8306 等) の時系列1分足を外部API接続なしで高速抽出し、
現行 Meta-labeling モデルおよび改良モデルの定量評価を行います。
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
logger = logging.getLogger("evaluate_local_dataset")


def extract_ticker_1m_bars_from_local(ticker_symbol: str, bars_dir: Path, max_days: int = 60) -> pd.DataFrame:
    """ローカル保存済みの Parquet 群から特定銘柄の1分足データを抽出する"""
    raw_code = ticker_symbol.replace(".T", "")
    target_codes = {raw_code + "0", raw_code}

    parquet_files = sorted(list(bars_dir.glob("*.parquet")))
    if not parquet_files:
        return pd.DataFrame()

    selected_files = [p for p in parquet_files if p.name != "download_status.json"][-max_days:]

    dfs = []
    for p_file in selected_files:
        try:
            day_df = pd.read_parquet(p_file)
            if "Code" not in day_df.columns:
                continue
            day_df["Code"] = day_df["Code"].astype(str)
            sub = day_df[day_df["Code"].isin(target_codes)]
            if not sub.empty:
                dfs.append(sub)
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    full_df = pd.concat(dfs, ignore_index=True)
    
    if "Date" in full_df.columns and "Time" in full_df.columns:
        full_df["datetime"] = pd.to_datetime(full_df["Date"] + " " + full_df["Time"])
    elif "Date" in full_df.columns:
        full_df["datetime"] = pd.to_datetime(full_df["Date"])
    
    full_df = full_df.rename(columns={
        "O": "open", "H": "high", "L": "low", "C": "close", "Vo": "volume", "V": "volume"
    })

    required_cols = ["datetime", "open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col not in full_df.columns:
            return pd.DataFrame()

    clean_df = full_df[required_cols].sort_values("datetime").drop_duplicates(subset=["datetime"]).reset_index(drop=True)
    return clean_df


def main():
    parser = argparse.ArgumentParser(
        description="Run local backtest evaluation using offline J-Quants parquet files."
    )
    parser.add_argument(
        "--bars-dir",
        type=str,
        default="market_data/jquants/bars_1m",
        help="Path to local 1m bars directory",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default="tickers.csv",
        help="Path to tickers CSV",
    )
    parser.add_argument(
        "--max-stocks",
        type=int,
        default=20,
        help="Number of stocks to evaluate (Default: 20)",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=60,
        help="Number of past days to extract per stock (Default: 60)",
    )

    args = parser.parse_args()
    bars_dir = ROOT_DIR / args.bars_dir
    tickers_file = ROOT_DIR / args.tickers

    if not tickers_file.exists():
        logger.error(f"Tickers file {tickers_file} not found!")
        return

    df_tickers = pd.read_csv(tickers_file)
    ticker_list = df_tickers["ticker"].tolist()

    logger.info("==============================================================")
    logger.info("   LOCAL OFFLINE DATASET BASELINE EVALUATION START")
    logger.info(f" Target Stock Candidates : {len(ticker_list)}")
    logger.info(f" Max Stocks to Evaluate  : {args.max_stocks}")
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

    total_trades = 0
    winning_trades = 0
    total_pnl = 0.0
    gross_profit = 0.0
    gross_loss = 0.0

    count_evaluated = 0
    pbar = tqdm(ticker_list, desc="Evaluating Stocks")
    out_summary = ROOT_DIR / "docs" / "baseline_evaluation_local_jquants_results.csv"
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    if out_summary.exists():
        out_summary.unlink()

    for ticker in pbar:
        if count_evaluated >= args.max_stocks:
            break

        df_bars = extract_ticker_1m_bars_from_local(ticker, bars_dir, max_days=args.max_days)
        if df_bars.empty or len(df_bars) < 100:
            continue

        count_evaluated += 1

        try:
            result = engine.run(ticker, df_bars)
            trades = result.get("trades", [])

            n_tr = len(trades)
            pnl = sum(getattr(t, "pnl", 0) for t in trades)
            wins = sum(1 for t in trades if getattr(t, "pnl", 0) > 0)
            g_prof = sum(getattr(t, "pnl", 0) for t in trades if getattr(t, "pnl", 0) > 0)
            g_loss = abs(sum(getattr(t, "pnl", 0) for t in trades if getattr(t, "pnl", 0) < 0))

            total_trades += n_tr
            winning_trades += wins
            total_pnl += pnl
            gross_profit += g_prof
            gross_loss += g_loss

            row_data = pd.DataFrame([{
                "ticker": ticker,
                "bars_count": len(df_bars),
                "trades": n_tr,
                "win_rate": (wins / n_tr) * 100 if n_tr > 0 else 0.0,
                "pnl": pnl,
                "pf": (g_prof / g_loss) if g_loss > 0 else np.nan,
            }])
            row_data.to_csv(out_summary, mode="a", index=False, header=not out_summary.exists())
        except Exception as e:
            logger.warning(f"Failed evaluation for {ticker}: {e}")
            continue

    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
    overall_pf = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)

    logger.info("\n==============================================================")
    logger.info("   LOCAL OFFLINE EVALUATION SUMMARY RESULTS")
    logger.info("==============================================================")
    logger.info(f" Successfully Evaluated Stocks : {count_evaluated}")
    logger.info(f" Total Trades Executed         : {total_trades}")
    logger.info(f" Overall Win Rate              : {win_rate:.2f}% ({winning_trades} / {total_trades})")
    logger.info(f" Gross Profit                  : +¥{gross_profit:,.0f}")
    logger.info(f" Gross Loss                    : -¥{gross_loss:,.0f}")
    logger.info(f" Total Net PnL                 : ¥{total_pnl:+,.0f}")
    logger.info(f" Profit Factor (PF)            : {overall_pf:.2f}")
    logger.info("==============================================================")


if __name__ == "__main__":
    main()
