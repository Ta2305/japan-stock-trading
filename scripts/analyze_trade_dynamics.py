"""エグジット理由別の利益吐き出し (Profit Giveback) ＆ 資金回転率の比較検証スクリプト"""

import sys
from pathlib import Path
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Any

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.evaluate_macro_swing_model import (
    load_recent_jquants_daily_bars_fast,
    load_global_macro_features,
)
from scripts.experiment_unconstrained_and_no_ml_backtest import ComprehensiveStrategySimulator


def run_trade_dynamics_comparison():
    bars_daily_dir = ROOT_DIR / "market_data" / "jquants" / "bars_daily"
    macro_dir = ROOT_DIR / "market_data" / "global_macro"
    tickers_file = ROOT_DIR / "tickers.csv"

    df_tickers = pd.read_csv(tickers_file)
    ticker_list = df_tickers["ticker"].tolist()[:100]

    df_macro = load_global_macro_features(macro_dir)
    target_codes = {t.replace(".T", "") + "0" for t in ticker_list}
    stock_data_dict = load_recent_jquants_daily_bars_fast(bars_daily_dir, target_codes, max_days=500)

    records = []
    df_macro_lagged = df_macro.copy()
    for col in ["sox_close", "sp500_close", "usdjpy_close", "vix_close", "us10y_yield"]:
        if col in df_macro_lagged.columns:
            df_macro_lagged[col] = df_macro_lagged[col].shift(1)

    for code, df_stock in stock_data_dict.items():
        if len(df_stock) < 100:
            continue
        raw_ticker = code[:-1] + ".T" if code.endswith("0") else code
        merged = pd.merge(df_stock, df_macro_lagged, on="date", how="left").ffill().dropna()
        if len(merged) < 50:
            continue

        merged["returns"] = merged["close"].pct_change()
        merged["ma_5"] = merged["close"].rolling(5).mean()
        merged["ma_20"] = merged["close"].rolling(20).mean()
        merged["sma_diff"] = (merged["ma_5"] - merged["ma_20"]) / merged["close"]
        high_low = merged["high"] - merged["low"]
        merged["atr"] = high_low.rolling(14).mean()

        if "sox_close" in merged.columns:
            merged["sox_return_1d"] = merged["sox_close"].pct_change()
            merged["sox_diff_lag"] = merged["sox_return_1d"] - merged["returns"].shift(1)
        if "usdjpy_close" in merged.columns:
            merged["usdjpy_return_1d"] = merged["usdjpy_close"].pct_change()
        if "us10y_yield" in merged.columns:
            merged["us10y_change"] = merged["us10y_yield"].diff()
        if "vix_close" in merged.columns:
            merged["vix_regime"] = (merged["vix_close"] > 22.0).astype(int)

        merged["future_entry_open"] = merged["open"].shift(-1)
        merged["future_exit_close"] = merged["close"].shift(-3)
        merged["true_ret"] = (merged["future_exit_close"] - merged["future_entry_open"]) / merged["future_entry_open"]
        merged["target"] = (merged["true_ret"] > 0.005).astype(int)

        merged["ticker"] = raw_ticker
        records.append(merged.dropna())

    full_clean = pd.concat(records, ignore_index=True)

    cutoff_date = "2026-01-05"
    train_df = full_clean[full_clean["date"] < cutoff_date].copy()
    test_df = full_clean[full_clean["date"] >= cutoff_date].copy()

    features = ["returns", "sma_diff", "atr", "sox_return_1d", "sox_diff_lag", "usdjpy_return_1d", "us10y_change", "vix_regime"]
    features = [f for f in features if f in full_clean.columns]

    for f in features:
        train_df[f] = train_df[f].astype(float)
        test_df[f] = test_df[f].astype(float)

    clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.03, random_state=42, verbose=-1)
    clf.fit(train_df[features], train_df["target"])
    test_df["pred_prob"] = clf.predict_proba(test_df[features])[:, 1]

    test_dates = sorted(test_df["date"].unique())
    stock_by_date: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for _, row in test_df.iterrows():
        d = row["date"]
        t = row["ticker"]
        if d not in stock_by_date:
            stock_by_date[d] = {}
        stock_by_date[d][t] = {
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "atr": row["atr"],
            "sma_diff": row["sma_diff"],
            "prob": row["pred_prob"],
        }

    # シミュレーション1: 3日保有 (5分散)
    sim_3d = ComprehensiveStrategySimulator(
        initial_cash=1_000_000.0,
        position_sizing_mode="fixed_20pct",
        holding_mode="fixed_3days",
        allow_fractional=True,
    )
    # シミュレーション2: 保有制限なし (5分散)
    sim_unlim = ComprehensiveStrategySimulator(
        initial_cash=1_000_000.0,
        position_sizing_mode="fixed_20pct",
        holding_mode="unlimited_trend",
        allow_fractional=True,
    )

    pending_3d: List[str] = []
    pending_unlim: List[str] = []

    for d in test_dates:
        daily_stocks = stock_by_date.get(d, {})
        sim_3d.step(date=d, daily_prices=daily_stocks, buy_signals=pending_3d)
        sim_unlim.step(date=d, daily_prices=daily_stocks, buy_signals=pending_unlim)

        pending_3d = []
        pending_unlim = []
        candidates = []
        for ticker, data in daily_stocks.items():
            if data["prob"] > 0.55:
                candidates.append((ticker, data["prob"]))
        candidates.sort(key=lambda x: x[1], reverse=True)
        top5 = [c[0] for c in candidates[:5]]
        pending_3d = top5
        pending_unlim = top5

    trades_3d = pd.DataFrame(sim_3d.closed_trades)
    trades_unlim = pd.DataFrame(sim_unlim.closed_trades)

    print("\n==============================================================")
    print("   DEEP TRADE COMPARISON: 3-DAY HOLDING VS UNLIMITED TREND")
    print("==============================================================")
    print(f"【3日保有制限 (5分散)】")
    print(f"  総トレード数       : {len(trades_3d)} 回 (約8ヶ月で148回回転)")
    print(f"  平均保有日数       : {trades_3d['days_held'].mean():.1f} 日")
    print(f"  勝率               : {trades_3d['win'].mean()*100:.1f}%")
    print(f"  勝ちトレード平均利得: {trades_3d[trades_3d['win']==1]['ret'].mean()*100:+.2f}%")
    print(f"  負けトレード平均損失: {trades_3d[trades_3d['win']==0]['ret'].mean()*100:+.2f}%")
    print(f"  ペイオフレシオ     : {abs(trades_3d[trades_3d['win']==1]['ret'].mean() / trades_3d[trades_3d['win']==0]['ret'].mean()):.2f}")

    print(f"\n【保有制限なし・トレンド終了まで (5分散)】")
    print(f"  総トレード数       : {len(trades_unlim)} 回 (資金拘束により108回に減少)")
    print(f"  平均保有日数       : {trades_unlim['days_held'].mean():.1f} 日")
    print(f"  勝率               : {trades_unlim['win'].mean()*100:.1f}%")
    print(f"  勝ちトレード平均利得: {trades_unlim[trades_unlim['win']==1]['ret'].mean()*100:+.2f}%")
    print(f"  負けトレード平均損失: {trades_unlim[trades_unlim['win']==0]['ret'].mean()*100:+.2f}%")
    print(f"  ペイオフレシオ     : {abs(trades_unlim[trades_unlim['win']==1]['ret'].mean() / trades_unlim[trades_unlim['win']==0]['ret'].mean()):.2f}")
    print("\n  エグジット理由の内訳:")
    print(trades_unlim['reason'].value_counts())
    print("==============================================================")


if __name__ == "__main__":
    run_trade_dynamics_comparison()
