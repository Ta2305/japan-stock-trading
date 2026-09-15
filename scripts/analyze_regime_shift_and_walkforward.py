"""市場レジームシフト (構造変化) の検証 ＆ Walk-Forward (ローリング再学習) 比較検証スクリプト

固定学習と Walk-Forward ローリング再学習、学習ウィンドウ期間の違いによるパフォーマンスを比較する。
"""

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
from scripts.experiment_long_term_10years_backtest import LongTermSimulator


def run_walkforward_analysis():
    bars_daily_dir = ROOT_DIR / "market_data" / "jquants" / "bars_daily"
    macro_dir = ROOT_DIR / "market_data" / "global_macro"
    tickers_file = ROOT_DIR / "tickers.csv"

    df_tickers = pd.read_csv(tickers_file)
    ticker_list = df_tickers["ticker"].tolist()[:100]

    df_macro = load_global_macro_features(macro_dir)
    target_codes = {t.replace(".T", "") + "0" for t in ticker_list}
    
    print("Loading 10-year dataset...")
    stock_data_dict = load_recent_jquants_daily_bars_fast(bars_daily_dir, target_codes, max_days=2500)

    records = []
    df_macro_lagged = df_macro.copy()
    for col in ["sox_close", "sp500_close", "usdjpy_close", "vix_close", "us10y_yield"]:
        if col in df_macro_lagged.columns:
            df_macro_lagged[col] = df_macro_lagged[col].shift(1)

    for code, df_stock in stock_data_dict.items():
        if len(df_stock) < 200:
            continue
        raw_ticker = code[:-1] + ".T" if code.endswith("0") else code
        merged = pd.merge(df_stock, df_macro_lagged, on="date", how="left").ffill().dropna()
        if len(merged) < 100:
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
            merged["vix_regime"] = (merged["vix_close"] > 21.0).astype(int)

        merged["future_entry_open"] = merged["open"].shift(-1)
        merged["future_exit_close"] = merged["close"].shift(-3)
        merged["true_ret"] = (merged["future_exit_close"] - merged["future_entry_open"]) / merged["future_entry_open"]
        merged["target"] = (merged["true_ret"] > 0.005).astype(int)

        merged["ticker"] = raw_ticker
        records.append(merged.dropna())

    full_clean = pd.concat(records, ignore_index=True)
    full_clean["date"] = pd.to_datetime(full_clean["date"])
    full_clean = full_clean.sort_values("date").reset_index(drop=True)

    features = ["returns", "sma_diff", "atr", "sox_return_1d", "sox_diff_lag", "usdjpy_return_1d", "us10y_change", "vix_regime"]
    features = [f for f in features if f in full_clean.columns]
    for f in features:
        full_clean[f] = full_clean[f].astype(float)

    # 4つの学習アプローチの比較 (テスト期間はすべて同じ 2024-01-01 〜 2026-08-07)
    test_start = pd.to_datetime("2024-01-01")

    approaches = [
        # (name, window_years, is_walkforward)
        ("① 7.5年固定学習 (2016-2023固定・前回)", 7.5, False),
        ("② 直近2年固定学習 (2022-2023固定)", 2.0, False),
        ("③ Walk-Forward ローリング2年 (毎年直近2年で再学習)", 2.0, True),
        ("④ Walk-Forward ローリング3年 (毎年直近3年で再学習)", 3.0, True),
        ("⑤ Walk-Forward 累積拡張 (過去全期間を毎年再学習)", 10.0, True),
    ]

    results = []

    for name, window_yrs, is_wf in approaches:
        test_df = full_clean[full_clean["date"] >= test_start].copy()
        test_df["pred_prob"] = 0.0

        if not is_wf:
            # 固定学習
            train_end = test_start
            train_start = train_end - pd.DateOffset(years=int(window_yrs), months=int((window_yrs%1)*12))
            train_data = full_clean[(full_clean["date"] >= train_start) & (full_clean["date"] < train_end)]
            
            clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.03, random_state=42, verbose=-1)
            clf.fit(train_data[features], train_data["target"])
            test_df["pred_prob"] = clf.predict_proba(test_df[features])[:, 1]

        else:
            # Walk-Forward: 年ごとにモデルを再学習
            # 2024年テスト用, 2025年テスト用, 2026年テスト用
            years = [2024, 2025, 2026]
            for yr in years:
                yr_test = test_df[test_df["date"].dt.year == yr]
                if yr_test.empty:
                    continue
                
                yr_cutoff = pd.to_datetime(f"{yr}-01-01")
                if window_yrs >= 10.0:
                    yr_train = full_clean[full_clean["date"] < yr_cutoff]
                else:
                    yr_start = yr_cutoff - pd.DateOffset(years=int(window_yrs))
                    yr_train = full_clean[(full_clean["date"] >= yr_start) & (full_clean["date"] < yr_cutoff)]
                
                clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.03, random_state=42, verbose=-1)
                clf.fit(yr_train[features], yr_train["target"])
                
                idxs = yr_test.index
                test_df.loc[idxs, "pred_prob"] = clf.predict_proba(yr_test[features])[:, 1]

        # シミュレーション実行
        test_dates = sorted(test_df["date"].dt.strftime("%Y-%m-%d").unique())
        stock_by_date: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for _, row in test_df.iterrows():
            d = row["date"].strftime("%Y-%m-%d")
            t = row["ticker"]
            if d not in stock_by_date:
                stock_by_date[d] = {}
            stock_by_date[d][t] = {
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "prob": row["pred_prob"],
            }

        sim = LongTermSimulator(
            initial_cash=1_000_000.0,
            max_positions=5,
            risk_per_trade_pct=0.20,
            holding_days=3,
            slippage_fee_pct=0.001,
            allow_fractional=True,
        )

        pending_buy: List[str] = []
        for d in test_dates:
            daily_stocks = stock_by_date.get(d, {})
            sim.step(date=d, daily_prices=daily_stocks, buy_signals=pending_buy)

            pending_buy = []
            candidates = []
            for ticker, data in daily_stocks.items():
                if data["prob"] > 0.55:
                    candidates.append((ticker, data["prob"]))
            candidates.sort(key=lambda x: x[1], reverse=True)
            pending_buy = [c[0] for c in candidates[:5]]

        eq_df = pd.DataFrame(sim.equity_curve)
        trades_df = pd.DataFrame(sim.closed_trades)

        final_eq = eq_df["total_equity"].iloc[-1] if not eq_df.empty else 1_000_000.0
        mult = final_eq / 1_000_000.0
        ret_pct = (mult - 1.0) * 100
        n_trades = len(trades_df)
        win_rate = (trades_df["win"].sum() / n_trades * 100) if n_trades > 0 else 0
        avg_ret = (trades_df["ret"].mean() * 100) if n_trades > 0 else 0

        eq_df["peak"] = eq_df["total_equity"].cummax()
        eq_df["drawdown"] = (eq_df["total_equity"] - eq_df["peak"]) / eq_df["peak"]
        max_dd = eq_df["drawdown"].min() * 100

        results.append({
            "学習アプローチ": name,
            "最終資産 (2.5年後)": f"¥{int(final_eq):,}",
            "累積リターン": f"{ret_pct:+.2f}%",
            "元手倍率": f"{mult:.2f}倍",
            "勝率": f"{win_rate:.1f}%",
            "総取引数": n_trades,
            "1トレード平均": f"{avg_ret:+.2f}%",
            "Max DD": f"{max_dd:.2f}%",
        })

    print("\n==========================================================================================================")
    print("   WALK-FORWARD & TRAINING WINDOW SENSITIVITY EXPERIMENT RESULTS (2024 - 2026)")
    print("==========================================================================================================")
    print(pd.DataFrame(results).to_string(index=False))
    print("==========================================================================================================\n")


if __name__ == "__main__":
    run_walkforward_analysis()
