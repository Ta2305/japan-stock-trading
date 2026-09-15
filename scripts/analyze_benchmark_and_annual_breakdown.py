"""年別詳細ブレイクダウン ＆ 日経平均インデックス比較 ＆ レバレッジ効果の検証スクリプト

年別損益内訳と、日経平均 Buy & Hold に対する現物1.0倍・信用レバレッジ2.5倍の2.5年間推移を比較する。
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
from scripts.experiment_improved_margin_leverage import ImprovedMarginSimulator


def run_benchmark_and_annual_analysis():
    bars_daily_dir = ROOT_DIR / "market_data" / "jquants" / "bars_daily"
    macro_dir = ROOT_DIR / "market_data" / "global_macro"
    tickers_file = ROOT_DIR / "tickers.csv"

    df_tickers = pd.read_csv(tickers_file)
    ticker_list = df_tickers["ticker"].tolist()[:100]

    df_macro = load_global_macro_features(macro_dir)
    target_codes = {t.replace(".T", "") + "0" for t in ticker_list}
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

    # 1. 2022-2023年学習モデル (アプローチ②) の年別パフォーマンス分析
    train_22_23 = full_clean[(full_clean["date"] >= "2022-01-01") & (full_clean["date"] < "2024-01-01")]
    test_24_26 = full_clean[full_clean["date"] >= "2024-01-01"].copy()

    clf_22_23 = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.03, random_state=42, verbose=-1)
    clf_22_23.fit(train_22_23[features], train_22_23["target"])
    test_24_26["prob_22_23"] = clf_22_23.predict_proba(test_24_26[features])[:, 1]

    # 年別集計 (2024, 2025, 2026)
    test_24_26["year"] = test_24_26["date"].dt.year
    print("\n=== [1. 2022-2023年学習モデルの年別シグナル成績 (勝率 & 平均リターン)] ===")
    for yr in [2024, 2025, 2026]:
        sub = test_24_26[(test_24_26["year"] == yr) & (test_24_26["prob_22_23"] > 0.55)]
        if len(sub) > 0:
            w_rate = (sub["true_ret"] > 0).mean() * 100
            a_ret = sub["true_ret"].mean() * 100
            print(f"  {yr}年: トレードシグナル数 = {len(sub):4d} 回 | 勝率 = {w_rate:5.1f}% | 1トレード平均 = {a_ret:+5.2f}%")

    # 2. 直近2年学習 (2024-2025年学習) で2026年を予測した場合 (先ほどまでの好調の理由)
    train_24_25 = full_clean[(full_clean["date"] >= "2024-01-01") & (full_clean["date"] < "2026-01-01")]
    test_26 = full_clean[full_clean["date"] >= "2026-01-01"].copy()

    clf_24_25 = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.03, random_state=42, verbose=-1)
    clf_24_25.fit(train_24_25[features], train_24_25["target"])
    test_26["prob_24_25"] = clf_24_25.predict_proba(test_26[features])[:, 1]

    sub_26 = test_26[test_26["prob_24_25"] > 0.55]
    print("\n=== [2. 2024-2025年学習モデルで2026年を予測した場合] ===")
    print(f"  2026年: トレードシグナル数 = {len(sub_26):4d} 回 | 勝率 = {(sub_26['true_ret']>0).mean()*100:5.1f}% | 1トレード平均 = {sub_26['true_ret'].mean()*100:+5.2f}%")

    # 3. 2.5年間 (2024〜2026年) での「現物 vs 信用レバレッジ2.5倍 (Walk-Forward)」の運用結果
    # Walk-Forward 3年で日次シミュレーション
    test_df_wf = full_clean[full_clean["date"] >= "2024-01-01"].copy()
    test_df_wf["pred_prob"] = 0.0

    for yr in [2024, 2025, 2026]:
        yr_test = test_df_wf[test_df_wf["date"].dt.year == yr]
        if yr_test.empty:
            continue
        yr_cutoff = pd.to_datetime(f"{yr}-01-01")
        yr_start = yr_cutoff - pd.DateOffset(years=3)
        yr_train = full_clean[(full_clean["date"] >= yr_start) & (full_clean["date"] < yr_cutoff)]
        
        clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.03, random_state=42, verbose=-1)
        clf.fit(yr_train[features], yr_train["target"])
        test_df_wf.loc[yr_test.index, "pred_prob"] = clf.predict_proba(yr_test[features])[:, 1]

    test_dates = sorted(test_df_wf["date"].dt.strftime("%Y-%m-%d").unique())
    stock_by_date: Dict[str, Dict[str, Dict[str, Any]]] = {}
    macro_by_date: Dict[str, Dict[str, float]] = {}

    for _, row in test_df_wf.iterrows():
        d = row["date"].strftime("%Y-%m-%d")
        t = row["ticker"]
        if d not in macro_by_date:
            macro_by_date[d] = {
                "vix_close": row.get("vix_close", 15.0),
                "sox_return_1d": row.get("sox_return_1d", 0.0),
            }
        if d not in stock_by_date:
            stock_by_date[d] = {}
        stock_by_date[d][t] = {
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "atr": row["atr"],
            "prob": row["pred_prob"],
        }

    # 現物 1.0倍 vs 信用 2.5倍
    sim_spot = ImprovedMarginSimulator(initial_cash=1_000_000.0, target_leverage=1.0, leverage_mode="fixed", allow_fractional=True)
    sim_margin = ImprovedMarginSimulator(initial_cash=1_000_000.0, target_leverage=2.5, leverage_mode="adaptive_new_orders", allow_fractional=True)

    pending_spot: List[str] = []
    pending_margin: List[str] = []

    for d in test_dates:
        daily_stocks = stock_by_date.get(d, {})
        m_info = macro_by_date.get(d, {})
        sim_spot.step(date=d, macro_info=m_info, daily_prices=daily_stocks, buy_signals=pending_spot)
        sim_margin.step(date=d, macro_info=m_info, daily_prices=daily_stocks, buy_signals=pending_margin)

        pending_spot = []
        pending_margin = []
        candidates = []
        for ticker, data in daily_stocks.items():
            if data["prob"] > 0.55:
                candidates.append((ticker, data["prob"]))
        candidates.sort(key=lambda x: x[1], reverse=True)
        top5 = [c[0] for c in candidates[:5]]
        pending_spot = top5
        pending_margin = top5

    eq_spot = pd.DataFrame(sim_spot.equity_curve)
    eq_margin = pd.DataFrame(sim_margin.equity_curve)

    final_spot = eq_spot["total_equity"].iloc[-1]
    final_margin = eq_margin["total_equity"].iloc[-1]

    print("\n=========================================================================")
    print("   2.5 YEARS OVERALL COMPARISON: SPOT VS MARGIN (2024 - 2026)")
    print("=========================================================================")
    print(f"【現物 1.0倍 (Walk-Forward 3年)】")
    print(f"  初期元手 : ¥1,000,000  ->  最終純資産 : ¥{int(final_spot):,} (+{(final_spot-1e6)/1e4:.2f}% / {final_spot/1e6:.2f}倍)")
    print(f"\n【信用レバレッジ 2.5倍 (動的益出し複利・Walk-Forward 3年)】")
    print(f"  初期元手 : ¥1,000,000  ->  最終純資産 : ¥{int(final_margin):,} (+{(final_margin-1e6)/1e4:.2f}% / {final_margin/1e6:.2f}倍)")
    print("=========================================================================\n")


if __name__ == "__main__":
    run_benchmark_and_annual_analysis()
