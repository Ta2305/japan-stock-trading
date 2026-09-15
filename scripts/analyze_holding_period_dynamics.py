"""保有日数別リターン推移 ＆ トレンド持続期間の分析スクリプト

エントリー後1日目〜10日目のリターン推移を集計し、保有日数ごとの平均リターン・勝率・損益分布を評価する。
"""

import sys
from pathlib import Path
import pandas as pd
import lightgbm as lgb

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.evaluate_macro_swing_model import (
    load_recent_jquants_daily_bars_fast,
    load_global_macro_features,
)


def analyze_holding_period_dynamics():
    bars_daily_dir = ROOT_DIR / "market_data" / "jquants" / "bars_daily"
    macro_dir = ROOT_DIR / "market_data" / "global_macro"
    tickers_file = ROOT_DIR / "tickers.csv"

    df_tickers = pd.read_csv(tickers_file)
    ticker_list = df_tickers["ticker"].tolist()[:100]

    df_macro = load_global_macro_features(macro_dir)
    target_codes = {t.replace(".T", "") + "0" for t in ticker_list}
    stock_data_dict = load_recent_jquants_daily_bars_fast(bars_daily_dir, target_codes, max_days=500)

    # 特徴量作成 (米国マクロ Lag 1)
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

        # 翌日Openエントリー
        merged["entry_open"] = merged["open"].shift(-1)
        
        # エントリー後 1日目〜10日目のCloseリターン
        for day in range(1, 11):
            merged[f"ret_day_{day}"] = (merged["close"].shift(-(day)) - merged["entry_open"]) / merged["entry_open"]
            merged[f"high_day_{day}"] = (merged["high"].shift(-(day)) - merged["entry_open"]) / merged["entry_open"]
            merged[f"low_day_{day}"] = (merged["low"].shift(-(day)) - merged["entry_open"]) / merged["entry_open"]

        merged["future_exit_close"] = merged["close"].shift(-3)
        merged["true_ret"] = (merged["future_exit_close"] - merged["entry_open"]) / merged["entry_open"]
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

    # シグナル発生サンプル (P > 0.55)
    signals_df = test_df[test_df["pred_prob"] > 0.55].copy()
    print(f"Test Period Triggered Signals: {len(signals_df)} events")

    # 1. 保有日数別 (1日〜10日) の平均リターン推移
    day_stats = []
    for day in range(1, 11):
        col = f"ret_day_{day}"
        avg_ret = signals_df[col].mean() * 100
        win_rate = (signals_df[col] > 0).mean() * 100
        pos_avg = signals_df[signals_df[col] > 0][col].mean() * 100
        neg_avg = signals_df[signals_df[col] <= 0][col].mean() * 100
        day_stats.append({
            "保有日数": f"{day}日目",
            "平均リターン": f"{avg_ret:+.2f}%",
            "勝率 (プラス割合)": f"{win_rate:.1f}%",
            "平均利益": f"{pos_avg:+.2f}%",
            "平均損失": f"{neg_avg:+.2f}%",
            "プロフィットファクター概算": f"{abs(pos_avg * win_rate / (neg_avg * (100 - win_rate) + 1e-5)):.2f}",
        })

    print("\n=========================================================================")
    print("   HOLDING PERIOD DYNAMICS: DAY 1 TO DAY 10 FORWARD RETURNS (2026)")
    print("=========================================================================")
    print(pd.DataFrame(day_stats).to_string(index=False))
    print("=========================================================================\n")


if __name__ == "__main__":
    analyze_holding_period_dynamics()
