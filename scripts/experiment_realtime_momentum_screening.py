"""事前データ (Point-in-Time) による動的モメンタム銘柄自動スクリーニング＆運用検証スクリプト (厳格版)

評価基準日より前のデータのみでスクリーニングし、2025年以前で学習して2026年未知期間で検証する。
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
    build_macro_enhanced_features,
)


def screen_promising_tickers(stock_data_dict: dict, cutoff_date: str = "2026-01-05") -> list[str]:
    scored_tickers = []

    for code, df_stock in stock_data_dict.items():
        df_sub = df_stock[df_stock["date"] < cutoff_date].copy()
        if len(df_sub) < 60:
            continue
        
        # 1. ボラティリティ (ATR) の拡大傾向
        high_low = df_sub["high"] - df_sub["low"]
        atr = high_low.rolling(14).mean()
        atr_ratio = atr.iloc[-1] / (atr.iloc[-60:].mean() + 1e-5)
        
        # 2. 出来高の急増 (Volume Surge)
        vol_recent = df_sub["volume"].iloc[-10:].mean()
        vol_past = df_sub["volume"].iloc[-60:-10].mean() + 1e-5
        vol_surge = vol_recent / vol_past

        # 3. 直近20日のモメンタム
        high_20 = df_sub["high"].iloc[-20:].max()
        curr_close = df_sub["close"].iloc[-1]
        proximity_to_high = curr_close / (high_20 + 1e-5)

        # 複合スコア
        score = (atr_ratio * 0.4) + (vol_surge * 0.4) + (proximity_to_high * 0.2)
        raw_ticker = code[:-1] + ".T" if code.endswith("0") else code
        scored_tickers.append((raw_ticker, score))

    scored_tickers.sort(key=lambda x: x[1], reverse=True)
    return [t[0] for t in scored_tickers[:20]]


def main():
    bars_daily_dir = ROOT_DIR / "market_data" / "jquants" / "bars_daily"
    macro_dir = ROOT_DIR / "market_data" / "global_macro"
    tickers_file = ROOT_DIR / "tickers.csv"

    df_tickers = pd.read_csv(tickers_file)
    ticker_list = df_tickers["ticker"].tolist()[:100]

    df_macro = load_global_macro_features(macro_dir)
    target_codes = {t.replace(".T", "") + "0" for t in ticker_list}
    stock_data_dict = load_recent_jquants_daily_bars_fast(bars_daily_dir, target_codes, max_days=500)

    cutoff_date = "2026-01-05"
    screened_tickers = screen_promising_tickers(stock_data_dict, cutoff_date=cutoff_date)
    print(f"Top Screened Tickers as of {cutoff_date}:\n{screened_tickers}\n")

    all_rows = []
    for code, df_stock in stock_data_dict.items():
        raw_ticker = code[:-1] + ".T" if code.endswith("0") else code
        if raw_ticker not in screened_tickers:
            continue
        
        df_feat = build_macro_enhanced_features(df_stock, df_macro)
        if df_feat.empty:
            continue
        df_feat["ticker"] = raw_ticker
        all_rows.append(df_feat)

    full_df = pd.concat(all_rows, ignore_index=True)
    train_df = full_df[full_df["date"] < cutoff_date].copy()
    test_df = full_df[full_df["date"] >= cutoff_date].copy()

    base_features = ["returns", "sma_diff", "atr"]
    macro_features = base_features + [c for c in ["sox_return_1d", "sox_diff_lag", "usdjpy_return_1d", "us10y_change", "vix_regime"] if c in full_df.columns]

    for f in macro_features:
        train_df[f] = train_df[f].astype(float)
        test_df[f] = test_df[f].astype(float)

    clf_macro = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.03, random_state=42, verbose=-1)
    clf_macro.fit(train_df[macro_features], train_df["target"])
    test_df["pred_prob"] = clf_macro.predict_proba(test_df[macro_features])[:, 1]
    trades_df = test_df[test_df["pred_prob"] > 0.55].copy()

    results = []
    initial_cap = 1_000_000.0

    for ticker, group in trades_df.groupby("ticker"):
        cap = initial_cap
        for ret in group["true_return"]:
            net_ret = ret - 0.001
            cap += (cap * 0.20) * net_ret
        mult = cap / initial_cap
        results.append({
            "ticker": ticker,
            "trades": len(group),
            "win_rate": round((group["true_return"] > 0).mean() * 100, 1),
            "avg_ret_%": round(group["true_return"].mean() * 100, 2),
            "final_capital_yen": int(cap),
            "multiplier": round(mult, 2)
        })

    rdf = pd.DataFrame(results).sort_values("multiplier", ascending=False)
    print("==============================================================")
    print("   AUDITED POINT-IN-TIME MOMENTUM SCREENING EVALUATION (2026)")
    print("==============================================================")
    print(rdf.to_string(index=False))
    print("==============================================================")


if __name__ == "__main__":
    main()
