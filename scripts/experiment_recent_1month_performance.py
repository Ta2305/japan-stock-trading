"""直近1ヶ月 (2026年7月1日〜2026年8月7日) における市場環境分析 ＆ Meta-labeling モデル運用実験スクリプト

学習データから除外した直近1ヶ月で Point-in-Time スクリーニングと Meta-labeling 取引を行い、損益・勝率・累積リターンを評価する。
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


def evaluate_recent_1month(ticker_limit: int = 100):
    bars_daily_dir = ROOT_DIR / "market_data" / "jquants" / "bars_daily"
    macro_dir = ROOT_DIR / "market_data" / "global_macro"
    tickers_file = ROOT_DIR / "tickers.csv"

    df_tickers = pd.read_csv(tickers_file)
    ticker_list = df_tickers["ticker"].tolist()[:ticker_limit]

    df_macro = load_global_macro_features(macro_dir)
    target_codes = {t.replace(".T", "") + "0" for t in ticker_list}
    
    # 過去500営業日分のデータをロード
    stock_data_dict = load_recent_jquants_daily_bars_fast(bars_daily_dir, target_codes, max_days=500)
    
    # 全日付の確認
    all_dates = set()
    for code, df in stock_data_dict.items():
        all_dates.update(df["date"].tolist())
    sorted_dates = sorted(list(all_dates))
    
    cutoff_date_1m = "2026-07-01"
    print(f"Total date range in dataset: {sorted_dates[0]} to {sorted_dates[-1]}")
    print(f"Evaluating Out-of-Sample Period: >= {cutoff_date_1m} (Total {len([d for d in sorted_dates if d >= cutoff_date_1m])} trading days)")

    # 1. 直近1ヶ月の主要インデックス・マクロ指数の推移
    macro_1m = df_macro[df_macro["date"] >= "2026-06-25"].copy()
    if not macro_1m.empty:
        print("\n--- [Macro Indices Performance: 2026-07 to 2026-08] ---")
        for col in ["sox_close", "sp500_close", "usdjpy_close", "vix_close", "us10y_yield"]:
            if col in macro_1m.columns:
                v_start = macro_1m[col].iloc[0]
                v_end = macro_1m[col].iloc[-1]
                pct = ((v_end - v_start) / v_start) * 100
                print(f"  {col:15s}: Start={v_start:.2f} -> End={v_end:.2f} ({pct:+.2f}%)")

    # 2. Point-in-Time スクリーニング (2026-07-01 直前時点のデータのみで銘柄選定)
    prior_stock_dict = {}
    for code, df in stock_data_dict.items():
        sub = df[df["date"] < cutoff_date_1m].copy()
        if len(sub) >= 60:
            prior_stock_dict[code] = sub

    scored_tickers = []
    for code, df_sub in prior_stock_dict.items():
        high_low = df_sub["high"] - df_sub["low"]
        atr = high_low.rolling(14).mean()
        atr_ratio = atr.iloc[-1] / (atr.iloc[-60:].mean() + 1e-5)
        
        vol_recent = df_sub["volume"].iloc[-10:].mean()
        vol_past = df_sub["volume"].iloc[-60:-10].mean() + 1e-5
        vol_surge = vol_recent / vol_past

        high_20 = df_sub["high"].iloc[-20:].max()
        curr_close = df_sub["close"].iloc[-1]
        proximity_to_high = curr_close / (high_20 + 1e-5)

        score = (atr_ratio * 0.4) + (vol_surge * 0.4) + (proximity_to_high * 0.2)
        raw_ticker = code[:-1] + ".T" if code.endswith("0") else code
        scored_tickers.append((raw_ticker, score))

    scored_tickers.sort(key=lambda x: x[1], reverse=True)
    selected_tickers = [t[0] for t in scored_tickers[:15]]
    print(f"\n--- [Automatically Screened Tickers as of {cutoff_date_1m}] ---")
    print(f"Top 15 Tickers: {selected_tickers}")

    # 3. 特徴量生成 & 学習データ (2026-07-01 以前) と テストデータ (2026-07-01 以降) の厳密分離
    all_rows = []
    for code, df_stock in stock_data_dict.items():
        raw_ticker = code[:-1] + ".T" if code.endswith("0") else code
        df_feat = build_macro_enhanced_features(df_stock, df_macro)
        if df_feat.empty:
            continue
        df_feat["ticker"] = raw_ticker
        all_rows.append(df_feat)

    full_df = pd.concat(all_rows, ignore_index=True)

    # 厳密な時系列日付分割
    train_df = full_df[full_df["date"] < cutoff_date_1m].copy()
    test_df = full_df[full_df["date"] >= cutoff_date_1m].copy()

    base_features = ["returns", "sma_diff", "atr"]
    macro_features = base_features + [c for c in ["sox_return_1d", "sox_diff_lag", "usdjpy_return_1d", "us10y_change", "vix_regime"] if c in full_df.columns]

    for f in macro_features:
        train_df[f] = train_df[f].astype(float)
        test_df[f] = test_df[f].astype(float)

    # LightGBM 学習 (2026-07-01 以前のデータのみで学習)
    clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
    clf.fit(train_df[macro_features], train_df["target"])

    test_df["pred_prob"] = clf.predict_proba(test_df[macro_features])[:, 1]

    # 直近1ヶ月のシグナル実行
    threshold = 0.55
    trades_all = test_df[test_df["pred_prob"] > threshold].copy()
    trades_screened = test_df[(test_df["pred_prob"] > threshold) & (test_df["ticker"].isin(selected_tickers))].copy()

    # 4. パフォーマンス集計
    print("\n==============================================================")
    print("   RECENT 1-MONTH EXPERIMENT RESULTS (2026-07-01 to 2026-08-07)")
    print("==============================================================")
    print(f"Total Test Samples (All Tickers) : {len(test_df)} rows")
    print(f"Total Trades Triggered           : {len(trades_all)} trades")
    if len(trades_all) > 0:
        win_rate_all = trades_all["target"].sum() / len(trades_all) * 100
        avg_ret_all = trades_all["future_max_return"].mean() * 100
        print(f"All Tickers Win Rate             : {win_rate_all:.2f}%")
        print(f"All Tickers Avg Trade Return     : {avg_ret_all:+.2f}%")

    print("\n--- [Screened 15 Tickers in Recent 1 Month] ---")
    print(f"Screened Trades Triggered        : {len(trades_screened)} trades")
    if len(trades_screened) > 0:
        win_rate_sc = trades_screened["target"].sum() / len(trades_screened) * 100
        avg_ret_sc = trades_screened["future_max_return"].mean() * 100
        print(f"Screened Win Rate                : {win_rate_sc:.2f}%")
        print(f"Screened Avg Trade Return        : {avg_ret_sc:+.2f}%")

        # 銘柄別集計
        by_ticker = []
        for ticker, group in trades_screened.groupby("ticker"):
            by_ticker.append({
                "ticker": ticker,
                "trades": len(group),
                "win_rate": round(group["target"].sum() / len(group) * 100, 1),
                "avg_ret_%": round(group["future_max_return"].mean() * 100, 2),
            })
        print(pd.DataFrame(by_ticker).to_string(index=False))

    # 5. 元手100万円での直近1ヶ月 資産推移シミュレーション
    initial_cap = 1_000_000.0
    cap_screened = initial_cap
    if len(trades_screened) > 0:
        for _, row in trades_screened.sort_values("date").iterrows():
            ret = row["future_max_return"]
            cap_screened += (cap_screened * 0.15) * ret

    print("\n--------------------------------------------------------------")
    print(f"【直近1ヶ月（2026年7月〜8月）ポートフォリオ損益 (初期元手: ¥1,000,000)】")
    print(f"  スクリーニング15銘柄 最終資産 : ¥{int(cap_screened):,}")
    print(f"  直近1ヶ月の月間リターン        : {(cap_screened - initial_cap) / initial_cap * 100:+.2f}%")
    print("==============================================================")


if __name__ == "__main__":
    evaluate_recent_1month()
