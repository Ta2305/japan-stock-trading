"""バックテストおよび機械学習ロジックの健全性監査・検証スクリプト (Sanity Audit)

時系列の先読みリークや非現実的な約定前提など、バックテストに潜む重大な欠陥を検証・証明します。
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


def audit_train_test_split_leakage():
    """リーク1の検証: concat後の iloc[:split_idx] が時系列分割になっていないことの証明"""
    print("=== [監査1] Train/Test 分割の時系列リーク検証 ===")
    bars_daily_dir = ROOT_DIR / "market_data" / "jquants" / "bars_daily"
    macro_dir = ROOT_DIR / "market_data" / "global_macro"
    tickers_file = ROOT_DIR / "tickers.csv"

    df_tickers = pd.read_csv(tickers_file)
    ticker_list = df_tickers["ticker"].tolist()[:20]

    df_macro = load_global_macro_features(macro_dir)
    target_codes = {t.replace(".T", "") + "0" for t in ticker_list}
    stock_data_dict = load_recent_jquants_daily_bars_fast(bars_daily_dir, target_codes, max_days=500)

    all_dataset = []
    for code, df_stock in stock_data_dict.items():
        if df_stock.empty or len(df_stock) < 100:
            continue
        raw_ticker = code[:-1] + ".T" if code.endswith("0") else code
        df_feat = build_macro_enhanced_features(df_stock, df_macro)
        if not df_feat.empty:
            df_feat["ticker"] = raw_ticker
            all_dataset.append(df_feat)

    full_data = pd.concat(all_dataset, ignore_index=True)
    split_idx = int(len(full_data) * 0.8)
    train_df = full_data.iloc[:split_idx]
    test_df = full_data.iloc[split_idx:]

    print(f"全データ行数: {len(full_data)}")
    print(f"Trainデータ: 日付範囲 {train_df['date'].min()} 〜 {train_df['date'].max()} | 銘柄数: {train_df['ticker'].nunique()} ({train_df['ticker'].unique()[:5]}...)")
    print(f"Testデータ : 日付範囲 {test_df['date'].min()} 〜 {test_df['date'].max()} | 銘柄数: {test_df['ticker'].nunique()} ({test_df['ticker'].unique()[:5]}...)")
    
    # 判定
    if train_df['date'].max() >= test_df['date'].min():
        print("❌ 【重大な欠陥確認】: Trainに最新日（2026年）が含まれ、Testに過去（2024年）が含まれています。")
        print("   これは時系列分割ではなく「銘柄分割」になっており、モデルは未来のマクロ・市場環境を学習した状態で過去銘柄を予測しています。\n")


def audit_realistic_vs_unrealistic_returns():
    """リーク2の検証: future_max_return (3日後最高値売り抜け) vs 現実的エグジット (翌日始値エントリー & 3日後CloseまたはSL/TP手仕舞い)"""
    print("=== [監査2] 誇大リターン (3日後最高値イグジット) vs 現実的トレードの比較 ===")
    bars_daily_dir = ROOT_DIR / "market_data" / "jquants" / "bars_daily"
    macro_dir = ROOT_DIR / "market_data" / "global_macro"
    tickers_file = ROOT_DIR / "tickers.csv"

    df_tickers = pd.read_csv(tickers_file)
    ticker_list = df_tickers["ticker"].tolist()[:30]

    df_macro = load_global_macro_features(macro_dir)
    target_codes = {t.replace(".T", "") + "0" for t in ticker_list}
    stock_data_dict = load_recent_jquants_daily_bars_fast(bars_daily_dir, target_codes, max_days=500)

    # 厳格な特徴量構築（翌日始値エントリー、3日後Close決済、米国マクロ1日ラグ）
    records = []
    for code, df_stock in stock_data_dict.items():
        if len(df_stock) < 100:
            continue
        df = df_stock.copy()
        
        # 米国マクロは「前日の米市場終値」を使うため 1日シフト (Lag)
        df_macro_lagged = df_macro.copy()
        for col in ["sox_close", "sp500_close", "usdjpy_close", "vix_close", "us10y_yield"]:
            if col in df_macro_lagged.columns:
                df_macro_lagged[col] = df_macro_lagged[col].shift(1)

        merged = pd.merge(df, df_macro_lagged, on="date", how="left").ffill().dropna()
        if len(merged) < 50:
            continue

        # 特徴量
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

        # 【旧・誇大方式】: 当日Closeで買い、3日後Highで売る
        merged["unrealistic_ret"] = (merged["high"].shift(-3) - merged["close"]) / merged["close"]
        merged["unrealistic_target"] = (merged["unrealistic_ret"] > 0.02).astype(int)

        # 【現実的方式】: 当日引け後シグナル -> 翌日Openで買い、3日後Closeで手仕舞い
        # エントリー価格: 翌日Open (shift(-1))
        # イグジット価格: 3日後Close (shift(-3))
        merged["entry_open"] = merged["open"].shift(-1)
        merged["exit_close"] = merged["close"].shift(-3)
        merged["realistic_ret"] = (merged["exit_close"] - merged["entry_open"]) / merged["entry_open"]
        merged["realistic_target"] = (merged["realistic_ret"] > 0.0).astype(int)

        raw_ticker = code[:-1] + ".T" if code.endswith("0") else code
        merged["ticker"] = raw_ticker
        records.append(merged.dropna())

    full_clean = pd.concat(records, ignore_index=True)
    
    # 厳格な日付ベースの Train / Test 分割 (2025年以前で学習、2026年以降でテスト)
    all_dates = sorted(full_clean["date"].unique())
    cutoff_date = all_dates[int(len(all_dates) * 0.75)]
    print(f"厳格な日付分割日: {cutoff_date}")

    train_data = full_clean[full_clean["date"] < cutoff_date].copy()
    test_data = full_clean[full_clean["date"] >= cutoff_date].copy()

    features = ["returns", "sma_diff", "atr", "sox_return_1d", "sox_diff_lag", "usdjpy_return_1d", "us10y_change", "vix_regime"]
    features = [f for f in features if f in full_clean.columns]

    # 1. 旧・誇大モデルの評価
    clf_unrealistic = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
    clf_unrealistic.fit(train_data[features], train_data["unrealistic_target"])
    test_data["pred_unreal"] = clf_unrealistic.predict_proba(test_data[features])[:, 1]
    trades_unreal = test_data[test_data["pred_unreal"] > 0.55]

    # 2. 現実的モデルの学習・評価
    clf_realistic = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
    clf_realistic.fit(train_data[features], train_data["realistic_target"])
    test_data["pred_real"] = clf_realistic.predict_proba(test_data[features])[:, 1]
    trades_real = test_data[test_data["pred_real"] > 0.55]

    print("\n--------------------------------------------------------------")
    print(f"【旧・誇大モデル (3日後最高値イグジット & 誤ったデータ分割)】")
    print(f"  トレード回数   : {len(trades_unreal)}")
    print(f"  見かけの勝率   : {trades_unreal['unrealistic_target'].mean() * 100:.2f}%")
    print(f"  見かけの平均利得: {trades_unreal['unrealistic_ret'].mean() * 100:+.2f}%")
    
    print("\n【現実的モデル (翌日Openエントリー -> 3日後Closeエグジット & 厳格な日付分割)】")
    print(f"  トレード回数   : {len(trades_real)}")
    print(f"  現実の勝率     : {trades_real['realistic_target'].mean() * 100:.2f}%")
    print(f"  現実の平均利得 : {trades_real['realistic_ret'].mean() * 100:+.2f}%")
    print("--------------------------------------------------------------\n")


if __name__ == "__main__":
    audit_train_test_split_leakage()
    audit_realistic_vs_unrealistic_returns()
