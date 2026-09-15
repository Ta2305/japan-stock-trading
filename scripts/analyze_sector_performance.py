"""銘柄別・セクター別バックテスト詳細パフォーマンス集計スクリプト
"""

import sys
import json
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

# セクターマッピング (主要銘柄)
SECTOR_MAP = {
    # 半導体・電子部品
    "80350": "半導体・製造装置 (東京エレクトロン)",
    "68570": "半導体・製造装置 (アドバンテスト)",
    "69200": "半導体・検査 (レーザーテック)",
    "61460": "半導体・切断 (ディスコ)",
    "67580": "電気機器・イメージセンサ (ソニーG)",
    "69810": "電子部品 (村田製作所)",
    "69760": "電子部品 (太陽誘電)",
    "40620": "電子部品・基板 (イビデン)",
    "67230": "車載半導体 (ルネサス)",
    # 金融・銀行
    "83060": "銀行業 (三菱UFJ)",
    "83160": "銀行業 (三井住友)",
    "84110": "銀行業 (みずほ)",
    "86300": "保険業 (SOMPO)",
    "87660": "保険業 (東京海上)",
    # 自動車・輸送用機器
    "72030": "自動車 (トヨタ自動車)",
    "72670": "自動車 (ホンダ)",
    "72700": "自動車 (SUBARU)",
    # 情報通信・IT
    "99840": "情報通信・投資 (ソフトバンクG)",
    "94320": "通信 (NTT)",
    "94330": "通信 (KDDI)",
    "43850": "IT・メルカリ",
    # 商社・エネルギー
    "80580": "卸売業 (三菱商事)",
    "80010": "卸売業 (伊藤忠商事)",
    "80310": "卸売業 (三井物産)",
    "16050": "鉱業・石油 (INPEX)",
    "50200": "石油・石炭 (ENEOS)",
    # 機械・重工・電線
    "70110": "防衛・重工 (三菱重工)",
    "70120": "防衛・重工 (川崎重工)",
    "63010": "建設機械 (コマツ)",
    "58030": "非鉄金属・電線 (フジクラ)",
}


def main():
    bars_daily_dir = ROOT_DIR / "market_data" / "jquants" / "bars_daily"
    macro_dir = ROOT_DIR / "market_data" / "global_macro"
    tickers_file = ROOT_DIR / "tickers.csv"

    df_tickers = pd.read_csv(tickers_file)
    ticker_list = df_tickers["ticker"].tolist()[:50]

    df_macro = load_global_macro_features(macro_dir)
    target_codes = {t.replace(".T", "") + "0" for t in ticker_list}
    stock_data_dict = load_recent_jquants_daily_bars_fast(bars_daily_dir, target_codes, max_days=500)

    all_rows = []

    for code, df_stock in stock_data_dict.items():
        if df_stock.empty or len(df_stock) < 100:
            continue
        df_feat = build_macro_enhanced_features(df_stock, df_macro)
        if df_feat.empty:
            continue
        
        raw_ticker = code[:-1] + ".T" if code.endswith("0") else code
        df_feat["ticker"] = raw_ticker
        df_feat["code_5d"] = code
        df_feat["sector"] = SECTOR_MAP.get(code, "その他製造・サービス")
        all_rows.append(df_feat)

    full_df = pd.concat(all_rows, ignore_index=True)
    split_idx = int(len(full_df) * 0.8)
    train_df = full_df.iloc[:split_idx]
    test_df = full_df.iloc[split_idx:].copy()

    base_features = ["returns", "sma_diff", "atr"]
    macro_features = base_features + [c for c in ["sox_return_1d", "sox_diff_lag", "usdjpy_return_1d", "us10y_change", "vix_regime"] if c in full_df.columns]

    for f in macro_features:
        train_df[f] = train_df[f].astype(float)
        test_df[f] = test_df[f].astype(float)

    clf_macro = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
    clf_macro.fit(train_df[macro_features], train_df["target"])
    test_df["pred_prob"] = clf_macro.predict_proba(test_df[macro_features])[:, 1]
    
    # シグナル判定 (>0.55)
    trades_df = test_df[test_df["pred_prob"] > 0.55].copy()

    # 銘柄別集計
    ticker_stats = []
    for ticker, group in trades_df.groupby("ticker"):
        n_tr = len(group)
        win_r = (group["target"].sum() / n_tr * 100) if n_tr > 0 else 0
        avg_ret = group["future_max_return"].mean() * 100
        sec_name = group["sector"].iloc[0]
        ticker_stats.append({
            "ticker": ticker,
            "sector": sec_name,
            "trades": n_tr,
            "win_rate": win_r,
            "avg_return": avg_ret
        })

    df_ticker_summary = pd.DataFrame(ticker_stats)
    
    # セクター別集計
    sector_stats = []
    for sector, group in trades_df.groupby("sector"):
        n_tr = len(group)
        win_r = (group["target"].sum() / n_tr * 100) if n_tr > 0 else 0
        avg_ret = group["future_max_return"].mean() * 100
        sector_stats.append({
            "sector": sector,
            "trades": n_tr,
            "win_rate": win_r,
            "avg_return": avg_ret
        })
    df_sector_summary = pd.DataFrame(sector_stats).sort_values("win_rate", ascending=False)

    print("=== SECTOR PERFORMANCE SUMMARY ===")
    print(df_sector_summary.to_string(index=False))

    print("\n=== TOP 5 WINNING TICKERS ===")
    print(df_ticker_summary.sort_values("win_rate", ascending=False).head(10).to_string(index=False))

    print("\n=== WORST TICKERS ===")
    print(df_ticker_summary.sort_values("win_rate", ascending=True).head(5).to_string(index=False))

    # JSON形式で詳細保存
    res_out = ROOT_DIR / "docs" / "sector_ticker_performance.json"
    res_out.parent.mkdir(parents=True, exist_ok=True)
    with open(res_out, "w", encoding="utf-8") as f:
        json.dump({
            "sector_summary": df_sector_summary.to_dict(orient="records"),
            "ticker_summary": df_ticker_summary.to_dict(orient="records")
        }, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
