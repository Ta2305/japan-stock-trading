"""下落トレンド・レンジ銘柄におけるボラティリティ取引検証スクリプト

下落・横ばいレンジ銘柄を対象に、ボラティリティとマクロ指標を活用した買い・空売り両方向の取引を検証する。
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

# 下落・レンジの代表的銘柄
BEAR_RANGE_TICKERS = [
    "2914.T",  # JT (レンジ・横ばい)
    "7974.T",  # 任天堂 (レンジ・上昇鈍化)
    "7267.T",  # ホンダ (調整・下落局面あり)
    "4502.T",  # 武田薬品 (レンジ・停滞)
    "9432.T",  # NTT (下落・低ボラティリティ)
    "4689.T",  # LINEヤフー (下落・調整)
]


def main():
    bars_daily_dir = ROOT_DIR / "market_data" / "jquants" / "bars_daily"
    macro_dir = ROOT_DIR / "market_data" / "global_macro"

    df_macro = load_global_macro_features(macro_dir)
    target_codes = {t.replace(".T", "") + "0" for t in BEAR_RANGE_TICKERS}
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
        all_rows.append(df_feat)

    if not all_rows:
        print("No stock data found for bear/range tickers!")
        return

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

    # ロング ＆ ショート判定 (確率 > 0.55 で買い, 確率 < 0.35 で空売り)
    results = []
    for ticker, group in test_df.groupby("ticker"):
        cap = 1_000_000.0
        n_long = 0
        n_short = 0
        wins = 0

        for _, row in group.iterrows():
            prob = row["pred_prob"]
            ret = row["future_max_return"]

            if prob > 0.55:  # ロング
                n_long += 1
                cap += (cap * 0.15) * ret
                if ret > 0:
                    wins += 1
            elif prob < 0.35:  # ショート (株価下落で利益)
                n_short += 1
                cap += (cap * 0.15) * (-ret)
                if ret < 0:
                    wins += 1

        tot_tr = n_long + n_short
        win_rate = (wins / tot_tr * 100) if tot_tr > 0 else 0.0
        mult = cap / 1000000.0

        results.append({
            "ticker": ticker,
            "total_trades": tot_tr,
            "long_trades": n_long,
            "short_trades": n_short,
            "win_rate": round(win_rate, 1),
            "final_capital_yen": int(cap),
            "multiplier": round(mult, 2)
        })

    rdf = pd.DataFrame(results).sort_values("multiplier", ascending=False)
    print("\n==============================================================")
    print("   EXPERIMENT 1: BEAR / RANGE STOCKS WITH BOTH LONG & SHORT")
    print("==============================================================")
    print(rdf.to_string(index=False))
    print("==============================================================")


if __name__ == "__main__":
    main()
