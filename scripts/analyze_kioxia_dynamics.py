"""キオクシア (285A) の長期トレンド・ボラティリティ構造 ＆ 戦略別シミュレーションスクリプト

キオクシアの2026年の値動きに対し、Buy & Hold、3日保有スイング (Meta-labeling)、急落頻度を分析する。
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


def analyze_kioxia_strategies():
    bars_daily_dir = ROOT_DIR / "market_data" / "jquants" / "bars_daily"
    macro_dir = ROOT_DIR / "market_data" / "global_macro"

    df_macro = load_global_macro_features(macro_dir)
    target_codes = {"285A0", "285A"}
    stock_dict = load_recent_jquants_daily_bars_fast(bars_daily_dir, target_codes, max_days=500)

    code = list(stock_dict.keys())[0]
    df_kioxia = stock_dict[code]

    print(f"Loaded Kioxia Data: {len(df_kioxia)} rows from {df_kioxia['date'].min()} to {df_kioxia['date'].max()}")

    # 1. 2026年の値動きのプルバック (下落調整) 分析
    df_2026 = df_kioxia[df_kioxia["date"] >= "2026-01-05"].copy().reset_index(drop=True)
    
    # ピークとドローダウン
    df_2026["peak"] = df_2026["close"].cummax()
    df_2026["dd"] = (df_2026["close"] - df_2026["peak"]) / df_2026["peak"]

    print("\n=== [キオクシア 2026年 Buy & Hold の推移] ===")
    print(f"年初株価 (2026-01-05) : ¥{df_2026['close'].iloc[0]:,}")
    print(f"最高値   (2026-06-22) : ¥{df_2026['close'].max():,} (年初比 {df_2026['close'].max()/df_2026['close'].iloc[0]:.2f}倍)")
    print(f"7月最安値 (2026-07-30) : ¥{df_2026['close'].min():,} (最高値から {df_2026['dd'].min()*100:.2f}% 暴落)")
    print(f"直近株価 (2026-08-07) : ¥{df_2026['close'].iloc[-1]:,} (年初比 {df_2026['close'].iloc[-1]/df_2026['close'].iloc[0]:.2f}倍)")

    # 2. 途中での「急落 (5営業日以内の下落幅)」の頻度
    df_2026["5d_return"] = df_2026["close"].pct_change(5)
    deep_pullbacks = df_2026[df_2026["5d_return"] < -0.15]
    print(f"\n2026年中の『5日間で -15%以上 急落した回数』: {len(deep_pullbacks)} 営業日")

    # 3. 3日スイング戦略でのキオクシア個別トレード結果
    df_feat = build_macro_enhanced_features(df_kioxia, df_macro)
    df_feat["ticker"] = "285A.T"

    # モデル学習 (2025年以前で学習 -> 2026年で推論)
    train_df = df_feat[df_feat["date"] < "2026-01-05"].copy()
    test_df = df_feat[df_feat["date"] >= "2026-01-05"].copy()

    features = ["returns", "sma_diff", "atr", "sox_return_1d", "sox_diff_lag", "usdjpy_return_1d", "us10y_change", "vix_regime"]
    features = [f for f in features if f in df_feat.columns]

    for f in features:
        train_df[f] = train_df[f].astype(float)
        test_df[f] = test_df[f].astype(float)

    clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.03, random_state=42, verbose=-1)
    clf.fit(train_df[features], train_df["target"])
    test_df["pred_prob"] = clf.predict_proba(test_df[features])[:, 1]

    # トレード抽出
    trades_3d = test_df[test_df["pred_prob"] > 0.55].copy()
    
    print("\n=== [キオクシア 3日保有スイング (Meta-labeling) の運用結果 (2026年)] ===")
    print(f"発生トレード数 : {len(trades_3d)} 回")
    win_rate = (trades_3d["true_return"] > 0).mean() * 100
    avg_ret = trades_3d["true_return"].mean() * 100
    print(f"勝率           : {win_rate:.1f}%")
    print(f"1トレード平均  : {avg_ret:+.2f}%")

    # 複利運用シミュレーション (キオクシア単体に毎回全額投入した場合)
    cap = 1_000_000.0
    for _, r in trades_3d.iterrows():
        ret = r["true_return"] - 0.001
        cap *= (1.0 + ret)
    print(f"全額複利最終資産 : ¥{int(cap):,} (元手の {cap/1_000_000.0:.2f}倍)")

    # 月別トレード回数
    trades_3d["month"] = pd.to_datetime(trades_3d["date"]).dt.strftime("%Y-%m")
    print("\n月別トレード回数と平均リターン:")
    monthly = trades_3d.groupby("month").agg(
        trades=("true_return", "count"),
        win_rate=("true_return", lambda x: (x > 0).mean() * 100),
        avg_ret=("true_return", lambda x: x.mean() * 100),
    )
    print(monthly.to_string())


if __name__ == "__main__":
    analyze_kioxia_strategies()
