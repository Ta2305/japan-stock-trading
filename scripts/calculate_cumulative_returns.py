"""個別銘柄別 累積資産倍率シミュレーションスクリプト (厳格・リーク排除版)

翌営業日寄付 (Open) エントリー -> 3営業日後大引け (Close) エグジット、日付ベースの Train/Test 分割、
往復スリッページ・手数料 (0.1%) を考慮した資金推移をシミュレーションします。
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


def calculate_portfolio_multiplier(df_trades: pd.DataFrame, initial_capital: float = 1_000_000.0) -> dict:
    capital = initial_capital
    fee_pct = 0.001

    for ret in df_trades["true_return"]:
        net_ret = ret - fee_pct
        capital += (capital * 0.20) * net_ret

    multiplier = capital / initial_capital
    win_count = (df_trades["true_return"] > 0).sum()
    return {
        "initial_capital": initial_capital,
        "final_capital": capital,
        "multiplier": multiplier,
        "total_trades": len(df_trades),
        "win_rate": (win_count / len(df_trades) * 100) if len(df_trades) > 0 else 0,
        "avg_ret": (df_trades["true_return"].mean() * 100) if len(df_trades) > 0 else 0,
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
        raw_ticker = code[:-1] + ".T" if code.endswith("0") else code
        df_feat = build_macro_enhanced_features(df_stock, df_macro)
        if df_feat.empty:
            continue
        df_feat["ticker"] = raw_ticker
        all_rows.append(df_feat)

    full_df = pd.concat(all_rows, ignore_index=True)
    
    # 厳格な日付分割
    cutoff_date = "2026-01-05"
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
        if len(group) == 0:
            continue
        stats = calculate_portfolio_multiplier(group, initial_capital=initial_cap)
        results.append({
            "ticker": ticker,
            "trades": stats["total_trades"],
            "win_rate": round(stats["win_rate"], 1),
            "avg_ret_%": round(stats["avg_ret"], 2),
            "final_capital_yen": int(stats["final_capital"]),
            "multiplier": round(stats["multiplier"], 2)
        })

    res_df = pd.DataFrame(results).sort_values("multiplier", ascending=False)

    print("\n==============================================================")
    print("   AUDITED CUMULATIVE ASSET MULTIPLIER RESULTS (OUT-OF-SAMPLE 2026)")
    print("==============================================================")
    print(res_df.to_string(index=False))
    print("==============================================================")


if __name__ == "__main__":
    main()
