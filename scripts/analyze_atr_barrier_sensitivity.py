"""ATR利確・損切り倍率の感度分析 (Grid Search & Sensitivity Analysis) スクリプト

様々な (利確ATR倍数, 損切りATR倍数) の組み合わせでグリッドサーチし、勝率・平均リターン・累積リターン等を評価する。
"""

import sys
from pathlib import Path
import pandas as pd
import lightgbm as lgb
from typing import Dict, Any

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.evaluate_macro_swing_model import (
    load_recent_jquants_daily_bars_fast,
    load_global_macro_features,
)


def run_atr_sensitivity_analysis():
    bars_daily_dir = ROOT_DIR / "market_data" / "jquants" / "bars_daily"
    macro_dir = ROOT_DIR / "market_data" / "global_macro"
    tickers_file = ROOT_DIR / "tickers.csv"

    df_tickers = pd.read_csv(tickers_file)
    ticker_list = df_tickers["ticker"].tolist()[:100]

    df_macro = load_global_macro_features(macro_dir)
    target_codes = {t.replace(".T", "") + "0" for t in ticker_list}
    stock_data_dict = load_recent_jquants_daily_bars_fast(bars_daily_dir, target_codes, max_days=500)

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
            merged["vix_regime"] = (merged["vix_close"] > 21.0).astype(int)

        merged["future_entry_open"] = merged["open"].shift(-1)
        merged["future_exit_close"] = merged["close"].shift(-3)
        merged["true_ret"] = (merged["future_exit_close"] - merged["future_entry_open"]) / merged["future_entry_open"]
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

    # グリッドサーチ設定 (TP倍率, SL倍率)
    grid_params = [
        (1.5, 1.0),
        (2.0, 1.0),
        (2.0, 1.5),
        (2.5, 1.5),  # 現行設定
        (3.0, 1.5),
        (3.0, 2.0),
        (3.5, 2.0),
        (4.0, 2.0),
    ]

    # シミュレーション用日次データ
    test_dates = sorted(test_df["date"].unique())
    stock_by_date: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for _, row in test_df.iterrows():
        d = row["date"]
        t = row["ticker"]
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

    results = []

    for tp_mult, sl_mult in grid_params:
        # Triple Barrier シミュレーション
        cash = 1_000_000.0
        positions = {}
        closed_trades = []
        equity_curve = []

        pending_buy = []

        for d in test_dates:
            daily_stocks = stock_by_date.get(d, {})
            current_prices = {t: dt["close"] for t, dt in daily_stocks.items()}

            # 1. 決済
            closed_tickers = []
            for ticker, pos in list(positions.items()):
                if ticker not in daily_stocks:
                    continue
                p_data = daily_stocks[ticker]
                pos["days_held"] += 1
                exit_triggered = False
                exit_price = p_data["close"]

                sl_price = pos["entry_price"] - (sl_mult * pos["atr"])
                tp_price = pos["entry_price"] + (tp_mult * pos["atr"])

                if p_data["low"] <= sl_price:
                    exit_triggered = True
                    exit_price = sl_price * 0.999
                elif p_data["high"] >= tp_price:
                    exit_triggered = True
                    exit_price = tp_price * 0.999
                elif pos["days_held"] >= 3:
                    exit_triggered = True
                    exit_price = p_data["close"] * 0.999

                if exit_triggered:
                    ret = (exit_price - pos["entry_price"]) / pos["entry_price"]
                    pnl = pos["shares"] * (exit_price - pos["entry_price"])
                    cash += pos["shares"] * exit_price
                    closed_trades.append({"win": 1 if pnl > 0 else 0, "ret": ret, "pnl": pnl})
                    closed_tickers.append(ticker)

            for t in closed_tickers:
                del positions[t]

            # 2. エントリー
            tot_eq = cash + sum(p["shares"] * current_prices.get(t, p["entry_price"]) for t, p in positions.items())
            slots = 5 - len(positions)
            candidates = [t for t in pending_buy if t not in positions and t in daily_stocks]

            if slots > 0 and candidates and cash > 1000:
                alloc = min(tot_eq * 0.20, cash)
                for ticker in candidates[:slots]:
                    open_p = daily_stocks[ticker]["open"] * 1.001
                    atr_v = daily_stocks[ticker]["atr"]
                    shares = int(alloc // open_p)
                    cost = shares * open_p
                    if shares > 0 and cash >= cost:
                        cash -= cost
                        positions[ticker] = {
                            "entry_price": open_p,
                            "atr": atr_v,
                            "shares": shares,
                            "days_held": 0,
                        }

            pending_buy = []
            cands = []
            for ticker, data in daily_stocks.items():
                if data["prob"] > 0.55:
                    cands.append((ticker, data["prob"]))
            cands.sort(key=lambda x: x[1], reverse=True)
            pending_buy = [c[0] for c in cands[:5]]

            tot_eq = cash + sum(p["shares"] * current_prices.get(t, p["entry_price"]) for t, p in positions.items())
            equity_curve.append(tot_eq)

        final_eq = equity_curve[-1] if equity_curve else 1_000_000.0
        ret_pct = (final_eq - 1_000_000.0) / 1_000_000.0 * 100
        mult = final_eq / 1_000_000.0
        t_df = pd.DataFrame(closed_trades)
        win_rate = (t_df["win"].mean() * 100) if not t_df.empty else 0
        avg_ret = (t_df["ret"].mean() * 100) if not t_df.empty else 0
        
        eq_s = pd.Series(equity_curve)
        peak_s = eq_s.cummax()
        dd_s = (eq_s - peak_s) / peak_s
        max_dd = dd_s.min() * 100

        results.append({
            "TP倍数 (利確)": f"+{tp_mult:.1f} ATR",
            "SL倍数 (損切り)": f"-{sl_mult:.1f} ATR",
            "リスクリワード比": f"1 : {tp_mult/sl_mult:.2f}",
            "最終資産": f"¥{int(final_eq):,}",
            "累積リターン": f"{ret_pct:+.2f}%",
            "元手倍率": f"{mult:.2f}倍",
            "勝率": f"{win_rate:.1f}%",
            "取引数": len(t_df),
            "1トレード平均": f"{avg_ret:+.2f}%",
            "Max DD": f"{max_dd:.2f}%",
        })

    print("\n==========================================================================================================")
    print("   ATR TRIPLE BARRIER SENSITIVITY GRID SEARCH RESULTS (2026 OUT-OF-SAMPLE)")
    print("==========================================================================================================")
    print(pd.DataFrame(results).to_string(index=False))
    print("==========================================================================================================\n")


if __name__ == "__main__":
    run_atr_sensitivity_analysis()
