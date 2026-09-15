"""Alpha-Max v2: 日経平均25日線トレンドフィルター ＆ 動的ディレバレッジ改善スクリプト

市場トレンドフィルターと動的ディレバレッジにより、調整相場や急落時のドローダウン削減効果を検証する。
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
from scripts.experiment_alpha_max_momentum_leverage import AlphaMaxPortfolioSimulator


class AlphaMaxV2Simulator(AlphaMaxPortfolioSimulator):
    def get_active_leverage(self, vix: float, sox_ret: float, nikkei_above_sma25: bool) -> float:
        if self.leverage_mode == "fixed":
            return self.target_leverage
        
        # 究極のマクロトレンド・レジーム判定
        if not nikkei_above_sma25 or vix > 21.0 or sox_ret < -0.025:
            # 市場全体が調整期または下落期: レバレッジ0倍 (完全ノーポジション待機)
            return 0.0
        elif vix > 18.0 or sox_ret < -0.01:
            # 警戒時: 1.0倍 (現物等倍)
            return 1.0
        elif vix < 16.0 and sox_ret > 0.0:
            # 超強気: 最大レバレッジ (2.0〜2.5倍)
            return self.target_leverage
        else:
            return min(self.target_leverage, 1.8)

    def step(
        self,
        date: str,
        macro_info: Dict[str, float],
        daily_prices: Dict[str, Dict[str, float]],
        buy_signals: List[str],
    ):
        current_prices = {t: d["close"] for t, d in daily_prices.items()}
        vix = macro_info.get("vix_close", 15.0)
        sox_ret = macro_info.get("sox_return_1d", 0.0)
        nikkei_above_sma25 = macro_info.get("nikkei_above_sma25", True)
        
        active_leverage = self.get_active_leverage(vix, sox_ret, nikkei_above_sma25)

        # 1. 決済チェック
        closed_tickers = []
        for ticker, pos in list(self.positions.items()):
            if ticker not in daily_prices:
                continue
            
            p_data = daily_prices[ticker]
            pos["days_held"] += 1
            exit_triggered = False
            exit_reason = ""
            exit_price = p_data["close"]

            atr = pos.get("entry_atr", p_data["atr"])

            if self.use_trailing_protection and p_data["high"] >= pos["entry_price"] + (1.5 * atr):
                pos["current_sl"] = max(pos.get("current_sl", pos["entry_price"] - 1.5 * atr), pos["entry_price"])

            current_sl = pos.get("current_sl", pos["entry_price"] - (1.5 * atr))
            if p_data["low"] <= current_sl:
                exit_triggered = True
                exit_reason = "Trailing Stop / Stop Loss"
                exit_price = current_sl * (1.0 - self.slippage_fee_pct)

            elif pos["days_held"] >= 3:
                exit_triggered = True
                exit_reason = "3-Day Time Limit"
                exit_price = p_data["close"] * (1.0 - self.slippage_fee_pct)

            if exit_triggered:
                interest_cost = pos["shares"] * pos["entry_price"] * (self.interest_rate_daily * pos["days_held"])
                gross_ret = (exit_price - pos["entry_price"]) / pos["entry_price"]
                pnl = pos["shares"] * (exit_price - pos["entry_price"]) - interest_cost
                self.cash += pnl

                self.closed_trades.append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"],
                    "exit_date": date,
                    "entry_price": pos["entry_price"],
                    "exit_price": exit_price,
                    "shares": pos["shares"],
                    "pnl": pnl,
                    "ret": gross_ret,
                    "days_held": pos["days_held"],
                    "win": 1 if pnl > 0 else 0,
                    "reason": exit_reason,
                })
                closed_tickers.append(ticker)

        for t in closed_tickers:
            del self.positions[t]

        total_equity = self.get_total_equity(current_prices)
        if total_equity <= 0:
            self.equity_curve.append({"date": date, "cash": 0, "positions_count": 0, "total_equity": 0})
            return

        # 2. 新規エントリー
        if active_leverage > 0:
            total_purchasing_power = total_equity * active_leverage
            current_pos_val = sum(pos["shares"] * current_prices.get(t, pos["entry_price"]) for t, pos in self.positions.items())
            remaining_purchasing_power = max(0.0, total_purchasing_power - current_pos_val)

            valid_candidates = [t for t in buy_signals if t not in self.positions and t in daily_prices]
            available_slots = self.max_positions - len(self.positions)

            if available_slots > 0 and valid_candidates and remaining_purchasing_power > 1000:
                max_to_buy = min(available_slots, len(valid_candidates))
                alloc_per_slot = remaining_purchasing_power / max_to_buy

                for ticker in valid_candidates[:max_to_buy]:
                    open_p = daily_prices[ticker]["open"] * (1.0 + self.slippage_fee_pct)
                    if open_p <= 0:
                        continue
                    shares = int(alloc_per_slot // open_p) if self.allow_fractional else int(alloc_per_slot // (open_p * 100)) * 100
                    if shares > 0:
                        atr_val = daily_prices[ticker]["atr"]
                        self.positions[ticker] = {
                            "entry_date": date,
                            "entry_price": open_p,
                            "entry_atr": atr_val,
                            "current_sl": open_p - (1.5 * atr_val),
                            "shares": shares,
                            "days_held": 0,
                        }

        # 3. 日次記録
        end_equity = self.get_total_equity(current_prices)
        self.equity_curve.append({
            "date": date,
            "cash": self.cash,
            "positions_count": len(self.positions),
            "total_equity": end_equity,
        })


def run_alpha_max_v2_experiments():
    bars_daily_dir = ROOT_DIR / "market_data" / "jquants" / "bars_daily"
    macro_dir = ROOT_DIR / "market_data" / "global_macro"
    tickers_file = ROOT_DIR / "tickers.csv"

    df_tickers = pd.read_csv(tickers_file)
    ticker_list = df_tickers["ticker"].tolist()[:100]

    df_macro = load_global_macro_features(macro_dir)
    target_codes = {t.replace(".T", "") + "0" for t in ticker_list}
    stock_data_dict = load_recent_jquants_daily_bars_fast(bars_daily_dir, target_codes, max_days=2500)

    # 日経平均 (99840 or S&P500) の25日線トレンド計算
    df_macro_lagged = df_macro.copy()
    if "sp500_close" in df_macro_lagged.columns:
        df_macro_lagged["sp500_sma25"] = df_macro_lagged["sp500_close"].rolling(25).mean()
        df_macro_lagged["market_above_sma25"] = (df_macro_lagged["sp500_close"] > df_macro_lagged["sp500_sma25"]).astype(bool)

    for col in ["sox_close", "sp500_close", "usdjpy_close", "vix_close", "us10y_yield", "market_above_sma25"]:
        if col in df_macro_lagged.columns:
            df_macro_lagged[col] = df_macro_lagged[col].shift(1)

    records = []
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

        vol_recent = merged["volume"].rolling(10).mean()
        vol_past = merged["volume"].rolling(50).mean() + 1e-5
        vol_surge = vol_recent / vol_past
        high_20 = merged["high"].rolling(20).max()
        proximity = merged["close"] / (high_20 + 1e-5)
        atr_ratio = merged["atr"] / (merged["atr"].rolling(60).mean() + 1e-5)
        merged["momentum_score"] = (atr_ratio * 0.35) + (vol_surge * 0.45) + (proximity * 0.20)

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

    test_df_wf = full_clean[full_clean["date"] >= "2024-01-01"].copy()
    test_df_wf["pred_prob"] = 0.0

    for yr in [2024, 2025, 2026]:
        yr_test = test_df_wf[test_df_wf["date"].dt.year == yr]
        if yr_test.empty:
            continue
        yr_cutoff = pd.to_datetime(f"{yr}-01-01")
        yr_start = yr_cutoff - pd.DateOffset(years=3)
        yr_train = full_clean[(full_clean["date"] >= yr_start) & (full_clean["date"] < yr_cutoff)]
        
        clf = lgb.LGBMClassifier(n_estimators=120, learning_rate=0.03, random_state=42, verbose=-1)
        clf.fit(yr_train[features], yr_train["target"])
        test_df_wf.loc[yr_test.index, "pred_prob"] = clf.predict_proba(yr_test[features])[:, 1]

    test_dates = sorted(test_df_wf["date"].dt.strftime("%Y-%m-%d").unique())
    stock_by_date: Dict[str, Dict[str, Dict[str, Any]]] = {}
    macro_by_date: Dict[str, Dict[str, Any]] = {}

    for _, row in test_df_wf.iterrows():
        d = row["date"].strftime("%Y-%m-%d")
        t = row["ticker"]
        if d not in macro_by_date:
            macro_by_date[d] = {
                "vix_close": row.get("vix_close", 15.0),
                "sox_return_1d": row.get("sox_return_1d", 0.0),
                "nikkei_above_sma25": row.get("market_above_sma25", True),
            }
        if d not in stock_by_date:
            stock_by_date[d] = {}
        
        combined_score = (row["pred_prob"] * 0.6) + (min(row["momentum_score"], 3.0) / 3.0 * 0.4)
        stock_by_date[d][t] = {
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "atr": row["atr"],
            "prob": row["pred_prob"],
            "combined_score": combined_score,
        }

    scenarios = [
        (5, 1.0, "fixed", False, "① 【基準】現物 1.0倍 5分散"),
        (2, 2.0, "adaptive_macro", True,  "② 【Alpha-Max v2】Top2モメンタム × レバ2.0倍 ＋ 市場トレンド防御"),
        (2, 2.5, "adaptive_macro", True,  "③ 【Alpha-Max v2】Top2モメンタム × レバ2.5倍 ＋ 市場トレンド防御"),
        (3, 2.5, "adaptive_macro", True,  "④ 【Alpha-Max v2】Top3モメンタム × レバ2.5倍 ＋ 市場トレンド防御"),
    ]

    results = []
    equity_curves_dict = {}

    for max_pos, lev, mode, trailing, label in scenarios:
        sim = AlphaMaxV2Simulator(
            initial_cash=1_000_000.0,
            max_positions=max_pos,
            target_leverage=lev,
            leverage_mode=mode,
            use_trailing_protection=trailing,
            interest_rate_daily=0.000077,
            slippage_fee_pct=0.001,
            allow_fractional=True,
        )

        pending_buy = []

        for d in test_dates:
            daily_stocks = stock_by_date.get(d, {})
            m_info = macro_by_date.get(d, {})
            sim.step(date=d, macro_info=m_info, daily_prices=daily_stocks, buy_signals=pending_buy)

            pending_buy = []
            candidates = []
            for ticker, data in daily_stocks.items():
                if data["prob"] > 0.55:
                    candidates.append((ticker, data["combined_score"]))
            candidates.sort(key=lambda x: x[1], reverse=True)
            pending_buy = [c[0] for c in candidates[:max_pos]]

        eq_df = pd.DataFrame(sim.equity_curve)
        trades_df = pd.DataFrame(sim.closed_trades)

        final_eq = eq_df["total_equity"].iloc[-1]
        mult = final_eq / 1_000_000.0
        ret_pct = (mult - 1.0) * 100
        n_trades = len(trades_df)
        win_rate = (trades_df["win"].sum() / n_trades * 100) if n_trades > 0 else 0
        avg_ret = (trades_df["ret"].mean() * 100) if n_trades > 0 else 0

        eq_df["peak"] = eq_df["total_equity"].cummax()
        eq_df["drawdown"] = (eq_df["total_equity"] - eq_df["peak"]) / eq_df["peak"]
        max_dd = eq_df["drawdown"].min() * 100

        results.append({
            "戦略シナリオ": label,
            "最終純資産 (2.5年後)": f"¥{int(final_eq):,}",
            "累積リターン": f"{ret_pct:+.2f}%",
            "元手倍率": f"{mult:.2f}倍",
            "勝率": f"{win_rate:.1f}%",
            "総取引数": n_trades,
            "1トレード平均": f"{avg_ret:+.2f}%",
            "Max DD": f"{max_dd:.2f}%",
        })
        equity_curves_dict[label] = eq_df

    print("\n==========================================================================================================")
    print("   ALPHA-MAX v2 STRATEGY EXPERIMENT RESULTS (2024 - 2026 : 2.5 YEARS OUT-OF-SAMPLE)")
    print("==========================================================================================================")
    print(pd.DataFrame(results).to_string(index=False))
    print("==========================================================================================================\n")

    print("=== [年別・節目ごとの資産残高推移 (2024年〜2026年)] ===")
    sample_dates = ["2024-01-04", "2024-06-28", "2024-12-30", "2025-06-30", "2025-12-30", "2026-06-30", "2026-08-04"]
    tracking_rows = []
    for d in sample_dates:
        row = {"Date": d}
        for lbl in [scenarios[0][4], scenarios[1][4], scenarios[2][4], scenarios[3][4]]:
            eq = equity_curves_dict[lbl]
            sub = eq[eq["date"] <= d]
            if not sub.empty:
                val = sub["total_equity"].iloc[-1]
                short_name = lbl.split("】")[1][:18]
                row[short_name] = f"¥{int(val):,}"
        tracking_rows.append(row)
    print(pd.DataFrame(tracking_rows).to_string(index=False))


if __name__ == "__main__":
    run_alpha_max_v2_experiments()
