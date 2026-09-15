"""改善版 信用取引レバレッジ＆益出し複利 (Improved Margin Compounding) バックテストスクリプト

急落時の寄付投げ売りを廃止し、新規エントリー時の動的レバレッジ制御と Top集中益出し複利の効果を検証する。
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


class ImprovedMarginSimulator:
    def __init__(
        self,
        initial_cash: float = 1_000_000.0,
        max_positions: int = 5,
        target_leverage: float = 2.5,
        sizing_mode: str = "fixed_slots",            # "fixed_slots" (5分散) or "top_concentrated" (上位集中)
        leverage_mode: str = "adaptive_new_orders",  # "fixed" or "adaptive_new_orders"
        interest_rate_daily: float = 0.000077,       # 信用買い金利 (年2.8%)
        slippage_fee_pct: float = 0.001,             # 往復スリッページ手数料 0.1%
        allow_fractional: bool = True,
    ):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.max_positions = max_positions
        self.target_leverage = target_leverage
        self.sizing_mode = sizing_mode
        self.leverage_mode = leverage_mode
        self.interest_rate_daily = interest_rate_daily
        self.slippage_fee_pct = slippage_fee_pct
        self.allow_fractional = allow_fractional

        self.positions: Dict[str, Dict[str, Any]] = {}
        self.closed_trades: List[Dict[str, Any]] = []
        self.equity_curve: List[Dict[str, Any]] = []

    def get_total_equity(self, current_prices: Dict[str, float]) -> float:
        unrealized_pnl = 0.0
        for ticker, pos in self.positions.items():
            curr_p = current_prices.get(ticker, pos["entry_price"])
            unrealized_pnl += pos["shares"] * (curr_p - pos["entry_price"])
        return self.cash + unrealized_pnl

    def get_new_order_leverage(self, vix: float, sox_ret: float) -> float:
        if self.leverage_mode == "fixed":
            return self.target_leverage
        
        # 新規エントリー枠の動的決定
        if vix > 21.0 or sox_ret < -0.025:
            return 0.0  # 危機時: 新規エントリー禁止 (ポジションは自然解消待ち)
        elif vix > 18.0 or sox_ret < -0.01:
            return 1.2  # 警戒時: 1.2倍
        elif vix < 16.0 and sox_ret > 0.0:
            return self.target_leverage  # 強気時: フルレバレッジ (2.5〜3.0倍)
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
        active_new_leverage = self.get_new_order_leverage(vix, sox_ret)

        # 1. 決済チェック (3日経過またはATR損切り)
        closed_tickers = []
        for ticker, pos in list(self.positions.items()):
            if ticker not in daily_prices:
                continue
            
            p_data = daily_prices[ticker]
            pos["days_held"] += 1
            exit_triggered = False
            exit_reason = ""
            exit_price = p_data["close"]

            # ATR損切りチェック (-1.5 ATR)
            sl_price = pos["entry_price"] - (1.5 * pos.get("entry_atr", p_data["atr"]))
            if p_data["low"] <= sl_price:
                exit_triggered = True
                exit_reason = "Stop Loss (-1.5 ATR)"
                exit_price = sl_price * (1.0 - self.slippage_fee_pct)

            elif pos["days_held"] >= 3:
                exit_triggered = True
                exit_reason = "3-Day Time Limit"
                exit_price = p_data["close"] * (1.0 - self.slippage_fee_pct)

            if exit_triggered:
                interest_cost = pos["shares"] * pos["entry_price"] * (self.interest_rate_daily * pos["days_held"])
                gross_ret = (exit_price - pos["entry_price"]) / pos["entry_price"]
                pnl = pos["shares"] * (exit_price - pos["entry_price"]) - interest_cost
                
                # 益出し: 確定損益をキャッシュ(保証金)に合算
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
            self.equity_curve.append({
                "date": date,
                "cash": 0,
                "positions_count": 0,
                "total_equity": 0,
            })
            return

        # 2. 新規エントリー執行 (益出し複利)
        if active_new_leverage > 0:
            total_purchasing_power = total_equity * active_new_leverage
            current_pos_val = sum(pos["shares"] * current_prices.get(t, pos["entry_price"]) for t, pos in self.positions.items())
            remaining_purchasing_power = max(0.0, total_purchasing_power - current_pos_val)

            valid_candidates = [t for t in buy_signals if t not in self.positions and t in daily_prices]

            if valid_candidates and remaining_purchasing_power > 1000:
                if self.sizing_mode == "fixed_slots":
                    available_slots = self.max_positions - len(self.positions)
                    if available_slots > 0:
                        alloc_per_slot = remaining_purchasing_power / available_slots
                        for ticker in valid_candidates[:available_slots]:
                            open_p = daily_prices[ticker]["open"] * (1.0 + self.slippage_fee_pct)
                            if open_p <= 0:
                                continue
                            shares = int(alloc_per_slot // open_p) if self.allow_fractional else int(alloc_per_slot // (open_p * 100)) * 100
                            if shares > 0:
                                self.positions[ticker] = {
                                    "entry_date": date,
                                    "entry_price": open_p,
                                    "entry_atr": daily_prices[ticker]["atr"],
                                    "shares": shares,
                                    "days_held": 0,
                                }

                elif self.sizing_mode == "top_concentrated":
                    # 上位1〜2銘柄に利用可能枠を集中投下
                    max_to_buy = min(len(valid_candidates), 2)
                    alloc_per_slot = remaining_purchasing_power / max_to_buy
                    for ticker in valid_candidates[:max_to_buy]:
                        open_p = daily_prices[ticker]["open"] * (1.0 + self.slippage_fee_pct)
                        if open_p <= 0:
                            continue
                        shares = int(alloc_per_slot // open_p) if self.allow_fractional else int(alloc_per_slot // (open_p * 100)) * 100
                        if shares > 0:
                            self.positions[ticker] = {
                                "entry_date": date,
                                "entry_price": open_p,
                                "entry_atr": daily_prices[ticker]["atr"],
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


def run_improved_experiments():
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

    test_dates = sorted(test_df["date"].unique())
    stock_by_date: Dict[str, Dict[str, Dict[str, Any]]] = {}
    macro_by_date: Dict[str, Dict[str, float]] = {}

    for _, row in test_df.iterrows():
        d = row["date"]
        t = row["ticker"]
        if d not in macro_by_date:
            macro_by_date[d] = {
                "vix_close": row.get("vix_close", 15.0),
                "sox_return_1d": row.get("sox_return_1d", 0.0),
            }
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

    # 改善検証シナリオ
    scenarios = [
        # (lev, sizing, mode, label)
        (1.0, "fixed_slots",     "fixed",               "① 現物 1.0倍 (5分散・基準)"),
        (2.0, "fixed_slots",     "fixed",               "② 固定信用 2.0倍 (5分散・益出し複利)"),
        (3.0, "fixed_slots",     "fixed",               "③ 固定信用 3.0倍 (5分散・益出し複利)"),
        (2.5, "fixed_slots",     "adaptive_new_orders", "④ 【改善】動的レバレッジ 2.5倍 (5分散・新規枠制御)"),
        (3.0, "fixed_slots",     "adaptive_new_orders", "⑤ 【改善】動的レバレッジ 3.0倍 (5分散・新規枠制御)"),
        (2.5, "top_concentrated", "adaptive_new_orders", "⑥ 【改善・集中】動的 2.5倍 (Top集中・STF益出し型)"),
        (3.0, "top_concentrated", "adaptive_new_orders", "⑦ 【改善・集中】動的 3.0倍 (Top集中・STF益出し型)"),
    ]

    results = []
    equity_curves_dict = {}

    for lev, sizing, mode, label in scenarios:
        sim = ImprovedMarginSimulator(
            initial_cash=1_000_000.0,
            max_positions=5,
            target_leverage=lev,
            sizing_mode=sizing,
            leverage_mode=mode,
            interest_rate_daily=0.000077,
            slippage_fee_pct=0.001,
            allow_fractional=True,
        )

        pending_buy: List[str] = []

        for d in test_dates:
            daily_stocks = stock_by_date.get(d, {})
            m_info = macro_by_date.get(d, {})
            sim.step(date=d, macro_info=m_info, daily_prices=daily_stocks, buy_signals=pending_buy)

            pending_buy = []
            candidates = []
            for ticker, data in daily_stocks.items():
                if data["prob"] > 0.55:
                    candidates.append((ticker, data["prob"]))
            candidates.sort(key=lambda x: x[1], reverse=True)
            pending_buy = [c[0] for c in candidates[:5]]

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
            "シナリオ": label,
            "最終資産": f"¥{int(final_eq):,}",
            "累積リターン": f"{ret_pct:+.2f}%",
            "元手倍率": f"{mult:.2f}倍",
            "勝率": f"{win_rate:.1f}%",
            "取引数": n_trades,
            "1トレード平均損益": f"{avg_ret:+.2f}%",
            "Max DD": f"{max_dd:.2f}%",
        })
        equity_curves_dict[label] = eq_df

    print("\n==========================================================================================================")
    print("   IMPROVED MARGIN COMPOUNDING EXPERIMENT RESULTS (2026 OUT-OF-SAMPLE)")
    print("==========================================================================================================")
    print(pd.DataFrame(results).to_string(index=False))
    print("==========================================================================================================\n")

    # 月別推移 (6月末ピークと7〜8月の推移)
    print("=== [月別・節目ごとの純資産残高推移 (2026年)] ===")
    sample_dates = ["2026-01-05", "2026-03-31", "2026-05-29", "2026-06-30", "2026-07-31", "2026-08-04"]
    tracking_rows = []
    for d in sample_dates:
        row = {"Date": d}
        for lbl in [scenarios[0][3], scenarios[1][3], scenarios[2][3], scenarios[4][3], scenarios[6][3]]:
            eq = equity_curves_dict[lbl]
            sub = eq[eq["date"] <= d]
            if not sub.empty:
                val = sub["total_equity"].iloc[-1]
                short_name = lbl.split(" (")[0][:18]
                row[short_name] = f"¥{int(val):,}"
        tracking_rows.append(row)
    print(pd.DataFrame(tracking_rows).to_string(index=False))


if __name__ == "__main__":
    run_improved_experiments()
