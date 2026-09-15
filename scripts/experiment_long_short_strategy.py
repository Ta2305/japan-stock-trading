"""信用取引・空売り (Short) を統合した双方向 (Long & Short) トレード戦略バックテストスクリプト (修正版)

ロング＋ショートの評価額計算を修正し、ML有無・資金管理・保有方式ごとのパフォーマンスを比較する。
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


class LongShortPortfolioSimulator:
    def __init__(
        self,
        initial_cash: float = 1_000_000.0,
        position_sizing_mode: str = "fixed_20pct",  # "fixed_20pct" (5分散) or "unconstrained_top" (集中)
        holding_mode: str = "fixed_3days",           # "fixed_3days" or "unlimited_trend"
        slippage_fee_pct: float = 0.001,            # 往復手数料スリッページ 0.1%
        stock_borrowing_rate_daily: float = 0.00003, # 空売り貸株料 (年1.15% = 約0.003%/日)
        allow_fractional: bool = True,
    ):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position_sizing_mode = position_sizing_mode
        self.holding_mode = holding_mode
        self.slippage_fee_pct = slippage_fee_pct
        self.stock_borrowing_rate_daily = stock_borrowing_rate_daily
        self.allow_fractional = allow_fractional
        
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.closed_trades: List[Dict[str, Any]] = []
        self.equity_curve: List[Dict[str, Any]] = []

    def get_total_equity(self, current_prices: Dict[str, float]) -> float:
        pos_total_val = 0.0
        for ticker, pos in self.positions.items():
            curr_p = current_prices.get(ticker, pos["entry_price"])
            if pos["side"] == "long":
                pos_total_val += pos["shares"] * curr_p
            elif pos["side"] == "short":
                # ショートの現在価値 = 投入元本 + (売り値 - 現在値)*株数
                pnl = pos["shares"] * (pos["entry_price"] - curr_p)
                pos_total_val += pos["invested_cash"] + pnl
        return self.cash + pos_total_val

    def step(
        self,
        date: str,
        daily_prices: Dict[str, Dict[str, float]],
        buy_signals: List[str],
        short_signals: List[str],
    ):
        current_prices = {t: d["close"] for t, d in daily_prices.items()}

        # 1. 決済チェック
        closed_tickers = []
        for ticker, pos in list(self.positions.items()):
            if ticker not in daily_prices:
                continue
            
            p_data = daily_prices[ticker]
            pos["days_held"] += 1
            side = pos["side"]
            exit_triggered = False
            exit_reason = ""
            exit_price = p_data["close"]

            if self.holding_mode == "fixed_3days":
                if pos["days_held"] >= 3:
                    exit_triggered = True
                    exit_reason = "3-Day Time Limit"
                    if side == "long":
                        exit_price = p_data["close"] * (1.0 - self.slippage_fee_pct)
                    else:
                        exit_price = p_data["close"] * (1.0 + self.slippage_fee_pct)

            elif self.holding_mode == "unlimited_trend":
                sma_diff = p_data.get("sma_diff", 0.0)
                atr = pos.get("entry_atr", p_data["atr"])

                if side == "long":
                    sl_price = pos["entry_price"] - (1.5 * atr)
                    if p_data["low"] <= sl_price:
                        exit_triggered = True
                        exit_reason = "Stop Loss (Long)"
                        exit_price = sl_price * (1.0 - self.slippage_fee_pct)
                    elif sma_diff < 0:  # デッドクロス
                        exit_triggered = True
                        exit_reason = "Trend End (SMA Cross Below)"
                        exit_price = p_data["close"] * (1.0 - self.slippage_fee_pct)

                elif side == "short":
                    sl_price = pos["entry_price"] + (1.5 * atr)
                    if p_data["high"] >= sl_price:
                        exit_triggered = True
                        exit_reason = "Stop Loss (Short)"
                        exit_price = sl_price * (1.0 + self.slippage_fee_pct)
                    elif sma_diff > 0:  # ゴールデンクロス (反発)
                        exit_triggered = True
                        exit_reason = "Trend End (SMA Cross Above)"
                        exit_price = p_data["close"] * (1.0 + self.slippage_fee_pct)

            if exit_triggered:
                borrowing_cost = pos["shares"] * pos["entry_price"] * (self.stock_borrowing_rate_daily * pos["days_held"]) if side == "short" else 0.0
                
                if side == "long":
                    gross_ret = (exit_price - pos["entry_price"]) / pos["entry_price"]
                    pnl = pos["shares"] * (exit_price - pos["entry_price"])
                    # ロング返済: 売却代金がキャッシュに戻る
                    self.cash += pos["shares"] * exit_price
                else:
                    gross_ret = (pos["entry_price"] - exit_price) / pos["entry_price"]
                    pnl = pos["shares"] * (pos["entry_price"] - exit_price) - borrowing_cost
                    # ショート返済: 担保元本 + 確定損益がキャッシュに戻る
                    self.cash += pos["invested_cash"] + pnl

                self.closed_trades.append({
                    "ticker": ticker,
                    "side": side,
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

        # 2. 新規エントリー執行 (Long & Short)
        total_equity = self.get_total_equity(current_prices)

        valid_longs = [t for t in buy_signals if t not in self.positions and t not in short_signals and t in daily_prices]
        valid_shorts = [t for t in short_signals if t not in self.positions and t not in buy_signals and t in daily_prices]

        entries = [(t, "long") for t in valid_longs] + [(t, "short") for t in valid_shorts]

        if entries and self.cash > 1000:
            if self.position_sizing_mode == "fixed_20pct":
                max_pos = 5
                available_slots = max_pos - len(self.positions)
                for ticker, side in entries[:available_slots]:
                    open_raw = daily_prices[ticker]["open"]
                    open_p = open_raw * (1.0 + self.slippage_fee_pct) if side == "long" else open_raw * (1.0 - self.slippage_fee_pct)
                    if open_p <= 0:
                        continue
                    target_amt = min(total_equity * 0.20, self.cash)
                    shares = int(target_amt // open_p) if self.allow_fractional else int(target_amt // (open_p * 100)) * 100
                    cost = shares * open_p
                    if shares > 0 and self.cash >= cost:
                        self.cash -= cost
                        self.positions[ticker] = {
                            "side": side,
                            "entry_date": date,
                            "entry_price": open_p,
                            "entry_atr": daily_prices[ticker]["atr"],
                            "shares": shares,
                            "invested_cash": cost,
                            "days_held": 0,
                        }

            elif self.position_sizing_mode == "unconstrained_top":
                max_to_buy = min(len(entries), 2)
                alloc_per_ticker = self.cash / max_to_buy
                for ticker, side in entries[:max_to_buy]:
                    open_raw = daily_prices[ticker]["open"]
                    open_p = open_raw * (1.0 + self.slippage_fee_pct) if side == "long" else open_raw * (1.0 - self.slippage_fee_pct)
                    if open_p <= 0:
                        continue
                    shares = int(alloc_per_ticker // open_p) if self.allow_fractional else int(alloc_per_ticker // (open_p * 100)) * 100
                    cost = shares * open_p
                    if shares > 0 and self.cash >= cost:
                        self.cash -= cost
                        self.positions[ticker] = {
                            "side": side,
                            "entry_date": date,
                            "entry_price": open_p,
                            "entry_atr": daily_prices[ticker]["atr"],
                            "shares": shares,
                            "invested_cash": cost,
                            "days_held": 0,
                        }

        # 3. 日次資産残高記録
        end_equity = self.get_total_equity(current_prices)
        self.equity_curve.append({
            "date": date,
            "cash": self.cash,
            "positions_count": len(self.positions),
            "total_equity": end_equity,
        })


def run_long_short_experiments():
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

        vol_recent = merged["volume"].rolling(10).mean()
        vol_past = merged["volume"].rolling(50).mean() + 1e-5
        vol_surge = vol_recent / vol_past
        high_20 = merged["high"].rolling(20).max()
        low_20 = merged["low"].rolling(20).min()
        proximity_high = merged["close"] / (high_20 + 1e-5)
        proximity_low = (merged["close"] - low_20) / (high_20 - low_20 + 1e-5)
        atr_ratio = merged["atr"] / (merged["atr"].rolling(60).mean() + 1e-5)

        merged["tech_score_long"] = (atr_ratio * 0.4) + (vol_surge * 0.4) + (proximity_high * 0.2)
        merged["primary_signal_long"] = (merged["sma_diff"] > 0) & (vol_surge > 1.1)

        merged["tech_score_short"] = (atr_ratio * 0.4) + (vol_surge * 0.4) + ((1.0 - proximity_low) * 0.2)
        merged["primary_signal_short"] = (merged["sma_diff"] < -0.01) & (vol_surge > 1.1)

        if "sox_close" in merged.columns:
            merged["sox_return_1d"] = merged["sox_close"].pct_change()
            merged["sox_diff_lag"] = merged["sox_return_1d"] - merged["returns"].shift(1)
        if "usdjpy_close" in merged.columns:
            merged["usdjpy_return_1d"] = merged["usdjpy_close"].pct_change()
        if "us10y_yield" in merged.columns:
            merged["us10y_change"] = merged["us10y_yield"].diff()
        if "vix_close" in merged.columns:
            merged["vix_regime"] = (merged["vix_close"] > 22.0).astype(int)

        merged["future_entry_open"] = merged["open"].shift(-1)
        merged["future_exit_close"] = merged["close"].shift(-3)
        merged["true_ret_long"] = (merged["future_exit_close"] - merged["future_entry_open"]) / merged["future_entry_open"]
        merged["true_ret_short"] = (merged["future_entry_open"] - merged["future_exit_close"]) / merged["future_entry_open"]

        merged["target_long"] = (merged["true_ret_long"] > 0.005).astype(int)
        merged["target_short"] = (merged["true_ret_short"] > 0.005).astype(int)

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

    clf_long = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.03, random_state=42, verbose=-1)
    clf_long.fit(train_df[features], train_df["target_long"])
    test_df["prob_long"] = clf_long.predict_proba(test_df[features])[:, 1]

    clf_short = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.03, random_state=42, verbose=-1)
    clf_short.fit(train_df[features], train_df["target_short"])
    test_df["prob_short"] = clf_short.predict_proba(test_df[features])[:, 1]

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
            "sma_diff": row["sma_diff"],
            "prob_long": row["prob_long"],
            "prob_short": row["prob_short"],
            "tech_score_long": row["tech_score_long"],
            "tech_score_short": row["tech_score_short"],
            "primary_long": row["primary_signal_long"],
            "primary_short": row["primary_signal_short"],
        }

    scenarios = [
        # (allow_short, use_ml, sizing_mode, holding_mode, label)
        (False, True, "fixed_20pct", "fixed_3days", "① [MLあり] 買いのみ (Long Only) 5分散 × 3日保有"),
        (True,  True, "fixed_20pct", "fixed_3days", "② [MLあり] 買い＆空売り (Long & Short) 5分散 × 3日保有"),
        (True,  True, "unconstrained_top", "fixed_3days", "③ [MLあり] 買い＆空売り (Long & Short) Top集中 × 3日保有"),
        (True,  True, "fixed_20pct", "unlimited_trend", "④ [MLあり] 買い＆空売り (Long & Short) 5分散 × トレンド終了まで"),
        (True,  True, "unconstrained_top", "unlimited_trend", "⑤ [MLあり] 買い＆空売り (Long & Short) Top集中 × トレンド終了まで"),
        (False, False, "fixed_20pct", "fixed_3days", "⑥ [MLなし] 買いのみ (Long Only) 5分散 × 3日保有 (純テクニカル)"),
        (True,  False, "fixed_20pct", "fixed_3days", "⑦ [MLなし] 買い＆空売り (Long & Short) 5分散 × 3日保有"),
        (True,  False, "unconstrained_top", "fixed_3days", "⑧ [MLなし] 買い＆空売り (Long & Short) Top集中 × 3日保有"),
    ]

    results = []
    equity_curves_dict = {}
    trades_dict = {}

    for allow_short, use_ml, sizing, holding, label in scenarios:
        sim = LongShortPortfolioSimulator(
            initial_cash=1_000_000.0,
            position_sizing_mode=sizing,
            holding_mode=holding,
            slippage_fee_pct=0.001,
            stock_borrowing_rate_daily=0.00003,
            allow_fractional=True,
        )

        pending_long: List[str] = []
        pending_short: List[str] = []

        for d in test_dates:
            daily_stocks = stock_by_date.get(d, {})
            sim.step(date=d, daily_prices=daily_stocks, buy_signals=pending_long, short_signals=pending_short)

            pending_long = []
            pending_short = []
            c_long = []
            c_short = []

            for ticker, data in daily_stocks.items():
                if use_ml:
                    if data["prob_long"] > 0.55:
                        c_long.append((ticker, data["prob_long"]))
                    if allow_short and data["prob_short"] > 0.55:
                        c_short.append((ticker, data["prob_short"]))
                else:
                    if data["primary_long"]:
                        c_long.append((ticker, data["tech_score_long"]))
                    if allow_short and data["primary_short"]:
                        c_short.append((ticker, data["tech_score_short"]))

            c_long.sort(key=lambda x: x[1], reverse=True)
            c_short.sort(key=lambda x: x[1], reverse=True)

            pending_long = [c[0] for c in c_long[:5]]
            pending_short = [c[0] for c in c_short[:5]] if allow_short else []

        eq_df = pd.DataFrame(sim.equity_curve)
        trades_df = pd.DataFrame(sim.closed_trades)

        final_eq = eq_df["total_equity"].iloc[-1]
        mult = final_eq / 1_000_000.0
        ret_pct = (mult - 1.0) * 100
        n_trades = len(trades_df)
        win_rate = (trades_df["win"].sum() / n_trades * 100) if n_trades > 0 else 0
        avg_ret = (trades_df["ret"].mean() * 100) if n_trades > 0 else 0

        long_trades = trades_df[trades_df["side"] == "long"] if not trades_df.empty else pd.DataFrame()
        short_trades = trades_df[trades_df["side"] == "short"] if not trades_df.empty else pd.DataFrame()

        eq_df["peak"] = eq_df["total_equity"].cummax()
        eq_df["drawdown"] = (eq_df["total_equity"] - eq_df["peak"]) / eq_df["peak"]
        max_dd = eq_df["drawdown"].min() * 100

        results.append({
            "シナリオ": label,
            "最終資産": f"¥{int(final_eq):,}",
            "累積リターン": f"{ret_pct:+.2f}%",
            "元手倍率": f"{mult:.2f}倍",
            "勝率": f"{win_rate:.1f}%",
            "総取引数": n_trades,
            "Long数": len(long_trades),
            "Short数": len(short_trades),
            "平均損益": f"{avg_ret:+.2f}%",
            "Max DD": f"{max_dd:.2f}%",
        })
        equity_curves_dict[label] = eq_df
        trades_dict[label] = trades_df

    print("\n======================================================================================================================")
    print("   LONG & SHORT (CREDIT TRADING) COMPREHENSIVE BACKTEST RESULTS (2026 OUT-OF-SAMPLE)")
    print("======================================================================================================================")
    print(pd.DataFrame(results).to_string(index=False))
    print("======================================================================================================================\n")

    # 7月 (急落月) における Long Only vs Long & Short の比較
    print("=== [2026年7月 (SOX急落・調整局面) における資産推移比較] ===")
    sample_dates = ["2026-06-30", "2026-07-15", "2026-07-31", "2026-08-04"]
    tracking_rows = []
    for d in sample_dates:
        row = {"Date": d}
        for lbl in [scenarios[0][4], scenarios[1][4], scenarios[2][4], scenarios[5][4]]:
            eq = equity_curves_dict[lbl]
            sub = eq[eq["date"] <= d]
            if not sub.empty:
                val = sub["total_equity"].iloc[-1]
                short_name = lbl.split("] ")[1][:16]
                row[short_name] = f"¥{int(val):,}"
        tracking_rows.append(row)
    print(pd.DataFrame(tracking_rows).to_string(index=False))


if __name__ == "__main__":
    run_long_short_experiments()
