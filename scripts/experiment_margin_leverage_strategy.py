"""信用取引レバレッジ＆益出し複利 (Compounding Margin Leverage) トレード戦略バックテストスクリプト

固定・動的レバレッジと益出し複利の効果を、2026年 Out-of-Sample 期間で比較検証する。
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


class MarginLeveragePortfolioSimulator:
    def __init__(
        self,
        initial_cash: float = 1_000_000.0,
        max_positions: int = 5,
        target_leverage: float = 2.0,                # レバレッジ目標倍率 (1.0〜3.0)
        leverage_mode: str = "fixed",                # "fixed" or "adaptive_macro"
        emergency_flush_on_crisis: bool = True,      # 危機レジーム突入時に即座に全ポジション解消するか
        interest_rate_daily: float = 0.000077,       # 信用買い金利 (年2.8% / 365 = 約0.0077%/日)
        slippage_fee_pct: float = 0.001,             # 往復スリッページ手数料 0.1%
        allow_fractional: bool = True,
    ):
        self.initial_cash = initial_cash
        self.cash = initial_cash                     # 未拘束キャッシュ
        self.max_positions = max_positions
        self.target_leverage = target_leverage
        self.leverage_mode = leverage_mode
        self.emergency_flush_on_crisis = emergency_flush_on_crisis
        self.interest_rate_daily = interest_rate_daily
        self.slippage_fee_pct = slippage_fee_pct
        self.allow_fractional = allow_fractional

        self.positions: Dict[str, Dict[str, Any]] = {}
        self.closed_trades: List[Dict[str, Any]] = []
        self.equity_curve: List[Dict[str, Any]] = []

    def get_total_equity(self, current_prices: Dict[str, float]) -> float:
        """純資産 (Net Asset Value = 保証金残高 + 建て玉の含み損益)"""
        unrealized_pnl = 0.0
        for ticker, pos in self.positions.items():
            curr_p = current_prices.get(ticker, pos["entry_price"])
            unrealized_pnl += pos["shares"] * (curr_p - pos["entry_price"])
        return self.cash + unrealized_pnl

    def get_active_leverage(self, vix: float, sox_ret: float) -> float:
        """マクロ指標に基づく動的レバレッジの決定"""
        if self.leverage_mode == "fixed":
            return self.target_leverage
        
        # 動的レバレッジ (Adaptive Macro Regime)
        if vix > 21.0 or sox_ret < -0.02:
            return 0.0  # 危機・下落相場: レバレッジ0倍 (キャッシュ退避)
        elif vix > 18.0 or sox_ret < -0.005:
            return 1.0  # 警戒相場: 現物等倍 (1.0倍)
        elif vix < 15.0 and sox_ret > 0.01:
            return min(self.target_leverage * 1.2, 3.0)  # 超強気: 最大3.0倍
        else:
            return self.target_leverage  # 通常強気: 設定レバレッジ (例 2.0〜2.5倍)

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
        current_leverage_limit = self.get_active_leverage(vix, sox_ret)

        # 1. 決済チェック (3日経過 または 危機時の緊急キャッシュ退避)
        closed_tickers = []
        for ticker, pos in list(self.positions.items()):
            if ticker not in daily_prices:
                continue
            
            p_data = daily_prices[ticker]
            pos["days_held"] += 1
            exit_triggered = False
            exit_reason = ""
            exit_price = p_data["close"]

            # 危機時の即座手仕舞い
            if self.emergency_flush_on_crisis and current_leverage_limit == 0.0:
                exit_triggered = True
                exit_reason = "Emergency Cash Flush (Crisis Macro)"
                exit_price = p_data["open"] * (1.0 - self.slippage_fee_pct)

            elif pos["days_held"] >= 3:
                exit_triggered = True
                exit_reason = "3-Day Time Limit"
                exit_price = p_data["close"] * (1.0 - self.slippage_fee_pct)

            if exit_triggered:
                interest_cost = pos["shares"] * pos["entry_price"] * (self.interest_rate_daily * pos["days_held"])
                gross_ret = (exit_price - pos["entry_price"]) / pos["entry_price"]
                pnl = pos["shares"] * (exit_price - pos["entry_price"]) - interest_cost
                
                # 純資産キャッシュに純損益を加算 (益出し)
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

        # 破産・追証チェック (純資産がゼロ以下になった場合は終了)
        total_equity = self.get_total_equity(current_prices)
        if total_equity <= 0:
            self.equity_curve.append({
                "date": date,
                "cash": 0,
                "positions_count": 0,
                "total_equity": 0,
                "leverage_limit": 0,
            })
            return

        # 2. 新規エントリー執行 (益出し複利による最大建て枠の配分)
        if current_leverage_limit > 0:
            # 信用取引の総建て枠 = 純資産 (Equity) * レバレッジ倍率
            total_purchasing_power = total_equity * current_leverage_limit
            
            # 現在の総建玉評価額
            current_total_pos_val = sum(pos["shares"] * current_prices.get(t, pos["entry_price"]) for t, pos in self.positions.items())
            remaining_purchasing_power = max(0.0, total_purchasing_power - current_total_pos_val)

            available_slots = self.max_positions - len(self.positions)
            valid_candidates = [t for t in buy_signals if t not in self.positions and t in daily_prices]

            if available_slots > 0 and valid_candidates and remaining_purchasing_power > 1000:
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
                            "shares": shares,
                            "days_held": 0,
                        }

        # 3. 日次資産残高記録
        end_equity = self.get_total_equity(current_prices)
        self.equity_curve.append({
            "date": date,
            "cash": self.cash,
            "positions_count": len(self.positions),
            "total_equity": end_equity,
            "leverage_limit": current_leverage_limit,
        })


def run_margin_leverage_experiments():
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
            "prob": row["pred_prob"],
        }

    # 比較シナリオ (レバレッジ倍率 ＆ 動的制御)
    scenarios = [
        # (target_lev, lev_mode, flush, label)
        (1.0, "fixed", False, "① 現物 1.0倍 (レバレッジなし・基準)"),
        (1.5, "fixed", False, "② 固定信用 1.5倍 (控えめレバレッジ)"),
        (2.0, "fixed", False, "③ 固定信用 2.0倍 (標準レバレッジ)"),
        (2.5, "fixed", False, "④ 固定信用 2.5倍 (積極レバレッジ)"),
        (3.0, "fixed", False, "⑤ 固定信用 3.0倍 (制度上限フルレバ)"),
        (2.5, "adaptive_macro", True,  "⑥ 【動的】マクロ連動レバレッジ 2.5倍 + 危機時緊急退避"),
        (3.0, "adaptive_macro", True,  "⑦ 【動的】マクロ連動レバレッジ 3.0倍 + 危機時緊急退避 (STF型)"),
    ]

    results = []
    equity_curves_dict = {}

    for lev, mode, flush, label in scenarios:
        sim = MarginLeveragePortfolioSimulator(
            initial_cash=1_000_000.0,
            max_positions=5,
            target_leverage=lev,
            leverage_mode=mode,
            emergency_flush_on_crisis=flush,
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
    print("   MARGIN LEVERAGE & COMPOUNDING ALLOCATION RESULTS (2026 OUT-OF-SAMPLE)")
    print("==========================================================================================================")
    print(pd.DataFrame(results).to_string(index=False))
    print("==========================================================================================================\n")

    # 月別推移 (6月末ピークと7〜8月のドローダウン)
    print("=== [月別・節目ごとの純資産残高推移 (2026年)] ===")
    sample_dates = ["2026-01-05", "2026-03-31", "2026-05-29", "2026-06-30", "2026-07-31", "2026-08-04"]
    tracking_rows = []
    for d in sample_dates:
        row = {"Date": d}
        for lbl in [scenarios[0][3], scenarios[2][3], scenarios[4][3], scenarios[6][3]]:
            eq = equity_curves_dict[lbl]
            sub = eq[eq["date"] <= d]
            if not sub.empty:
                val = sub["total_equity"].iloc[-1]
                short_name = lbl.split(" (")[0][:18]
                row[short_name] = f"¥{int(val):,}"
        tracking_rows.append(row)
    print(pd.DataFrame(tracking_rows).to_string(index=False))


if __name__ == "__main__":
    run_margin_leverage_experiments()
