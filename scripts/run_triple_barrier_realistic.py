"""ATR利確・損切り (Triple Barrier) を組み込んだ現実的ポートフォリオ・バックテスト"""

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


class RealisticTripleBarrierSimulator:
    def __init__(
        self,
        initial_cash: float = 1_000_000.0,
        max_positions: int = 5,
        risk_per_trade_pct: float = 0.20,
        max_holding_days: int = 5,
        sl_atr_mult: float = 1.5,
        tp_atr_mult: float = 2.5,
        slippage_fee_pct: float = 0.001,
    ):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.max_positions = max_positions
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_holding_days = max_holding_days
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.slippage_fee_pct = slippage_fee_pct
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.closed_trades: List[Dict[str, Any]] = []
        self.equity_curve: List[Dict[str, Any]] = []

    def get_total_equity(self, current_prices: Dict[str, float]) -> float:
        pos_val = 0.0
        for ticker, pos in self.positions.items():
            curr_p = current_prices.get(ticker, pos["entry_price"])
            pos_val += pos["shares"] * curr_p
        return self.cash + pos_val

    def step(
        self,
        date: str,
        daily_prices: Dict[str, Dict[str, float]],
        buy_signals: List[str],
    ):
        current_prices = {t: d["close"] for t, d in daily_prices.items()}

        # 1. 決済チェック (High/LowでのSL/TP判定 & 期限切れ判定)
        closed_tickers = []
        for ticker, pos in list(self.positions.items()):
            if ticker not in daily_prices:
                continue
            
            p_data = daily_prices[ticker]
            pos["days_held"] += 1

            exit_triggered = False
            exit_price = p_data["close"]
            exit_reason = ""

            # 損切りチェック (当日安値 Low が SL を下回ったか)
            if p_data["low"] <= pos["sl_price"]:
                exit_triggered = True
                exit_price = pos["sl_price"] * (1.0 - self.slippage_fee_pct)
                exit_reason = "Stop Loss"
            # 利確チェック (当日高値 High が TP を上回ったか)
            elif p_data["high"] >= pos["tp_price"]:
                exit_triggered = True
                exit_price = pos["tp_price"] * (1.0 - self.slippage_fee_pct)
                exit_reason = "Take Profit"
            # 期限切れ (最大保有日数経過)
            elif pos["days_held"] >= self.max_holding_days:
                exit_triggered = True
                exit_price = p_data["close"] * (1.0 - self.slippage_fee_pct)
                exit_reason = "Time Limit"

            if exit_triggered:
                gross_ret = (exit_price - pos["entry_price"]) / pos["entry_price"]
                pnl = pos["shares"] * (exit_price - pos["entry_price"])
                self.cash += pos["shares"] * exit_price

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

        # 2. 新規エントリー
        total_equity = self.get_total_equity(current_prices)

        for ticker in buy_signals:
            if ticker in self.positions:
                continue
            if len(self.positions) >= self.max_positions:
                break

            if ticker not in daily_prices:
                continue

            open_p = daily_prices[ticker]["open"] * (1.0 + self.slippage_fee_pct)
            atr = daily_prices[ticker]["atr"]
            if open_p <= 0 or atr <= 0:
                continue

            target_pos_size = total_equity * self.risk_per_trade_pct
            invest_amount = min(target_pos_size, self.cash)

            shares = int(invest_amount // (open_p * 100)) * 100
            if shares <= 0 and invest_amount >= open_p:
                shares = int(invest_amount // open_p)

            cost = shares * open_p
            if shares > 0 and self.cash >= cost:
                self.cash -= cost
                sl_p = open_p - (self.sl_atr_mult * atr)
                tp_p = open_p + (self.tp_atr_mult * atr)
                self.positions[ticker] = {
                    "entry_date": date,
                    "entry_price": open_p,
                    "sl_price": sl_p,
                    "tp_price": tp_p,
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
        })


def run():
    bars_daily_dir = ROOT_DIR / "market_data" / "jquants" / "bars_daily"
    macro_dir = ROOT_DIR / "market_data" / "global_macro"
    tickers_file = ROOT_DIR / "tickers.csv"

    df_tickers = pd.read_csv(tickers_file)
    ticker_list = df_tickers["ticker"].tolist()[:50]

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
            merged["vix_regime"] = (merged["vix_close"] > 22.0).astype(int)

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
    stock_by_date: Dict[str, Dict[str, Dict[str, float]]] = {}
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

    sim = RealisticTripleBarrierSimulator(
        initial_cash=1_000_000.0,
        max_positions=5,
        risk_per_trade_pct=0.20,
        max_holding_days=5,
        sl_atr_mult=1.5,
        tp_atr_mult=2.5,
        slippage_fee_pct=0.001,
    )

    pending_buy_signals: List[str] = []
    for d in test_dates:
        daily_stocks = stock_by_date.get(d, {})
        sim.step(date=d, daily_prices=daily_stocks, buy_signals=pending_buy_signals)

        pending_buy_signals = []
        candidates = []
        for ticker, data in daily_stocks.items():
            if data["prob"] > 0.55:
                candidates.append((ticker, data["prob"]))
        candidates.sort(key=lambda x: x[1], reverse=True)
        pending_buy_signals = [c[0] for c in candidates[:5]]

    eq_df = pd.DataFrame(sim.equity_curve)
    trades_df = pd.DataFrame(sim.closed_trades)

    final_equity = eq_df["total_equity"].iloc[-1]
    multiplier = final_equity / sim.initial_cash
    win_rate = (trades_df["win"].sum() / len(trades_df) * 100) if len(trades_df) > 0 else 0
    avg_ret = (trades_df["ret"].mean() * 100) if len(trades_df) > 0 else 0

    eq_df["peak"] = eq_df["total_equity"].cummax()
    eq_df["drawdown"] = (eq_df["total_equity"] - eq_df["peak"]) / eq_df["peak"]
    max_dd = eq_df["drawdown"].min() * 100

    print("==============================================================")
    print("   TRIPLE BARRIER REALISTIC PORTFOLIO RESULTS (2026 OUT-OF-SAMPLE)")
    print("==============================================================")
    print(f"初期元手資金     : ¥{int(sim.initial_cash):,}")
    print(f"最終資産残高     : ¥{int(final_equity):,}")
    print(f"資金累積倍率     : {multiplier:.2f} 倍 (リターン: {(multiplier - 1.0)*100:+.2f}%)")
    print(f"総トレード数     : {len(trades_df)} 回")
    print(f"勝率             : {win_rate:.2f}%")
    print(f"平均トレード損益 : {avg_ret:+.2f}%")
    print(f"最大ドローダウン : {max_dd:.2f}%")
    print("エグジット理由内訳:")
    print(trades_df["reason"].value_counts())
    print("==============================================================")


if __name__ == "__main__":
    run()
