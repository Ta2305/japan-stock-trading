"""10年間フルデータ (2016年〜2026年) を用いた長期 Out-of-Sample バックテストスクリプト

2016〜2023年で学習し、2024〜2026年の長期 Out-of-Sample 期間で運用成績を評価する。
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


class LongTermSimulator:
    def __init__(
        self,
        initial_cash: float = 1_000_000.0,
        max_positions: int = 5,
        risk_per_trade_pct: float = 0.20,
        holding_days: int = 3,
        slippage_fee_pct: float = 0.001,
        allow_fractional: bool = True,
    ):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.max_positions = max_positions
        self.risk_per_trade_pct = risk_per_trade_pct
        self.holding_days = holding_days
        self.slippage_fee_pct = slippage_fee_pct
        self.allow_fractional = allow_fractional

        self.positions: Dict[str, Dict[str, Any]] = {}
        self.closed_trades: List[Dict[str, Any]] = []
        self.equity_curve: List[Dict[str, Any]] = []

    def get_total_equity(self, current_prices: Dict[str, float]) -> float:
        pos_val = sum(pos["shares"] * current_prices.get(t, pos["entry_price"]) for t, pos in self.positions.items())
        return self.cash + pos_val

    def step(
        self,
        date: str,
        daily_prices: Dict[str, Dict[str, float]],
        buy_signals: List[str],
    ):
        current_prices = {t: d["close"] for t, d in daily_prices.items()}

        # 1. 決済
        closed_tickers = []
        for ticker, pos in list(self.positions.items()):
            if ticker not in daily_prices:
                continue
            
            p_data = daily_prices[ticker]
            pos["days_held"] += 1

            if pos["days_held"] >= self.holding_days:
                exit_price = p_data["close"] * (1.0 - self.slippage_fee_pct)
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
                })
                closed_tickers.append(ticker)

        for t in closed_tickers:
            del self.positions[t]

        # 2. エントリー
        total_equity = self.get_total_equity(current_prices)
        valid_candidates = [t for t in buy_signals if t not in self.positions and t in daily_prices]
        available_slots = self.max_positions - len(self.positions)

        if available_slots > 0 and valid_candidates and self.cash > 1000:
            for ticker in valid_candidates[:available_slots]:
                open_p = daily_prices[ticker]["open"] * (1.0 + self.slippage_fee_pct)
                if open_p <= 0:
                    continue
                target_amt = min(total_equity * self.risk_per_trade_pct, self.cash)
                shares = int(target_amt // open_p) if self.allow_fractional else int(target_amt // (open_p * 100)) * 100
                cost = shares * open_p
                if shares > 0 and self.cash >= cost:
                    self.cash -= cost
                    self.positions[ticker] = {
                        "entry_date": date,
                        "entry_price": open_p,
                        "shares": shares,
                        "days_held": 0,
                    }

        # 3. 資産記録
        end_equity = self.get_total_equity(current_prices)
        self.equity_curve.append({
            "date": date,
            "cash": self.cash,
            "positions_count": len(self.positions),
            "total_equity": end_equity,
        })


def run_long_term_evaluation():
    bars_daily_dir = ROOT_DIR / "market_data" / "jquants" / "bars_daily"
    macro_dir = ROOT_DIR / "market_data" / "global_macro"
    tickers_file = ROOT_DIR / "tickers.csv"

    df_tickers = pd.read_csv(tickers_file)
    ticker_list = df_tickers["ticker"].tolist()[:100]

    df_macro = load_global_macro_features(macro_dir)
    target_codes = {t.replace(".T", "") + "0" for t in ticker_list}
    
    # 過去全期間 (max_days=2500, 10年分フルロード)
    print("Loading 10-year J-Quants full dataset (2016-2026)...")
    stock_data_dict = load_recent_jquants_daily_bars_fast(bars_daily_dir, target_codes, max_days=2500)
    print(f"Loaded {len(stock_data_dict)} stocks across 10 years.")

    records = []
    df_macro_lagged = df_macro.copy()
    for col in ["sox_close", "sp500_close", "usdjpy_close", "vix_close", "us10y_yield"]:
        if col in df_macro_lagged.columns:
            df_macro_lagged[col] = df_macro_lagged[col].shift(1)

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

    # 厳格な長期分割:
    # Train: 2016年08月 〜 2023年12月 (約7.5年間)
    # Test : 2024年01月 〜 2026年08月 (直近2.5年間・約640営業日)
    cutoff_date = "2024-01-04"
    train_df = full_clean[full_clean["date"] < cutoff_date].copy()
    test_df = full_clean[full_clean["date"] >= cutoff_date].copy()

    features = ["returns", "sma_diff", "atr", "sox_return_1d", "sox_diff_lag", "usdjpy_return_1d", "us10y_change", "vix_regime"]
    features = [f for f in features if f in full_clean.columns]

    for f in features:
        train_df[f] = train_df[f].astype(float)
        test_df[f] = test_df[f].astype(float)

    print(f"Train Dataset: {train_df['date'].min()} to {train_df['date'].max()} ({len(train_df)} rows)")
    print(f"Test Dataset : {test_df['date'].min()} to {test_df['date'].max()} ({len(test_df)} rows)")

    # モデル学習
    clf = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.03, random_state=42, verbose=-1)
    clf.fit(train_df[features], train_df["target"])
    test_df["pred_prob"] = clf.predict_proba(test_df[features])[:, 1]

    # 日次シミュレーション
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
            "prob": row["pred_prob"],
        }

    sim = LongTermSimulator(
        initial_cash=1_000_000.0,
        max_positions=5,
        risk_per_trade_pct=0.20,
        holding_days=3,
        slippage_fee_pct=0.001,
        allow_fractional=True,
    )

    pending_buy: List[str] = []

    for d in test_dates:
        daily_stocks = stock_by_date.get(d, {})
        sim.step(date=d, daily_prices=daily_stocks, buy_signals=pending_buy)

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

    print("\n=========================================================================")
    print("   LONG-TERM OUT-OF-SAMPLE BACKTEST RESULTS (2024 - 2026 : 2.5 YEARS)")
    print("=========================================================================")
    print(f"学習期間 (Train)      : {train_df['date'].min()} 〜 {train_df['date'].max()} (約7.5年)")
    print(f"検証期間 (Test)       : {test_dates[0]} 〜 {test_dates[-1]} (2.5年間・{len(test_dates)} 営業日)")
    print(f"初期元手資金          : ¥1,000,000")
    print(f"最終資産残高          : ¥{int(final_eq):,}")
    print(f"累積リターン          : {ret_pct:+.2f}% (元手の {mult:.2f} 倍)")
    print(f"総執行トレード数      : {n_trades} 回")
    print(f"実質勝率 (Win Rate)   : {win_rate:.1f}%")
    print(f"1トレード平均損益     : {avg_ret:+.2f}%")
    print(f"最大ドローダウン (Max DD): {max_dd:.2f}%")
    print("=========================================================================\n")

    # 年別パフォーマンス
    trades_df["year"] = pd.to_datetime(trades_df["exit_date"]).dt.year
    print("--- [年別の運用実績] ---")
    yearly = trades_df.groupby("year").agg(
        trades=("win", "count"),
        win_rate=("win", lambda x: round(x.mean() * 100, 1)),
        avg_ret=("ret", lambda x: round(x.mean() * 100, 2)),
    )
    print(yearly.to_string())


if __name__ == "__main__":
    run_long_term_evaluation()
