"""初期元手資金の規模別 ＆ 単元株制約による運用パフォーマンス比較検証スクリプト

初期資金規模 (100万〜3000万) と単元株/単元未満株の制約が、ポートフォリオの分散性・リターン・倍率に与える影響を検証する。
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


class RealisticCapitalScaleSimulator:
    def __init__(
        self,
        initial_cash: float = 1_000_000.0,
        max_positions: int = 5,
        risk_per_trade_pct: float = 0.20,
        holding_days: int = 3,
        slippage_fee_pct: float = 0.001,
        allow_fractional: bool = False,  # 単元未満株 (1株単位) の許可フラグ
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
        self.skipped_trades_due_to_capital = 0

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

        # 1. イグジット
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

        for ticker in buy_signals:
            if ticker in self.positions:
                continue
            if len(self.positions) >= self.max_positions:
                break

            if ticker not in daily_prices:
                continue

            open_p = daily_prices[ticker]["open"] * (1.0 + self.slippage_fee_pct)
            if open_p <= 0:
                continue

            target_pos_size = total_equity * self.risk_per_trade_pct
            invest_amount = min(target_pos_size, self.cash)

            if self.allow_fractional:
                # 1株単位（単元未満株）
                shares = int(invest_amount // open_p)
            else:
                # 日本株 通常単元（100株単位）
                shares = int(invest_amount // (open_p * 100)) * 100

            if shares <= 0:
                self.skipped_trades_due_to_capital += 1
                continue

            cost = shares * open_p
            if self.cash >= cost:
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


def run_capital_scaling_experiment():
    bars_daily_dir = ROOT_DIR / "market_data" / "jquants" / "bars_daily"
    macro_dir = ROOT_DIR / "market_data" / "global_macro"
    tickers_file = ROOT_DIR / "tickers.csv"

    df_tickers = pd.read_csv(tickers_file)
    ticker_list = df_tickers["ticker"].tolist()[:100]  # 上位100銘柄に拡張

    df_macro = load_global_macro_features(macro_dir)
    target_codes = {t.replace(".T", "") + "0" for t in ticker_list}
    stock_data_dict = load_recent_jquants_daily_bars_fast(bars_daily_dir, target_codes, max_days=500)

    # 特徴量作成 (米国マクロ 1日ラグ)
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

        # 目的変数
        merged["future_entry_open"] = merged["open"].shift(-1)
        merged["future_exit_close"] = merged["close"].shift(-3)
        merged["true_ret"] = (merged["future_exit_close"] - merged["future_entry_open"]) / merged["future_entry_open"]
        merged["target"] = (merged["true_ret"] > 0.005).astype(int)

        merged["ticker"] = raw_ticker
        records.append(merged.dropna())

    full_clean = pd.concat(records, ignore_index=True)

    # 厳格な日付分割 (Train: 2025年末まで / Test: 2026年1月〜8月)
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
            "prob": row["pred_prob"],
        }

    # 各種初期資金額でテスト
    capital_levels = [
        (1_000_000, False, "100万円 (単元株100株単位)"),
        (3_000_000, False, "300万円 (単元株100株単位)"),
        (5_000_000, False, "500万円 (単元株100株単位)"),
        (10_000_000, False, "1,000万円 (単元株100株単位)"),
        (30_000_000, False, "3,000万円 (単元株100株単位)"),
        (1_000_000, True, "100万円 (単元未満株・1株単位)"),
    ]

    summary_results = []

    for cap, allow_frac, label in capital_levels:
        sim = RealisticCapitalScaleSimulator(
            initial_cash=float(cap),
            max_positions=5,
            risk_per_trade_pct=0.20,
            holding_days=3,
            slippage_fee_pct=0.001,
            allow_fractional=allow_frac,
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

        final_eq = eq_df["total_equity"].iloc[-1] if not eq_df.empty else cap
        net_profit = final_eq - cap
        ret_pct = (final_eq - cap) / cap * 100
        mult = final_eq / cap
        n_trades = len(trades_df)
        win_rate = (trades_df["win"].sum() / n_trades * 100) if n_trades > 0 else 0
        avg_ret = (trades_df["ret"].mean() * 100) if n_trades > 0 else 0

        eq_df["peak"] = eq_df["total_equity"].cummax()
        eq_df["drawdown"] = (eq_df["total_equity"] - eq_df["peak"]) / eq_df["peak"]
        max_dd = eq_df["drawdown"].min() * 100

        summary_results.append({
            "資金条件": label,
            "初期資金": f"¥{cap:,}",
            "最終資産": f"¥{int(final_eq):,}",
            "純利益": f"¥{int(net_profit):+,}",
            "累積リターン": f"{ret_pct:+.2f}%",
            "元手倍率": f"{mult:.2f}倍",
            "約定トレード数": n_trades,
            "資金不足スキップ数": sim.skipped_trades_due_to_capital,
            "勝率": f"{win_rate:.1f}%",
            "1トレード平均損益": f"{avg_ret:+.2f}%",
            "Max DD": f"{max_dd:.2f}%",
        })

    res_df = pd.DataFrame(summary_results)
    print("\n==============================================================")
    print("   CAPITAL SCALING & LOT SIZE EXPERIMENT (2026 OUT-OF-SAMPLE)")
    print("==============================================================")
    print(res_df.to_string(index=False))
    print("==============================================================")


if __name__ == "__main__":
    run_capital_scaling_experiment()
