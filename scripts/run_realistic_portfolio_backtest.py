"""現実的ポートフォリオ・バックテスト＆資産推移シミュレーションスクリプト

時系列日付分割・米国マクロ指標の1営業日ラグ (Lag 1)・翌日Open執行・往復スリッページ手数料 (0.1%) を考慮し、
最大5銘柄・1ポジション最大20%の資金拘束付きポートフォリオ運用をシミュレーションします。
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


class RealisticBacktestSimulator:
    def __init__(
        self,
        initial_cash: float = 1_000_000.0,
        max_positions: int = 5,
        risk_per_trade_pct: float = 0.20,
        holding_days: int = 3,
        slippage_fee_pct: float = 0.001,  # 往復0.1%
    ):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.max_positions = max_positions
        self.risk_per_trade_pct = risk_per_trade_pct
        self.holding_days = holding_days
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
        """1営業日のシミュレーションステップ"""
        current_prices = {t: d["close"] for t, d in daily_prices.items()}

        # 1. 既存ポジションの決済チェック (寄付または引けでの手仕舞い)
        closed_tickers = []
        for ticker, pos in list(self.positions.items()):
            if ticker not in daily_prices:
                continue
            
            p_data = daily_prices[ticker]
            pos["days_held"] += 1

            # イグジット判定 (3営業日経過した日の引けでエグジット)
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

        # 2. 新規エントリーの執行 (前日シグナルに基づき、当日Openで成行買い)
        total_equity = self.get_total_equity(current_prices)

        for ticker in buy_signals:
            if ticker in self.positions:
                continue  # 既に保有中なら見送り
            if len(self.positions) >= self.max_positions:
                break     # ポジション枠上限

            if ticker not in daily_prices:
                continue

            open_p = daily_prices[ticker]["open"] * (1.0 + self.slippage_fee_pct)
            if open_p <= 0:
                continue

            target_pos_size = total_equity * self.risk_per_trade_pct
            invest_amount = min(target_pos_size, self.cash)

            # 日本株単元株 (100株単位) を考慮
            shares = int(invest_amount // (open_p * 100)) * 100
            if shares <= 0 and invest_amount >= open_p:
                shares = int(invest_amount // open_p)  # 単元未満考慮

            cost = shares * open_p
            if shares > 0 and self.cash >= cost:
                self.cash -= cost
                self.positions[ticker] = {
                    "entry_date": date,
                    "entry_price": open_p,
                    "shares": shares,
                    "days_held": 0,
                }

        # 3. 日次資産残高の記録
        end_equity = self.get_total_equity(current_prices)
        self.equity_curve.append({
            "date": date,
            "cash": self.cash,
            "positions_count": len(self.positions),
            "total_equity": end_equity,
        })


def run_realistic_simulation():
    bars_daily_dir = ROOT_DIR / "market_data" / "jquants" / "bars_daily"
    macro_dir = ROOT_DIR / "market_data" / "global_macro"
    tickers_file = ROOT_DIR / "tickers.csv"

    df_tickers = pd.read_csv(tickers_file)
    ticker_list = df_tickers["ticker"].tolist()[:50]

    df_macro = load_global_macro_features(macro_dir)
    target_codes = {t.replace(".T", "") + "0" for t in ticker_list}
    stock_data_dict = load_recent_jquants_daily_bars_fast(bars_daily_dir, target_codes, max_days=500)

    # 1. 厳格な特徴量作成 (米国マクロ 1日ラグ)
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

        # 目的変数: 当日引けシグナル -> 翌日Openで買い、3日後Closeで売ったときの純リターンが正か
        merged["future_entry_open"] = merged["open"].shift(-1)
        merged["future_exit_close"] = merged["close"].shift(-3)
        merged["true_ret"] = (merged["future_exit_close"] - merged["future_entry_open"]) / merged["future_entry_open"]
        merged["target"] = (merged["true_ret"] > 0.005).astype(int)  # 手数料込みで+0.5%以上

        merged["ticker"] = raw_ticker
        records.append(merged.dropna())

    full_clean = pd.concat(records, ignore_index=True)

    # 2. 厳格な日付分割 (Train: 2024年〜2025年末 / Test: 2026年1月〜2026年8月)
    cutoff_date = "2026-01-05"

    train_df = full_clean[full_clean["date"] < cutoff_date].copy()
    test_df = full_clean[full_clean["date"] >= cutoff_date].copy()

    features = ["returns", "sma_diff", "atr", "sox_return_1d", "sox_diff_lag", "usdjpy_return_1d", "us10y_change", "vix_regime"]
    features = [f for f in features if f in full_clean.columns]

    for f in features:
        train_df[f] = train_df[f].astype(float)
        test_df[f] = test_df[f].astype(float)

    # モデル学習
    clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.03, random_state=42, verbose=-1)
    clf.fit(train_df[features], train_df["target"])

    test_df["pred_prob"] = clf.predict_proba(test_df[features])[:, 1]

    # 3. 日次時系列シミュレーションの実行 (2026年1月〜8月の全営業日)
    test_dates = sorted(test_df["date"].unique())
    
    # 銘柄ごとの日足データ辞書
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

    sim = RealisticBacktestSimulator(
        initial_cash=1_000_000.0,
        max_positions=5,
        risk_per_trade_pct=0.20,
        holding_days=3,
        slippage_fee_pct=0.001,
    )

    # 前日に発生したシグナルを翌日に執行するキュー
    pending_buy_signals: List[str] = []

    for d in test_dates:
        daily_stocks = stock_by_date.get(d, {})
        
        # 1. 前日のシグナルで当日寄付エントリー＆3日経過エグジット
        sim.step(
            date=d,
            daily_prices=daily_stocks,
            buy_signals=pending_buy_signals,
        )

        # 2. 当日引け値に基づいてシグナル計算 (翌日寄付エントリー用)
        pending_buy_signals = []
        candidates = []
        for ticker, data in daily_stocks.items():
            if data["prob"] > 0.55:  # 確信度55%以上
                candidates.append((ticker, data["prob"]))
        
        # 確率が高い順にソート
        candidates.sort(key=lambda x: x[1], reverse=True)
        pending_buy_signals = [c[0] for c in candidates[:5]]

    # 4. バックテスト集計
    eq_df = pd.DataFrame(sim.equity_curve)
    trades_df = pd.DataFrame(sim.closed_trades)

    final_equity = eq_df["total_equity"].iloc[-1] if not eq_df.empty else sim.initial_cash
    multiplier = final_equity / sim.initial_cash
    total_trades = len(trades_df)
    win_rate = (trades_df["win"].sum() / total_trades * 100) if total_trades > 0 else 0
    avg_trade_ret = (trades_df["ret"].mean() * 100) if total_trades > 0 else 0

    # 最大ドローダウンの計算
    eq_df["peak"] = eq_df["total_equity"].cummax()
    eq_df["drawdown"] = (eq_df["total_equity"] - eq_df["peak"]) / eq_df["peak"]
    max_dd = eq_df["drawdown"].min() * 100

    print("==============================================================")
    print("   REALISTIC PORTFOLIO BACKTEST RESULTS (OUT-OF-SAMPLE 2026)")
    print("==============================================================")
    print(f"テスト期間 (未知データ)     : {test_dates[0]} 〜 {test_dates[-1]} (約8ヶ月・{len(test_dates)} 営業日)")
    print(f"初期元手資金                 : ¥{int(sim.initial_cash):,}")
    print(f"最終資産残高                 : ¥{int(final_equity):,}")
    print(f"資金累積倍率                 : {multiplier:.2f} 倍 (リターン: {(multiplier - 1.0)*100:+.2f}%)")
    print(f"総執行トレード数             : {total_trades} 回")
    print(f"実質勝率 (Win Rate)          : {win_rate:.2f}%")
    print(f"1トレード平均損益率 (手数料後): {avg_trade_ret:+.2f}%")
    print(f"最大ドローダウン (Max DD)    : {max_dd:.2f}%")
    print("==============================================================")

    # 上位利益銘柄
    if not trades_df.empty:
        print("\n--- 銘柄別 損益サマリー (Top 10) ---")
        by_ticker = trades_df.groupby("ticker").agg(
            trades=("win", "count"),
            win_rate=("win", lambda x: round(x.mean() * 100, 1)),
            total_pnl=("pnl", "sum"),
            avg_ret=("ret", lambda x: round(x.mean() * 100, 2)),
        ).sort_values("total_pnl", ascending=False).head(10)
        print(by_ticker.to_string())


if __name__ == "__main__":
    run_realistic_simulation()
