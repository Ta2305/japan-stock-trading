"""ポジション制限撤廃 (集中投資) ＆ 保有期間制限撤廃 (トレンド追従) ＆ ML有無の総合比較検証スクリプト

ポジション集中・保有期間・ML有無の組み合わせ8通りを比較し、日別・月別の資産推移も追跡する。
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


class ComprehensiveStrategySimulator:
    def __init__(
        self,
        initial_cash: float = 1_000_000.0,
        position_sizing_mode: str = "fixed_20pct",  # "fixed_20pct" (5分散) or "unconstrained_top" (集中全額)
        holding_mode: str = "fixed_3days",           # "fixed_3days" or "unlimited_trend" (トレンド終了まで)
        slippage_fee_pct: float = 0.001,
        allow_fractional: bool = True,              # 1株単位 (単元未満株・資本効率最大化)
    ):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position_sizing_mode = position_sizing_mode
        self.holding_mode = holding_mode
        self.slippage_fee_pct = slippage_fee_pct
        self.allow_fractional = allow_fractional
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

        # 1. 既存ポジションの決済判定
        closed_tickers = []
        for ticker, pos in list(self.positions.items()):
            if ticker not in daily_prices:
                continue
            
            p_data = daily_prices[ticker]
            pos["days_held"] += 1
            exit_triggered = False
            exit_reason = ""
            exit_price = p_data["close"]

            # モード別エグジット判定
            if self.holding_mode == "fixed_3days":
                if pos["days_held"] >= 3:
                    exit_triggered = True
                    exit_reason = "3-Day Time Limit"
                    exit_price = p_data["close"] * (1.0 - self.slippage_fee_pct)

            elif self.holding_mode == "unlimited_trend":
                # トレンド終了判定: 5日MAが20日MAを下回る (SMAデッドクロス) または ATR損切り (1.5*ATR)
                sma_diff = p_data.get("sma_diff", 0.0)
                sl_price = pos["entry_price"] - (1.5 * pos.get("entry_atr", p_data["atr"]))
                
                if p_data["low"] <= sl_price:
                    exit_triggered = True
                    exit_reason = "Stop Loss (1.5 ATR)"
                    exit_price = sl_price * (1.0 - self.slippage_fee_pct)
                elif sma_diff < 0:  # トレンド反転・終了
                    exit_triggered = True
                    exit_reason = "Trend End (SMA Cross Below)"
                    exit_price = p_data["close"] * (1.0 - self.slippage_fee_pct)

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

        # 2. 新規エントリー執行
        total_equity = self.get_total_equity(current_prices)

        # エントリー銘柄の選定とサイズ決定
        valid_candidates = [t for t in buy_signals if t not in self.positions and t in daily_prices]

        if valid_candidates and self.cash > 1000:
            if self.position_sizing_mode == "fixed_20pct":
                # 従来方式: 最大5銘柄、各20%
                max_pos = 5
                available_slots = max_pos - len(self.positions)
                for ticker in valid_candidates[:available_slots]:
                    open_p = daily_prices[ticker]["open"] * (1.0 + self.slippage_fee_pct)
                    if open_p <= 0:
                        continue
                    target_amt = min(total_equity * 0.20, self.cash)
                    shares = int(target_amt // open_p) if self.allow_fractional else int(target_amt // (open_p * 100)) * 100
                    cost = shares * open_p
                    if shares > 0 and self.cash >= cost:
                        self.cash -= cost
                        self.positions[ticker] = {
                            "entry_date": date,
                            "entry_price": open_p,
                            "entry_atr": daily_prices[ticker]["atr"],
                            "shares": shares,
                            "days_held": 0,
                        }

            elif self.position_sizing_mode == "unconstrained_top":
                # 縛り撤廃方式: 最も確信度が高い銘柄群に余力キャッシュを100%全力配分
                # （候補が1銘柄ならその1銘柄に余力100%全額、複数なら上位最大2〜3銘柄に余力を均等全額配分）
                max_to_buy = min(len(valid_candidates), 2)  # 最上位1〜2銘柄に全力集中
                alloc_per_ticker = self.cash / max_to_buy
                for ticker in valid_candidates[:max_to_buy]:
                    open_p = daily_prices[ticker]["open"] * (1.0 + self.slippage_fee_pct)
                    if open_p <= 0:
                        continue
                    shares = int(alloc_per_ticker // open_p) if self.allow_fractional else int(alloc_per_ticker // (open_p * 100)) * 100
                    cost = shares * open_p
                    if shares > 0 and self.cash >= cost:
                        self.cash -= cost
                        self.positions[ticker] = {
                            "entry_date": date,
                            "entry_price": open_p,
                            "entry_atr": daily_prices[ticker]["atr"],
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


def run_comprehensive_experiments():
    bars_daily_dir = ROOT_DIR / "market_data" / "jquants" / "bars_daily"
    macro_dir = ROOT_DIR / "market_data" / "global_macro"
    tickers_file = ROOT_DIR / "tickers.csv"

    df_tickers = pd.read_csv(tickers_file)
    ticker_list = df_tickers["ticker"].tolist()[:100]

    df_macro = load_global_macro_features(macro_dir)
    target_codes = {t.replace(".T", "") + "0" for t in ticker_list}
    stock_data_dict = load_recent_jquants_daily_bars_fast(bars_daily_dir, target_codes, max_days=500)

    # 特徴量作成 (米国マクロ Lag 1)
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

        # テクニカル一次モメンタムスコア (MLなし用)
        vol_recent = merged["volume"].rolling(10).mean()
        vol_past = merged["volume"].rolling(50).mean() + 1e-5
        vol_surge = vol_recent / vol_past
        high_20 = merged["high"].rolling(20).max()
        proximity = merged["close"] / (high_20 + 1e-5)
        atr_ratio = merged["atr"] / (merged["atr"].rolling(60).mean() + 1e-5)
        merged["tech_score"] = (atr_ratio * 0.4) + (vol_surge * 0.4) + (proximity * 0.2)
        # 一次シグナル: SMA乖離 > 0 かつ 出来高サージ > 1.1
        merged["primary_signal"] = (merged["sma_diff"] > 0) & (vol_surge > 1.1)

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

    # LightGBM モデル学習
    clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.03, random_state=42, verbose=-1)
    clf.fit(train_df[features], train_df["target"])
    test_df["pred_prob"] = clf.predict_proba(test_df[features])[:, 1]

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
            "prob": row["pred_prob"],
            "tech_score": row["tech_score"],
            "primary_signal": row["primary_signal"],
        }

    # 8つの実験設定
    scenarios = [
        # (use_ml, sizing_mode, holding_mode, label)
        (True, "fixed_20pct", "fixed_3days", "① [MLあり] 20%制限(5分散) × 3日保有 (現行ベース)"),
        (True, "unconstrained_top", "fixed_3days", "② [MLあり] 20%制限撤廃(Top集中) × 3日保有"),
        (True, "fixed_20pct", "unlimited_trend", "③ [MLあり] 20%制限(5分散) × 保有制限なし(トレンド終了まで)"),
        (True, "unconstrained_top", "unlimited_trend", "④ [MLあり] 20%制限撤廃(Top集中) × 保有制限なし(両方撤廃)"),
        
        (False, "fixed_20pct", "fixed_3days", "⑤ [MLなし] 20%制限(5分散) × 3日保有 (純テクニカル)"),
        (False, "unconstrained_top", "fixed_3days", "⑥ [MLなし] 20%制限撤廃(Top集中) × 3日保有"),
        (False, "fixed_20pct", "unlimited_trend", "⑦ [MLなし] 20%制限(5分散) × 保有制限なし(トレンド終了まで)"),
        (False, "unconstrained_top", "unlimited_trend", "⑧ [MLなし] 20%制限撤廃(Top集中) × 保有制限なし(両方撤廃)"),
    ]

    results = []
    equity_curves_dict = {}

    for use_ml, sizing, holding, label in scenarios:
        sim = ComprehensiveStrategySimulator(
            initial_cash=1_000_000.0,
            position_sizing_mode=sizing,
            holding_mode=holding,
            slippage_fee_pct=0.001,
            allow_fractional=True,
        )

        pending_buy_signals: List[str] = []

        for d in test_dates:
            daily_stocks = stock_by_date.get(d, {})
            sim.step(date=d, daily_prices=daily_stocks, buy_signals=pending_buy_signals)

            pending_buy_signals = []
            candidates = []

            for ticker, data in daily_stocks.items():
                if use_ml:
                    # MLあり: P(Win) > 0.55
                    if data["prob"] > 0.55:
                        candidates.append((ticker, data["prob"]))
                else:
                    # MLなし: 一次シグナルがTrueのもの
                    if data["primary_signal"]:
                        candidates.append((ticker, data["tech_score"]))

            # スコア降順ソート
            candidates.sort(key=lambda x: x[1], reverse=True)
            pending_buy_signals = [c[0] for c in candidates[:5]]

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
            "平均損益": f"{avg_ret:+.2f}%",
            "Max DD": f"{max_dd:.2f}%",
        })
        equity_curves_dict[label] = eq_df

    print("\n==========================================================================================================")
    print("   COMPREHENSIVE BACKTEST MATRIX: POSITION SIZING, HOLDING PERIOD, AND ML IMPACT (2026 OUT-OF-SAMPLE)")
    print("==========================================================================================================")
    print(pd.DataFrame(results).to_string(index=False))
    print("==========================================================================================================\n")

    # 資産推移の月別トラッキング (シナリオ① vs シナリオ② vs シナリオ④)
    print("=== [月別・節目ごとの資産残高トラッキング (2026年)] ===")
    sample_dates = ["2026-01-05", "2026-03-31", "2026-05-29", "2026-06-30", "2026-07-31", "2026-08-04"]
    
    tracking_rows = []
    for d in sample_dates:
        row = {"Date": d}
        for lbl in [scenarios[0][3], scenarios[1][3], scenarios[3][3], scenarios[4][3]]:
            eq = equity_curves_dict[lbl]
            sub = eq[eq["date"] <= d]
            if not sub.empty:
                val = sub["total_equity"].iloc[-1]
                short_name = lbl.split("] ")[1][:18]
                row[short_name] = f"¥{int(val):,}"
        tracking_rows.append(row)
    print(pd.DataFrame(tracking_rows).to_string(index=False))


if __name__ == "__main__":
    run_comprehensive_experiments()
