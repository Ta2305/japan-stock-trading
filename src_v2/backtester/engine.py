"""バックテスト実行エンジンモジュール (イベント駆動・Bar-by-Bar)

指標算出、戦略シグナル取得、Meta-labeling (提案B) 判定、
ATRベースSL/TP・トレイリング判定、呼値・スリッページ考慮の注文執行を一貫して行います。
"""

from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd

from src_v2.backtester.portfolio import Portfolio, Trade
from src_v2.backtester.report import calculate_performance_metrics, print_backtest_report
from src_v2.strategy import AdvancedTrendFollowStrategy, AdvancedMTFRegimeStrategy
from src_v2.ml import MetaLabelingModel, build_ml_features
from src_v2.risk import calculate_sl_tp, update_trailing_stop, calculate_position_size
from src_v2.utils import get_tick_size


class BacktestEngine:
    """提案A / 提案A+ / 提案B 対応統合バックテストエンジン"""

    def __init__(self, config: Dict[str, Any], model_dir: Optional[str] = "saved_models"):
        self.config = config.get("default", config)

        strat_type = self.config.get("strategy_type", "mtf_regime")
        if strat_type == "mtf_regime":
            self.strategy = AdvancedMTFRegimeStrategy(self.config)
        else:
            self.strategy = AdvancedTrendFollowStrategy(self.config)

        # Meta-labeling 機械学習モデル (提案B) の自動ロード
        self.meta_model: Optional[MetaLabelingModel] = None
        if model_dir and Path(model_dir).exists() and (Path(model_dir) / "meta_model.txt").exists():
            try:
                model = MetaLabelingModel()
                model.load(model_dir)
                self.meta_model = model
                print(f"[ML Engine] Successfully loaded Meta-labeling model from {model_dir}/")
            except Exception as e:
                print(f"[ML Engine] Warning: Failed to load meta model from {model_dir}: {e}")

    def run(self, ticker: str, df: pd.DataFrame) -> Dict[str, Any]:
        """指定銘柄データに対してバックテストを実行する。

        Parameters
        ----------
        ticker : str
            銘柄コード
        df : pd.DataFrame
            1分足データ (open, high, low, close, volume)

        Returns
        -------
        Dict[str, Any]
            バックテスト評価結果
        """
        if df.empty:
            raise ValueError(f"Empty dataframe for ticker {ticker}")

        # 1. テクニカル指標およびML用特徴量の準備
        prep_df = self.strategy.prepare_indicators(df)
        feat_df = build_ml_features(prep_df)

        risk_cfg = self.config.get("risk_management", {})
        exec_cfg = self.config.get("execution", {})

        initial_cash = risk_cfg.get("initial_cash", 1_000_000.0)
        sl_mult = risk_cfg.get("stop_loss_atr_multiplier", 2.0)
        tp_mult = risk_cfg.get("take_profit_atr_multiplier", 3.0)
        use_trailing = risk_cfg.get("trailing_stop_enabled", True)
        trailing_mult = risk_cfg.get("trailing_stop_atr_multiplier", 1.5)
        risk_pct = risk_cfg.get("risk_per_trade_pct", 1.0)
        use_slippage = exec_cfg.get("use_slippage", True)

        portfolio = Portfolio(initial_cash=initial_cash)

        n_rows = len(prep_df)

        for i in range(n_rows):
            row = prep_df.iloc[i]
            dt = (
                pd.to_datetime(row["datetime"])
                if "datetime" in row
                else pd.to_datetime(prep_df.index[i])
            )
            curr_high = row["high"]
            curr_low = row["low"]
            curr_close = row["close"]
            curr_atr = row["atr"] if not pd.isna(row["atr"]) else 0.0

            tick = get_tick_size(curr_close)
            slippage = tick if use_slippage else 0.0

            # ----------------------------------------------------
            # 1. ポジション保有中の場合の処理 (SL/TP/トレイリングチェック)
            # ----------------------------------------------------
            if portfolio.has_position:
                trade = portfolio.position
                portfolio.update_price(curr_close)

                # トレイリングストップの更新
                if use_trailing and curr_atr > 0:
                    portfolio.current_sl = update_trailing_stop(
                        current_price=curr_close,
                        high_since_entry=portfolio.high_since_entry,
                        low_since_entry=portfolio.low_since_entry,
                        position_type=trade.position_type,
                        atr=curr_atr,
                        trailing_atr_multiplier=trailing_mult,
                        current_sl=portfolio.current_sl,
                    )

                # 決済条件のチェック
                exit_triggered = False
                exit_price = curr_close
                exit_reason = ""

                sl = portfolio.current_sl
                tp = portfolio.current_tp

                if trade.position_type == "long":
                    if sl is not None and curr_low <= sl:
                        exit_triggered = True
                        exit_price = sl - slippage
                        exit_reason = "Stop Loss"
                    elif tp is not None and curr_high >= tp:
                        exit_triggered = True
                        exit_price = tp - slippage
                        exit_reason = "Take Profit"

                elif trade.position_type == "short":
                    if sl is not None and curr_high >= sl:
                        exit_triggered = True
                        exit_price = sl + slippage
                        exit_reason = "Stop Loss"
                    elif tp is not None and curr_low <= tp:
                        exit_triggered = True
                        exit_price = tp + slippage
                        exit_reason = "Take Profit"

                # 15:29 (引け1分前) での強制ポジションクローズ
                if not exit_triggered and dt.time() >= pd.to_datetime("15:29").time():
                    exit_triggered = True
                    exit_price = curr_close - (
                        slippage if trade.position_type == "long" else -slippage
                    )
                    exit_reason = "Force Close Before Session End"

                if exit_triggered:
                    portfolio.close_position(
                        exit_time=dt, exit_price=exit_price, exit_reason=exit_reason
                    )
                    continue

            # ----------------------------------------------------
            # 2. 戦略シグナルの生成および Meta-labeling ML 判定 (提案B)
            # ----------------------------------------------------
            curr_pos = portfolio.current_position_type
            sig = self.strategy.generate_signal_at(
                prep_df, i, current_position=curr_pos
            )
            action = sig["action"]

            # エグジットシグナル処理
            if action in ["sell", "cover"] and portfolio.has_position:
                exit_p = curr_close - (
                    slippage if curr_pos == "long" else -slippage
                )
                portfolio.close_position(
                    exit_time=dt, exit_price=exit_p, exit_reason=sig["reason"]
                )

            # エントリーシグナル処理
            elif action in ["buy", "short"] and not portfolio.has_position:
                # Meta-labeling 機械学習によるシグナルフィルタリング (提案B)
                if self.meta_model is not None:
                    feat_row = feat_df.iloc[[i]]
                    pass_ml = self.meta_model.should_filter_signal(feat_row)
                    if not pass_ml:
                        # 機械学習が「ダマシ・低勝率トレード」と判断したため見送り
                        continue

                pos_type = "long" if action == "buy" else "short"

                calc_sl = (
                    curr_close - (sl_mult * curr_atr)
                    if pos_type == "long"
                    else curr_close + (sl_mult * curr_atr)
                )

                shares = calculate_position_size(
                    account_cash=portfolio.cash,
                    entry_price=curr_close,
                    stop_loss_price=calc_sl,
                    risk_per_trade_pct=risk_pct,
                )

                if shares > 0 and curr_atr > 0:
                    entry_p = curr_close + (
                        slippage if pos_type == "long" else -slippage
                    )
                    real_sl, real_tp = calculate_sl_tp(
                        entry_price=entry_p,
                        position_type=pos_type,
                        atr=curr_atr,
                        sl_atr_multiplier=sl_mult,
                        tp_atr_multiplier=tp_mult,
                    )
                    portfolio.open_position(
                        ticker=ticker,
                        position_type=pos_type,
                        entry_time=dt,
                        entry_price=entry_p,
                        shares=shares,
                        sl_price=real_sl,
                        tp_price=real_tp,
                    )

        metrics = calculate_performance_metrics(
            portfolio.closed_trades, initial_cash=initial_cash
        )
        return {"ticker": ticker, "metrics": metrics, "trades": portfolio.closed_trades}
