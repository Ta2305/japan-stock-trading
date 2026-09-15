"""高度化トレンドフォロー戦略モジュール (提案A)

- ATR正規化されたKAMA傾き (Normalized Slope)
- ADX トレンド強度フィルター
- 日次リセット型 VWAP 順張りフィルター
- 時間帯（寄り付き直後・引け際回避）フィルター
を組み合わせた実用的でスケーラブルなルールベース戦略です。
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from src_v2.strategy.base import BaseStrategy
from src_v2.indicators import (
    calculate_atr,
    calculate_kama,
    calculate_adx,
    calculate_vwap,
    compute_consecutive_streaks,
)
from src_v2.utils import get_time_regime


class AdvancedTrendFollowStrategy(BaseStrategy):
    """提案A: ルールベース高度化トレンドフォロー戦略"""

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """指標計算の追加"""
        df = df.copy()

        # パラメータ取得
        ind_cfg = self.config.get("indicators", {})
        kama_er = ind_cfg.get("kama_er_period", 10)
        kama_fast = ind_cfg.get("kama_fast", 2)
        kama_slow = ind_cfg.get("kama_slow", 30)
        atr_period = ind_cfg.get("atr_period", 14)
        adx_period = ind_cfg.get("adx_period", 14)

        # 1. ATR計算
        df["atr"] = calculate_atr(df, period=atr_period)

        # 2. KAMA計算
        df["kama"] = calculate_kama(
            df["close"], er_period=kama_er, fast_sc=kama_fast, slow_sc=kama_slow
        )

        # 3. KAMAの傾きとATRによる正規化
        kama_diff = df["kama"].diff()
        # normalized_slope = 傾き(円) / ATR(円) -> 全銘柄共通スケール
        df["normalized_slope"] = kama_diff / df["atr"].replace(0, np.nan)

        # 4. ADX計算
        adx_df = calculate_adx(df, period=adx_period)
        df["adx"] = adx_df["adx"]
        df["plus_di"] = adx_df["plus_di"]
        df["minus_di"] = adx_df["minus_di"]

        # 5. VWAP計算
        df["vwap"] = calculate_vwap(df)

        # 6. 連続足・連続KAMA数値
        df["kama_streak"] = compute_consecutive_streaks(df["kama"])
        df["candle_streak"] = compute_consecutive_streaks(df["close"])

        return df

    def generate_signal_at(
        self,
        df: pd.DataFrame,
        idx: int,
        current_position: Optional[str] = None,
    ) -> Dict[str, Any]:
        """インデックス時点でのシグナル判定"""
        row = df.iloc[idx]
        dt = (
            pd.to_datetime(row["datetime"])
            if "datetime" in row
            else pd.to_datetime(df.index[idx])
        )

        entry_cfg = self.config.get("entry_conditions", {}).get("trend_follow", {})
        min_slope = entry_cfg.get("slope_atr_ratio_threshold", 0.3)
        min_adx = entry_cfg.get("adx_min", 20.0)
        use_vwap = entry_cfg.get("vwap_alignment", True)
        min_streak = entry_cfg.get("consecutive_ma_min", 2)

        time_cfg = self.config.get("time_filters", {})
        no_trade_regimes = time_cfg.get("no_trade_regimes", ["out_of_hours", "closing"])

        regime = get_time_regime(dt)
        if regime in no_trade_regimes:
            return {"action": "hold", "reason": f"Time regime filter ({regime})"}

        price = row["close"]
        atr = row["atr"] if not np.isnan(row["atr"]) else 0.0
        norm_slope = (
            row["normalized_slope"] if not np.isnan(row["normalized_slope"]) else 0.0
        )
        adx = row["adx"] if not np.isnan(row["adx"]) else 0.0
        vwap = row["vwap"] if not np.isnan(row["vwap"]) else price
        kama_streak = row["kama_streak"]

        if atr <= 0:
            return {"action": "hold", "reason": "Insufficient ATR"}

        # トレンド強度フィルター
        if adx < min_adx:
            return {"action": "hold", "reason": f"Low ADX ({adx:.1f} < {min_adx})"}

        # ---- ポジションなしの場合 (エントリー判定) ----
        if current_position is None or current_position == "none":
            # ロング条件
            long_cond = (
                norm_slope >= min_slope
                and kama_streak >= min_streak
                and (not use_vwap or price >= vwap)
            )

            # ショート条件
            short_cond = (
                norm_slope <= -min_slope
                and kama_streak <= -min_streak
                and (not use_vwap or price <= vwap)
            )

            if long_cond:
                return {
                    "action": "buy",
                    "reason": f"Long Trend (Slope/ATR={norm_slope:.2f}, ADX={adx:.1f})",
                    "atr": atr,
                    "price": price,
                }
            elif short_cond:
                return {
                    "action": "short",
                    "reason": f"Short Trend (Slope/ATR={norm_slope:.2f}, ADX={adx:.1f})",
                    "atr": atr,
                    "price": price,
                }

        # ---- ポジションありの場合 (反転エグジット判定) ----
        elif current_position == "long":
            # 傾きが反転した場合
            if norm_slope < 0:
                return {
                    "action": "sell",
                    "reason": "Slope Reversed to Negative",
                    "atr": atr,
                    "price": price,
                }
        elif current_position == "short":
            # 傾きが反転した場合
            if norm_slope > 0:
                return {
                    "action": "cover",
                    "reason": "Slope Reversed to Positive",
                    "atr": atr,
                    "price": price,
                }

        return {"action": "hold", "reason": "No condition met", "atr": atr, "price": price}
