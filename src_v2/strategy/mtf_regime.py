"""マルチタイムフレーム (MTF) & 時間帯レジーム統合戦略 (提案A+)

- 1分足 KAMA ＋ 5分足 EMA トレンド順張り (MTF Confirmation)
- Relative Volume (RVOL) 出来高スパイク確認
- 時間帯（寄り付き/日中/引け際）レジームに応じた動的フィルター調整
- ATR正規化による全銘柄共通スケーリング
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from src_v2.strategy.base import BaseStrategy
from src_v2.indicators import (
    calculate_atr,
    calculate_kama,
    calculate_ema,
    calculate_adx,
    calculate_vwap,
    calculate_rvol,
    compute_consecutive_streaks,
)
from src_v2.utils import get_time_regime


class AdvancedMTFRegimeStrategy(BaseStrategy):
    """提案A+: MTF 順張り & RVOL 出来高 統合戦略"""

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """1分足および5分足マルチタイムフレーム指標の計算"""
        df = df.copy()

        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.set_index("datetime", drop=False, inplace=True)

        ind_cfg = self.config.get("indicators", {})
        kama_er = ind_cfg.get("kama_er_period", 10)
        kama_fast = ind_cfg.get("kama_fast", 2)
        kama_slow = ind_cfg.get("kama_slow", 30)
        atr_period = ind_cfg.get("atr_period", 14)
        adx_period = ind_cfg.get("adx_period", 14)

        # 1. 1分足基本指標
        df["atr"] = calculate_atr(df, period=atr_period)
        df["kama"] = calculate_kama(
            df["close"], er_period=kama_er, fast_sc=kama_fast, slow_sc=kama_slow
        )

        # KAMAの傾きとATRによる正規化
        kama_diff = df["kama"].diff()
        df["normalized_slope"] = kama_diff / df["atr"].replace(0, np.nan)

        # ADX / VWAP / RVOL
        adx_df = calculate_adx(df, period=adx_period)
        df["adx"] = adx_df["adx"]
        df["vwap"] = calculate_vwap(df)
        df["rvol"] = calculate_rvol(df, window=20)

        df["kama_streak"] = compute_consecutive_streaks(df["kama"])

        # 2. 5分足マルチタイムフレーム (MTF) 指標の算出
        df_5m = (
            df.resample("5min")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )

        df_5m["ema_5m"] = calculate_ema(df_5m["close"], period=20)
        df_5m["ema_5m_slope"] = df_5m["ema_5m"].diff()

        # 5分足の指標を1分足データに前方補間 (ffill) でマージ
        df["ema_5m"] = df_5m["ema_5m"].reindex(df.index, method="ffill")
        df["ema_5m_slope"] = df_5m["ema_5m_slope"].reindex(df.index, method="ffill")

        return df.reset_index(drop=True) if "datetime" in df.columns else df

    def generate_signal_at(
        self,
        df: pd.DataFrame,
        idx: int,
        current_position: Optional[str] = None,
    ) -> Dict[str, Any]:
        """インデックス時点での総合シグナル判定"""
        row = df.iloc[idx]
        dt = (
            pd.to_datetime(row["datetime"])
            if "datetime" in row
            else pd.to_datetime(df.index[idx])
        )

        entry_cfg = self.config.get("entry_conditions", {}).get("trend_follow", {})
        base_min_slope = entry_cfg.get("slope_atr_ratio_threshold", 0.4)
        min_adx = entry_cfg.get("adx_min", 20.0)
        min_rvol = entry_cfg.get("min_rvol", 1.2)
        use_vwap = entry_cfg.get("vwap_alignment", True)

        regime = get_time_regime(dt)
        if regime in ["out_of_hours", "closing"]:
            return {"action": "hold", "reason": f"Time regime filter ({regime})"}

        # 時間帯ごとの感度調整 (寄り付きは条件を少し緩和, 日中は厳格化)
        if regime == "opening":
            min_slope = base_min_slope * 0.8
            rvol_thresh = min_rvol
        elif regime in ["midday_morning", "midday_afternoon"]:
            min_slope = base_min_slope * 1.2
            rvol_thresh = min_rvol * 1.2
        else:
            min_slope = base_min_slope
            rvol_thresh = min_rvol

        price = row["close"]
        atr = row["atr"] if not np.isnan(row["atr"]) else 0.0
        norm_slope = (
            row["normalized_slope"] if not np.isnan(row["normalized_slope"]) else 0.0
        )
        adx = row["adx"] if not np.isnan(row["adx"]) else 0.0
        rvol = row["rvol"] if not np.isnan(row["rvol"]) else 1.0
        vwap = row["vwap"] if not np.isnan(row["vwap"]) else price
        m5_slope = (
            row["ema_5m_slope"] if not np.isnan(row["ema_5m_slope"]) else 0.0
        )

        if atr <= 0:
            return {"action": "hold", "reason": "Insufficient ATR"}

        # トレンド強度 & 出来高フィルター
        if adx < min_adx:
            return {"action": "hold", "reason": f"Low ADX ({adx:.1f})"}
        if rvol < rvol_thresh:
            return {"action": "hold", "reason": f"Low RVOL ({rvol:.2f})"}

        # ---- ポジションなしの場合 (エントリー判定) ----
        if current_position is None or current_position == "none":
            # ロング条件: 1分足KAMA上昇 + 5分足EMA上昇 + VWAPの上 + RVOLスパイク
            long_cond = (
                norm_slope >= min_slope
                and m5_slope > 0
                and (not use_vwap or price >= vwap)
            )

            # ショート条件: 1分足KAMA下落 + 5分足EMA下落 + VWAPの下 + RVOLスパイク
            short_cond = (
                norm_slope <= -min_slope
                and m5_slope < 0
                and (not use_vwap or price <= vwap)
            )

            if long_cond:
                return {
                    "action": "buy",
                    "reason": f"Long MTF (Slope={norm_slope:.2f}, 5mSlope={m5_slope:.2f}, RVOL={rvol:.2f})",
                    "atr": atr,
                    "price": price,
                }
            elif short_cond:
                return {
                    "action": "short",
                    "reason": f"Short MTF (Slope={norm_slope:.2f}, 5mSlope={m5_slope:.2f}, RVOL={rvol:.2f})",
                    "atr": atr,
                    "price": price,
                }

        # ---- ポジションありの場合 (手仕舞い判定) ----
        elif current_position == "long":
            if norm_slope < 0 or m5_slope < 0:
                return {
                    "action": "sell",
                    "reason": "MTF Trend Reversed to Negative",
                    "atr": atr,
                    "price": price,
                }
        elif current_position == "short":
            if norm_slope > 0 or m5_slope > 0:
                return {
                    "action": "cover",
                    "reason": "MTF Trend Reversed to Positive",
                    "atr": atr,
                    "price": price,
                }

        return {"action": "hold", "reason": "No condition met", "atr": atr, "price": price}
