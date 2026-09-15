"""Triple Barrier Method ラベリングモジュール

López de Prado (2018) "Advances in Financial Machine Learning" に基づく
Triple Barrier Method ラベリングを提供します。

一次モデル（提案A/A+）のシグナル発生時点に対し、以下の3つのバリアでラベル付けを行います：
1. 上方バリア（利益確定: N * ATR） -> 成功 (1)
2. 下方バリア（損切り: M * ATR）   -> 失敗 (0)
3. 時間バリア（保有期限: K 分）     -> 期限時点の収益の正負で判定 (1 or 0)
"""

from typing import Tuple
import numpy as np
import pandas as pd


def apply_triple_barrier_labels(
    df: pd.DataFrame,
    events: pd.DataFrame,
    pt_sl_ratio: Tuple[float, float] = (2.0, 1.5),
    max_holding_bars: int = 15,
) -> pd.DataFrame:
    """一次モデルが発行したシグナルイベントに対し、Triple Barrierラベルを生成する。

    Parameters
    ----------
    df : pd.DataFrame
        'close', 'high', 'low', 'atr' を含む1分足データ
    events : pd.DataFrame
        'datetime' (または Index), 'side' (+1 for Long, -1 for Short), 'price', 'atr' を持つシグナル発生イベント
    pt_sl_ratio : Tuple[float, float], optional
        (Take Profit ATR倍数, Stop Loss ATR倍数), by default (2.0, 1.5)
    max_holding_bars : int, optional
        時間バリアの最大保有バー数（分）, by default 15

    Returns
    -------
    pd.DataFrame
        イベントごとの 'target' (1: 勝ちトレード, 0: 負けトレード), 'ret' (収益率), 'touch_barrier' (触れたバリア)
    """
    if events.empty or df.empty:
        return pd.DataFrame()

    df_close = df["close"].to_numpy()
    df_high = df["high"].to_numpy()
    df_low = df["low"].to_numpy()
    df_atr = df["atr"].to_numpy()

    dt_to_idx = {dt: i for i, dt in enumerate(df["datetime"])}

    labels = []

    for _, event in events.iterrows():
        event_dt = event["datetime"]
        if event_dt not in dt_to_idx:
            continue

        start_idx = dt_to_idx[event_dt]
        side = event["side"]  # 1 (Long) or -1 (Short)
        entry_price = event.get("price", df_close[start_idx])
        atr = event.get("atr", df_atr[start_idx])

        if np.isnan(atr) or atr <= 0 or entry_price <= 0:
            continue

        pt_mult, sl_mult = pt_sl_ratio
        pt_target = entry_price + (side * pt_mult * atr)
        sl_target = entry_price - (side * sl_mult * atr)

        end_idx = min(start_idx + max_holding_bars, len(df_close) - 1)

        target_label = 0
        touch_barrier = "time"
        ret = 0.0

        for i in range(start_idx + 1, end_idx + 1):
            curr_high = df_high[i]
            curr_low = df_low[i]

            if side == 1:  # Long
                if curr_high >= pt_target:
                    target_label = 1
                    touch_barrier = "pt"
                    ret = (pt_target - entry_price) / entry_price
                    break
                elif curr_low <= sl_target:
                    target_label = 0
                    touch_barrier = "sl"
                    ret = (sl_target - entry_price) / entry_price
                    break
            elif side == -1:  # Short
                if curr_low <= pt_target:
                    target_label = 1
                    touch_barrier = "pt"
                    ret = (entry_price - pt_target) / entry_price
                    break
                elif curr_high >= sl_target:
                    target_label = 0
                    touch_barrier = "sl"
                    ret = (entry_price - sl_target) / entry_price
                    break

        if touch_barrier == "time":
            final_price = df_close[end_idx]
            ret = side * (final_price - entry_price) / entry_price
            target_label = 1 if ret > 0 else 0

        labels.append(
            {
                "event_datetime": event_dt,
                "side": side,
                "entry_price": entry_price,
                "target": target_label,  # Meta-label: 1 = Win, 0 = Loss
                "ret": ret,
                "touch_barrier": touch_barrier,
            }
        )

    return pd.DataFrame(labels)
