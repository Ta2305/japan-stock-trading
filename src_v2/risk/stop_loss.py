"""ストップロス & テイクプロフィット モジュール

ATR (Average True Range) に基づく動的損切り (Stop Loss) / 利益確定 (Take Profit) / トレイリングストップの算出を行います。
"""

from typing import Tuple, Optional


def calculate_sl_tp(
    entry_price: float,
    position_type: str,
    atr: float,
    sl_atr_multiplier: float = 2.0,
    tp_atr_multiplier: float = 3.0,
) -> Tuple[float, float]:
    """エントリー価格とATRに基づいてストップロス(SL)およびテイクプロフィット(TP)価格を算出する。

    Parameters
    ----------
    entry_price : float
        約定価格
    position_type : str
        'long' (買) または 'short' (売)
    atr : float
        エントリー時点のATR
    sl_atr_multiplier : float, optional
        SL用のATR倍数, by default 2.0
    tp_atr_multiplier : float, optional
        TP用のATR倍数, by default 3.0

    Returns
    -------
    Tuple[float, float]
        (stop_loss_price, take_profit_price)
    """
    if position_type == "long":
        sl = entry_price - (sl_atr_multiplier * atr)
        tp = entry_price + (tp_atr_multiplier * atr)
    elif position_type == "short":
        sl = entry_price + (sl_atr_multiplier * atr)
        tp = entry_price - (tp_atr_multiplier * atr)
    else:
        raise ValueError(f"Invalid position_type: {position_type}")

    return sl, tp


def update_trailing_stop(
    current_price: float,
    high_since_entry: float,
    low_since_entry: float,
    position_type: str,
    atr: float,
    trailing_atr_multiplier: float = 1.5,
    current_sl: Optional[float] = None,
) -> float:
    """トレイリングストップ価格を更新する。

    Parameters
    ----------
    current_price : float
        現在価格
    high_since_entry : float
        保有期間中の最高値
    low_since_entry : float
        保有期間中の最安値
    position_type : str
        'long' または 'short'
    atr : float
        直近のATR
    trailing_atr_multiplier : float, optional
        トレイリング用のATR倍数, by default 1.5
    current_sl : Optional[float], optional
        現在のストップロス価格（下がりすぎを防止）

    Returns
    -------
    float
        更新されたストップロス価格
    """
    if position_type == "long":
        new_sl = high_since_entry - (trailing_atr_multiplier * atr)
        if current_sl is not None:
            return max(current_sl, new_sl)
        return new_sl
    elif position_type == "short":
        new_sl = low_since_entry + (trailing_atr_multiplier * atr)
        if current_sl is not None:
            return min(current_sl, new_sl)
        return new_sl
    else:
        raise ValueError(f"Invalid position_type: {position_type}")
