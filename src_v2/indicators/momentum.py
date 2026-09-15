"""モメンタム系インジケーターモジュール

RSI (Relative Strength Index) と ADX (Average Directional Index) を提供します。
ADXはトレンド強度の判定に使用し、レンジ vs トレンドのレジームスイッチングで重要となります。
"""

import numpy as np
import pandas as pd
from src_v2.indicators.volatility import calculate_true_range


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (RSI) を計算する。"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average Directional Index (ADX) および +DI, -DI を計算する。

    Parameters
    ----------
    df : pd.DataFrame
        'high', 'low', 'close' カラムを含むデータフレーム
    period : int, optional
        計算期間, by default 14

    Returns
    -------
    pd.DataFrame
        'adx', 'plus_di', 'minus_di' カラムを含むデータフレーム
    """
    high = df["high"]
    low = df["low"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = calculate_true_range(df)

    tr_smoothed = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    plus_dm_smoothed = pd.Series(plus_dm, index=df.index).ewm(
        alpha=1.0 / period, min_periods=period, adjust=False
    ).mean()
    minus_dm_smoothed = pd.Series(minus_dm, index=df.index).ewm(
        alpha=1.0 / period, min_periods=period, adjust=False
    ).mean()

    plus_di = 100.0 * (plus_dm_smoothed / tr_smoothed.replace(0, np.nan))
    minus_di = 100.0 * (minus_dm_smoothed / tr_smoothed.replace(0, np.nan))

    dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    res = pd.DataFrame(
        {
            "adx": adx.fillna(0.0),
            "plus_di": plus_di.fillna(0.0),
            "minus_di": minus_di.fillna(0.0),
        },
        index=df.index,
    )
    return res
