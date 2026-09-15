"""移動平均関連インジケーターモジュール

SMA, EMA, および KAMA (Kaufman Adaptive Moving Average) を提供します。
KAMAはボラティリティに応じて平滑化定数を自動調整し、1分足などのノイズが多いデータでダマシを抑えます。
"""

import numpy as np
import pandas as pd


def calculate_sma(series: pd.Series, period: int = 14) -> pd.Series:
    """単純移動平均 (SMA) を計算する。"""
    return series.rolling(window=period, min_periods=period).mean()


def calculate_ema(series: pd.Series, period: int = 14) -> pd.Series:
    """指数移動平均 (EMA) を計算する。"""
    return series.ewm(span=period, adjust=False).mean()


def calculate_kama(
    series: pd.Series, er_period: int = 10, fast_sc: int = 2, slow_sc: int = 30
) -> pd.Series:
    """Kaufman Adaptive Moving Average (KAMA) を計算する。

    Parameters
    ----------
    series : pd.Series
        価格（終値等）の時系列データ
    er_period : int, optional
        Efficiency Ratio (効率比) の計算期間, by default 10
    fast_sc : int, optional
        高速スパン（トレンド用EMA相当）, by default 2
    slow_sc : int, optional
        低速スパン（レンジ用EMA相当）, by default 30

    Returns
    -------
    pd.Series
        KAMA の時系列データ
    """
    change = (series - series.shift(er_period)).abs()
    volatility = (series - series.shift(1)).abs().rolling(window=er_period).sum()

    er = np.where(volatility != 0, change / volatility, 0.0)

    fast_alpha = 2.0 / (fast_sc + 1.0)
    slow_alpha = 2.0 / (slow_sc + 1.0)
    sc = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2

    kama = np.zeros(len(series))
    kama[:] = np.nan

    vals = series.to_numpy()
    first_valid = er_period

    if len(vals) > first_valid:
        kama[first_valid - 1] = vals[first_valid - 1]
        for i in range(first_valid, len(vals)):
            kama[i] = kama[i - 1] + sc[i] * (vals[i] - kama[i - 1])

    return pd.Series(kama, index=series.index, name="kama")
