"""ボラティリティ関連インジケーターモジュール

True Range および Average True Range (ATR) の算出関数を提供します。
1分足等での動的なストップロス・テイクプロフィット計算および閾値の正規化に使用します。
"""

import numpy as np
import pandas as pd


def calculate_true_range(df: pd.DataFrame) -> pd.Series:
    """True Range (TR) を計算する。

    TR = max(High - Low, |High - Prior Close|, |Low - Prior Close|)

    Parameters
    ----------
    df : pd.DataFrame
        'high', 'low', 'close' カラムを含むデータフレーム

    Returns
    -------
    pd.Series
        True Range の時系列データ
    """
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (ATR) を計算する（Wilderの指数移動平均）。

    Parameters
    ----------
    df : pd.DataFrame
        'high', 'low', 'close' カラムを含むデータフレーム
    period : int, optional
        計算期間, by default 14

    Returns
    -------
    pd.Series
        ATR の時系列データ
    """
    tr = calculate_true_range(df)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
