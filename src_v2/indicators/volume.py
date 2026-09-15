"""出来高系インジケーターモジュール

日次・セッションリセット型 VWAP (Volume Weighted Average Price) および Relative Volume (RVOL) の算出を提供します。
"""

import numpy as np
import pandas as pd


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """日次（セッションごと）リセット型 VWAP を計算する。

    Parameters
    ----------
    df : pd.DataFrame
        'datetime' (またはDatetimeIndex), 'high', 'low', 'close', 'volume' を含むデータフレーム

    Returns
    -------
    pd.Series
        VWAP の時系列データ
    """
    if "datetime" in df.columns:
        dt_series = pd.to_datetime(df["datetime"])
    elif isinstance(df.index, pd.DatetimeIndex):
        dt_series = df.index.to_series()
    else:
        typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
        pv = typical_price * df["volume"]
        cum_pv = pv.cumsum()
        cum_vol = df["volume"].cumsum()
        return cum_pv / cum_vol.replace(0, np.nan)

    date_group = dt_series.dt.date

    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical_price * df["volume"]

    cum_pv = pv.groupby(date_group).cumsum()
    cum_vol = df["volume"].groupby(date_group).cumsum()

    vwap = cum_pv / cum_vol.replace(0, np.nan)
    return vwap.rename("vwap")


def calculate_rvol(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Relative Volume (RVOL: 相対出来高比率) を計算する。

    RVOL = 現在の出来高 / 過去N期間の移動平均出来高

    Parameters
    ----------
    df : pd.DataFrame
        'volume' カラムを含むデータフレーム
    window : int, optional
        ローリング移動平均の期間, by default 20

    Returns
    -------
    pd.Series
        RVOL の時系列データ
    """
    vol = df["volume"]
    mean_vol = vol.rolling(window=window, min_periods=5).mean()
    rvol = vol / mean_vol.replace(0, np.nan)
    return rvol.fillna(1.0).rename("rvol")
