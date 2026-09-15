"""Meta-labeling 特徴量構築パイプライン

1分足株価データおよび一次モデルシグナルから、機械学習二次モデルに投入する定常化特徴量を一元生成します。
"""

import numpy as np
import pandas as pd

from src_v2.indicators import (
    calculate_atr,
    calculate_kama,
    calculate_rsi,
    calculate_adx,
    calculate_vwap,
    calculate_rvol,
    compute_consecutive_streaks,
    calculate_ema,
)


FEATURE_COLUMNS = [
    "normalized_slope",
    "vwap_deviation",
    "rsi_14",
    "adx_14",
    "rvol_20",
    "ema_5m_slope",
    "kama_streak",
    "volatility_ratio",
    "time_sin",
    "time_cos",
    "is_opening",
    "is_afternoon",
]


def build_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """データフレームからML用特徴量行列を算出・付与する。

    Parameters
    ----------
    df : pd.DataFrame
        'datetime', 'open', 'high', 'low', 'close', 'volume' を持つ1分足データ

    Returns
    -------
    pd.DataFrame
        特徴量が付与されたデータフレーム
    """
    df = df.copy()

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        dt_index = df["datetime"]
    else:
        dt_index = pd.to_datetime(df.index)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"]

    # 1. 基本インジケーター
    atr = calculate_atr(df, period=14)
    kama = calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30)
    kama_diff = kama.diff()

    df["atr"] = atr
    df["kama"] = kama
    df["normalized_slope"] = kama_diff / atr.replace(0, np.nan)

    vwap = calculate_vwap(df)
    df["vwap"] = vwap
    df["vwap_deviation"] = (close - vwap) / vwap.replace(0, np.nan)

    df["rsi_14"] = calculate_rsi(close, period=14)

    adx_df = calculate_adx(df, period=14)
    df["adx_14"] = adx_df["adx"]

    df["rvol_20"] = calculate_rvol(df, window=20)
    df["kama_streak"] = compute_consecutive_streaks(kama)
    df["volatility_ratio"] = atr / close.replace(0, np.nan)

    # 2. 5分足マルチタイムフレーム特徴量
    if "datetime" in df.columns:
        df.set_index("datetime", drop=False, inplace=True)

    df_5m = (
        df.resample("5min")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )
    df_5m["ema_5m"] = calculate_ema(df_5m["close"], period=20)
    df_5m["ema_5m_slope"] = df_5m["ema_5m"].diff()

    df["ema_5m_slope"] = df_5m["ema_5m_slope"].reindex(df.index, method="ffill")

    if "datetime" in df.columns and df.index.name == "datetime":
        df.reset_index(drop=True, inplace=True)

    # 3. 時間帯周期エンコーディング (Time-of-day features)
    minutes = dt_index.dt.hour * 60 + dt_index.dt.minute
    total_minutes = 24 * 60
    df["time_sin"] = np.sin(2 * np.pi * minutes / total_minutes)
    df["time_cos"] = np.cos(2 * np.pi * minutes / total_minutes)

    # フラグ特徴量
    hours = dt_index.dt.hour
    df["is_opening"] = ((hours == 9) & (dt_index.dt.minute <= 30)).astype(int)
    df["is_afternoon"] = (hours >= 12).astype(int)

    # 欠損値補完
    df.fillna(0.0, inplace=True)
    return df
