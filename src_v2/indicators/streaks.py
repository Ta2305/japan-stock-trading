"""連続足・連続移動平均インジケーターモジュール

ローソク足の連続上昇/下落数、移動平均線の連続上昇/下落数のカウントを提供します。
"""

import numpy as np
import pandas as pd


def compute_consecutive_streaks(series: pd.Series) -> pd.Series:
    """時系列データの連続変化数（連続上昇: >0, 連続下落: <0）を計算する。

    Parameters
    ----------
    series : pd.Series
        価格や移動平均線などの時系列データ

    Returns
    -------
    pd.Series
        連続変化数（例: 3連続上昇なら +3, 2連続下落なら -2）
    """
    diff = series.diff()
    direction = np.sign(diff).fillna(0).astype(int)

    streaks = np.zeros(len(series), dtype=int)
    curr_streak = 0

    dir_vals = direction.to_numpy()

    for i in range(1, len(dir_vals)):
        d = dir_vals[i]
        if d == 0:
            curr_streak = 0
        elif d > 0:
            if curr_streak > 0:
                curr_streak += 1
            else:
                curr_streak = 1
        else:
            if curr_streak < 0:
                curr_streak -= 1
            else:
                curr_streak = -1
        streaks[i] = curr_streak

    return pd.Series(streaks, index=series.index, name="streaks")
