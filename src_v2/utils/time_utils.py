"""時間帯・セッション判定ユーティリティ

東証の取引セッション（前場・後場）および時間帯別レジーム（寄り付き、昼前、引け際など）の判定を行います。
"""

from datetime import time, datetime


def is_trading_session(dt: datetime) -> bool:
    """指定時刻が東証の取引時間内（前場: 9:00-11:30, 後場: 12:30-15:30）か判定する。"""
    t = dt.time()
    morning = time(9, 0) <= t <= time(11, 30)
    afternoon = time(12, 30) <= t <= time(15, 30)
    return morning or afternoon


def get_time_regime(dt: datetime) -> str:
    """時刻に応じた時間帯レジーム名を返す。

    - "opening": 寄り付き直後 (9:00 - 9:30)
    - "midday_morning": 前場中盤 (9:30 - 11:30)
    - "afternoon_open": 後場寄り付き (12:30 - 13:00)
    - "midday_afternoon": 後場中盤 (13:00 - 15:00)
    - "closing": 引け際 (15:00 - 15:30)
    - "out_of_hours": 時間外
    """
    t = dt.time()
    if time(9, 0) <= t < time(9, 30):
        return "opening"
    elif time(9, 30) <= t <= time(11, 30):
        return "midday_morning"
    elif time(12, 30) <= t < time(13, 0):
        return "afternoon_open"
    elif time(13, 0) <= t < time(15, 0):
        return "midday_afternoon"
    elif time(15, 0) <= t <= time(15, 30):
        return "closing"
    else:
        return "out_of_hours"
