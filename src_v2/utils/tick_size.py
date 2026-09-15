"""東証呼値（Tick Size）計算ユーティリティ

東京証券取引所の価格帯別最小呼値（1 tick）を算出します。
スリッページや注文価格調整、バリア計算に使用します。
"""


def get_tick_size(price: float) -> float:
    """株価に応じた東証の最小呼値 (1 tick) を返す。

    Parameters
    ----------
    price : float
        現在の株価

    Returns
    -------
    float
        最小呼値 (円)
    """
    if price <= 3000:
        return 1.0
    elif price <= 5000:
        return 5.0
    elif price <= 30000:
        return 10.0
    elif price <= 50000:
        return 50.0
    elif price <= 300000:
        return 100.0
    elif price <= 500000:
        return 500.0
    else:
        return 1000.0
