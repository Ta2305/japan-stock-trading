"""ポジションサイジング モジュール

Fixed Fractional 法（口座資金の一定割合をリスクに晒す手法）に基づき、
損切り幅と資金に応じた最適取引株数（100株単位）を算出します。
"""

import math


def calculate_position_size(
    account_cash: float,
    entry_price: float,
    stop_loss_price: float,
    risk_per_trade_pct: float = 1.0,
    unit_size: int = 100,
    max_portfolio_pct: float = 50.0,
) -> int:
    """許容リスク額とストップロス幅から取引株数（単元数）を算出する。

    Parameters
    ----------
    account_cash : float
        現在の口座利用可能資金 (円)
    entry_price : float
        想定エントリー価格
    stop_loss_price : float
        想定ストップロス価格
    risk_per_trade_pct : float, optional
        1トレードあたりの許容損失割合 (%), by default 1.0
    unit_size : int, optional
        最小売買単位（日本株は原則100株）, by default 100
    max_portfolio_pct : float, optional
        1銘柄に投入できる最大資金割合 (%), by default 50.0

    Returns
    -------
    int
        発注株数 (単元整倍数, 例: 100, 200, 300...)
    """
    if account_cash <= 0 or entry_price <= 0:
        return 0

    risk_per_share = abs(entry_price - stop_loss_price)
    if risk_per_share <= 0:
        return 0

    max_risk_amount = account_cash * (risk_per_trade_pct / 100.0)
    shares_by_risk = max_risk_amount / risk_per_share

    max_position_cash = account_cash * (max_portfolio_pct / 100.0)
    shares_by_cash = max_position_cash / entry_price

    target_shares = min(shares_by_risk, shares_by_cash)
    units = math.floor(target_shares / unit_size)
    return max(0, units * unit_size)
