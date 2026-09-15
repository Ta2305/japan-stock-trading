"""ポートフォリオ・ポジション管理モジュール

ポジション状態、エントリー・エグジットの追跡、ATRベースSL/TPの管理および取引記録（Trades）の保持を行います。
"""

from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Trade:
    ticker: str
    position_type: str  # 'long' or 'short'
    entry_time: datetime
    entry_price: float
    shares: int
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    is_closed: bool = False


class Portfolio:
    """ポートフォリオおよび単一/複数銘柄ポジション管理クラス"""

    def __init__(self, initial_cash: float = 1_000_000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position: Optional[Trade] = None
        self.closed_trades: List[Trade] = []

        # 保有中の価格高値/安値（トレイリングストップ用）
        self.high_since_entry: float = 0.0
        self.low_since_entry: float = 0.0
        self.current_sl: Optional[float] = None
        self.current_tp: Optional[float] = None

    def open_position(
        self,
        ticker: str,
        position_type: str,
        entry_time: datetime,
        entry_price: float,
        shares: int,
        sl_price: Optional[float] = None,
        tp_price: Optional[float] = None,
    ) -> Trade:
        """新規ポジションの開始"""
        trade = Trade(
            ticker=ticker,
            position_type=position_type,
            entry_time=entry_time,
            entry_price=entry_price,
            shares=shares,
        )
        self.position = trade
        self.high_since_entry = entry_price
        self.low_since_entry = entry_price
        self.current_sl = sl_price
        self.current_tp = tp_price
        return trade

    def update_price(self, current_price: float):
        """保有中ポジションの最高値/最安値を更新"""
        if self.position is not None:
            self.high_since_entry = max(self.high_since_entry, current_price)
            self.low_since_entry = min(self.low_since_entry, current_price)

    def close_position(
        self, exit_time: datetime, exit_price: float, exit_reason: str
    ) -> Trade:
        """ポジションの決済"""
        if self.position is None:
            raise ValueError("No active position to close")

        trade = self.position
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        trade.is_closed = True

        if trade.position_type == "long":
            trade.pnl = (exit_price - trade.entry_price) * trade.shares
            trade.pnl_pct = ((exit_price / trade.entry_price) - 1.0) * 100.0
        else:  # short
            trade.pnl = (trade.entry_price - exit_price) * trade.shares
            trade.pnl_pct = (1.0 - (exit_price / trade.entry_price)) * 100.0

        self.cash += trade.pnl
        self.closed_trades.append(trade)
        self.position = None
        self.current_sl = None
        self.current_tp = None
        return trade

    @property
    def has_position(self) -> bool:
        return self.position is not None

    @property
    def current_position_type(self) -> Optional[str]:
        return self.position.position_type if self.position else None
