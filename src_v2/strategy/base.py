"""戦略基底クラスモジュール

すべての売買戦略の基底抽象クラス (BaseStrategy) を提供します。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd


class BaseStrategy(ABC):
    """トレーディング戦略の抽象基底クラス"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """データフレームに必要なテクニカル指標を計算・付与する。"""
        pass

    @abstractmethod
    def generate_signal_at(
        self,
        df: pd.DataFrame,
        idx: int,
        current_position: Optional[str] = None,
    ) -> Dict[str, Any]:
        """指定インデックス時点での売買シグナルを生成する。

        Parameters
        ----------
        df : pd.DataFrame
            テクニカル指標算出済みのデータフレーム
        idx : int
            判定対象の行インデックス
        current_position : Optional[str], optional
            現在のポジション ('long', 'short', None)

        Returns
        -------
        Dict[str, Any]
            シグナル辞書例:
            {
                'action': 'buy' | 'sell' | 'short' | 'cover' | 'hold',
                'reason': str,
                'atr': float,
                'price': float
            }
        """
        pass
