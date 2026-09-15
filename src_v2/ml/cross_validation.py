"""Purged K-Fold Cross Validation モジュール

López de Prado (2018) "Advances in Financial Machine Learning" に基づく
Purging (過度な重なり排除) & Embargo (エンバーゴ) 付きの時系列クロスバリデーションです。
"""

from typing import Generator, Tuple, Optional
import numpy as np
import pandas as pd


class PurgedGroupTimeSeriesSplit:
    """Purged and Embargoed Time Series Splitter"""

    def __init__(self, n_splits: int = 5, pct_embargo: float = 0.01, purge_window: int = 15):
        self.n_splits = n_splits
        self.pct_embargo = pct_embargo
        self.purge_window = purge_window

    def split(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None, groups: Optional[pd.Series] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """学習・テストデータのインデックスペアを生成する。"""
        n_samples = len(X)
        indices = np.arange(n_samples)
        embargo_size = int(n_samples * self.pct_embargo)
        test_size = n_samples // self.n_splits

        for i in range(self.n_splits):
            test_start = i * test_size
            test_end = (i + 1) * test_size if i < self.n_splits - 1 else n_samples
            test_idx = indices[test_start:test_end]

            # Purging & Embargo 適用
            train_mask = np.ones(n_samples, dtype=bool)

            # 1. テスト区間の前後のパージング (Purging)
            purge_start = max(0, test_start - self.purge_window)
            purge_end = min(n_samples, test_end + self.purge_window + embargo_size)

            train_mask[purge_start:purge_end] = False

            train_idx = indices[train_mask]
            yield train_idx, test_idx
