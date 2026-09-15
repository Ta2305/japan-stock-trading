"""バックテストレポート・メトリクス算出モジュール

Sharpe Ratio, Sortino Ratio, Profit Factor, Max Drawdown などの評価指標の計算と結果レポートを出力します。
"""

from typing import List, Dict, Any
import numpy as np
import pandas as pd
from src_v2.backtester.portfolio import Trade


def calculate_performance_metrics(
    trades: List[Trade], initial_cash: float = 1_000_000.0
) -> Dict[str, Any]:
    """トレード履歴から詳細なパフォーマンス指標を算出する。

    Parameters
    ----------
    trades : List[Trade]
        決済済みトレードのリスト
    initial_cash : float, optional
        初期資金, by default 1_000_000.0

    Returns
    -------
    Dict[str, Any]
        パフォーマンス指標の辞書
    """
    total_trades = len(trades)
    if total_trades == 0:
        return {
            "total_trades": 0,
            "win_rate_pct": 0.0,
            "total_pnl_yen": 0.0,
            "total_return_pct": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_yen": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
        }

    pnls = [t.pnl for t in trades]
    pnls_array = np.array(pnls)

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    win_count = len(wins)
    loss_count = len(losses)
    win_rate = (win_count / total_trades) * 100.0

    total_gross_profit = sum(wins) if wins else 0.0
    total_gross_loss = abs(sum(losses)) if losses else 0.0

    profit_factor = (
        (total_gross_profit / total_gross_loss) if total_gross_loss > 0 else np.nan
    )

    total_pnl = sum(pnls)
    total_return_pct = (total_pnl / initial_cash) * 100.0

    # 資産曲線の作成とドローダウン計算
    equity_curve = [initial_cash]
    curr = initial_cash
    for pnl in pnls:
        curr += pnl
        equity_curve.append(curr)

    eq_series = pd.Series(equity_curve)
    peak = eq_series.cummax()
    drawdown = eq_series - peak
    drawdown_pct = (drawdown / peak) * 100.0

    max_dd_yen = abs(drawdown.min())
    max_dd_pct = abs(drawdown_pct.min())

    # リターン配列から Sharpe / Sortino 計算
    returns = eq_series.pct_change().dropna()
    if len(returns) > 1 and returns.std() > 0:
        # 年率換算 (1分足ベースを簡易的に日次リターン換算)
        mean_ret = returns.mean()
        std_ret = returns.std()
        sharpe = (mean_ret / std_ret) * np.sqrt(252 * 300)  # 約75600分/年

        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino = (
            (mean_ret / downside_std) * np.sqrt(252 * 300)
            if downside_std > 0
            else np.nan
        )
    else:
        sharpe = 0.0
        sortino = 0.0

    return {
        "total_trades": total_trades,
        "win_count": win_count,
        "loss_count": loss_count,
        "win_rate_pct": win_rate,
        "total_pnl_yen": total_pnl,
        "total_return_pct": total_return_pct,
        "profit_factor": profit_factor,
        "max_drawdown_yen": max_dd_yen,
        "max_drawdown_pct": max_dd_pct,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
    }


def print_backtest_report(ticker: str, metrics: Dict[str, Any]):
    """レポートを整形して標準出力に出力する。"""
    print("=" * 60)
    print(f"       BACKTEST REPORT (Proposal A) : {ticker}")
    print("=" * 60)
    print(f" Total Trades          : {metrics['total_trades']}")
    print(
        f" Win / Loss Count      : {metrics.get('win_count', 0)} / {metrics.get('loss_count', 0)}"
    )
    print(f" Win Rate              : {metrics['win_rate_pct']:.2f}%")
    print(f" Total PnL (Yen)       : {metrics['total_pnl_yen']:,.0f} JPY")
    print(f" Total Return          : {metrics['total_return_pct']:.2f}%")
    print(f" Profit Factor         : {metrics['profit_factor']:.2f}")
    print(
        f" Max Drawdown          : -{metrics['max_drawdown_yen']:,.0f} JPY (-{metrics['max_drawdown_pct']:.2f}%)"
    )
    print(f" Sharpe Ratio          : {metrics['sharpe_ratio']:.2f}")
    print(f" Sortino Ratio         : {metrics['sortino_ratio']:.2f}")
    print("=" * 60)
