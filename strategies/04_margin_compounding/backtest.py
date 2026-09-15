"""04. 信用レバレッジ ＆ 益出し複利 (Margin Compounding) 戦略 バックテストスクリプト"""

import sys
from pathlib import Path

STRATEGY_DIR = Path(__file__).resolve().parent
ROOT_DIR = STRATEGY_DIR.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.experiment_improved_margin_leverage import run_improved_experiments

if __name__ == "__main__":
    run_improved_experiments()
