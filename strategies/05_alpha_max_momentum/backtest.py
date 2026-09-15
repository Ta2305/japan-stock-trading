"""05. Alpha-Max 最強モメンタム集中 ＆ 動的信用レバレッジ戦略 バックテストスクリプト"""

import sys
from pathlib import Path

STRATEGY_DIR = Path(__file__).resolve().parent
ROOT_DIR = STRATEGY_DIR.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.experiment_alpha_max_v2_strategy import run_alpha_max_v2_experiments

if __name__ == "__main__":
    run_alpha_max_v2_experiments()
