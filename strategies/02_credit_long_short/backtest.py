"""02. 信用取引・空売り統合 (Credit Long & Short) 戦略 バックテストスクリプト"""

import sys
from pathlib import Path

STRATEGY_DIR = Path(__file__).resolve().parent
ROOT_DIR = STRATEGY_DIR.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.experiment_long_short_strategy import run_long_short_experiments

if __name__ == "__main__":
    run_long_short_experiments()
