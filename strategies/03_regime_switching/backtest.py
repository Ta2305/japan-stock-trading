"""03. マクロ連動・レジームスイッチング戦略 バックテストスクリプト"""

import sys
from pathlib import Path

STRATEGY_DIR = Path(__file__).resolve().parent
ROOT_DIR = STRATEGY_DIR.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.experiment_regime_switching_strategy import run_regime_switching_experiments

if __name__ == "__main__":
    run_regime_switching_experiments()
