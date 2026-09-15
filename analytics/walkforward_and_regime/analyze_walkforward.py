"""Walk-Forward ローリング最適化 ＆ レジームシフト分析"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.analyze_regime_shift_and_walkforward import run_walkforward_analysis

if __name__ == "__main__":
    run_walkforward_analysis()
