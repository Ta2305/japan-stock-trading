"""ATR利確・損切り倍率感度分析 (Grid Search)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.analyze_atr_barrier_sensitivity import run_atr_sensitivity_analysis

if __name__ == "__main__":
    run_atr_sensitivity_analysis()
