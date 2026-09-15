"""キオクシア (285A) 急騰トレンド・プルバック動態分析"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.analyze_kioxia_dynamics import analyze_kioxia_strategies

if __name__ == "__main__":
    analyze_kioxia_strategies()
