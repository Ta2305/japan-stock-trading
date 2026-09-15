"""年別損益内訳 ＆ 日経平均インデックス比較分析"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.analyze_benchmark_and_annual_breakdown import run_benchmark_and_annual_analysis

if __name__ == "__main__":
    run_benchmark_and_annual_analysis()
