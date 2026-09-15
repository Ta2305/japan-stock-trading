"""無料データソース (yfinance ＆ FRED) のデイリーデータ自動更新スクリプト

夕方モード (日本株)・朝モード (グローバルマクロ)・全更新モードを --mode で切り替えて実行します。

使用例:
  .venv/bin/python scripts/daily_data_updater.py --mode all
"""

import sys
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

log_dir = ROOT_DIR / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "daily_data_update.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("daily_updater")


def run_command(cmd: list[str], task_name: str) -> bool:
    logger.info(f"▶ [{task_name}] 開始: {' '.join(cmd)}")
    try:
        res = subprocess.run(cmd, check=True, cwd=str(ROOT_DIR))
        logger.info(f"✔ [{task_name}] 正常完了 (Exit Code 0)")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ [{task_name}] 失敗 (Exit Code {e.returncode})")
        return False
    except Exception as e:
        logger.error(f"❌ [{task_name}] 予期せぬエラー: {e}")
        return False


def update_japan_stocks():
    """日本株 (1分足・日足・分割) の更新 (yfinance)"""
    cmd = [str(ROOT_DIR / ".venv" / "bin" / "python"), "scripts/download_market_data.py"]
    return run_command(cmd, "日本株マーケットデータ更新")


def update_global_macro():
    """グローバルマクロ (SOX・S&P500・米金利・ドル円・VIX) の更新 (yfinance + FRED)"""
    cmd = [str(ROOT_DIR / ".venv" / "bin" / "python"), "scripts/download_global_macro.py"]
    return run_command(cmd, "グローバルマクロ・金利データ更新")


def main():
    parser = argparse.ArgumentParser(description="Daily automated data updater for free data sources.")
    parser.add_argument(
        "--mode",
        choices=["all", "evening", "morning"],
        default="all",
        help="Update mode: 'evening' (Japan stocks), 'morning' (Global macro), or 'all'",
    )
    args = parser.parse_args()

    logger.info("=========================================================================")
    logger.info(f"   DAILY DATA UPDATER START [Mode: {args.mode}]")
    logger.info(f"   Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=========================================================================")

    success_jp = True
    success_macro = True

    if args.mode in ["all", "evening"]:
        success_jp = update_japan_stocks()

    if args.mode in ["all", "morning"]:
        success_macro = update_global_macro()

    logger.info("=========================================================================")
    if success_jp and success_macro:
        logger.info("🎉 すべてのデータ更新が正常に完了いたしました！")
    else:
        logger.warning("⚠️ 一部のデータ更新でエラーが発生しました。ログをご確認ください。")
    logger.info("=========================================================================\n")


if __name__ == "__main__":
    main()
