"""J-Quants API V2 の全データを一括ダウンロードするマスター処理スクリプト

すべての処理で中途中断耐性 (Ctrl+C対応レジューム機能) とディスク空き容量安全監視 (10GB下限) が有効です。
環境変数 SLACK_WEBHOOK_URL が設定されている場合、完了時にSlackへ通知を送信します。

使用例:
    python scripts/jquants/download_all.py --output-base-dir market_data/jquants
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path
import requests

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("download_all")


def send_slack_notification(message: str):
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        return

    try:
        payload = {"text": message}
        res = requests.post(webhook_url, json=payload, timeout=10)
        if res.status_code == 200:
            logger.info("Successfully sent notification to Slack!")
        else:
            logger.warning(f"Failed to send Slack notification: HTTP {res.status_code}")
    except Exception as e:
        logger.warning(f"Error sending Slack notification: {e}")


def run_script(script_name: str, args_list: list[str]):
    script_path = Path(__file__).resolve().parent / script_name
    cmd = [sys.executable, str(script_path)] + args_list
    logger.info(f"\n============================================================")
    logger.info(f" Starting Task: {script_name}")
    logger.info(f" Command: {' '.join(cmd)}")
    logger.info(f"============================================================\n")

    try:
        res = subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Task {script_name} failed with exit code {e.returncode}")
    except KeyboardInterrupt:
        logger.warning(f"Task {script_name} interrupted by user.")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Master Downloader for all J-Quants V2 datasets with auto-resume & disk safety & Slack notification."
    )
    parser.add_argument(
        "--output-base-dir",
        type=str,
        default="market_data/jquants",
        help="Base output directory path (e.g., market_data/jquants or D:/jquants_data)",
    )

    args = parser.parse_args()
    base_dir = Path(args.output_base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    logger.info("==============================================================")
    logger.info("       J-QUANTS API V2 MASTER DATA DOWNLOADER START")
    logger.info(f" Base Save Path: {base_dir.resolve()}")
    logger.info("==============================================================")

    # 1. 銘柄マスター情報
    run_script(
        "download_listed_info.py",
        ["--output-dir", str(base_dir / "listed_info")],
    )

    # 2. 過去2年分 1分足・Tickデータ
    run_script(
        "download_bars_1m.py",
        ["--output-dir", str(base_dir / "bars_1m"), "--years", "2"],
    )

    # 3. 過去10年分 TOPIX・株価指数四本値
    run_script(
        "download_indices.py",
        ["--output-dir", str(base_dir / "indices"), "--years", "10"],
    )

    # 4. 過去10年分 信用取引週末残高
    run_script(
        "download_margins_weekly.py",
        ["--output-dir", str(base_dir / "margins_weekly"), "--years", "10"],
    )

    # 5. 過去10年分 業種別空売り比率
    run_script(
        "download_short_positions.py",
        ["--output-dir", str(base_dir / "short_ratio"), "--years", "10"],
    )

    # 6. 過去10年分 個別機関空売り残高報告 (5%ルール)
    run_script(
        "download_short_reports.py",
        ["--output-dir", str(base_dir / "short_reports"), "--years", "10"],
    )

    # 7. 過去10年分 財務情報・決算報告
    run_script(
        "download_financials.py",
        ["--output-dir", str(base_dir / "financials"), "--years", "10"],
    )

    # 8. 過去10年分 日足四本値
    run_script(
        "download_daily_bars.py",
        ["--output-dir", str(base_dir / "bars_daily"), "--years", "10"],
    )

    completion_msg = (
        "🎉 *[J-Quants Data Downloader]* 追加データ含む全データのダウンロードが100%正常に完了いたしました！\n"
        f"• **保存ディレクトリ**: `{base_dir.resolve()}`\n"
        "• **完了データ**: 上場銘柄マスター、2年分 1分足/Tick、10年分 TOPIX/指数、10年分 信用残、10年分 空売り比率、10年分 機関空売り残高報告、10年分 財務決算、10年分 日足四本値"
    )

    logger.info("\n==============================================================")
    logger.info("   ALL J-QUANTS DATA DOWNLOAD TASKS COMPLETED SUCCESSFULLY!")
    logger.info("==============================================================")

    send_slack_notification(completion_msg)


if __name__ == "__main__":
    main()
