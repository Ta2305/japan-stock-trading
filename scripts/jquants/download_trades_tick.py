"""J-Quants API V2 約定データ (Raw Tick / 歩み値) 一括取得スクリプト (過去2年分 / レジューム対応)

過去2年分の東証全上場銘柄の約定データ (/equities/trades) を日次で一括取得し、日別 Parquet ファイルとして保存します。
途中で Ctrl+C 等で中断されても、未取得の日付から自動的に再開します。

使用例:
    python scripts/jquants/download_trades_tick.py --output-dir D:/jquants_data/trades_tick --years 2
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.jquants.client import JQuantsClientV2

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("download_trades_tick")


def get_business_days(start_date: datetime, end_date: datetime) -> list[str]:
    curr = start_date
    date_list = []
    while curr <= end_date:
        if curr.weekday() < 5:
            date_list.append(curr.strftime("%Y-%m-%d"))
        curr += timedelta(days=1)
    return date_list


def main():
    parser = argparse.ArgumentParser(
        description="Download 2-year Raw Trade (Tick) data via J-Quants V2 with auto-resume."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="market_data/jquants/trades_tick",
        help="Output directory path (e.g., D:/jquants_data/trades_tick)",
    )
    parser.add_argument(
        "--years",
        type=float,
        default=2.0,
        help="Number of past years to download (Default: 2.0 years)",
    )
    parser.add_argument(
        "--delay-sec",
        type=float,
        default=0.8,
        help="Delay seconds between daily requests (Default: 0.8s)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_file = output_dir / "download_status.json"

    completed_dates = set()
    if status_file.exists():
        try:
            with open(status_file, "r") as f:
                completed_dates = set(json.load(f).get("completed_dates", []))
        except Exception:
            pass

    for p_file in output_dir.glob("*.parquet"):
        completed_dates.add(p_file.stem)

    end_dt = datetime.now() - timedelta(days=1)
    days_back = int(365 * args.years) if args.years > 0 else 7
    start_dt = end_dt - timedelta(days=days_back)
    all_dates = get_business_days(start_dt, end_dt)

    pending_dates = [d for d in all_dates if d not in completed_dates]

    logger.info("=== J-Quants Raw Trade (Tick) Data Downloader (/equities/trades) ===")
    logger.info(f"Target Directory   : {output_dir.resolve()}")
    logger.info(f"Date Range         : {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
    logger.info(f"Total Business Days : {len(all_dates)}")
    logger.info(f"Already Downloaded  : {len(completed_dates)}")
    logger.info(f"Pending to Download : {len(pending_dates)}")

    if not pending_dates:
        logger.info("All target trade dates are already downloaded!")
        return

    client = JQuantsClientV2()

    pbar = tqdm(pending_dates, desc="Downloading Raw Trades (Tick)")
    for date_str in pbar:
        pbar.set_postfix_str(f"Date: {date_str}")
        try:
            trades_data = client.get_data_list(
                "/equities/trades",
                params={"date": date_str.replace("-", "")},
                delay_sec=args.delay_sec,
            )

            if trades_data:
                df = pd.DataFrame(trades_data)
                out_file = output_dir / f"{date_str}.parquet"
                df.to_parquet(out_file, index=False)

            completed_dates.add(date_str)
            with open(status_file, "w") as f:
                json.dump({"completed_dates": sorted(list(completed_dates))}, f, indent=2)

        except KeyboardInterrupt:
            logger.warning("\nDownload interrupted by user (Ctrl+C). Progress saved.")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error downloading trade data for date {date_str}: {e}")
            continue

    logger.info(f"\nCompleted! Total files: {len(completed_dates)} saved in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
