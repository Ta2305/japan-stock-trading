"""J-Quants API V2 指数四本値データ (TOPIX, 日経平均等) 取得スクリプト (過去10年分 / レジューム対応)

過去10年分の指数四本値データ (/indices/bars/daily) を取得し、
指定ディレクトリへ保存します。
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

from scripts.jquants.client import JQuantsClientV2, check_disk_space

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("download_indices")


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
        description="Download 10-year Index Daily Bars (TOPIX, Nikkei225, etc.) via J-Quants V2."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="market_data/jquants/indices",
        help="Output directory path (e.g., market_data/jquants/indices)",
    )
    parser.add_argument(
        "--years",
        type=float,
        default=10.0,
        help="Number of past years to download (Default: 10.0 years)",
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
    days_back = int(365 * args.years)
    start_dt = end_dt - timedelta(days=days_back)
    all_dates = get_business_days(start_dt, end_dt)

    pending_dates = [d for d in all_dates if d not in completed_dates]

    logger.info("=== J-Quants Index Daily Bars Downloader (/indices/bars/daily) ===")
    logger.info(f"Target Directory   : {output_dir.resolve()}")
    logger.info(f"Total Business Days : {len(all_dates)}")
    logger.info(f"Pending to Fetch   : {len(pending_dates)}")

    if not pending_dates:
        logger.info("All index bar dates are already downloaded!")
        return

    client = JQuantsClientV2()

    pbar = tqdm(pending_dates, desc="Downloading Index Bars")
    for date_str in pbar:
        if not check_disk_space(output_dir):
            break

        pbar.set_postfix_str(f"Date: {date_str}")
        try:
            idx_data = client.get_data_list(
                "/indices/bars/daily",
                params={"date": date_str.replace("-", "")},
                delay_sec=0.4,
            )

            if idx_data:
                df = pd.DataFrame(idx_data)
                out_file = output_dir / f"{date_str}.parquet"
                df.to_parquet(out_file, index=False)

            completed_dates.add(date_str)
            with open(status_file, "w") as f:
                json.dump({"completed_dates": sorted(list(completed_dates))}, f, indent=2)

        except Exception as e:
            logger.error(f"Error fetching index bars for date {date_str}: {e}")
            continue

    logger.info(f"\nCompleted! Saved index daily bars to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
