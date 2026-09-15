"""J-Quants API V2 業種別空売り比率 (/markets/short-ratio) 取得スクリプト (過去10年分 / レジューム対応)

使用例:
    python scripts/jquants/download_short_positions.py --output-dir D:/jquants_data/short_ratio --years 10
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
logger = logging.getLogger("download_short_positions")


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
        description="Download 10-year Sector Short Ratio data via J-Quants V2."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="market_data/jquants/short_ratio",
        help="Output directory path (e.g., D:/jquants_data/short_ratio)",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=10,
        help="Number of past years to download (Default: 10 years)",
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
    start_dt = end_dt - timedelta(days=365 * args.years if args.years > 0 else 7)
    all_dates = get_business_days(start_dt, end_dt)

    pending_dates = [d for d in all_dates if d not in completed_dates]

    logger.info("=== J-Quants Sector Short Ratio Downloader (/markets/short-ratio) ===")
    logger.info(f"Target Directory   : {output_dir.resolve()}")
    logger.info(f"Total Business Days : {len(all_dates)}")
    logger.info(f"Pending to Fetch   : {len(pending_dates)}")

    if not pending_dates:
        logger.info("All short ratio dates are already downloaded!")
        return

    client = JQuantsClientV2()

    pbar = tqdm(pending_dates, desc="Downloading Short Ratio")
    for date_str in pbar:
        pbar.set_postfix_str(f"Date: {date_str}")
        try:
            param_date = date_str.replace("-", "")

            # 業種別空売り比率
            rat_data = client.get_data_list(
                "/markets/short-ratio",
                params={"date": param_date},
                delay_sec=0.4,
            )

            if rat_data:
                df = pd.DataFrame(rat_data)
                out_file = output_dir / f"{date_str}.parquet"
                df.to_parquet(out_file, index=False)

            completed_dates.add(date_str)
            with open(status_file, "w") as f:
                json.dump({"completed_dates": sorted(list(completed_dates))}, f, indent=2)

        except KeyboardInterrupt:
            logger.warning("\nDownload interrupted by user (Ctrl+C). Progress saved.")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error fetching short ratio data for date {date_str}: {e}")
            continue

    logger.info(f"\nCompleted! Downloaded short ratio files to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
