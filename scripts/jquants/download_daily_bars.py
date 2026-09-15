"""J-Quants API V2 日足四本値データ取得スクリプト (過去10年分 / レジューム・ディスク残量安全監視対応)"""

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
logger = logging.getLogger("download_daily_bars")


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
        description="Download 10-year Daily Bar data via J-Quants V2 with auto-resume & disk safety."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="market_data/jquants/bars_daily",
        help="Output directory path (e.g., market_data/jquants/bars_daily)",
    )
    parser.add_argument(
        "--years",
        type=float,
        default=10.0,
        help="Number of past years to download (Default: 10.0 years)",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=10.0,
        help="Minimum required free disk space in GB for safety (Default: 10.0 GB)",
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

    logger.info("=== J-Quants Daily Bars Downloader ===")
    logger.info(f"Target Directory   : {output_dir.resolve()}")
    logger.info(f"Total Business Days : {len(all_dates)}")
    logger.info(f"Pending to Fetch   : {len(pending_dates)}")
    logger.info(f"Disk Safety Limit   : Keep at least {args.min_free_gb:.1f} GB Free")

    if not pending_dates:
        logger.info("All daily bar dates are already downloaded!")
        return

    client = JQuantsClientV2()

    pbar = tqdm(pending_dates, desc="Downloading Daily Bars")
    for date_str in pbar:
        if not check_disk_space(output_dir, min_free_gb=args.min_free_gb):
            logger.critical(f"Stopping daily bars download safely due to low disk space threshold ({args.min_free_gb:.1f} GB).")
            break

        pbar.set_postfix_str(f"Date: {date_str}")
        try:
            daily_data = client.get_data_list(
                "/equities/bars/daily",
                params={"date": date_str.replace("-", "")},
                delay_sec=0.4,
            )

            if daily_data:
                df = pd.DataFrame(daily_data)
                out_file = output_dir / f"{date_str}.parquet"
                df.to_parquet(out_file, index=False)

            completed_dates.add(date_str)
            with open(status_file, "w") as f:
                json.dump({"completed_dates": sorted(list(completed_dates))}, f, indent=2)

        except KeyboardInterrupt:
            logger.warning("\nDownload interrupted by user (Ctrl+C). Progress saved.")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error fetching daily bars for date {date_str}: {e}")
            continue

    logger.info(f"\nTask Finished! Saved daily bar files to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
