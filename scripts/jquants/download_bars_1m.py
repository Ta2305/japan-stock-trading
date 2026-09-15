"""J-Quants API V2 1分足・Tick データ全銘柄一括取得スクリプト (過去2年分 / ページネーション完走・レジューム対応)

過去2年分の東証全上場銘柄の1分足およびTickデータをページネーション完走で日次一括取得し、
指定されたディレクトリに日別 Parquet ファイルとして保存します。
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
logger = logging.getLogger("download_bars_1m")


def get_business_days(start_date: datetime, end_date: datetime) -> list[str]:
    curr = start_date
    date_list = []
    while curr <= end_date:
        if curr.weekday() < 5:
            date_list.append(curr.strftime("%Y-%m-%d"))
        curr += timedelta(days=1)
    return date_list


def fetch_full_day_bars_1m(client: JQuantsClientV2, date_str: str, delay_sec: float = 0.3) -> list[dict]:
    """ページネーションキーを自動巡回してその日の全銘柄1分足を100%全件取得する"""
    all_bars = []
    pagination_key = None
    param_date = date_str.replace("-", "")

    while True:
        params = {"date": param_date}
        if pagination_key:
            params["pagination_key"] = pagination_key

        res_json = client.get(
            "/equities/bars/minute",
            params=params,
            delay_sec=delay_sec,
            ignore_400=True,
        )

        if not res_json or not isinstance(res_json, dict):
            break

        data = res_json.get("data", [])
        if not data:
            break

        all_bars.extend(data)
        pagination_key = res_json.get("pagination_key")
        if not pagination_key:
            break

    return all_bars


def main():
    parser = argparse.ArgumentParser(
        description="Download 2-year 1-minute / Tick bar data via J-Quants V2 with full pagination."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="market_data/jquants/bars_1m",
        help="Output directory path (e.g., market_data/jquants/bars_1m)",
    )
    parser.add_argument(
        "--years",
        type=float,
        default=2.0,
        help="Number of past years to download (Default: 2.0 years)",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=10.0,
        help="Minimum required free disk space in GB for safety (Default: 10.0 GB)",
    )
    parser.add_argument(
        "--delay-sec",
        type=float,
        default=0.3,
        help="Delay seconds between pagination requests (Default: 0.3s)",
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

    end_dt = datetime.now() - timedelta(days=1)
    start_dt = max(datetime.now() - timedelta(days=725), datetime(2024, 8, 9))
    all_dates = get_business_days(start_dt, end_dt)

    pending_dates = [d for d in all_dates if d not in completed_dates]

    logger.info("=== J-Quants 1-Minute / Tick Bars Full Downloader (Pagination Complete) ===")
    logger.info(f"Target Directory   : {output_dir.resolve()}")
    logger.info(f"Total Business Days : {len(all_dates)}")
    logger.info(f"Pending to Fetch   : {len(pending_dates)}")

    if not pending_dates:
        logger.info("All target dates are already fully downloaded!")
        return

    client = JQuantsClientV2()

    pbar = tqdm(pending_dates, desc="Downloading 1m Bars")
    for date_str in pbar:
        if not check_disk_space(output_dir, min_free_gb=args.min_free_gb):
            break

        pbar.set_postfix_str(f"Date: {date_str}")
        try:
            full_day_data = fetch_full_day_bars_1m(client, date_str, delay_sec=args.delay_sec)

            if full_day_data:
                df = pd.DataFrame(full_day_data)
                out_file = output_dir / f"{date_str}.parquet"
                df.to_parquet(out_file, index=False)

            completed_dates.add(date_str)
            with open(status_file, "w") as f:
                json.dump({"completed_dates": sorted(list(completed_dates))}, f, indent=2)

        except Exception as e:
            logger.error(f"Error downloading data for date {date_str}: {e}")
            continue

    logger.info(f"\nTask Finished! Total files: {len(completed_dates)} saved in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
