"""J-Quants API V2 上場銘柄マスター情報取得スクリプト"""

import sys
import argparse
import logging
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.jquants.client import JQuantsClientV2

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("download_listed_info")


def main():
    parser = argparse.ArgumentParser(
        description="Download TSE Listed Companies Master Info via J-Quants API V2."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="market_data/jquants/listed_info",
        help="Output directory path (e.g., D:/jquants_data/listed_info)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing J-Quants V2 API Client...")
    client = JQuantsClientV2()

    logger.info("Fetching TSE Listed Equities Master list...")
    data = client.get_data_list("/equities/master")

    if not data:
        logger.error("No listed equities info received from API.")
        return

    df = pd.DataFrame(data)
    logger.info(f"Successfully retrieved {len(df)} listed company records.")

    csv_path = output_dir / "listed_info.csv"
    parquet_path = output_dir / "listed_info.parquet"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_parquet(parquet_path, index=False)

    logger.info(f"Saved master info to:")
    logger.info(f" - CSV: {csv_path.resolve()}")
    logger.info(f" - Parquet: {parquet_path.resolve()}")


if __name__ == "__main__":
    main()
