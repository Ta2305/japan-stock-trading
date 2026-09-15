"""金融庁 EDINET API から大量保有報告書 (5%ルール) ・有価証券報告書 (大株主状況・政策保有株式) を取得するスクリプト

使用方法:
    python scripts/edinet/download_edinet_reports.py --type 5pct --output-dir market_data/edinet/5pct_reports
"""

import time
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
import requests
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("edinet_downloader")

EDINET_API_BASE_URL = "https://disclosure.edinet-fsa.go.jp/api/v2"


def fetch_edinet_document_list(date_str: str, api_key: str = None) -> list[dict]:
    """指定日付 (YYYY-MM-DD) に提出された EDINET 書類一覧を取得する"""
    url = f"{EDINET_API_BASE_URL}/documents.json"
    params = {"date": date_str, "type": 2}
    if api_key:
        params["Subscription-Key"] = api_key

    try:
        res = requests.get(url, params=params, timeout=20)
        res.raise_for_status()
        json_data = res.json()
        return json_data.get("results", [])
    except Exception as e:
        logger.warning(f"Failed to fetch EDINET document list for date {date_str}: {e}")
        return []


def download_edinet_document(doc_id: str, output_path: Path, api_key: str = None) -> bool:
    """指定された書類ID (doc_id) の提出本文 (zip/pdf/csv) をダウンロードする"""
    url = f"{EDINET_API_BASE_URL}/documents/{doc_id}"
    params = {"type": 1}  # 本文取得
    if api_key:
        params["Subscription-Key"] = api_key

    try:
        res = requests.get(url, params=params, timeout=30)
        res.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(res.content)
        return True
    except Exception as e:
        logger.warning(f"Failed to download document {doc_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="EDINET API Downloader for 5% Large Shareholding & Annual Reports (Major Shareholders)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="market_data/edinet",
        help="Output directory path (e.g., market_data/edinet)",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["5pct", "annual", "all"],
        default="5pct",
        help="Target report type: '5pct' (大量保有報告書), 'annual' (有価証券報告書:大株主/政策保有株), 'all'",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of past days to search EDINET documents (Default: 30 days)",
    )

    args = parser.parse_args()
    base_dir = Path(args.output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Financial Services Agency EDINET API Downloader ===")
    logger.info(f"Target Save Directory: {base_dir.resolve()}")
    logger.info(f"Target Filter Type   : {args.type}")
    logger.info(f"Search Past Days     : {args.days} days")

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=args.days)

    doc_records = []
    curr_dt = start_dt

    while curr_dt <= end_dt:
        date_str = curr_dt.strftime("%Y-%m-%d")
        results = fetch_edinet_document_list(date_str)

        for doc in results:
            doc_code = doc.get("docTypeCode")
            sec_code = doc.get("secCode")
            doc_id = doc.get("docID")
            doc_description = doc.get("docDescription", "")
            filer_name = doc.get("filerName", "")

            # 5%ルール (大量保有報告書) -> docTypeCode '030000'
            # 有価証券報告書 (大株主・政策保有株) -> docTypeCode '010000'
            is_5pct = doc_code == "030000" or "大量保有報告書" in doc_description or "変更報告書" in doc_description
            is_annual = doc_code == "010000" or "有価証券報告書" in doc_description

            if (args.type == "5pct" and is_5pct) or (args.type == "annual" and is_annual) or (args.type == "all" and (is_5pct or is_annual)):
                doc_records.append({
                    "submit_date": date_str,
                    "doc_id": doc_id,
                    "sec_code": sec_code,
                    "filer_name": filer_name,
                    "doc_code": doc_code,
                    "doc_description": doc_description,
                    "submit_time": doc.get("submitDateTime"),
                })

        time.sleep(0.3)
        curr_dt += timedelta(days=1)

    if not doc_records:
        logger.info("No matching EDINET reports found in the target period.")
        return

    df_docs = pd.DataFrame(doc_records)
    logger.info(f"Found {len(df_docs)} matching EDINET report submissions!")

    meta_file = base_dir / f"edinet_{args.type}_list.csv"
    df_docs.to_csv(meta_file, index=False, encoding="utf-8-sig")
    logger.info(f"Saved EDINET submission metadata list to: {meta_file.resolve()}")


if __name__ == "__main__":
    main()
