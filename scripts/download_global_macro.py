"""グローバルマクロ・マルチアセットデータ自動一括取得スクリプト

Yahoo Finance (yfinance) および FRED API から主要アセットの価格・金利・為替・
ボラティリティ・コモディティデータを一括取得し、market_data/global_macro/ へ Parquet 形式で保存します。
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import yfinance as yf
import requests

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("download_global_macro")

# Yahoo Finance 対象アセット一覧
YFINANCE_ASSETS = {
    # 主要国株価指数 (Index)
    "S&P500": "^GSPC",
    "NASDAQ100": "^NDX",
    "SOX_Semiconductor": "^SOX",
    "KOSPI": "^KS11",
    "HangSeng": "^HSI",
    "DAX": "^GDAXI",
    # 米国主要セクターETF
    "SMH_Semicon_ETF": "SMH",
    "XLF_Financial_ETF": "XLF",
    "XLE_Energy_ETF": "XLE",
    # 為替 (FX Rates)
    "USD_JPY": "JPY=X",
    "EUR_JPY": "EURJPY=X",
    "Dollar_Index": "DX=F",
    # コモディティ (Commodities)
    "WTI_Crude_Oil": "CL=F",
    "Gold": "GC=F",
    "Copper": "HG=F",
    # ボラティリティ (Volatility)
    "VIX_Index": "^VIX",
    "VXN_Index": "^VXN",
}

# FRED (セントルイス連銀) 金利・マクロ経済指標一覧
FRED_SERIES = {
    "US10Y_Treasury_Yield": "DGS10",  # 米10年債利回り (%)
    "US02Y_Treasury_Yield": "DGS2",   # 米2年債利回り (%)
    "US_Yield_Spread_10Y_2Y": "T10Y2Y",  # 10年-2年イールドスプレッド
    "Fed_Funds_Rate": "FEDFUNDS",     # FF政策金利
}


def fetch_fred_series(series_id: str, api_key: str) -> pd.DataFrame:
    """FRED API 経由で経済・金利指標を取得する"""
    url = f"https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": "2015-01-01",
    }
    try:
        res = requests.get(url, params=params, timeout=20)
        res.raise_for_status()
        data = res.json().get("observations", [])
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        df = df[["date", "value"]].copy()
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        df = df.dropna().sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        logger.warning(f"Failed to fetch FRED series {series_id}: {e}")
        return pd.DataFrame()


def download_yfinance_assets(output_dir: Path) -> dict:
    """Yahoo Finance から各主要アセットの日足四本値および直近分足を一括ダウンロードする"""
    logger.info("=== Downloading Yahoo Finance Global Macro Assets ===")
    results = {}

    for name, symbol in YFINANCE_ASSETS.items():
        try:
            ticker = yf.Ticker(symbol)
            df_daily = ticker.history(period="10y", interval="1d")
            if not df_daily.empty:
                df_daily = df_daily.reset_index()
                df_daily.columns = [c.lower() for c in df_daily.columns]
                out_daily = output_dir / "daily" / f"{name}.parquet"
                out_daily.parent.mkdir(parents=True, exist_ok=True)
                df_daily.to_parquet(out_daily, index=False)
                logger.info(f"  -> Saved 10-year daily data for {name}: {len(df_daily)} rows ({out_daily.resolve()})")
                results[name] = len(df_daily)
        except Exception as e:
            logger.error(f"Error fetching {name} ({symbol}): {e}")

    return results


def download_fred_data(output_dir: Path, api_key: str) -> dict:
    """FRED (セントルイス連銀) から米金利・イールドスプレッドを一括取得する"""
    logger.info("=== Downloading FRED Treasury Rates & Yield Spreads ===")
    results = {}

    fred_dir = output_dir / "daily"
    fred_dir.mkdir(parents=True, exist_ok=True)

    for name, series_id in FRED_SERIES.items():
        df = fetch_fred_series(series_id, api_key)
        if not df.empty:
            out_file = fred_dir / f"{name}.parquet"
            df.to_parquet(out_file, index=False)
            logger.info(f"  -> Saved FRED series for {name}: {len(df)} rows ({out_file.resolve()})")
            results[name] = len(df)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Download Global Macro & Multi-Asset Market Data via yfinance and FRED API."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="market_data/global_macro",
        help="Output directory path (Default: market_data/global_macro)",
    )
    parser.add_argument(
        "--fred-api-key",
        type=str,
        default=os.getenv("FRED_API_KEY"),
        help="FRED API Key. If omitted, the FRED_API_KEY environment variable is used.",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("==============================================================")
    logger.info("    GLOBAL MACRO & MULTI-ASSET DATA DOWNLOADER START")
    logger.info(f" Target Directory: {output_dir.resolve()}")
    logger.info("==============================================================")

    yf_res = download_yfinance_assets(output_dir)

    if args.fred_api_key:
        fred_res = download_fred_data(output_dir, args.fred_api_key)
    else:
        logger.warning(
            "FRED API key not provided; skipping Treasury/Fed series. "
            "Set the FRED_API_KEY environment variable or pass --fred-api-key."
        )
        fred_res = {}

    logger.info("\n==============================================================")
    logger.info("   GLOBAL MACRO DATA DOWNLOAD COMPLETED SUCCESSFULLY!")
    logger.info(f" yfinance Assets Downloaded : {len(yf_res)} / {len(YFINANCE_ASSETS)}")
    logger.info(f" FRED Series Downloaded     : {len(fred_res)} / {len(FRED_SERIES)}")
    logger.info("==============================================================")


if __name__ == "__main__":
    main()
