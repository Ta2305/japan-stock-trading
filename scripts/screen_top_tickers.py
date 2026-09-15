"""東証主要銘柄スクリーニングスクリプト

東証プライム・グロース・スタンダード市場の上場銘柄から「20営業日平均売買代金 >= 10億円/日」かつ「最新株価 >= 500円」の銘柄を抽出し、`tickers.csv` を更新します。
yfinance のアクセス制限（HTTP 429）およびURL制限（HTTP 414）を回避するため、50銘柄ずつの適切なバッチ処理とディレイ挿入を行って安全に取得します。

使用方法:
    python scripts/screen_top_tickers.py --min-trading-value-yen 1000000000 --min-price 500
"""

import sys
import time
import re
import urllib.request
import argparse
from pathlib import Path
import pandas as pd
import yfinance as yf

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

JPX_PAGE_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/01.html"


def get_latest_excel_url() -> str:
    """JPXのページから最新の東証上場銘柄一覧ExcelのダウンロードURLを動的に取得します。"""
    try:
        req = urllib.request.Request(
            JPX_PAGE_URL,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            },
        )
        with urllib.request.urlopen(req) as response:
            html = response.read().decode("utf-8")

        match = re.search(r'href="([^"]*data_j\.xls)"', html)
        if not match:
            raise ValueError("HTML内に 'data_j.xls' へのリンクが見つかりませんでした。")

        link = match.group(1)
        if link.startswith("http"):
            return link
        else:
            return f"https://www.jpx.co.jp{link}"
    except Exception as e:
        print(f"Error fetching JPX URL: {e}", file=sys.stderr)
        return ""


def fetch_tse_stock_list() -> list[str]:
    """JPX公式Excelから東証上場の株式銘柄リストを取得する。"""
    print("Fetching TSE listed stock list from JPX...")
    excel_url = get_latest_excel_url()
    if not excel_url:
        print("Falling back to core tickers...")
        return ["7203.T", "8306.T", "9984.T", "6758.T", "6857.T", "8035.T", "6146.T", "7011.T", "6920.T", "8316.T"]

    try:
        df = pd.read_excel(excel_url)
        valid_markets = [
            "プライム（内国株式）",
            "スタンダード（内国株式）",
            "グロース（内国株式）",
            "プライム（外国株式）",
            "スタンダード（外国株式）",
            "グロース（外国株式）",
        ]
        filtered_df = df[df["市場・商品区分"].isin(valid_markets)].copy()

        tickers = []
        for code in filtered_df["コード"]:
            code_str = str(code).strip().split(".")[0]
            if len(code_str) == 4:
                tickers.append(f"{code_str}.T")

        tickers = sorted(list(set(tickers)))
        print(f"Extracted {len(tickers)} candidate stocks from JPX list.")
        return tickers
    except Exception as e:
        print(f"Error reading JPX excel: {e}")
        return ["7203.T", "8306.T", "9984.T", "6758.T", "6857.T", "8035.T", "6146.T", "7011.T", "6920.T", "8316.T"]


def screen_tickers(
    tickers: list[str],
    min_trading_value_yen: float = 1_000_000_000,
    min_price: float = 500,
    batch_size: int = 50,
    delay_sec: float = 1.5,
) -> pd.DataFrame:
    """指定銘柄群に対し、yfinance 日足一括データからスクリーニングを行う。"""
    passed_stocks = []

    total_batches = (len(tickers) + batch_size - 1) // batch_size
    print(f"\nStarting screening in {total_batches} batches (Batch size: {batch_size}, Delay: {delay_sec}s)...")

    for i in range(0, len(tickers), batch_size):
        batch_tickers = tickers[i : i + batch_size]
        batch_num = i // batch_size + 1

        try:
            # 50銘柄一括で直近1ヶ月間の日足データを取得
            data = yf.download(
                batch_tickers,
                period="1mo",
                interval="1d",
                group_by="ticker",
                progress=False,
                threads=True,
            )

            if data.empty:
                continue

            for tk in batch_tickers:
                try:
                    if len(batch_tickers) == 1:
                        stock_df = data
                    else:
                        if tk not in data.columns.levels[0]:
                            continue
                        stock_df = data[tk]

                    if stock_df.empty or "Close" not in stock_df or "Volume" not in stock_df:
                        continue

                    clean_df = stock_df.dropna(subset=["Close", "Volume"])
                    if len(clean_df) < 5:
                        continue

                    # 直近終値
                    last_price = float(clean_df["Close"].iloc[-1])

                    # 日次売買代金 = 終値 * 出来高
                    daily_trading_val = clean_df["Close"] * clean_df["Volume"]
                    avg_trading_val = float(daily_trading_val.mean())

                    # 条件判定
                    if last_price >= min_price and avg_trading_val >= min_trading_value_yen:
                        passed_stocks.append(
                            {
                                "ticker": tk,
                                "last_price": last_price,
                                "avg_trading_value_yen": avg_trading_val,
                            }
                        )
                except Exception:
                    continue

        except Exception as e:
            print(f"Warning: Batch {batch_num} failed: {e}")

        # レート制限回避ディレイ
        time.sleep(delay_sec)
        if batch_num % 5 == 0 or batch_num == total_batches:
            print(f"Processed batch {batch_num}/{total_batches} | Selected so far: {len(passed_stocks)}")

    res_df = pd.DataFrame(passed_stocks)
    if not res_df.empty:
        res_df = res_df.sort_values(by="avg_trading_value_yen", ascending=False).reset_index(drop=True)
    return res_df


def main():
    parser = argparse.ArgumentParser(description="Screen TSE stocks by trading value and price.")
    parser.add_argument(
        "--min-trading-value-yen",
        type=float,
        default=1_000_000_000,
        help="Minimum 20-day average trading value in JPY (Default: 1,000,000,000)",
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=500,
        help="Minimum stock price in JPY (Default: 500)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="tickers.csv",
        help="Output CSV path (Default: tickers.csv)",
    )

    args = parser.parse_args()

    # 1. 東証上場銘柄リストの取得
    candidate_tickers = fetch_tse_stock_list()

    # 2. スクリーニング実行
    print(f"\nTarget Criteria:")
    print(f" - Min Trading Value : {args.min_trading_value_yen:,.0f} JPY / day")
    print(f" - Min Stock Price   : {args.min_price:,.0f} JPY")

    result_df = screen_tickers(
        candidate_tickers,
        min_trading_value_yen=args.min_trading_value_yen,
        min_price=args.min_price,
        batch_size=50,
        delay_sec=1.5,
    )

    if result_df.empty:
        print("No stocks passed the screening criteria.")
        return

    print("\n" + "=" * 75)
    print(f"          SCREENING COMPLETED: {len(result_df)} STOCKS PASSED")
    print("=" * 75)
    print(result_df.head(20).to_string(index=False))
    print("=" * 75)

    # 3. tickers.csv の更新
    output_path = Path(args.output_csv)
    result_df[["ticker"]].to_csv(output_path, index=False)
    print(f"\nSuccessfully updated {output_path} with {len(result_df)} screened tickers.")


if __name__ == "__main__":
    main()
