import sys
import re
import urllib.request
import pandas as pd
from pathlib import Path

# JPXの統計情報ページ（このページにExcelファイルへのリンクがある）
JPX_PAGE_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/01.html"
OUTPUT_PATH = Path(__file__).resolve().parent / "tickers.csv"

def get_latest_excel_url() -> str:
    """JPXのページから最新の東証上場銘柄一覧ExcelのダウンロードURLを動的に取得します。"""
    try:
        # User-Agentを設定してページをフェッチ（ブロック防止）
        req = urllib.request.Request(
            JPX_PAGE_URL,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        )
        with urllib.request.urlopen(req) as response:
            html = response.read().decode("utf-8")
            
        # href="..." の中から 'data_j.xls' に終わるリンクを検索
        match = re.search(r'href="([^"]*data_j\.xls)"', html)
        if not match:
            raise ValueError("HTML内に 'data_j.xls' へのリンクが見つかりませんでした。")
            
        link = match.group(1)
        if link.startswith("http"):
            return link
        else:
            return f"https://www.jpx.co.jp{link}"
    except Exception as e:
        print(f"エラー: JPXサイトから最新のExcelダウンロードURLの取得に失敗しました: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    print("=== JPX上場銘柄一覧の取得スクリプト ===")
    
    jpx_excel_url = get_latest_excel_url()
    
    print(f"最新のExcelデータをダウンロード中...")
    print(f"URL: {jpx_excel_url}")
    
    try:
        # pandas.read_excel は xlrd がインストールされていれば直接URLから読み込み可能です
        # ※ファイル形式は .xls（古いExcel形式）のため、xlrdライブラリが必要です
        df = pd.read_excel(jpx_excel_url)
    except Exception as e:
        print(f"エラー: Excelデータのダウンロードまたは読み込みに失敗しました: {e}", file=sys.stderr)
        print("依存ライブラリの xlrd がインストールされているか確認してください。", file=sys.stderr)
        sys.exit(1)
        
    print(f"ダウンロード完了。全データ件数: {len(df)} 件")
    
    # 期待されるカラム名:
    # 「コード」「銘柄名」「市場・商品区分」「33業種区分」など
    required_cols = ["コード", "市場・商品区分"]
    for col in required_cols:
        if col not in df.columns:
            print(f"エラー: Excelのシート内に期待するカラム '{col}' が見つかりません。", file=sys.stderr)
            print(f"取得したカラム一覧: {df.columns.tolist()}", file=sys.stderr)
            sys.exit(1)

    # 1. 国内株式のみを抽出（ETF・ETN、REIT、インフラファンド等を除外する）
    # ※デイトレード用にETFも含めたい場合は、このフィルタリングを調整してください。
    valid_markets = [
        "プライム（内国株式）",
        "スタンダード（内国株式）",
        "グロース（内国株式）",
        "プライム（外国株式）",
        "スタンダード（外国株式）",
        "グロース（外国株式）"
    ]
    
    # 国内株式（プライム、スタンダード、グロース）のみに絞り込む
    print("市場区分でフィルタリングを行っています（ETF/REIT等を除外して国内株式のみに限定します）...")
    filtered_df = df[df["市場・商品区分"].isin(valid_markets)].copy()
    print(f"フィルタリング後件数: {len(filtered_df)} 件")
    
    # 2. 4桁の銘柄コードの整形（数値から文字列へ変換し、末尾に '.T' を付与）
    tickers = []
    for code in filtered_df["コード"]:
        code_str = str(code).strip()
        # 小数点が含まれてしまう場合の対処 (例: 7203.0 -> 7203)
        if "." in code_str:
            code_str = code_str.split(".")[0]
            
        # 4桁のコードであることを保証 (最近の上場銘柄でアルファベットが含まれるケース 130A 等も許容するため、桁数で判定)
        if len(code_str) == 4:
            tickers.append(f"{code_str}.T")
            
    tickers = sorted(list(set(tickers)))
    
    # 3. CSV形式で保存
    output_df = pd.DataFrame({"ticker": tickers})
    output_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    
    print(f"=== 正常終了 ===")
    print(f"すべての日本株（内国・外国株合計 {len(tickers)} 銘柄）のティッカーシンボルを以下に保存しました:")
    print(f"👉 {OUTPUT_PATH.resolve()}")
    print("これで download_market_data.py を実行すれば、全銘柄のダウンロードが開始されます。")
    print("※注意: 全銘柄のダウンロードには時間がかかります。必要に応じて config.py の並列数やディレイを調整してください。")

if __name__ == "__main__":
    main()
