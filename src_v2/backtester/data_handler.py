"""データ読み込み・前処理モジュール

保存済みの1分足Parquetデータのロード、株式分割調整、指標計算の準備を担当します。
"""

from pathlib import Path
from typing import Optional, List
import pandas as pd


def load_ticker_data(
    ticker: str,
    data_dir: str = "market_data",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """指定銘柄のParquetデータを全結合してロードする。

    Parameters
    ----------
    ticker : str
        銘柄コード (例: "7203", "7203.T")
    data_dir : str, optional
        データディレクトリ, by default "market_data"
    start_date : Optional[str], optional
        開始日 ("YYYY-MM-DD")
    end_date : Optional[str], optional
        終了日 ("YYYY-MM-DD")

    Returns
    -------
    pd.DataFrame
        時系列ソート済みの1分足データフレーム
    """
    clean_ticker = ticker.split(".")[0]
    dotted_ticker = f"{clean_ticker}.T"
    base_path = Path(data_dir)

    # 直近よくあるパス構造
    candidate_dirs = [
        base_path / "jp" / "1m" / dotted_ticker,
        base_path / "jp" / "1m" / clean_ticker,
        base_path / dotted_ticker,
        base_path / clean_ticker,
    ]

    ticker_dir = None
    for cdir in candidate_dirs:
        if cdir.exists() and any(cdir.glob("*.parquet")):
            ticker_dir = cdir
            break

    if ticker_dir is None:
        # ディレクトリ探索 (再帰検索)
        for t_name in [dotted_ticker, clean_ticker]:
            matches = list(base_path.rglob(t_name))
            for m in matches:
                if m.is_dir() and any(m.glob("*.parquet")):
                    ticker_dir = m
                    break
            if ticker_dir:
                break

    if ticker_dir is None:
        raise FileNotFoundError(f"No parquet data directory found for ticker: {ticker} in {data_dir}")

    parquet_files = sorted(ticker_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {ticker_dir}")

    df_list = []
    for pfile in parquet_files:
        date_str = pfile.stem  # YYYY-MM-DD
        if start_date and date_str < start_date:
            continue
        if end_date and date_str > end_date:
            continue

        try:
            df = pd.read_parquet(pfile)
            if not df.empty and "datetime" in df.columns:
                df_list.append(df)
        except Exception as e:
            print(f"Warning: Failed to read {pfile}: {e}")

    if not df_list:
        return pd.DataFrame()

    full_df = pd.concat(df_list, axis=0).reset_index(drop=True)

    # datetime変換とソート
    full_df["datetime"] = pd.to_datetime(full_df["datetime"])
    full_df = full_df.sort_values("datetime").reset_index(drop=True)

    # 株式分割調整 (splits.csv が存在する場合)
    splits_file = ticker_dir / "splits.csv"
    if splits_file.exists():
        try:
            splits_df = pd.read_csv(splits_file)
            if not splits_df.empty:
                full_df = adjust_splits(full_df, splits_df)
        except Exception as e:
            print(f"Warning: Failed to apply splits for {clean_ticker}: {e}")

    return full_df


def get_available_tickers(data_dir: str = "market_data") -> List[str]:
    """保存済みデータが存在する銘柄コード一覧を取得する。"""
    base_path = Path(data_dir)
    tickers = set()

    for pfile in base_path.rglob("*.parquet"):
        ticker_name = pfile.parent.name
        if ticker_name not in ["1m", "jp", "market_data"]:
            tickers.add(ticker_name)

    return sorted(list(tickers))


def adjust_splits(df: pd.DataFrame, splits_df: pd.DataFrame) -> pd.DataFrame:
    """株式分割履歴に基づいて過去の株価・出来高を調整する。"""
    if splits_df.empty:
        return df

    df = df.copy()
    date_col = "Date" if "Date" in splits_df.columns else ("date" if "date" in splits_df.columns else splits_df.columns[0])
    split_col = "Stock Splits" if "Stock Splits" in splits_df.columns else splits_df.columns[1]

    splits_df[date_col] = pd.to_datetime(splits_df[date_col]).dt.tz_localize(None)

    for _, row in splits_df.iterrows():
        split_date = row[date_col]
        try:
            factor = float(row[split_col])
        except (ValueError, TypeError):
            continue

        if factor > 0 and factor != 1.0:
            mask = df["datetime"].dt.tz_localize(None) < split_date
            df.loc[mask, ["open", "high", "low", "close"]] /= factor
            df.loc[mask, "volume"] *= factor

    return df
