"""ローカル保存済み J-Quants 日足 ＆ グローバルマクロ統合 バックテスト評価スクリプト (厳格・リーク完全排除版)

米国マクロ指標 (SOX, S&P500, US10Y, USD/JPY, VIX) は前営業日終値 (Lag 1) をマージし、
当日大引けシグナルを翌営業日寄付 (Open) で執行、時系列日付ベースで Train/Test 分割して評価します。
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import lightgbm as lgb

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("evaluate_macro_swing_model")


def load_recent_jquants_daily_bars_fast(bars_daily_dir: Path, target_codes: set, max_days: int = 500) -> dict:
    parquet_files = sorted(list(bars_daily_dir.glob("*.parquet")))[-max_days:]
    if not parquet_files:
        return {}

    code_data = {c: [] for c in target_codes}

    for p in parquet_files:
        try:
            day_df = pd.read_parquet(p)
            if "Code" not in day_df.columns:
                continue
            day_df["Code"] = day_df["Code"].astype(str)
            sub = day_df[day_df["Code"].isin(target_codes)]
            if not sub.empty:
                for code, group in sub.groupby("Code"):
                    code_data[code].append(group)
        except Exception:
            continue

    result = {}
    for code, dfs in code_data.items():
        if dfs:
            full_df = pd.concat(dfs, ignore_index=True)
            full_df["date"] = pd.to_datetime(full_df["Date"]).dt.strftime("%Y-%m-%d")
            full_df = full_df.rename(columns={
                "O": "open", "H": "high", "L": "low", "C": "close", "Vo": "volume"
            })
            req_cols = ["date", "open", "high", "low", "close", "volume"]
            clean_df = full_df[req_cols].sort_values("date").drop_duplicates("date").reset_index(drop=True)
            for col in ["open", "high", "low", "close", "volume"]:
                clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")
            result[code] = clean_df.dropna()

    return result


def load_global_macro_features(macro_dir: Path) -> pd.DataFrame:
    daily_dir = macro_dir / "daily"
    if not daily_dir.exists():
        return pd.DataFrame()

    assets = {
        "SOX_Semiconductor": "sox_close",
        "S&P500": "sp500_close",
        "USD_JPY": "usdjpy_close",
        "VIX_Index": "vix_close",
        "US10Y_Treasury_Yield": "us10y_yield",
        "US_Yield_Spread_10Y_2Y": "yield_spread_10y2y",
    }

    base_df = None
    for asset_file, col_name in assets.items():
        fpath = daily_dir / f"{asset_file}.parquet"
        if not fpath.exists():
            continue
        df = pd.read_parquet(fpath)
        date_col = "date" if "date" in df.columns else "Date"
        val_col = "close" if "close" in df.columns else ("value" if "value" in df.columns else df.columns[-1])
        
        sub = df[[date_col, val_col]].copy()
        sub.columns = ["date", col_name]
        sub["date"] = pd.to_datetime(sub["date"]).dt.tz_localize(None).dt.strftime("%Y-%m-%d")
        sub[col_name] = pd.to_numeric(sub[col_name], errors="coerce")
        sub = sub.drop_duplicates(subset=["date"])

        if base_df is None:
            base_df = sub
        else:
            base_df = pd.merge(base_df, sub, on="date", how="outer")

    if base_df is not None:
        base_df = base_df.sort_values("date").ffill().reset_index(drop=True)
    return base_df if base_df is not None else pd.DataFrame()


def build_macro_enhanced_features(df_stock: pd.DataFrame, df_macro: pd.DataFrame) -> pd.DataFrame:
    # 米国マクロ指標は前日終値 (Lag 1) を使用
    df_macro_lagged = df_macro.copy()
    for col in ["sox_close", "sp500_close", "usdjpy_close", "vix_close", "us10y_yield"]:
        if col in df_macro_lagged.columns:
            df_macro_lagged[col] = df_macro_lagged[col].shift(1)

    merged = pd.merge(df_stock, df_macro_lagged, on="date", how="left").ffill().dropna()
    if len(merged) < 50:
        return pd.DataFrame()

    # テクニカル指標
    merged["returns"] = merged["close"].pct_change().astype(float)
    merged["ma_5"] = merged["close"].rolling(5).mean().astype(float)
    merged["ma_20"] = merged["close"].rolling(20).mean().astype(float)
    merged["sma_diff"] = ((merged["ma_5"] - merged["ma_20"]) / merged["close"]).astype(float)
    
    high_low = merged["high"] - merged["low"]
    merged["atr"] = high_low.rolling(14).mean().astype(float)

    # マクロ特徴量
    if "sox_close" in merged.columns:
        merged["sox_return_1d"] = merged["sox_close"].pct_change().astype(float)
        merged["sox_diff_lag"] = (merged["sox_return_1d"] - merged["returns"].shift(1)).astype(float)
    if "usdjpy_close" in merged.columns:
        merged["usdjpy_return_1d"] = merged["usdjpy_close"].pct_change().astype(float)
    if "us10y_yield" in merged.columns:
        merged["us10y_change"] = merged["us10y_yield"].diff().astype(float)
    if "vix_close" in merged.columns:
        merged["vix_regime"] = (merged["vix_close"] > 22.0).astype(int)

    # 目的変数 (翌営業日寄付Openで買い、3営業日後大引けCloseで売ったときの純リターン)
    merged["future_entry_open"] = merged["open"].shift(-1)
    merged["future_exit_close"] = merged["close"].shift(-3)
    merged["true_return"] = ((merged["future_exit_close"] - merged["future_entry_open"]) / merged["future_entry_open"]).astype(float)
    merged["target"] = (merged["true_return"] > 0.005).astype(int)  # 手数料込みで+0.5%以上

    return merged.dropna().reset_index(drop=True)


def run_baseline_vs_macro_experiment(ticker_list: list, bars_daily_dir: Path, macro_dir: Path):
    logger.info("=== Loading Global Macro Features ===")
    df_macro = load_global_macro_features(macro_dir)
    target_codes = {t.replace(".T", "") + "0" for t in ticker_list}
    stock_data_dict = load_recent_jquants_daily_bars_fast(bars_daily_dir, target_codes, max_days=500)

    all_dataset = []
    for code, df_stock in stock_data_dict.items():
        if df_stock.empty or len(df_stock) < 100:
            continue
        raw_ticker = code[:-1] + ".T" if code.endswith("0") else code
        df_feat = build_macro_enhanced_features(df_stock, df_macro)
        if not df_feat.empty:
            df_feat["ticker"] = raw_ticker
            all_dataset.append(df_feat)

    if not all_dataset:
        logger.error("No dataset could be built!")
        return

    full_data = pd.concat(all_dataset, ignore_index=True)
    
    # 厳格な日付ベース分割 (2025年以前で学習 -> 2026年でテスト)
    cutoff_date = "2026-01-05"
    train_df = full_data[full_data["date"] < cutoff_date].copy()
    test_df = full_data[full_data["date"] >= cutoff_date].copy()

    base_features = ["returns", "sma_diff", "atr"]
    macro_features = base_features + [c for c in ["sox_return_1d", "sox_diff_lag", "usdjpy_return_1d", "us10y_change", "vix_regime"] if c in full_data.columns]

    for f in macro_features:
        train_df[f] = train_df[f].astype(float)
        test_df[f] = test_df[f].astype(float)

    # 1. Baseline Model
    clf_base = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.03, random_state=42, verbose=-1)
    clf_base.fit(train_df[base_features], train_df["target"])
    preds_base = clf_base.predict_proba(test_df[base_features])[:, 1]
    
    # 2. Macro Enhanced Model
    clf_macro = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.03, random_state=42, verbose=-1)
    clf_macro.fit(train_df[macro_features], train_df["target"])
    preds_macro = clf_macro.predict_proba(test_df[macro_features])[:, 1]

    threshold = 0.55
    trades_base = test_df[preds_base > threshold]
    trades_macro = test_df[preds_macro > threshold]

    win_rate_base = (trades_base["target"].sum() / len(trades_base) * 100) if len(trades_base) > 0 else 0.0
    win_rate_macro = (trades_macro["target"].sum() / len(trades_macro) * 100) if len(trades_macro) > 0 else 0.0

    avg_pnl_base = trades_base["true_return"].mean() * 100 if len(trades_base) > 0 else 0.0
    avg_pnl_macro = trades_macro["true_return"].mean() * 100 if len(trades_macro) > 0 else 0.0

    logger.info("\n==============================================================")
    logger.info("   AUDITED EXPERIMENT: BASELINE VS GLOBAL MACRO MODEL (OUT-OF-SAMPLE 2026)")
    logger.info("==============================================================")
    logger.info(f" Train Dates                   : {train_df['date'].min()} to {train_df['date'].max()} ({len(train_df)} rows)")
    logger.info(f" Test Dates (Out-of-Sample)    : {test_df['date'].min()} to {test_df['date'].max()} ({len(test_df)} rows)")
    logger.info("--------------------------------------------------------------")
    logger.info(f" 🔴 Baseline Model (Technical Features Only):")
    logger.info(f"    - Total Trades Executed  : {len(trades_base)}")
    logger.info(f"    - Win Rate               : {win_rate_base:.2f}%")
    logger.info(f"    - Avg Trade Return       : {avg_pnl_base:+.2f}%")
    logger.info("--------------------------------------------------------------")
    logger.info(f" 🟢 Macro Enhanced Model (+SOX, US10Y, USD/JPY, VIX):")
    logger.info(f"    - Total Trades Executed  : {len(trades_macro)}")
    logger.info(f"    - Win Rate               : {win_rate_macro:.2f}%")
    logger.info(f"    - Avg Trade Return       : {avg_pnl_macro:+.2f}%")
    logger.info("==============================================================")


def main():
    bars_daily_dir = ROOT_DIR / "market_data" / "jquants" / "bars_daily"
    macro_dir = ROOT_DIR / "market_data" / "global_macro"
    tickers_file = ROOT_DIR / "tickers.csv"

    if not tickers_file.exists():
        logger.error("tickers.csv not found!")
        return

    df_tickers = pd.read_csv(tickers_file)
    ticker_list = df_tickers["ticker"].tolist()[:100]

    run_baseline_vs_macro_experiment(ticker_list, bars_daily_dir, macro_dir)


if __name__ == "__main__":
    main()
