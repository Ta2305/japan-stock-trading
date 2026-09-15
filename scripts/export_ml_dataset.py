"""ML学習用データセットエクスポートスクリプト

`market_data/` 内の全銘柄データから、一次モデル（提案A+）のシグナル、特徴量、および Triple Barrier ラベルを構築し、
Google Colab / Kaggle などの外部学習環境に転送可能な `ml_dataset.parquet` としてエクスポートします。

使用方法:
    python scripts/export_ml_dataset.py --output ml_dataset.parquet
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src_v2.config import load_config, DATA_DIR
from src_v2.backtester.data_handler import load_ticker_data, get_available_tickers
from src_v2.strategy import AdvancedMTFRegimeStrategy
from src_v2.ml.feature_builder import build_ml_features, FEATURE_COLUMNS
from src_v2.ml.labeler import apply_triple_barrier_labels


def export_dataset(output_path: str = "ml_dataset.parquet"):
    config = load_config()
    strategy = AdvancedMTFRegimeStrategy(config.get("default", config))

    tickers = get_available_tickers(str(DATA_DIR))
    print(f"Exporting ML dataset for {len(tickers)} tickers...")

    all_samples = []

    for ticker in tqdm(tickers, desc="Processing Tickers"):
        try:
            df = load_ticker_data(ticker, data_dir=str(DATA_DIR))
            if df.empty or len(df) < 100:
                continue

            # 1. 特徴量の構築
            feat_df = build_ml_features(df)

            # 2. 一次モデルのシグナルイベント抽出
            prep_df = strategy.prepare_indicators(df)
            events = []

            for i in range(len(prep_df)):
                sig = strategy.generate_signal_at(prep_df, i)
                action = sig["action"]
                if action in ["buy", "short"]:
                    row = prep_df.iloc[i]
                    dt = row["datetime"] if "datetime" in row else prep_df.index[i]
                    side = 1 if action == "buy" else -1
                    events.append(
                        {
                            "datetime": dt,
                            "side": side,
                            "price": sig["price"],
                            "atr": sig["atr"],
                            "orig_idx": i,
                        }
                    )

            if not events:
                continue

            events_df = pd.DataFrame(events)

            # 3. Triple Barrier ラベリング
            labels_df = apply_triple_barrier_labels(
                df=feat_df,
                events=events_df,
                pt_sl_ratio=(2.0, 1.5),
                max_holding_bars=15,
            )

            if labels_df.empty:
                continue

            # 4. イベント時点の特徴量とラベルを結合
            merged = pd.merge(
                events_df, labels_df, left_on="datetime", right_on="event_datetime"
            )

            for _, row in merged.iterrows():
                orig_idx = int(row["orig_idx"])
                feat_row = feat_df.iloc[orig_idx][FEATURE_COLUMNS].to_dict()
                feat_row["ticker"] = ticker
                feat_row["datetime"] = row["datetime"]
                feat_row["side"] = row["side_x"]
                feat_row["target"] = int(row["target"])
                feat_row["ret"] = float(row["ret"])
                all_samples.append(feat_row)

        except Exception as e:
            print(f"Warning: Failed processing {ticker}: {e}")

    if not all_samples:
        print("No events generated for dataset.")
        return

    full_dataset = pd.DataFrame(all_samples)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if out_file.suffix == ".parquet":
        full_dataset.to_parquet(out_file, index=False)
    else:
        full_dataset.to_csv(out_file, index=False)

    print(f"\nSuccessfully exported ML dataset: {out_file}")
    print(f"Total Samples: {len(full_dataset)}")
    print(f"Target Distribution: Win (1) = {sum(full_dataset['target'] == 1)}, Loss (0) = {sum(full_dataset['target'] == 0)}")
    print(f"Features Count: {len(FEATURE_COLUMNS)}")


def main():
    parser = argparse.ArgumentParser(description="Export ML Dataset for Colab/Kaggle Training")
    parser.add_argument("--output", type=str, default="ml_dataset.parquet", help="Output path (.parquet or .csv)")
    args = parser.parse_args()

    export_dataset(args.output)


if __name__ == "__main__":
    main()
