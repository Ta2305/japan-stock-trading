"""Meta-labeling モデル学習・評価スクリプト

`export_ml_dataset.py` で作成したデータセットを読み込み、Purged Group TimeSeries Cross-Validation で
LightGBM Meta-labeling モデルの学習・評価を行い、モデルを保存します。
Google Colab / Kaggle などの外部GPU環境でもそのまま実行可能です。

使用方法:
    python scripts/train_meta_model.py --dataset ml_dataset.parquet --gpu
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src_v2.ml import (
    MetaLabelingModel,
    PurgedGroupTimeSeriesSplit,
    FEATURE_COLUMNS,
)


def train_pipeline(
    dataset_path: str = "ml_dataset.parquet",
    output_dir: str = "saved_models",
    use_gpu: bool = False,
    threshold: float = 0.52,
):
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    print(f"Loading dataset: {dataset_path}...")
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

    print(f"Dataset loaded: {len(df)} samples, {len(FEATURE_COLUMNS)} features.")

    X = df[FEATURE_COLUMNS]
    y = df["target"]

    # Purged TimeSeries CV で評価
    cv = PurgedGroupTimeSeriesSplit(n_splits=5, pct_embargo=0.01, purge_window=15)

    cv_accs = []
    cv_precisions = []

    print("\n--- Running Purged K-Fold Cross Validation ---")
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = MetaLabelingModel(
            probability_threshold=threshold, use_gpu=use_gpu
        )
        model.train(X_train, y_train, eval_set=(X_val, y_val))

        val_probs = model.predict_proba(X_val)
        val_preds = (val_probs >= threshold).astype(int)

        # 評価指標
        acc = np.mean(val_preds == y_val)
        selected_mask = val_preds == 1
        precision = (
            np.sum((val_preds == 1) & (y_val == 1)) / np.sum(selected_mask)
            if np.sum(selected_mask) > 0
            else 0.0
        )

        cv_accs.append(acc)
        cv_precisions.append(precision)

        print(
            f"Fold {fold+1}: Accuracy = {acc*100:.2f}%, Precision(WinRate) = {precision*100:.2f}%, Selected = {np.sum(selected_mask)}/{len(y_val)}"
        )

    print(f"\nMean CV Accuracy   : {np.mean(cv_accs)*100:.2f}%")
    print(f"Mean CV Precision  : {np.mean(cv_precisions)*100:.2f}%")

    # 全データで最終モデルの学習と保存
    print("\nTraining final model on full dataset...")
    final_model = MetaLabelingModel(
        probability_threshold=threshold, use_gpu=use_gpu
    )
    final_model.train(X, y)
    final_model.save(output_dir)

    print(f"Model saved to: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM Meta-labeling Model")
    parser.add_argument("--dataset", type=str, default="ml_dataset.parquet", help="Path to input dataset")
    parser.add_argument("--output-dir", type=str, default="saved_models", help="Directory to save model")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration (for Colab / Kaggle)")
    parser.add_argument("--threshold", type=float, default=0.52, help="Probability threshold for signal filtering")

    args = parser.parse_args()

    train_pipeline(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        use_gpu=args.gpu,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
