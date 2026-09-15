"""Kaggle GPU Kernel 用 LightGBM Meta-labeling 学習スクリプト

`kaggle kernels push` で送信すると、Kaggle サーバー上で GPU を使用して
モデルの学習・保存を全自動で行います。入出力は Kaggle の規約に従います。
"""

import os
import glob
import json
import pandas as pd
import numpy as np
import lightgbm as lgb


def main():
    print("=== Kaggle GPU Meta-labeling Model Trainer ===")

    # /kaggle/input 配下のすべての .parquet / .csv ファイルを再帰的に探索
    dataset_files = glob.glob("/kaggle/input/**/*.parquet", recursive=True) + glob.glob(
        "/kaggle/input/**/*.csv", recursive=True
    )
    print(f"Found input dataset files: {dataset_files}")

    if not dataset_files:
        raise FileNotFoundError(
            "No .parquet or .csv dataset file found anywhere under /kaggle/input/"
        )

    dataset_file = dataset_files[0]
    print(f"Loading dataset: {dataset_file}")

    df = (
        pd.read_parquet(dataset_file)
        if dataset_file.endswith(".parquet")
        else pd.read_csv(dataset_file)
    )

    feature_cols = [
        "normalized_slope",
        "vwap_deviation",
        "rsi_14",
        "adx_14",
        "rvol_20",
        "ema_5m_slope",
        "kama_streak",
        "volatility_ratio",
        "time_sin",
        "time_cos",
        "is_opening",
        "is_afternoon",
    ]

    X = df[feature_cols]
    y = df["target"]

    print(
        f"Dataset loaded successfully: {len(X)} samples, Win (1) = {sum(y==1)}, Loss (0) = {sum(y==0)}"
    )

    # 1. GPU設定でのモデル試行
    params_gpu = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
        "n_estimators": 500,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 30,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 0.5,
        "random_state": 42,
        "verbose": -1,
    }

    try:
        print("\nTraining LightGBM model on Kaggle GPU...")
        model = lgb.LGBMClassifier(**params_gpu)
        model.fit(X, y)
    except Exception as e:
        print(f"GPU training encounter fallback exception: {e}")
        print("Falling back to CPU multi-threading...")
        params_cpu = params_gpu.copy()
        params_cpu["device"] = "cpu"
        params_cpu["n_jobs"] = -1
        model = lgb.LGBMClassifier(**params_cpu)
        model.fit(X, y)

    # 作業出力ディレクトリへモデルと設定を保存
    out_dir = "/kaggle/working/saved_models"
    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(out_dir, "meta_model.txt")
    model.booster_.save_model(model_path)

    meta_config = {
        "probability_threshold": 0.52,
        "feature_names": feature_cols,
        "params": params_gpu,
    }
    config_path = os.path.join(out_dir, "meta_config.json")
    with open(config_path, "w") as f:
        json.dump(meta_config, f, indent=2)

    print(f"\nTraining completed successfully! Saved model to: {model_path}")
    print(f"Saved config to: {config_path}")


if __name__ == "__main__":
    main()
