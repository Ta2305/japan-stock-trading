"""LightGBM Meta-labeling モデルクラス

一次モデルのシグナルに対し「従うべきか否か（1 or 0）」を予測する確率分類モデルです。
CPU / GPU (Colab / Kaggle CUDA) の自動切替に対応します。
"""

from typing import Dict, Any, Optional, Tuple, List
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

from src_v2.ml.feature_builder import FEATURE_COLUMNS


class MetaLabelingModel:
    """LightGBMベースの二次分類（Meta-labeling）モデル"""

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        probability_threshold: float = 0.52,
        use_gpu: bool = False,
    ):
        self.probability_threshold = probability_threshold
        self.use_gpu = use_gpu

        # デフォルトハイパーパラメータ
        default_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "n_estimators": 300,
            "learning_rate": 0.03,
            "num_leaves": 15,
            "max_depth": 4,
            "min_child_samples": 30,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.5,
            "reg_lambda": 0.5,
            "random_state": 42,
            "verbose": -1,
        }

        if use_gpu:
            default_params["device"] = "gpu"

        if params:
            default_params.update(params)

        self.params = default_params
        self.model: Optional[lgb.LGBMClassifier] = None
        self.booster: Optional[lgb.Booster] = None
        self.feature_names: List[str] = FEATURE_COLUMNS

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
    ) -> Dict[str, Any]:
        """モデルの学習を実行する。"""
        X_train = (
            X[self.feature_names]
            if all(c in X.columns for c in self.feature_names)
            else X
        )

        self.model = lgb.LGBMClassifier(**self.params)

        if eval_set:
            X_val, y_val = eval_set
            X_val_clean = (
                X_val[self.feature_names]
                if all(c in X_val.columns for c in self.feature_names)
                else X_val
            )
            self.model.fit(
                X_train,
                y,
                eval_X=X_val_clean,
                eval_y=y_val,
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
        else:
            self.model.fit(X_train, y)

        self.booster = self.model.booster_

        train_preds = self.predict_proba(X_train)
        train_acc = np.mean((train_preds >= self.probability_threshold) == y)

        return {"train_accuracy": train_acc, "n_samples": len(X_train)}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """勝率確率 P(Win) を予測する。"""
        if self.booster is None and self.model is None:
            raise ValueError("Model is not loaded or trained yet.")

        X_clean = (
            X[self.feature_names]
            if all(c in X.columns for c in self.feature_names)
            else X
        )

        if self.booster is not None:
            probs = self.booster.predict(X_clean)
            return np.array(probs)
        else:
            return self.model.predict_proba(X_clean)[:, 1]

    def should_filter_signal(self, X_single_row: pd.DataFrame) -> bool:
        """単一の発生シグナルに従うべきか（P(Win) >= threshold）判定する。"""
        probs = self.predict_proba(X_single_row)
        prob = probs[0] if isinstance(probs, np.ndarray) and len(probs) > 0 else float(probs)
        return prob >= self.probability_threshold

    def save(self, save_dir: str):
        """モデルおよび設定の保存"""
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)

        if self.model and hasattr(self.model, "booster_"):
            self.model.booster_.save_model(str(path / "meta_model.txt"))
        elif self.booster:
            self.booster.save_model(str(path / "meta_model.txt"))

        meta_info = {
            "probability_threshold": self.probability_threshold,
            "feature_names": self.feature_names,
            "params": self.params,
        }
        with open(path / "meta_config.json", "w") as f:
            json.dump(meta_info, f, indent=2)

    def load(self, load_dir: str):
        """モデルおよび設定のロード"""
        path = Path(load_dir)
        config_path = path / "meta_config.json"
        model_path = path / "meta_model.txt"

        if not config_path.exists() or not model_path.exists():
            raise FileNotFoundError(f"Model files not found in {load_dir}")

        with open(config_path, "r") as f:
            meta_info = json.load(f)

        self.probability_threshold = meta_info.get("probability_threshold", 0.52)
        self.feature_names = meta_info.get("feature_names", FEATURE_COLUMNS)

        self.booster = lgb.Booster(model_file=str(model_path))
