"""設定・パス共通モジュール"""

import json
from pathlib import Path
from typing import Dict, Any

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "configs"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "strategy_config_v2.json"
DATA_DIR = BASE_DIR / "market_data"


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """設定JSONファイルを読み込む。"""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)
