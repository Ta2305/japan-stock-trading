"""Windows (WSL) + kabuステーション API リアルタイム・デモトレードメインプログラム

kabuステーション API からの PUSH 配信（1分足・Tick）をリアルタイム受信しながら、
保存済み Meta-labeling 機械学習モデル (saved_models/meta_model.txt) でリアルタイム推論し、
シグナル発生時にカブコム API 経由でリアルタイムデモトレード（シミュレーション注文）を実行します。

使用方法:
    python run_live_demo_trader.py --config configs/kabu_config.json
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import lightgbm as lgb

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

from kabu_station.rest_client import KabuRestClient
from kabu_station.websocket_client import KabuWebSocketClient
from src_v2.strategy.trend_follow import AdvancedTrendFollowStrategy

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("run_live_demo_trader")


class LiveDemoTrader:
    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        kabu_cfg = self.config.get("kabu_station", {})
        self.api_password = os.getenv("KABU_API_PASSWORD") or kabu_cfg.get("api_password", "")
        self.trading_password = os.getenv("KABU_TRADING_PASSWORD") or kabu_cfg.get("trading_password", "")
        self.port = int(os.getenv("KABU_PORT") or kabu_cfg.get("port", 18081))
        self.is_demo = kabu_cfg.get("is_demo_environment", True)

        t_cfg = self.config.get("trading_params", {})
        self.ml_threshold = t_cfg.get("ml_threshold", 0.55)
        self.use_ml_filter = t_cfg.get("use_ml_filter", True)
        self.trade_size_shares = t_cfg.get("trade_size_shares", 100)

        # 1. 概念・機械学習モデルのロード (Cross-Platform)
        self.meta_model = None
        model_path = ROOT_DIR / "saved_models" / "meta_model.txt"
        if model_path.exists():
            try:
                self.meta_model = lgb.Booster(model_file=str(model_path))
                logger.info(f"🟢 Successfully Loaded Meta-labeling LightGBM Model from: {model_path.resolve()}")
            except Exception as e:
                logger.warning(f"Failed to load LightGBM model: {e}")
        else:
            logger.warning(f"No saved_models/meta_model.txt found at {model_path}. Running with technical strategy only.")

        # 2. 戦略インスタンス
        strat_cfg = {"short_window": 5, "long_window": 20, "atr_period": 14}
        self.strategy = AdvancedTrendFollowStrategy(config=strat_cfg)

        # 3. kabuステーション REST & WebSocket クライアント
        self.rest_client = KabuRestClient(api_password=self.api_password, port=self.port, is_demo=self.is_demo)
        self.ws_client = KabuWebSocketClient(port=self.port, on_bar_completed_callback=self.on_realtime_bar)

        # 銘柄ごとの過去時系列バッファ: { ticker: DataFrame }
        self.price_history = {}
        # 保有ポジション: { ticker: { "side": "buy", "qty": 100, "entry_price": 2500 } }
        self.positions = {}

    def on_realtime_bar(self, ticker: str, new_bar: dict):
        """1分足が確定するたびに呼ばれるリアルタイム推論 ＆ デモ発注ロジック"""
        logger.info(f"⚡ [1m Bar Completed] {ticker} | Time={new_bar['datetime']} | Close={new_bar['close']} | Vol={new_bar['volume']}")

        if ticker not in self.price_history:
            self.price_history[ticker] = pd.DataFrame([new_bar])
        else:
            self.price_history[ticker] = pd.concat(
                [self.price_history[ticker], pd.DataFrame([new_bar])], ignore_index=True
            ).tail(100)  # 最新100本を保持

        df_history = self.price_history[ticker]
        if len(df_history) < 25:
            logger.info(f"Building history for {ticker}: {len(df_history)}/25 bars")
            return

        # 1. テクニカル指標計算
        prep_df = self.strategy.prepare_indicators(df_history)
        latest_idx = len(prep_df) - 1
        curr_pos = self.positions.get(ticker, {}).get("side")

        # 2. テクニカルシグナル判定
        sig = self.strategy.generate_signal_at(prep_df, latest_idx, current_position=curr_pos)
        action = sig["action"]

        if action == "hold":
            return

        logger.info(f"🎯 [Signal Detected] {ticker} | Action={action} | Reason={sig['reason']}")

        # 3. Meta-labeling 機械学習フィルター
        if action in ["buy", "short"] and self.use_ml_filter and self.meta_model is not None:
            # 特徴量抽出
            row = prep_df.iloc[[latest_idx]]
            sma_diff = (row["sma_short"].values[0] - row["sma_long"].values[0]) / row["close"].values[0] if "sma_short" in row else 0.0
            atr_val = row["atr"].values[0] if "atr" in row else 0.0
            ret_val = row["close"].pct_change().values[0] if len(df_history) > 1 else 0.0

            feat_vector = np.array([[ret_val, sma_diff, atr_val]], dtype=float)
            prob = self.meta_model.predict(feat_vector)[0]

            logger.info(f"🧠 [Meta-ML Inference] {ticker} | ML Success Probability = {prob:.4f} (Threshold = {self.ml_threshold})")

            if prob < self.ml_threshold:
                logger.info(f"⛔ [ML Signal Filtered] Trade Skipped for {ticker} (Low ML Probability: {prob:.4f} < {self.ml_threshold})")
                return

        # 4. kabuステーション API デモ発注の実行
        if action == "buy":
            logger.info(f"🚀 [Executing Live Order] Sending BUY order for {ticker} ({self.trade_size_shares} shares)")
            res = self.rest_client.send_order(
                symbol=ticker,
                side="buy",
                qty=self.trade_size_shares,
                trading_password=self.trading_password,
                execution_type="market"
            )
            self.positions[ticker] = {"side": "buy", "qty": self.trade_size_shares, "entry_price": new_bar["close"]}

        elif action in ["sell", "cover"] and ticker in self.positions:
            logger.info(f"🚀 [Executing Live Order] Closing Position for {ticker}")
            res = self.rest_client.send_order(
                symbol=ticker,
                side="sell",
                qty=self.positions[ticker]["qty"],
                trading_password=self.trading_password,
                execution_type="market"
            )
            del self.positions[ticker]

    def start(self):
        """デモトレードシステムの起動"""
        logger.info("==============================================================")
        logger.info("    KABU STATION REALTIME LIVE DEMO TRADER STARTING")
        logger.info(f" Environment    : {'DEMO / SIMULATION' if self.is_demo else 'LIVE REAL-MONEY'}")
        logger.info(f" API Port       : {self.port}")
        logger.info(f" ML Filtering   : {'ENABLED' if self.use_ml_filter else 'DISABLED'} (Prob Threshold={self.ml_threshold})")
        logger.info("==============================================================")

        # REST 認証
        try:
            self.rest_client.auth()
        except Exception:
            logger.error("Could not authenticate with kabuステーション API. Make sure kabuステーション is running on Windows!")
            return

        # 銘柄読み込み & 登録
        tickers_file = ROOT_DIR / "tickers.csv"
        if not tickers_file.exists():
            logger.error("tickers.csv not found!")
            return

        df_tickers = pd.read_csv(tickers_file)
        target_tickers = df_tickers["ticker"].tolist()[:10]  # 直近10銘柄をストリーミング登録

        logger.info(f"Registering target tickers for PUSH stream: {target_tickers}")
        self.rest_client.register_symbols(target_tickers)

        # WebSocket リアルタイムストリーミング開始
        self.ws_client.start()

        logger.info("🟢 Trading Bot is active and listening for market ticks. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping Live Demo Trader...")
            self.ws_client.stop()


def main():
    parser = argparse.ArgumentParser(description="Run Live Demo Trader with kabuステーション API.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/kabu_config.json",
        help="Path to kabu_config.json",
    )
    args = parser.parse_args()

    trader = LiveDemoTrader(args.config)
    trader.start()


if __name__ == "__main__":
    main()
