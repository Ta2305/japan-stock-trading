"""kabuステーション API V1 WebSocket リアルタイムストリーミングモジュール

WebSocket 接続 (ws://localhost:{port}/kabusapi/websocket) を通じて
Tick・約定・最良気配値をリアルタイム受信し、1分足 OHLCV データへ自動集計してコールバック関数へ引き渡します。
"""

import json
import time
import logging
import threading
from datetime import datetime
import pandas as pd
import websocket

logger = logging.getLogger("kabu_websocket_client")


class KabuWebSocketClient:
    def __init__(self, port: int = 18081, on_bar_completed_callback=None):
        self.port = port
        self.ws_url = f"ws://localhost:{port}/kabusapi/websocket"
        self.ws = None
        self.thread = None
        self.on_bar_completed_callback = on_bar_completed_callback

        # 1分足作成用バッファ: { symbol: { "current_minute": "14:05", "open": p, ... } }
        self.minute_buffers = {}
        self.is_running = False

    def _on_message(self, ws, message):
        try:
            msg = json.loads(message)
            symbol = msg.get("Symbol")
            price = msg.get("CurrentPrice")
            vol = msg.get("TradingVolume", 0)
            time_str = msg.get("CurrentPriceTime")  # e.g., "14:05:23" or "2026-08-14T14:05:23+09:00"

            if not symbol or price is None or price <= 0:
                return

            dt_now = datetime.now()
            minute_str = dt_now.strftime("%Y-%m-%d %H:%M")
            ticker_symbol = f"{symbol}.T"

            if ticker_symbol not in self.minute_buffers:
                self.minute_buffers[ticker_symbol] = {
                    "minute": minute_str,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": vol,
                    "last_vol": vol,
                }
            else:
                buf = self.minute_buffers[ticker_symbol]
                if buf["minute"] != minute_str:
                    # 前の1分足が完成！コールバックへ通知
                    completed_bar = {
                        "datetime": pd.to_datetime(buf["minute"]),
                        "open": buf["open"],
                        "high": buf["high"],
                        "low": buf["low"],
                        "close": buf["close"],
                        "volume": max(0, vol - buf["last_vol"]),
                    }
                    if self.on_bar_completed_callback:
                        self.on_bar_completed_callback(ticker_symbol, completed_bar)

                    # 新しい分のバッファ初期化
                    self.minute_buffers[ticker_symbol] = {
                        "minute": minute_str,
                        "open": price,
                        "high": price,
                        "low": price,
                        "close": price,
                        "volume": vol,
                        "last_vol": vol,
                    }
                else:
                    # 同じ1分枠内で四本値を更新
                    buf["high"] = max(buf["high"], price)
                    buf["low"] = min(buf["low"], price)
                    buf["close"] = price
                    buf["volume"] = max(0, vol - buf["last_vol"])

        except Exception as e:
            logger.error(f"Error parsing WebSocket message: {e}")

    def _on_error(self, ws, error):
        logger.error(f"WebSocket Error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        logger.info(f"WebSocket Closed: status={close_status_code}, msg={close_msg}")
        self.is_running = False

    def _on_open(self, ws):
        logger.info("WebSocket Stream Connection Established! Listening for market ticks...")
        self.is_running = True

    def start(self):
        """バックグラウンドスレッドで WebSocket 接続を開く"""
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self.thread = threading.Thread(target=self.ws.run_forever, daemon=True)
        self.thread.start()

    def stop(self):
        if self.ws:
            self.ws.close()
        self.is_running = False
