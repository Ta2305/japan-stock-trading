"""kabuステーション API V1 REST クライアントモジュール

auカブコム証券の kabuステーション API と REST 通信を行い、
トークン発行、銘柄登録、時価情報取得、注文発注（デモトレード・実発注）、口座照会を実行します。
"""

import logging
import requests

logger = logging.getLogger("kabu_rest_client")


class KabuRestClient:
    def __init__(self, api_password: str, port: int = 18081, is_demo: bool = True):
        self.api_password = api_password
        self.port = port
        self.is_demo = is_demo
        self.base_url = f"http://localhost:{port}/kabusapi"
        self.token = None

    def auth(self) -> str:
        """API用トークンを発行・更新する"""
        url = f"{self.base_url}/token"
        headers = {"content-type": "application/json"}
        body = {"APIPassword": self.api_password}
        try:
            res = requests.post(url, json=body, headers=headers, timeout=10)
            res.raise_for_status()
            data = res.json()
            self.token = data.get("Token")
            logger.info("Successfully authenticated with kabuステーション API!")
            return self.token
        except Exception as e:
            logger.error(f"Failed to authenticate with kabuステーション API: {e}")
            raise e

    def get_headers(self) -> dict:
        if not self.token:
            self.auth()
        return {"content-type": "application/json", "X-API-KEY": self.token}

    def register_symbols(self, symbol_list: list[str]) -> bool:
        """WebSocketストリーミング受信用の対象銘柄を登録する"""
        url = f"{self.base_url}/register"
        headers = self.get_headers()

        symbols_body = []
        for s in symbol_list:
            code = s.replace(".T", "")
            symbols_body.append({"Symbol": code, "Exchange": 1})  # 1: 東証

        body = {"Symbols": symbols_body}
        try:
            res = requests.put(url, json=body, headers=headers, timeout=10)
            res.raise_for_status()
            logger.info(f"Registered {len(symbol_list)} symbols to kabuステーション API PUSH stream.")
            return True
        except Exception as e:
            logger.error(f"Failed to register symbols: {e}")
            return False

    def send_order(
        self,
        symbol: str,
        side: str,  # "buy" or "sell"
        qty: int,
        trading_password: str,
        execution_type: str = "market",  # "market" (成行) or "limit" (指値)
        price: float = 0.0,
    ) -> dict:
        """注文を発注する（デモ環境設定時は検証環境でのシミュレーション注文）"""
        url = f"{self.base_url}/sendorder"
        headers = self.get_headers()

        code = symbol.replace(".T", "")
        # side: 2=買, 1=売
        side_code = "2" if side.lower() == "buy" else "1"

        body = {
            "Password": trading_password,
            "Symbol": code,
            "Exchange": 1,         # 1: 東証
            "SecurityType": 1,     # 1: 株式
            "Side": side_code,
            "CashMargin": 1,       # 1: 現物
            "MarginTradeType": 0,
            "DelivType": 2,        # 2: お預り金
            "FundType": "AA",
            "AccountType": 4,      # 4: 特定口座
            "Qty": qty,
            "FrontOrderType": 10 if execution_type == "market" else 20,  # 10: 成行, 20: 指値
            "Price": 0 if execution_type == "market" else price,
            "ExpireDay": 0,        # 0: 当日中
        }

        try:
            res = requests.post(url, json=body, headers=headers, timeout=10)
            res.raise_for_status()
            result = res.json()
            logger.info(f"Order Sent! Symbol={symbol}, Side={side}, Qty={qty}, OrderID={result.get('OrderId')}")
            return result
        except Exception as e:
            logger.error(f"Failed to send order for {symbol}: {e}")
            return {"error": str(e)}
