"""J-Quants API V2 認証および通信共通クライアントモジュール

環境変数 JQUANTS_API_KEY (または JQUANTS_MAIL_ADDRESS / JQUANTS_PASSWORD) から
トークンを自動ロードし、自動リトライ・レート制限・ディスク残量安全監視付きのセッションを提供します。
"""

import os
import time
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import requests

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

logger = logging.getLogger("jquants_client")

JQUANTS_V2_BASE_URL = "https://api.jquants.com/v2"


def check_disk_space(target_path: Path, min_free_gb: float = 10.0) -> bool:
    try:
        total, used, free = shutil.disk_usage(target_path.resolve().anchor or str(target_path.resolve()))
        free_gb = free / (1024 ** 3)
        if free_gb < min_free_gb:
            logger.critical(
                f"\n[DISK SAFETY ALERT] Remaining disk space ({free_gb:.2f} GB) is below safety limit ({min_free_gb:.1f} GB)!"
            )
            return False
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
    return True


class JQuantsClientV2:
    """J-Quants API V2 クライアント"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("JQUANTS_API_KEY")
        self.mail_address = os.getenv("JQUANTS_MAIL_ADDRESS")
        self.password = os.getenv("JQUANTS_PASSWORD")
        self.id_token: Optional[str] = None

        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "JapanStockTradingSystem/2.0 (python-requests)"}
        )

        self._authenticate()

    def _authenticate(self):
        if self.api_key:
            self.session.headers.update({"x-api-key": self.api_key})
            logger.info("Authenticated using JQUANTS_API_KEY.")
            return

        if self.mail_address and self.password:
            try:
                auth_url = "https://api.jquants.com/v1/token/auth_user"
                res = self.session.post(
                    auth_url,
                    json={"mailaddress": self.mail_address, "password": self.password},
                )
                res.raise_for_status()
                refresh_token = res.json().get("refreshToken")

                ref_url = (
                    f"https://api.jquants.com/v1/token/auth_refresh?refreshtoken={refresh_token}"
                )
                ref_res = self.session.post(ref_url)
                ref_res.raise_for_status()
                self.id_token = ref_res.json().get("idToken")
                self.session.headers.update({"Authorization": f"Bearer {self.id_token}"})
                logger.info("Authenticated using Mail Address & Password tokens.")
                return
            except Exception as e:
                logger.error(f"Failed token auth via Mail/Password: {e}")

        logger.warning(
            "No valid J-Quants credentials found in environment variables (JQUANTS_API_KEY, JQUANTS_MAIL_ADDRESS, JQUANTS_PASSWORD)."
        )

    def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        max_retries: int = 5,
        delay_sec: float = 0.3,
        ignore_400: bool = False,
    ) -> Dict[str, Any]:
        """レート制限および自動バックオフ付きの GET リクエスト"""
        url = (
            endpoint
            if endpoint.startswith("http")
            else f"{JQUANTS_V2_BASE_URL}{endpoint if endpoint.startswith('/') else '/' + endpoint}"
        )

        for attempt in range(1, max_retries + 1):
            try:
                time.sleep(delay_sec)
                response = self.session.get(url, params=params, timeout=30)

                if response.status_code == 400 and ignore_400:
                    return {}

                if response.status_code == 429:
                    wait_time = attempt * 3.0
                    logger.warning(
                        f"Rate limit exceeded (HTTP 429). Retrying in {wait_time}s... (Attempt {attempt}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                return response.json()

            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 400 and ignore_400:
                    return {}
                if attempt == max_retries:
                    if ignore_400:
                        return {}
                    raise e
                time.sleep(1.0)
            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"Request failed to {url}: {e} (Attempt {attempt}/{max_retries})"
                )
                if attempt == max_retries:
                    raise e
                time.sleep(attempt * 1.5)

        return {}

    def get_data_list(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        delay_sec: float = 0.3,
        ignore_400: bool = False,
    ) -> List[Dict[str, Any]]:
        """API V2 のデータリスト ('data' キーまたは各情報キー) を抽出するヘルパー"""
        res = self.get(endpoint, params=params, delay_sec=delay_sec, ignore_400=ignore_400)
        if isinstance(res, list):
            return res
        if isinstance(res, dict):
            if "data" in res and isinstance(res["data"], list):
                return res["data"]
            for v in res.values():
                if isinstance(v, list):
                    return v
        return []
