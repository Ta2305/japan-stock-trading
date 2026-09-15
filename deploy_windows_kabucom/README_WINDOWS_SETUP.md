# Windows (WSL) ＋ kabuステーション API リアルタイム・デモトレード完全セットアップガイド

本パッケージ（`deploy_windows_kabucom/`）は、Windows（WSL / Python）環境上で auカブコム証券の **kabuステーション API** と連携し、**リアルタイム PUSH ストリーミングデータ（1分足/Tick）を受信しながら学習済み Meta-labeling 機械学習モデルでリアルタイム推論を行い、デモトレード（シミュレーション自動発注）を実行する環境**が一括パッケージ化されたものです。

---

## 1. 🤖 機械学習モデルの互換性について

> **結論**: Linux / WSL 上で学習・作成されたモデルファイル（`saved_models/meta_model.txt`）は、**Windows 側にコピー・ダウンロードすれば一切の再学習なしに100%そのまま使用可能**です！

LightGBM の Booster モデルテキスト（`meta_model.txt`）は**プラットフォーム非依存（Cross-Platform）**のフォーマットです。Windows 側の Python 環境で `lgb.Booster(model_file="saved_models/meta_model.txt")` を呼び出すだけで、リアルタイムで秒単位の高速推論が動作します。

---

## 2. 📂 ディレクトリ構造

```
deploy_windows_kabucom/
├── README_WINDOWS_SETUP.md          # 本ガイドファイル
├── .env.example                     # 認証情報などの環境変数テンプレート
├── requirements_windows.txt         # 依存ライブラリ一覧
├── run_live_demo_trader.py          # リアルタイム・デモトレードメイン起動スクリプト
├── tickers.csv                      # 対象銘柄リスト (高流動性銘柄)
├── configs/                         # kabuステーション API 接続 ＆ パラメータ設定
│   └── kabu_config.json
└── kabu_station/                    # kabuステーション API v1 REST/WebSocket 連携モジュール
    ├── rest_client.py               # トークン発行・銘柄登録・注文発注
    └── websocket_client.py          # WebSocket リアルタイム1分足 PUSH 受信
```

> **⚠️ 本リポジトリの公開版には `src_v2/`（推論エンジン）・`saved_models/`（学習済みモデル）・`market_data/`（マクロデータ）は含まれません**（データ・モデルは Git 管理対象外）。実行前に以下を用意してください：
>
> 1. リポジトリ直下の `src_v2/` を、本ディレクトリ配下にコピーする（`cp -r ../src_v2 ./src_v2`）
> 2. 学習済みモデル `meta_model.txt` / `meta_config.json` を `saved_models/` に配置する（学習手順はルート `README.md` を参照）
> 3. マクロデータ `market_data/global_macro/` を配置する

---

## 3. 🚀 セットアップ ＆ 実行手順

### Step 1: Windows 側で kabuステーションを起動し、API設定をONにする

1. Windows 上で **kabuステーション** を起動し、ログインします。
2. 画面右上の **「設定」 $\rightarrow$ 「API設定」** を開きます。
3. 以下の設定を行います：
   * **API機能**: 「利用する」にチェック
   * **検証用環境ポート**: `18081`（デモトレード用）または 本番環境 `18080`
   * **APIパスワード**: 任意の英数字パスワードを設定（例: `my_api_pass_123`）

---

### Step 2: 本パッケージ（`deploy_windows_kabucom`）を Windows / WSL へコピー

本ディレクトリ `deploy_windows_kabucom/` 全体を、Windows 側の任意のフォルダ（例: `C:\Users\username\japan-stock-trading\` または WSL 上の `~/japan-stock-trading/`）へコピーしてください。

---

### Step 3: 認証情報の設定（`.env`）

`.env.example` をコピーして `.env` を作成し、Step 1 で設定した値を記入します（`.env` は Git 管理対象外です）：

```bash
cp .env.example .env
```

```dotenv
KABU_API_PASSWORD=設定したAPIパスワード
KABU_TRADING_PASSWORD=注文用暗証番号
KABU_HOST=127.0.0.1
KABU_PORT=18081
```

接続先や売買パラメータを変えたい場合は `configs/kabu_config.json` を編集します（パスワードは環境変数が優先されます）：

```json
{
  "kabu_station": {
    "port": 18081,
    "is_demo_environment": true
  },
  "trading_params": {
    "max_positions": 5,
    "trade_size_shares": 100,
    "use_ml_filter": true,
    "ml_threshold": 0.55
  }
}
```

---

### Step 4: 依存ライブラリのインストール

Windows ターミナル（Command Prompt / PowerShell）または WSL ターミナルで以下を実行します：

```bash
cd deploy_windows_kabucom
pip install -r requirements_windows.txt
```

---

### Step 5: リアルタイム・デモトレードの起動

ザラ場中（平日 9:00〜15:30）に以下のコマンドを実行すると、kabuステーション API からリアルタイムで 1分足・Tick が PUSH 受信され、Meta-labeling 機械学習でフィルタリングされた自動デモ注文が開始されます！

```bash
python run_live_demo_trader.py --config configs/kabu_config.json
```

#### ログ出力イメージ:
```
==============================================================
    KABU STATION REALTIME LIVE DEMO TRADER STARTING
 Environment    : DEMO / SIMULATION (Port: 18081)
 ML Filtering   : ENABLED (Prob Threshold = 0.55)
🟢 Successfully Loaded Meta-labeling LightGBM Model from saved_models/meta_model.txt
==============================================================
Successfully authenticated with kabuステーション API!
Registered 10 symbols to kabuステーション API PUSH stream.
WebSocket Stream Connection Established! Listening for market ticks...
⚡ [1m Bar Completed] 7203.T | Time=2026-08-14 09:05 | Close=2540.0 | Vol=12000
🎯 [Signal Detected] 7203.T | Action=buy | Reason=MA Golden Cross
🧠 [Meta-ML Inference] 7203.T | ML Success Probability = 0.7850 (Threshold = 0.55)
🚀 [Executing Live Order] Sending BUY order for 7203.T (100 shares)
Order Sent! Symbol=7203.T, Side=buy, Qty=100, OrderID=202608140001
```
