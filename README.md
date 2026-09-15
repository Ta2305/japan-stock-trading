# 日本株 自動売買システム (Japan Stock Trading System)

J-Quants API（日本株の日足・財務・信用データ）とグローバルマクロ指標（SOX指数・米金利・ドル円・VIX）を統合し、
**モメンタム戦略による一次シグナル** と **LightGBM による Meta-labeling（二次判定）** を組み合わせて、
日本株を自動売買するためのリサーチ・バックテスト・本番執行の一式です。

> 本リポジトリは研究・学習目的のポートフォリオです。投資助言ではなく、いかなる運用成果も保証しません。

---

## 1. 概要

本システムは「**2段階意思決定フレームワーク (Two-Stage Meta-labeling)**」を採用しています。

1. **一次判定 (Primary Strategy)**: テクニカル指標（KAMA, ATR, ADX, VWAP, RVOL 等）で買い候補を検出する。
2. **二次判定 (Meta-labeling)**: LightGBM が「その候補が実際に勝てるか」の確率を推論し、ダマシを棄却する。
3. **執行・リスク管理**: 翌営業日の寄付で成行エントリーし、ATRベースの Triple Barrier（利確 / 損切り / 時間切れ）で決済する。

データ収集からモデル学習、バックテスト、リアルタイム・デモトレードまでを一貫して実行できます。

---

## 2. なぜこの設計にしたのか（開発の背景と設計思想）

### 2-1. 当初の想定（1分足デイトレ）から、日足スイングへ

開発当初は「1分足を用いた高頻度デイトレード」を想定し、yfinance から1分足データを収集していました。
しかし検証を進める過程で、次の事実が判明しました。

- 1分足の短期ノイズはダマシが多く、取引コスト（スリッページ・呼値）の影響が相対的に大きい。
- 一方で、**日足スイング（数日保有）× 機械学習フィルタ**のバックテストの方が安定して高い成績を示した。

そのため、現在のシステムは **日足スイングを主軸** とし、1分足はリアルタイム執行（板情報・約定）用途として位置づけています。

### 2-2. なぜ「2段階」なのか

テクニカルのブレイクアウト戦略は、単体では「偽のブレイクアウト（ダマシ）」を多く含みます。
そこで、世界的に標準的な **Meta-labeling**（López de Prado, *Advances in Financial Machine Learning*, 2018）を採用し、

- 一次モデル = 「エントリー候補の発見」（再現率重視）
- 二次モデル = 「候補の中から勝てるものだけを選別」（精度重視）

と役割を分離しました。二次モデルは「勝率確率 P(Win) ≥ 閾値」のときだけエントリーを承認します。

### 2-3. グローバルマクロを「前日終値（Lag 1）」で統合する

日本株は前夜の米国市場（SOX・S&P500・米金利・VIX・ドル円）の影響を強く受けます。
本システムは、これらを **前夜に確定した終値（Lag 1）としてのみ** 特徴量に結合します。
当日の日本株取引時間中に未確定の海外指標を参照する「先読み（Look-ahead Bias）」は一切行いません。

### 2-4. バックテストの健全性を最優先する

初期のプロトタイプには、以下のような非現実的な仮定（＝実運用では再現不可能な過大評価）が含まれていました。
これらをすべて是正した上で成績を評価しています。

| 当初の欠陥 | 是正後の実装 |
| :--- | :--- |
| Train/Test の銘柄順分割（リーク） | **時系列日付で厳格に分割**（過去で学習 → 未知期間で検証） |
| 「3日後の最高値で利確」という神業約定 | **翌日寄付（Open）成行エントリー → ATR利確/損切り/Triple Barrier** |
| 米国マクロ指標の同日先読みマージ | **前夜終値（Lag 1）のみ結合** |
| 資金拘束の無視（無限に建てられる） | **最大5銘柄分散・余力不足時はスキップ** |

さらに、往復スリッページ（0.1%）、信用買方金利（年2.8%）、貸株料（年1.15%）をすべて控除しています。

### 2-5. 「単元株100株」という現実との戦い

日本株には **単元株制度（原則100株単位）** があります。この制約は成績を大きく左右します。

- **元手100万円（1枠20万円）** の場合、株価2,000円以上の値がさ株（AI・半導体など最強のモメンタム銘柄）は1単元すら買えず、
  シグナルの大半を「資金不足」でスキップしてしまい、リターンが伸びない。
- **元手500万円以上**、または **単元未満株（1株単位）** を活用すると、優良シグナルを網羅でき、本来の性能が発揮される。

この発見から、現実的な資金制約を織り込んだポートフォリオ・シミュレーションを重視しています。

---

## 3. システム構成

```
┌─────────────────────────────┐
│  データ収集                   │
│  ・J-Quants API (日足/財務/信用) │
│  ・yfinance/FRED (SOX/米金利/  │
│    ドル円/VIX ※Lag 1)         │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  一次判定 (Primary Strategy)   │
│  KAMA / ATR / ADX / VWAP /RVOL │
└──────────────┬──────────────┘
               ▼ 買い候補
┌─────────────────────────────┐
│  二次判定 (LightGBM           │
│  Meta-labeling)              │
│  P(Win) >= 閾値 で承認         │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  執行・リスク管理              │
│  翌日寄付エントリー            │
│  ATR Triple Barrier 決済      │
│  最大5銘柄・単元株サイジング     │
└─────────────────────────────┘
```

---

## 4. ディレクトリ構成

```text
japan-stock-trading/
├── README.md
├── pyproject.toml / requirements.txt
├── .env.example                # 環境変数テンプレート
├── tickers.csv                 # 対象銘柄リスト
│
├── src_v2/                     # コアパッケージ（本システムの中核）
│   ├── indicators/             # ATR / KAMA / EMA / SMA / RSI / ADX / VWAP / RVOL / 連続足
│   ├── utils/                  # 東証呼値（tick size）・セッション/時間帯レジーム
│   ├── risk/                   # ATRベース SL/TP・トレイリング・100株単位ポジションサイジング
│   ├── strategy/               # 一次戦略（トレンドフォロー / MTFレジーム）
│   ├── ml/                     # Meta-labeling（Triple Barrier / Purged CV / LightGBM）
│   └── backtester/             # イベント駆動 Bar-by-Bar バックテストエンジン
│
├── scripts/                    # 実行スクリプト
│   ├── jquants/                # J-Quants データ取得ツールキット
│   ├── edinet/                 # EDINET（大量保有報告等）取得
│   ├── download_global_macro.py# グローバルマクロ取得（yfinance + FRED）
│   ├── daily_data_updater.py   # 日次データ更新
│   ├── generate_tickers.py     # 銘柄リスト生成
│   ├── screen_top_tickers.py   # 流動性スクリーニング
│   ├── run_backtest_v2.py      # バックテスト（銘柄別 Parquet）
│   ├── run_backtest_jquants_v2.py # バックテスト（J-Quants 1分足）
│   ├── export_ml_dataset.py    # ML学習用データセット出力
│   ├── train_meta_model.py     # Meta-labeling モデル学習
│   ├── audit_code_integrity.py # リーク・健全性監査
│   ├── analyze_*.py            # 各種分析（ATR感度・年別内訳・保有期間など）
│   └── experiment_*.py         # 戦略実験（後述の strategies/ から呼び出し）
│
├── strategies/                 # 戦略別バックテスト（README + backtest.py）
│   ├── 01_long_only_swing/
│   ├── 02_credit_long_short/
│   ├── 03_regime_switching/
│   ├── 04_margin_compounding/
│   └── 05_alpha_max_momentum/
│
├── analytics/                  # 分析エントリポイント（scripts/ の薄いラッパー）
├── configs/                    # 戦略パラメータ（strategy_config_v2.json）
├── kaggle_kernel/              # 外部GPU（Kaggle）学習用カーネル
├── deploy_windows_kabucom/     # Windows + kabuステーション API リアルタイム・デモトレード
└── market_data/                # 取得データの保存先（Git管理対象外）
```

> **データ・学習済みモデル・調査ドキュメントはリポジトリに含まれません**（`.gitignore` で除外）。
> コードとフレームワークのみを公開し、各自の環境でデータ取得・学習を行える構成にしています。

---

## 5. セットアップ

### 5-1. 必要環境

- Python **3.10 以上**
- [uv](https://docs.astral.sh/uv/)（推奨）または pip

### 5-2. インストール

```bash
# uv（推奨）
uv sync

# pip の場合
pip install -r requirements.txt
```

### 5-3. 環境変数の設定（重要）

認証情報（APIキー・パスワード）はコードに一切ハードコードしていません。すべて環境変数から読み込みます。

```bash
cp .env.example .env
```

`.env` に必要な値を記入します（`.env` は `.gitignore` 済みでコミットされません）。

| 変数 | 用途 | 必須 |
| :--- | :--- | :---: |
| `JQUANTS_API_KEY` | J-Quants API キー（または `JQUANTS_MAIL_ADDRESS` + `JQUANTS_PASSWORD`） | ○（日本株データ取得時） |
| `FRED_API_KEY` | FRED（米金利・マクロ）APIキー | △（未設定時は FRED 系列をスキップ） |
| `KABU_API_PASSWORD` / `KABU_TRADING_PASSWORD` | kabuステーション API パスワード / 注文暗証番号 | ○（実取引デモ時） |
| `SLACK_WEBHOOK_URL` | データ取得完了通知（任意） | - |
| `KAGGLE_USERNAME` / `KAGGLE_DATASET_SLUG` | Kaggle 外部GPU学習（任意） | - |

- J-Quants API: https://jpx-jquants.com/
- FRED API キー: https://fred.stlouisfed.org/docs/api/api_key.html

### 5-4. データの準備

```bash
# 1. 日本株データ（J-Quants: 日足・財務・信用・1分足/Tick 等）
uv run python scripts/jquants/download_all.py --output-base-dir market_data/jquants

# 2. グローバルマクロ（SOX / 米金利 / ドル円 / VIX など）
uv run python scripts/download_global_macro.py

# 3. 対象銘柄リスト（東証上場銘柄から生成 / 流動性で選抜）
uv run python scripts/generate_tickers.py
uv run python scripts/screen_top_tickers.py   # 売買代金・株価でスクリーニング
```

---

## 6. 使い方

### 6-1. バックテスト

```bash
# J-Quants の1分足データを用いて、Meta-labeling モデルの性能を評価
uv run python scripts/run_backtest_jquants_v2.py --tickers tickers.csv --max-stocks 50 --max-days 60

# 銘柄ディレクトリ形式の Parquet データを用いる場合
uv run python scripts/run_backtest_v2.py --all
uv run python scripts/run_backtest_v2.py --ticker 7203
```

### 6-2. 機械学習（Meta-labeling）

```bash
# 1. 学習用データセットのエクスポート（Triple Barrier ラベル付き）
uv run python scripts/export_ml_dataset.py --output ml_dataset.parquet

# 2. Purged K-Fold CV で学習・評価し、モデルを保存（saved_models/）
uv run python scripts/train_meta_model.py --dataset ml_dataset.parquet
uv run python scripts/train_meta_model.py --dataset ml_dataset.parquet --gpu   # GPU 環境
```

外部GPU（Kaggle / Colab）を利用する場合は `kaggle_kernel/` を参照してください
（`kernel-metadata.json` の `YOUR_KAGGLE_USERNAME` を自分のアカウントに置き換えてから push します）。

### 6-3. 戦略別バックテスト

`strategies/` に、検証してきた戦略を段階的に整理しています（各ディレクトリの `README.md` に仮説・結果・考察を記載）。

| # | 戦略 | 概要 |
| :--- | :--- | :--- |
| 01 | 現物ロング・3日スイング | ベースライン。Meta-labeling で高勝率銘柄を選別し3営業日保有 |
| 02 | 信用 Long & Short | 空売りを統合した双方向戦略（下落相場のヘッジ） |
| 03 | マクロ・レジームスイッチング | VIX/SOX に応じて買いのみ / 空売りのみへ完全切替 |
| 04 | 信用レバレッジ＆益出し複利 | 利確益を保証金に算入し建玉枠を雪だるま式に拡大 |
| 05 | Alpha-Max モメンタム集中 | Point-in-Time スクリーニング × 動的レバレッジ × 防御的トレイリング |

```bash
uv run python strategies/01_long_only_swing/backtest.py
uv run python strategies/05_alpha_max_momentum/backtest.py
```

### 6-4. 分析

`analytics/` にパラメータ感度や市場構造の分析エントリを配置しています。

```bash
uv run python analytics/atr_sensitivity/analyze_atr.py
uv run python analytics/walkforward_and_regime/analyze_walkforward.py
uv run python analytics/benchmark_breakdown/analyze_benchmark.py
uv run python analytics/kioxia_case_study/analyze_kioxia.py
```

リークや非現実的仮定が混入していないかを機械的に監査するスクリプトもあります。

```bash
uv run python scripts/audit_code_integrity.py
```

### 6-5. リアルタイム・デモトレード（Windows + kabuステーション API）

`deploy_windows_kabucom/` に、kabuステーション API の PUSH 配信を受信しながら
Meta-labeling でリアルタイム推論し、デモ口座へ自動発注するサンプルを同梱しています。
詳細は [`deploy_windows_kabucom/README_WINDOWS_SETUP.md`](deploy_windows_kabucom/README_WINDOWS_SETUP.md) を参照してください。

---

## 7. 主要モジュール（`src_v2/`）

| モジュール | 内容 |
| :--- | :--- |
| `indicators/` | ATR, KAMA, SMA/EMA, RSI, ADX(+DI/-DI), VWAP(セッションリセット), RVOL, 連続足 |
| `utils/` | 東証の価格帯別呼値（tick size）、取引セッション・時間帯レジーム判定 |
| `risk/` | ATR倍数ベースの SL/TP・トレイリングストップ、Fixed Fractional ポジションサイジング |
| `strategy/` | `AdvancedTrendFollowStrategy`（提案A）/ `AdvancedMTFRegimeStrategy`（提案A+） |
| `ml/` | Triple Barrier ラベリング、12次元特徴量生成、Purged Group TimeSeries CV、LightGBM モデル |
| `backtester/` | イベント駆動 Bar-by-Bar エンジン、スリッページ考慮、パフォーマンス指標算出 |

---

## 8. 設計上の約束事

- **先読み禁止**: 特徴量に使うデータは当該時点までに確定した値のみ（マクロは Lag 1）。
- **時系列分割**: 学習は過去、検証は未来。シャッフル分割は行わない。
- **実コスト控除**: 往復スリッページ（0.1%）、信用金利・貸株料を厳密に差し引く。
- **現実的な約定**: 前日引け後にシグナル判定 → **翌営業日寄付（Open）成行**。
- **資金制約の尊重**: 最大保有数・単元株・余力不足スキップを厳密にモデル化。
- **秘密情報の排除**: 認証情報は `.env` から読み込み、コード・履歴に残さない。

---

## 9. 注意事項・免責

- 本リポジトリは **研究・学習目的** であり、投資助言・勧誘ではありません。実際の取引は自己責任で行ってください。
- バックテストの成績は過去データに基づくシミュレーションであり、将来の運用成果を保証しません。
- 市場データ・学習済みモデルは同梱していません。各自の契約・環境で取得してください。
- 認証情報（`.env`）は絶対にコミットしないでください。
