# `src_v2` パッケージ仕様・実装状況

本パッケージ (`src_v2`) は、日本株の売買戦略・バックテスト・機械学習二次モデル（Meta-labeling）のコアモジュール群です。
1分足・日足いずれのOHLCVデータにも適用でき、指標計算からリスク管理・イベント駆動バックテストまでを提供します。

---

## 📂 モジュール構成と実装状況

```text
src_v2/
├── __init__.py
├── config.py                       # 設定ファイル読み込み・データディレクトリパス管理
│
├── indicators/                     # 高度テクニカル指標モジュール
│   ├── __init__.py
│   ├── volatility.py               # ATR (Average True Range), True Range 算出
│   ├── moving_averages.py          # KAMA (Kaufman Adaptive MA), SMA, EMA 算出
│   ├── momentum.py                 # ADX, +DI/-DI, RSI 算出
│   ├── volume.py                   # VWAP (日次・セッションリセット型), RVOL (相対出来高比率) 算出
│   └── streaks.py                  # 連続足・連続KAMA数カウント
│
├── utils/                          # ユーティリティ・市場ルール
│   ├── __init__.py
│   ├── tick_size.py                # 東証公式価格帯別最小呼値（1 tick）計算
│   └── time_utils.py               # 東証前場/後場セッション判定・時間帯レジーム判定
│
├── risk/                           # リスク管理・資金管理
│   ├── __init__.py
│   ├── stop_loss.py                # ATR倍数ベース動的SL/TP & トレイリングストップ
│   └── position_sizer.py           # 口座資金許容リスク率 (Fixed Fractional) に基づく100株単位発注量計算
│
├── strategy/                       # 売買戦略モジュール
│   ├── __init__.py
│   ├── base.py                     # 戦略の共通抽象基底クラス (BaseStrategy)
│   ├── trend_follow.py             # 提案A: ATR正規化KAMA傾き + ADX + VWAP 順張り戦略
│   └── mtf_regime.py               # 提案A+: 1分足KAMA ＋ 5分足EMA (MTF) ＋ RVOL ＋ 時間帯レジーム統合戦略
│
├── ml/                             # 提案B: 機械学習 (Meta-labeling) モジュール
│   ├── __init__.py
│   ├── labeler.py                  # Triple Barrier Method (López de Prado 2018) ラベリング
│   ├── feature_builder.py          # 定常化12次元特徴量（KAMA傾き/ATR, VWAP乖離率, ADX, RVOL, 5分足傾き等）抽出
│   ├── cross_validation.py         # Purged Group TimeSeries Split (Purging & Embargo 付きCV)
│   └── meta_labeling.py            # LightGBM Meta-labeling モデル (CPU/GPU 自動切り替え対応)
│
└── backtester/                     # イベント駆動型バックテストエンジン
    ├── __init__.py
    ├── data_handler.py             # Parquetデータロード・ネストディレクトリ対応・株式分割自動調整
    ├── portfolio.py                # ポジション管理・損益計算・バーバイバー評価
    ├── engine.py                   # Bar-by-Bar シミュレーションエンジン
    └── report.py                   # Sharpe Ratio, Sortino Ratio, Profit Factor, Max Drawdown レポート出力
```

---

## 🛠️ 各モジュールの詳細仕様

### 1. Indicators (`src_v2/indicators/`)
- **`calculate_atr(df, period=14)`**: 1分足の動的ボラティリティを計測。
- **`calculate_kama(series, er_period=10, fast_sc=2, slow_sc=30)`**: ノイズを排除する適応型移動平均。
- **`calculate_adx(df, period=14)`**: トレンド強度を判定。
- **`calculate_vwap(df)`**: 東証の9:00および12:30にセッション累計をリセットするVWAP。
- **`calculate_rvol(df, window=20)`**: 直近出来高 / 過去20期間平均出来高。出来高スパイク（大口参入）を検知。

### 2. Utils (`src_v2/utils/`)
- **`get_tick_size(price)`**: 東証の基準に応じた最小呼値 (1 tick) を返却（スリッページ計算用）。
- **`get_time_regime(dt)`**: 寄り付き (9:00-9:30)、日中 (9:30-15:00)、引け前 (15:00-15:30) などのレジーム判定。

### 3. Risk Management (`src_v2/risk/`)
- **`calculate_sl_tp(...)`**: `Entry Price ± (N * ATR)` で損切り・利確価格を算出。
- **`update_trailing_stop(...)`**: 保有期間中の最高値/最安値に追従してSL価格を繰り上げ/繰り下げ。
- **`calculate_position_size(...)`**: 許容損失額から単元（100株単位）の最適発注株数を計算。

### 4. Strategy (`src_v2/strategy/`)
- **`AdvancedTrendFollowStrategy`**: `normalized_slope = (KAMA_diff) / ATR` による全銘柄共通スケーリング順張り。
- **`AdvancedMTFRegimeStrategy`**: 1分足KAMA ＋ 5分足EMAの方向一致、RVOLスパイク、時間帯レジームに応じた動的フィルター適用。

### 5. Machine Learning (`src_v2/ml/`)
- **`apply_triple_barrier_labels(...)`**: 一次モデルシグナルに対し、未来15分間の利確/損切/時間期限で二次ターゲット (`1`: 成功, `0`: ダマシ) を付与。
- **`build_ml_features(df)`**: 定常化された12次元の特徴量行列を生成。
- **`PurgedGroupTimeSeriesSplit`**: 時系列の自己相関やオーバーラップによる漏洩（Look-ahead bias）をパージング＆エンバーゴで防止。
- **`MetaLabelingModel`**: LightGBMを用いた二次分類モデル。ColabやKaggle等のGPU環境では `device="gpu"` で全開動作。

### 6. Backtester (`src_v2/backtester/`)
- **`BacktestEngine`**: 1分ごとの Bar-by-Bar シミュレーション。呼値スリッページおよび15:29強制決済を厳密処理。
- **`calculate_performance_metrics(...)`**: 勝率, Profit Factor, Max Drawdown, Sharpe Ratio, Sortino Ratio を算出。

---

## 🚀 スクリプト連携

`src_v2` パッケージは、プロジェクト直下の `scripts/` ディレクトリ内の各種スクリプトから利用されます：

- **`scripts/run_backtest_v2.py --all`**: `src_v2/backtester` を用いて一括バックテストを実行
- **`scripts/export_ml_dataset.py`**: `src_v2/ml` を用いて `ml_dataset.parquet` をエクスポート
- **`scripts/train_meta_model.py`**: `src_v2/ml/meta_labeling.py` を用いてモデルを学習
