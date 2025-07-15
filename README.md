# snowdepth_regression

気象データから積雪深を予測する機械学習システムです。統計的手法と機械学習手法の両方を提供し、異なるアプローチで積雪深の予測を行うことができます。

## 概要

このリポジトリは、気象観測データ（気温、湿度、降水量、風速など）を用いて積雪深を予測する5つの異なる手法を実装しています。研究者や気象予報士、データサイエンティストが雪の蓄積予測に使用できる包括的なツールセットです。

### 主な機能
- **複数の予測手法の比較**: 統計的手法（単回帰・重回帰・非線形回帰）と機械学習手法（ニューラルネットワーク）
- **異なるフレームワークの実験**: scikit-learnとPyTorchの両方でのニューラルネットワーク実装
- **実際の気象データ処理**: CSV形式の気象観測データと積雪深データの処理
- **インタラクティブな予測**: コマンドライン経由でのリアルタイム積雪深予測
- **モデル性能評価**: 組み込みの評価指標と可視化ツール

## プロジェクト構成

### 統計的手法

#### 1. 単回帰分析 (`snowdepth_regression_single/`)
- **ファイル**: `weather_predict.py`
- **手法**: 降雪量のみを使用した線形単回帰
- **入力変数**: 降雪量（snow_falling）
- **データセット**: `snowdepth_r060410.csv` (338レコード、6変数)
- **使用方法**: `python weather_predict.py snowdepth_r060410.csv`
- **特徴**: 最もシンプルな予測手法、降雪量と積雪深の直接的な関係を分析

#### 2. 重回帰分析 (`snowdepth_regression_multiple/`)
- **ファイル**: `weather_predict_regression.py`
- **手法**: 複数の気象変数を使用した線形重回帰
- **入力変数**: 月、日、気圧、気温、湿度、降水量、降雪量（7変数）
- **データセット**: `snowdepth_r060410.csv`
- **使用方法**: `python weather_predict_regression.py snowdepth_r060410.csv`
- **特徴**: データの標準化、訓練/テスト分割、MSE・RMSE・R²による評価

#### 3. 非線形回帰分析 (`snowdepth_nonlinear_regression/`)
- **ファイル**: `weather_predict_regression.py`
- **手法**: 多項式特徴量（3次）を使用した非線形回帰
- **入力変数**: 降水量、気温、湿度、降雪量（4変数）
- **データセット**: `data_2023_2024.csv` (733レコード、5変数)
- **使用方法**: `python weather_predict_regression.py data_2023_2024.csv`
- **特徴**: 非線形関係の捕捉、気温10度以上で積雪深0の条件付きロジック
- **付属ツール**: `list_plot.py` - データ可視化ユーティリティ

### 機械学習手法

#### 4. scikit-learn ニューラルネットワーク (`snowdepth_neuralnetwork/`)
- **ファイル**: `snowdepth.py`
- **手法**: MLPRegressorを使用したニューラルネットワーク
- **アーキテクチャ**: 11入力 → 128 → 64 → 32 → 1出力
- **入力変数**: 月、日、陸上気圧、海上気圧、降水量、気温、湿度、風速、風向、降雪量、融雪量（11変数）
- **データセット**: `data_2016_2025.csv` (3,408レコード、12変数)
- **使用方法**: `python snowdepth.py data_2016_2025.csv output_dir`
- **特徴**: 早期停止、学習曲線の可視化、RMSE・R²評価
- **付属ツール**: `wind_direction.py` - 日本語風向を数値角度に変換

#### 5. PyTorch ニューラルネットワーク (`snowdepth_neuralnetwork_pytorch/`)
- **ファイル**: `train.py`, `config.py`
- **手法**: PyTorchを使用したカスタムニューラルネットワーク
- **アーキテクチャ**: 13入力 → 128 → 64 → 32 → 1出力（ドロップアウト0.2付き）
- **入力変数**: 月、日、陸上気圧、海上気圧、降水量、気温、湿度、風速、風向、日照時間合計、日照時間、降雪量、融雪量（13変数）
- **データセット**: `data_2016_2025.csv` (3,408レコード、14変数)
- **使用方法**: `python train.py data_2016_2025.csv output_dir`
- **特徴**: Adam最適化、バッチ学習、モデル保存、予測結果の可視化
- **性能**: Train R²: 0.985, Test R²: 0.815, Test RMSE: 11.009

## データセット仕様

### 1. snowdepth_r060410.csv
- **レコード数**: 338
- **変数**: atmosphere, temperature, humidity, precipitation, snow_falling, snow_depth
- **用途**: 単回帰・重回帰分析

### 2. data_2023_2024.csv
- **レコード数**: 733
- **変数**: precipitation, temperature, humidity, snow_falling, snow_depth
- **用途**: 非線形回帰分析

### 3. data_2016_2025.csv
- **レコード数**: 3,408
- **変数**: month, day, land_atmosphere, sea_atmosphere, precipitation, temperature, humidity, wind_speed, wind_direction, sum_insolation, sum_sunlight, snow_falling, melted_snow, snow_depth
- **用途**: ニューラルネットワーク（scikit-learn・PyTorch）

## インストールと使用方法

### 依存関係のインストール

各ディレクトリには`library_install.txt`ファイルがあります：

```bash
# 統一インストール（推奨）
pip install pandas scikit-learn matplotlib seaborn numpy ipython torch

# または個別インストール
pip install -r snowdepth_regression_single/library_install.txt
pip install -r snowdepth_regression_multiple/library_install.txt
pip install -r snowdepth_nonlinear_regression/library_install.txt
pip install -r snowdepth_neuralnetwork/library_install.txt
```

### 実行例

```bash
# 単回帰分析
cd snowdepth_regression_single
python weather_predict.py snowdepth_r060410.csv

# 重回帰分析
cd snowdepth_regression_multiple
python weather_predict_regression.py snowdepth_r060410.csv

# 非線形回帰分析
cd snowdepth_nonlinear_regression
python weather_predict_regression.py data_2023_2024.csv

# scikit-learn ニューラルネットワーク
cd snowdepth_neuralnetwork
python snowdepth.py data_2016_2025.csv output_images

# PyTorch ニューラルネットワーク
cd snowdepth_neuralnetwork_pytorch
python train.py data_2016_2025.csv output_results
```

## 手法の選択指針

| 手法 | 精度 | 計算速度 | データ要件 | 解釈性 | 用途 |
|------|------|----------|------------|--------|------|
| 単回帰 | 低 | 高速 | 少 | 高 | 基本的な関係分析 |
| 重回帰 | 中 | 高速 | 中 | 高 | 線形関係の分析 |
| 非線形回帰 | 中-高 | 中速 | 中 | 中 | 非線形関係の捕捉 |
| scikit-learn NN | 高 | 中速 | 多 | 低 | 高精度予測 |
| PyTorch NN | 高 | 中速 | 多 | 低 | カスタム実装・研究 |

## ユーティリティツール

### wind_direction.py
日本語の風向表記を数値角度に変換します：
```bash
python wind_direction.py input_data.csv
# 出力: input_data_angle.csv
```

### list_plot.py
入力変数と積雪深の関係を可視化します：
```bash
python list_plot.py data_2023_2024.csv output_dir
```

## 技術仕様

### 入力変数の詳細
- **month, day**: 日付情報（1-12月、1-31日）
- **atmosphere / land_atmosphere / sea_atmosphere**: 気圧（hPa）
- **temperature**: 気温（°C）
- **humidity**: 湿度（%）
- **precipitation**: 降水量（mm）
- **wind_speed**: 風速（m/s）
- **wind_direction**: 風向（度、0-360）
- **sum_insolation**: 日射量合計
- **sum_sunlight**: 日照時間合計
- **snow_falling**: 降雪量（cm）
- **melted_snow**: 融雪量（cm）

### 出力変数
- **snow_depth**: 積雪深（cm）

## 開発・貢献

このプロジェクトは積雪深予測の研究・実用化を目的としています。各実装は独立しており、比較研究や手法の改良に活用できます。

### ディレクトリ構造
```
snowdepth_regression/
├── README.md                           # このファイル
├── requirements.txt                    # 統合依存関係
├── snowdepth_regression_single/        # 単回帰分析
├── snowdepth_regression_multiple/      # 重回帰分析
├── snowdepth_nonlinear_regression/     # 非線形回帰分析
├── snowdepth_neuralnetwork/            # scikit-learn NN
└── snowdepth_neuralnetwork_pytorch/    # PyTorch NN
```

各ディレクトリの詳細な使用方法は、それぞれのREADMEファイルをご参照ください。
