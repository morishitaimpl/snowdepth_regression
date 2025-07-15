import numpy as np
import pandas as pd
import sys
import pathlib
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

import config as cf

def load_model(model_path):
    """学習済みモデルとスケーラーを読み込む"""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    model = cf.SnowDepthPredictor(input_size=13)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scaler_x = checkpoint['scaler_x']
    scaler_y = checkpoint['scaler_y']
    input_features = checkpoint['input_features']
    
    return model, scaler_x, scaler_y, input_features

def preprocess_data(data, input_features, scaler_x):
    """入力データの前処理"""
    x = data[input_features].values
    x_scaled = scaler_x.transform(x)
    x_tensor = torch.FloatTensor(x_scaled)
    return x_tensor

def make_predictions(model, x_tensor, scaler_y):
    """予測を実行"""
    with torch.no_grad():
        y_pred_scaled = model(x_tensor).numpy()
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
    return y_pred.flatten()

def evaluate_predictions(y_true, y_pred):
    """予測結果の評価"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"平均二乗誤差 (MSE): {mse:.3f}")
    print(f"平均平方根誤差 (RMSE): {rmse:.3f}")
    print(f"決定係数 (R²): {r2:.3f}")
    
    return mse, rmse, r2

def plot_predictions(y_true, y_pred, output_path=None):
    """予測結果の可視化"""
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('実際の積雪深')
    plt.ylabel('予測積雪深')
    plt.title('実際値 vs 予測値')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('予測積雪深')
    plt.ylabel('残差')
    plt.title('残差プロット')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.hist(residuals, bins=30, alpha=0.7)
    plt.xlabel('残差')
    plt.ylabel('頻度')
    plt.title('残差の分布')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(y_true, label='実際値', alpha=0.7)
    plt.plot(y_pred, label='予測値', alpha=0.7)
    plt.xlabel('サンプル番号')
    plt.ylabel('積雪深')
    plt.title('時系列比較')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"予測結果のグラフを {output_path} に保存しました")
    else:
        plt.show()
    
    plt.close()

def save_predictions(data, y_pred, output_path):
    """予測結果をCSVファイルに保存"""
    result_df = data.copy()
    result_df['predicted_snow_depth'] = y_pred
    result_df.to_csv(output_path, index=False)
    print(f"予測結果を {output_path} に保存しました")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("使用方法: python predict_snowdepth.py <model_path> <data_path> [output_dir]")
        print("例: python predict_snowdepth.py output/model.pth data_2016_2025.csv results/")
        sys.exit(1)
    
    model_path = sys.argv[1]
    data_path = sys.argv[2]
    output_dir = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else pathlib.Path("evaluation_results")
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    print("=== 積雪深予測評価システム ===")
    print(f"モデル: {model_path}")
    print(f"データ: {data_path}")
    print(f"出力先: {output_dir}")
    print()
    
    print("学習済みモデルを読み込み中...")
    model, scaler_x, scaler_y, input_features = load_model(model_path)
    print(f"入力特徴量: {input_features}")
    print()
    
    # データの読み込み
    print("評価データを読み込み中...")
    data = pd.read_csv(data_path)
    print(f"データサイズ: {data.shape}")
    
    missing_features = [f for f in input_features if f not in data.columns]
    if missing_features:
        print(f"エラー: 以下の特徴量がデータに含まれていません: {missing_features}")
        sys.exit(1)
    
    if 'snow_depth' not in data.columns:
        print("警告: 'snow_depth'列が見つかりません。予測のみを実行します。")
        evaluate_mode = False
    else:
        evaluate_mode = True
    
    print("データを前処理中...")
    x_tensor = preprocess_data(data, input_features, scaler_x)
    
    # 予測の実行
    print("予測を実行中...")
    y_pred = make_predictions(model, x_tensor, scaler_y)
    print(f"予測完了: {len(y_pred)}件")
    print()
    
    if evaluate_mode:
        y_true = data['snow_depth'].values
        print("=== 評価結果 ===")
        mse, rmse, r2 = evaluate_predictions(y_true, y_pred)
        print()
        
        plot_path = output_dir / "prediction_evaluation.png"
        plot_predictions(y_true, y_pred, plot_path)
    else:
        print("実際値がないため、評価をスキップします。")
    
    csv_path = output_dir / "predictions.csv"
    save_predictions(data, y_pred, csv_path)
    
    print("\n=== 予測統計 ===")
    print(f"予測値の平均: {np.mean(y_pred):.3f}")
    print(f"予測値の標準偏差: {np.std(y_pred):.3f}")
    print(f"予測値の最小値: {np.min(y_pred):.3f}")
    print(f"予測値の最大値: {np.max(y_pred):.3f}")
    
    print(f"\n評価完了。結果は {output_dir} に保存されました。")
