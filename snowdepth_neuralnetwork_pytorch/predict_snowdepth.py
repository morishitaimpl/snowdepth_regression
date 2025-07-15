import numpy as np
import sys
import pathlib
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime

import config as cf

if len(sys.argv) < 2:
    print("使用方法: python predict_snowdepth.py <model_path> [output_dir]")
    print("例: python predict_snowdepth.py output/model.pth results/")
    sys.exit(1)

model_path = sys.argv[1]
output_dir = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path("evaluation_results")

if not output_dir.exists():
    output_dir.mkdir(parents=True)

print("=== 積雪深予測評価システム ===")
print(f"モデル: {model_path}")
print(f"出力先: {output_dir}")
print(f"エポック数設定: {cf.epochSize}")
print(f"バッチサイズ設定: {cf.batchSize}")
print()

print("学習済みモデルを読み込み中...")
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

input_size = len(checkpoint['input_features']) if 'input_features' in checkpoint else 13
model = cf.neuralnetwork(input_size=input_size)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

scaler_x = checkpoint['scaler_x']
scaler_y = checkpoint['scaler_y']
input_features = checkpoint['input_features']

print(f"入力特徴量数: {len(input_features)}")
print(f"入力特徴量: {input_features}")
print()

print("=== 気象データ入力 ===")
print("以下の項目を順番に入力してください:")

input_data = []
for i, feature in enumerate(input_features):
    while True:
        try:
            value = float(input(f"[{i+1}/{len(input_features)}] {feature}: "))
            input_data.append(value)
            break
        except ValueError:
            print("エラー: 数値を入力してください。")

print()

print("データを前処理中...")
x_array = np.array(input_data).reshape(1, -1)
x_scaled = scaler_x.transform(x_array)
x_tensor = torch.FloatTensor(x_scaled)

print(f"入力データ形状: {x_array.shape}")
print(f"標準化後データ形状: {x_scaled.shape}")

# 予測の実行
print("予測を実行中...")
with torch.no_grad():
    y_pred_scaled = model(x_tensor).numpy()
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

predicted_snow_depth = y_pred.flatten()[0]
print("予測完了")
print()

print("=== 予測結果 ===")
print(f"予測積雪深: {predicted_snow_depth:.3f} cm")
print()

actual_value_input = input("実際の積雪深がわかる場合は入力してください（スキップする場合はEnter）: ")
evaluation_performed = False

if actual_value_input.strip():
    try:
        actual_snow_depth = float(actual_value_input)
        evaluation_performed = True
        
        error = abs(actual_snow_depth - predicted_snow_depth)
        mse = (actual_snow_depth - predicted_snow_depth) ** 2
        rmse = np.sqrt(mse)
        
        print("=== 評価結果 ===")
        print(f"実際の積雪深: {actual_snow_depth:.3f} cm")
        print(f"予測積雪深: {predicted_snow_depth:.3f} cm")
        print(f"絶対誤差: {error:.3f} cm")
        print(f"二乗誤差: {mse:.3f}")
        print(f"平方根誤差: {rmse:.3f}")
        
        plt.figure(figsize=(8, 6))
        plt.bar(['実際値', '予測値'], [actual_snow_depth, predicted_snow_depth], 
               color=['blue', 'red'], alpha=0.7)
        plt.ylabel('積雪深 (cm)')
        plt.title('実際値 vs 予測値')
        plt.grid(True, alpha=0.3)
        
        plot_path = output_dir / "prediction_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"比較グラフを {plot_path} に保存しました")
        plt.close()
        
    except ValueError:
        print("エラー: 数値の入力が無効でした。評価をスキップします。")

result_data = {
    'model_path': model_path,
    'prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'input_features': input_features,
    'input_values': input_data,
    'predicted_snow_depth': predicted_snow_depth
}

if evaluation_performed:
    result_data['actual_snow_depth'] = actual_snow_depth
    result_data['absolute_error'] = error
    result_data['mse'] = mse
    result_data['rmse'] = rmse

result_path = output_dir / "prediction_result.txt"
with open(result_path, 'w', encoding='utf-8') as f:
    f.write("=== 積雪深予測結果 ===\n")
    f.write(f"モデル: {result_data['model_path']}\n")
    f.write(f"予測日時: {result_data['prediction_time']}\n")
    f.write(f"使用設定 - エポック数: {cf.epochSize}, バッチサイズ: {cf.batchSize}\n\n")
    
    f.write("入力データ:\n")
    for feature, value in zip(result_data['input_features'], result_data['input_values']):
        f.write(f"  {feature}: {value}\n")
    
    f.write(f"\n予測積雪深: {result_data['predicted_snow_depth']:.3f} cm\n")
    
    if 'actual_snow_depth' in result_data:
        f.write(f"実際の積雪深: {result_data['actual_snow_depth']:.3f} cm\n")
        f.write(f"絶対誤差: {result_data['absolute_error']:.3f} cm\n")
        f.write(f"二乗誤差: {result_data['mse']:.3f}\n")
        f.write(f"平方根誤差: {result_data['rmse']:.3f}\n")

print(f"予測結果を {result_path} に保存しました")
print(f"\n評価完了。結果は {output_dir} に保存されました。")
