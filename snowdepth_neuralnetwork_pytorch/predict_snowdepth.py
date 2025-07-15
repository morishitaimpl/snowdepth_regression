import numpy as np
import sys
import pathlib
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

import config as cf

if __name__ == "__main__":
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
    print()
    
    print("学習済みモデルを読み込み中...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    model = cf.neuralnetwork(input_size=cf.input_size if hasattr(cf, 'input_size') else 13)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scaler_x = checkpoint['scaler_x']
    scaler_y = checkpoint['scaler_y']
    input_features = checkpoint['input_features']
    
    print(f"入力特徴量: {input_features}")
    print()
    
    print("=== 気象データ入力 ===")
    print("以下の項目を順番に入力してください:")
    
    input_data = []
    for feature in input_features:
        while True:
            try:
                value = float(input(f"{feature}: "))
                input_data.append(value)
                break
            except ValueError:
                print("数値を入力してください。")
    
    print()
    
    print("データを前処理中...")
    x_array = np.array(input_data).reshape(1, -1)
    x_scaled = scaler_x.transform(x_array)
    x_tensor = torch.FloatTensor(x_scaled)
    
    # 予測の実行
    print("予測を実行中...")
    with torch.no_grad():
        y_pred_scaled = model(x_tensor).numpy()
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    predicted_snow_depth = y_pred.flatten()[0]
    print(f"予測完了")
    print()
    
    print("=== 予測結果 ===")
    print(f"予測積雪深: {predicted_snow_depth:.3f} cm")
    print()
    
    actual_value_input = input("実際の積雪深がわかる場合は入力してください（スキップする場合はEnter）: ")
    if actual_value_input.strip():
        try:
            actual_snow_depth = float(actual_value_input)
            
            mse = (actual_snow_depth - predicted_snow_depth) ** 2
            rmse = np.sqrt(mse)
            
            print("=== 評価結果 ===")
            print(f"実際の積雪深: {actual_snow_depth:.3f} cm")
            print(f"予測積雪深: {predicted_snow_depth:.3f} cm")
            print(f"誤差: {abs(actual_snow_depth - predicted_snow_depth):.3f} cm")
            print(f"二乗誤差: {mse:.3f}")
            print(f"平方根誤差: {rmse:.3f}")
            
            plt.figure(figsize=(8, 6))
            plt.bar(['実際値', '予測値'], [actual_snow_depth, predicted_snow_depth], 
                   color=['blue', 'red'], alpha=0.7)
            plt.ylabel('積雪深 (cm)')
            plt.title('実際値 vs 予測値')
            plt.grid(True, alpha=0.3)
            
            plot_path = output_dir / "prediction_comparison.png"
            plt.savefig(plot_path)
            print(f"比較グラフを {plot_path} に保存しました")
            plt.close()
            
        except ValueError:
            print("数値の入力が無効でした。評価をスキップします。")
    
    result_data = {
        'input_features': input_features,
        'input_values': input_data,
        'predicted_snow_depth': predicted_snow_depth
    }
    
    if actual_value_input.strip():
        try:
            result_data['actual_snow_depth'] = float(actual_value_input)
            result_data['error'] = abs(float(actual_value_input) - predicted_snow_depth)
        except ValueError:
            pass
    
    result_path = output_dir / "prediction_result.txt"
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("=== 積雪深予測結果 ===\n")
        f.write(f"モデル: {model_path}\n")
        f.write(f"予測日時: {pathlib.Path().cwd()}\n\n")
        
        f.write("入力データ:\n")
        for feature, value in zip(input_features, input_data):
            f.write(f"  {feature}: {value}\n")
        
        f.write(f"\n予測積雪深: {predicted_snow_depth:.3f} cm\n")
        
        if 'actual_snow_depth' in result_data:
            f.write(f"実際の積雪深: {result_data['actual_snow_depth']:.3f} cm\n")
            f.write(f"誤差: {result_data['error']:.3f} cm\n")
    
    print(f"予測結果を {result_path} に保存しました")
    print(f"\n評価完了。結果は {output_dir} に保存されました。")
