import numpy as np
import pandas as pd
import sys, pathlib
sys.dont_write_bytecode = True
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

import config as cf

if len(sys.argv) < 3:
    print("使用方法: python train.py <data_file> <output_dir>")
    print("例: python train.py data_2016_2025.csv output/")
    sys.exit(1)

data_file_path = sys.argv[1]
output_dir = pathlib.Path(sys.argv[2])
if not output_dir.exists():
    output_dir.mkdir(parents=True)

# データの読み込み
print("データを読み込み中...")
data_file = pd.read_csv(data_file_path)
print(f"データサイズ: {data_file.shape}")

# 入力変数と出力変数の設定
int_var = ["month", "day","land_atmosphere", "sea_atmosphere", "precipitation", "temperature", "humidity", "wind_speed", "wind_direction", "sum_insolation", "sum_sunlight", "snow_falling", "melted_snow"] #入力
out_var = data_file["snow_depth"] #出力

# 入力/出力変数をデータフレーム化
x = data_file[int_var]
y = out_var

print(f"入力特徴量: {int_var}")
print(f"入力データ形状: {x.shape}")
print(f"出力データ形状: {y.shape}")

# データの分割
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# データの標準化
scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_train_scaled = scaler_x.fit_transform(x_train)
x_test_scaled = scaler_x.transform(x_test)

y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

x_train_tensor = torch.FloatTensor(x_train_scaled)
x_test_tensor = torch.FloatTensor(x_test_scaled)
y_train_tensor = torch.FloatTensor(y_train_scaled)
y_test_tensor = torch.FloatTensor(y_test_scaled)

model = cf.neuralnetwork(input_size=len(int_var))
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print("学習を開始します...")
train_losses = []
test_losses = []

for epoch in range(cf.epochSize):
    model.train()
    
    optimizer.zero_grad()
    train_pred = model(x_train_tensor).flatten()
    train_loss = criterion(train_pred, y_train_tensor)
    train_loss.backward()
    optimizer.step()
    
    model.eval()
    with torch.no_grad():
        test_pred = model(x_test_tensor).flatten()
        test_loss = criterion(test_pred, y_test_tensor)
    
    train_losses.append(train_loss.item())
    test_losses.append(test_loss.item())
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{cf.epochSize}], Train Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

print("学習完了")

model.eval()
with torch.no_grad():
    y_train_pred_scaled = model(x_train_tensor).numpy().flatten()
    y_test_pred_scaled = model(x_test_tensor).numpy().flatten()
    
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"\n=== 評価結果 ===")
print(f"Train MSE: {train_mse:.3f}")
print(f"Test MSE: {test_mse:.3f}")
print(f"Train RMSE: {train_rmse:.3f}")
print(f"Test RMSE: {test_rmse:.3f}")
print(f"Train R²: {train_r2:.3f}")
print(f"Test R²: {test_r2:.3f}")

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.grid(True)
plt.savefig(output_dir / "loss_curve.png")
plt.close()

model_path = output_dir / "model.pth"
checkpoint = {
    'model_state_dict': model.state_dict(),
    'scaler_x': scaler_x,
    'scaler_y': scaler_y,
    'input_features': int_var,
    'train_mse': train_mse,
    'test_mse': test_mse,
    'train_rmse': train_rmse,
    'test_rmse': test_rmse,
    'train_r2': train_r2,
    'test_r2': test_r2
}

torch.save(checkpoint, model_path)
print(f"モデルを {model_path} に保存しました")
print(f"損失曲線を {output_dir / 'loss_curve.png'} に保存しました")
