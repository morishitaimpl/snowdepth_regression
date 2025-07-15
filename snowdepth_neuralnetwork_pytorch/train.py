import numpy as np
import pandas as pd
import sys, pathlib
sys.dont_write_bytecode = True
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import config as cf

if len(sys.argv) < 2:
    print("使用方法: python train.py <data_file.csv> [output_dir]")
    sys.exit(1)

data_file_path = sys.argv[1]
output_dir = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path("output")

if not output_dir.exists():
    output_dir.mkdir(parents=True)

# データの読み込み
print("データを読み込み中...")
data_file = pd.read_csv(data_file_path)
print(f"データサイズ: {data_file.shape}")

# 入力変数と出力変数の設定
int_var = ["month", "day","land_atmosphere", "sea_atmosphere", "precipitation", "temperature", "humidity", "wind_speed", "wind_direction", "sum_insolation", "sum_sunlight", "snow_falling", "melted_snow"]
out_var = "snow_depth"

# 入力/出力変数をデータフレーム化
x = data_file[int_var].values
y = data_file[out_var].values.reshape(-1, 1)

# データの分割
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# データの標準化
scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_train_scaled = scaler_x.fit_transform(x_train)
x_test_scaled = scaler_x.transform(x_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

x_train_tensor = torch.FloatTensor(x_train_scaled)
y_train_tensor = torch.FloatTensor(y_train_scaled)
x_test_tensor = torch.FloatTensor(x_test_scaled)
y_test_tensor = torch.FloatTensor(y_test_scaled)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=cf.batchSize, shuffle=True)

model = cf.neuralnetwork(input_size=len(int_var))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("学習を開始...")
train_losses = []

for epoch in range(cf.epochSize):
    model.train()
    epoch_loss = 0.0
    
    for batch_x, batch_y in train_loader:
        # batch_x_noisy = cf.add_noise(batch_x, noise_level=0.01)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{cf.epochSize}], Loss: {avg_loss:.6f}")

plt.figure(figsize=(10, 6))
plt.plot(train_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig(output_dir / "train_loss.png")
plt.close()

model.eval()
with torch.no_grad():
    y_train_pred_scaled = model(x_train_tensor).numpy()
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)
    
    y_test_pred_scaled = model(x_test_tensor).numpy()
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\n=== 評価結果 ===")
print(f"Train RMSE: {train_rmse:.3f}")
print(f"Test RMSE: {test_rmse:.3f}")
print(f"Train R²: {train_r2:.3f}")
print(f"Test R²: {test_r2:.3f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
plt.xlabel("実際の積雪深")
plt.ylabel("予測積雪深")
plt.title(f"訓練データ (R² = {train_r2:.3f})")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("実際の積雪深")
plt.ylabel("予測積雪深")
plt.title(f"テストデータ (R² = {test_r2:.3f})")
plt.grid(True)

plt.tight_layout()
plt.savefig(output_dir / "prediction_results.png")
plt.close()

torch.save({
    'model_state_dict': model.state_dict(),
    'scaler_x': scaler_x,
    'scaler_y': scaler_y,
    'input_features': int_var,
    'train_rmse': train_rmse,
    'test_rmse': test_rmse,
    'train_r2': train_r2,
    'test_r2': test_r2
}, output_dir / "model.pth")

print(f"\nモデルと結果を {output_dir} に保存しました。")
