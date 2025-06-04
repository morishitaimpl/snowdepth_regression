# ニューラルネットワーク
from sklearn.neural_network import MLPRegressor
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from IPython.display import display
import sys, os, pathlib
import numpy as np

# データの読み込み
file = pd.read_csv(sys.argv[1])
# display(df.head())
# df.shape
out_dir = pathlib.Path(sys.argv[2])
if(not out_dir.exists()): out_dir.mkdir()


# 入力変数と出力変数の設定
int_var = ["month", "day","land_atmosphere", "sea_atmosphere", "precipitation", "temperature", "humidity", "wind_speed", "wind_direction", "snow_falling", "melted_snow"] #入力
out_var = file["snow_depth"] #出力

# 入力/出力変数をデータフレーム化
x = file[int_var]
x.head()
y = out_var
y.head()

# データの分割
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# データの標準化
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
# print(x_train_scaled)
# print(x_test_scaled)

epochSize = 500

# ニューラルネットワーク
model = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation="relu", solver="adam", alpha=0.002, learning_rate_init=0.001, random_state=60, early_stopping=True, max_iter=epochSize)
model.fit(x_train_scaled, y_train)
train_score = model.score(x_train_scaled, y_train)
test_score = model.score(x_test_scaled, y_test)
print(f"train_score : {train_score:.3f}")
print(f"test_score : {test_score:.3f}")

#train_loss
plt.plot(model.loss_curve_)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig(out_dir / "train_loss.png")

# 予測
y_train_pred = model.predict(x_train_scaled)
y_test_pred = model.predict(x_test_scaled)

# MSEの計算
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
# print(f"train_mse : {train_mse:.3f}")
# print(f"test_mse : {test_mse:.3f}")

#RMSEの計算
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
print(f"train_rmse : {train_rmse:.3f}")
print(f"test_rmse : {test_rmse:.3f}")