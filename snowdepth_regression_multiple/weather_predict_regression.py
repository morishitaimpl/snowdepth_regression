#重回帰分析
import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import sys, os
import numpy as np

def predict_snow_depth(model, scaler, input_data):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    return prediction[0][0]

# データの読み込み
df = pd.read_csv(sys.argv[1])
# display(df.head())

# 入力変数と出力変数の設定
int_var = ["month", "day", "atmosphere", "temperature", "humidity", "precipitation", "snow_falling"] #入力
out_var = ["snow_depth"] #出力

# 入力/出力変数をデータフレーム化
x = df[int_var]
x.head()
y = df[out_var]
y.head()

# データの分割
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# データの標準化
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 回帰モデルの学習
model = LinearRegression()
model.fit(x_train_scaled, y_train)

# モデルの評価
print("\nモデルの評価:")
print("訓練データの決定係数: {:.3f}".format(model.score(x_train_scaled, y_train)))
print("テストデータの決定係数: {:.3f}".format(model.score(x_test_scaled, y_test)))

# 予測値の計算
y_pred_train = model.predict(x_train_scaled)
y_pred_test = model.predict(x_test_scaled)

# MSEの計算 #平均2乗誤差
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
print("\nMSE:")
print("訓練データMSE: {:.3f}".format(train_mse))
print("テストデータMSE: {:.3f}".format(test_mse))

#RMSEの計算 #平方根
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
print("\nRMSE:")
print("訓練データRMSE: {:.3f}".format(train_rmse))
print("テストデータRMSE: {:.3f}".format(test_rmse))
    
# 決定係数
print("決定係数: {:.3f}".format(model.score(x_test_scaled, y_test)))
w = pd.Series(index=int_var, data=model.coef_[0])
np.linalg.norm(w, ord=2) # 重みベクトルwの L2ノルム

# 予測値の表示
y_pred = model.predict(x_train_scaled)
y_pred[1:10] # 予測値の上位10件を表示

# 切片と回帰係数の表示
print("\nパラメーター:")
print("切片: {:.3f}".format(model.intercept_[0]))
print("\n各変数の回帰係数:")
for var, coef in zip(int_var, model.coef_[0]):
    print(f"{var}: {coef:.3f}")

while True:
    try:
        # 入力データの収集
        print("\n気象データを入力してください:")
        month = int(input("月 (1-12): "))
        day = int(input("日 (1-31): "))
        atmosphere = float(input("気圧 (hPa): "))
        temperature = float(input("気温 (°C): "))
        humidity = float(input("湿度 (%): "))
        precipitation = float(input("降水量 (mm): "))
        snow_falling = float(input("降雪量 (cm): "))
        
        # 入力データの辞書を作成
        input_data = {
            "month": month,
            "day": day,
            "atmosphere": atmosphere,
            "temperature": temperature,
            "humidity": humidity,
            "precipitation": precipitation,
            "snow_falling": snow_falling
        }
        
        # 予測の実行
        predicted_depth = predict_snow_depth(model, scaler, input_data)
        print(f"\n予測された積雪深: {predicted_depth:.2f} cm")
    
    except ValueError:
        print("入力が正しくありません。数値を入力してください。")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
