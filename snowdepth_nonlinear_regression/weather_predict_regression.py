#非線形回帰分析
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt
from IPython.display import display
import sys, os
import numpy as np

def predict_snow_depth(model, scaler, poly, input_data):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    input_poly = poly.transform(input_scaled)
    prediction = model.predict(input_poly)
    return prediction[0][0]

# データの読み込み
df = pd.read_csv(sys.argv[1])
display(df.head())
# df.shape

# 入力変数と出力変数の設定
int_var = ["precipitation", "temperature", "humidity", "snow_falling"] #入力
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

# 非線形特徴量の生成
poly = PolynomialFeatures(degree=3, include_bias=False)
x_train_poly = poly.fit_transform(x_train_scaled)
x_test_poly = poly.transform(x_test_scaled)

# 線形重回帰モデル（非線形特徴量を使用）
model = LinearRegression()
model.fit(x_train_poly, y_train)

# モデルの評価
print("\nモデルの評価:")
print("訓練データの決定係数: {:.3f}".format(model.score(x_train_poly, y_train)))
print("テストデータの決定係数: {:.3f}".format(model.score(x_test_poly, y_test)))

# 予測値の計算
y_pred_train = model.predict(x_train_poly)
y_pred_test = model.predict(x_test_poly)

# MSEの計算
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
print("\nMSE:")
print("訓練データMSE: {:.3f}".format(train_mse))
print("テストデータMSE: {:.3f}".format(test_mse))

# RMSEの計算
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
print("\nRMSE:")
print("訓練データRMSE: {:.3f}".format(train_rmse))
print("テストデータRMSE: {:.3f}".format(test_rmse))


# 特徴量の重要度を表示
# print("\n回帰モデルの係数:")
feature_names = poly.get_feature_names_out(int_var)
feature_importance = pd.DataFrame({
    '特徴量': feature_names,
    '係数': model.coef_[0]
})
# 係数の絶対値でソートし、上位10件を表示
top_features = feature_importance.sort_values('係数', key=abs, ascending=False).head(10)
# print("\n上位10件の特徴量と係数:")
# print(top_features.to_string(index=False))

# 切片の表示
print("\nパラメータ-")
print(f"切片: {model.intercept_[0]:.4f}")

# モデルの性能指標
# print("\nモデルの性能指標:")
metrics_df = pd.DataFrame({
    '指標': ['訓練データの決定係数', 'テストデータの決定係数', '訓練データRMSE', 'テストデータRMSE'],
    '値': [model.score(x_train_poly, y_train), 
           model.score(x_test_poly, y_test),
           train_rmse,
           test_rmse]
})
# print(metrics_df.to_string(index=False))

while True:
    try:
        # 入力データの収集
        print("\n気象データを入力してください:")
        precipitation = float(input("降水量 (mm): "))
        temperature = float(input("気温 (°C): "))
        humidity = float(input("湿度 (%): "))
        snow_falling = float(input("降雪量 (cm): "))
        
        # 入力データの辞書を作成
        input_data = {
            "precipitation": precipitation,
            "temperature": temperature,
            "humidity": humidity,
            "snow_falling": snow_falling
        }

        # 予測の実行
        predicted_depth = predict_snow_depth(model, scaler, poly, input_data)

        # 気温が10度以上の時積雪深が０
        if temperature >= 10:
            predicted_depth = 0
        
        print(f"\n予測された積雪深: {predicted_depth:.2f} cm")

    except ValueError:
        print("入力が正しくありません。数値を入力してください。")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")


