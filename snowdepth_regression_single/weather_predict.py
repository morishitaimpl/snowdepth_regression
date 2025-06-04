#単回帰分析
import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
import sys, os, pathlib
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def predict_snow_depth(snow_fall_amount, model):
    """降雪量から積雪深を予測する関数"""
    # 列名を含むDataFrameとして予測データを作成
    pred_data = pd.DataFrame([[snow_fall_amount]], columns=['snow_falling'])
    pred = model.predict(pred_data)
    return pred[0][0]

weather_data_path = sys.argv[1]
# out_dir_path = pathlib.Path(sys.argv[2]) 
# if not os.path.exists(out_dir_path): os.mkdir(out_dir_path)

df = pd.read_csv(weather_data_path) #データ読み込み
# display(df.head())

# heat_map = df.corr(method="pearson")
# sns.heatmap(heat_map, center=0, square = True, annot=True, cmap="OrRd",fmt="1.1f") #ヒートマップ
# plt_img = plt.savefig(out_dir_path/ 'heat_map.png', bbox_inches="tight") #画像は19行目で指定したフォルダ名に入ります。

int_var = "snow_falling" #入力
out_var = "snow_depth" #出力

#入力/出力変数をデータフレーム化
x = df[[int_var]]
y = df[[out_var]]

#線形単回帰モデル
model = LinearRegression()
model.fit(x, y)

print('回帰直線の切片', model.intercept_)
print('回帰係数', model.coef_)
print('決定係数', model.score(x, y))
print('積雪深（式）', f'y = {model.coef_[0][0]:.2f}x + {model.intercept_[0]:.2f}')

# 予測の実行
while True:
    try:
        snow_fall = float(input("降雪量を入力してください（cm）: "))
        predicted_depth = predict_snow_depth(snow_fall, model)
        print(f"予測される積雪深: {predicted_depth:.2f} cm")
    except ValueError:
        print("数値を入力してください")
    except KeyboardInterrupt:
        break
    # 処理を終えたい時は、ctl+"c"キーで終えることができます。