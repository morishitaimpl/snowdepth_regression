import pandas as pd
import matplotlib.pyplot as plt
import sys, os, pathlib

file_path = pd.read_csv(sys.argv[1])
save_dir = pathlib.Path(sys.argv[2])
if(not save_dir.exists()): save_dir.mkdir()

# 入力変数と出力変数の設定
int_var = ["precipitation", "temperature", "humidity", "snow_falling"] #入力
out_var = ["snow_depth"] #出力

# 入力/出力変数をデータフレーム化
x = file_path[int_var]
x.head()
y = file_path[out_var]
y.head()

plt.xlabel('int_var')
plt.ylabel('out_var')
plt.plot(x, y, label=int_var)
plt.legend()
plt.savefig(save_dir / "plot.png")