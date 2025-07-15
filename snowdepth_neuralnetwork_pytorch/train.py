import numpy as np
import sys, pathlib
sys.dont_write_bytecode = True
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import config as cf

data_file = sys.argv[1]

# 入力変数と出力変数の設定
int_var = ["month", "day","land_atmosphere", "sea_atmosphere", "precipitation", "temperature", "humidity", "wind_speed", "wind_direction", "sum_insolation", "sum_sunlight", "snow_falling", "melted_snow"] #入力
out_var = data_file["snow_depth"] #出力

# 入力/出力変数をデータフレーム化
x = data_file[int_var]
y = out_var

# データの分割
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# データの標準化
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = cf.neuralnetwork()