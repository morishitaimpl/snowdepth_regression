import pandas as pd
import csv, os, sys

csv = pd.read_csv(sys.argv[1])

# 風向きと角度の対応を辞書として定義
wind_direction_map = {
    "北": 0,
    "北北東": 22.5,
    "北東": 45,
    "東北東": 67.5,
    "東": 90,
    "東南東": 112.5,
    "南東": 135,
    "南南東": 157.5,
    "南": 180,
    "南南西": 202.5,
    "南西": 225,
    "西南西": 247.5,
    "西": 270,
    "西北西": 292.5,
    "北西": 315,
    "北北西": 337.5
}

# 風向き列を角度に変換
csv["wind_direction"] = csv["wind_direction"].map(wind_direction_map)

output_filename = os.path.splitext(sys.argv[1])[0] + "_angle.csv"

# 変換後のデータをCSVファイルとして保存
csv.to_csv(output_filename, index=False)