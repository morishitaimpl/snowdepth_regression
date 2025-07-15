import numpy as np
import pandas as pd
import sys
import pathlib
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime
import os

import config as cf

def load_model(model_path):
    """学習済みモデルとスケーラーを読み込む"""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    model = cf.SnowDepthPredictor(input_size=13)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    scaler_x = checkpoint['scaler_x']
    scaler_y = checkpoint['scaler_y']
    input_features = checkpoint['input_features']
    
    return model, scaler_x, scaler_y, input_features

def preprocess_data(data, input_features, scaler_x):
    """入力データの前処理"""
    x = data[input_features].values
    x_scaled = scaler_x.transform(x)
    x_tensor = torch.FloatTensor(x_scaled)
    return x_tensor

def make_predictions(model, x_tensor, scaler_y):
    """予測を実行"""
    with torch.no_grad():
        y_pred_scaled = model(x_tensor).numpy()
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
    return y_pred.flatten()

def evaluate_predictions(y_true, y_pred):
    """予測結果の評価"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"平均二乗誤差 (MSE): {mse:.3f}")
    print(f"平均平方根誤差 (RMSE): {rmse:.3f}")
    print(f"決定係数 (R²): {r2:.3f}")
    
    return mse, rmse, r2

def plot_predictions(y_true, y_pred, output_path=None):
    """予測結果の可視化"""
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('実際の積雪深')
    plt.ylabel('予測積雪深')
    plt.title('実際値 vs 予測値')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('予測積雪深')
    plt.ylabel('残差')
    plt.title('残差プロット')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.hist(residuals, bins=30, alpha=0.7)
    plt.xlabel('残差')
    plt.ylabel('頻度')
    plt.title('残差の分布')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(y_true, label='実際値', alpha=0.7)
    plt.plot(y_pred, label='予測値', alpha=0.7)
    plt.xlabel('サンプル番号')
    plt.ylabel('積雪深')
    plt.title('時系列比較')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"予測結果のグラフを {output_path} に保存しました")
    else:
        plt.show()
    
    plt.close()

def save_predictions(data, y_pred, output_path):
    """予測結果をCSVファイルに保存"""
    result_df = data.copy()
    result_df['predicted_snow_depth'] = y_pred
    result_df.to_csv(output_path, index=False)
    print(f"予測結果を {output_path} に保存しました")

def predict_single_day(model, scaler_x, scaler_y, input_features, input_data):
    """単日のデータから積雪深を予測"""
    df = pd.DataFrame([input_data], columns=input_features)
    x_tensor = preprocess_data(df, input_features, scaler_x)
    y_pred = make_predictions(model, x_tensor, scaler_y)
    return y_pred[0]

class SnowDepthInteractive:
    def __init__(self):
        self.model = None
        self.scaler_x = None
        self.scaler_y = None
        self.input_features = None
        self.field_labels = {
            'month': '月 (1-12)',
            'day': '日 (1-31)',
            'land_atmosphere': '陸上気圧 (hPa)',
            'sea_atmosphere': '海上気圧 (hPa)',
            'precipitation': '降水量 (mm)',
            'temperature': '気温 (°C)',
            'humidity': '湿度 (%)',
            'wind_speed': '風速 (m/s)',
            'wind_direction': '風向 (度)',
            'sum_insolation': '日射量 (MJ/m²)',
            'sum_sunlight': '日照時間 (時間)',
            'snow_falling': '降雪量 (cm)',
            'melted_snow': '融雪量 (cm)'
        }
        self.prediction_history = []
    
    def clear_screen(self):
        """画面をクリア"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def print_header(self):
        """ヘッダーを表示"""
        print("=" * 60)
        print("           積雪深予測システム - インタラクティブモード")
        print("=" * 60)
        print()
    
    def load_model_interactive(self):
        """インタラクティブにモデルを読み込み"""
        while True:
            print("モデルファイルのパスを入力してください:")
            print("(例: test_output/model.pth)")
            model_path = input("モデルパス: ").strip()
            
            if not model_path:
                print("エラー: モデルパスを入力してください")
                continue
            
            if not os.path.exists(model_path):
                print(f"エラー: ファイルが見つかりません: {model_path}")
                continue
            
            try:
                self.model, self.scaler_x, self.scaler_y, self.input_features = load_model(model_path)
                print(f"\n✓ モデルを正常に読み込みました")
                print(f"✓ 入力特徴量: {self.input_features}")
                print()
                return True
            except Exception as e:
                print(f"エラー: モデルの読み込みに失敗しました: {str(e)}")
                retry = input("再試行しますか？ (y/n): ").strip().lower()
                if retry != 'y':
                    return False
    
    def input_daily_data(self):
        """日次データを入力"""
        print("気象データを入力してください:")
        print("-" * 40)
        
        input_data = []
        for field in self.input_features:
            label = self.field_labels[field]
            while True:
                try:
                    value_str = input(f"{label}: ").strip()
                    if not value_str:
                        print("エラー: 値を入力してください")
                        continue
                    value = float(value_str)
                    input_data.append(value)
                    break
                except ValueError:
                    print("エラー: 数値を入力してください")
        
        return input_data
    
    def predict_and_display(self, input_data):
        """予測を実行して結果を表示"""
        try:
            predicted_depth = predict_single_day(
                self.model, self.scaler_x, self.scaler_y, 
                self.input_features, input_data
            )
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            print("\n" + "=" * 50)
            print("           予測結果")
            print("=" * 50)
            print(f"予測時刻: {timestamp}")
            print(f"予測積雪深: {predicted_depth:.2f} cm")
            print("=" * 50)
            
            self.prediction_history.append({
                'timestamp': timestamp,
                'input_data': input_data,
                'predicted_depth': predicted_depth
            })
            
            return predicted_depth
            
        except Exception as e:
            print(f"エラー: 予測に失敗しました: {str(e)}")
            return None
    
    def show_history(self):
        """予測履歴を表示"""
        if not self.prediction_history:
            print("予測履歴がありません")
            return
        
        print("\n" + "=" * 60)
        print("           予測履歴")
        print("=" * 60)
        
        for i, record in enumerate(self.prediction_history, 1):
            print(f"\n{i}. {record['timestamp']}")
            print(f"   入力データ:")
            for field, value in zip(self.input_features, record['input_data']):
                print(f"     {self.field_labels[field]}: {value}")
            print(f"   予測積雪深: {record['predicted_depth']:.2f} cm")
            print("-" * 40)
    
    def save_history(self):
        """予測履歴をCSVファイルに保存"""
        if not self.prediction_history:
            print("保存する履歴がありません")
            return
        
        output_path = input("保存先ファイル名 (例: predictions_history.csv): ").strip()
        if not output_path:
            output_path = "predictions_history.csv"
        
        try:
            records = []
            for record in self.prediction_history:
                row = {}
                for field, value in zip(self.input_features, record['input_data']):
                    row[field] = value
                row['predicted_snow_depth'] = record['predicted_depth']
                row['timestamp'] = record['timestamp']
                records.append(row)
            
            df = pd.DataFrame(records)
            df.to_csv(output_path, index=False)
            print(f"✓ 予測履歴を {output_path} に保存しました")
            
        except Exception as e:
            print(f"エラー: 保存に失敗しました: {str(e)}")
    
    def show_menu(self):
        """メニューを表示"""
        print("\n" + "-" * 40)
        print("メニュー:")
        print("1. 新しい予測を実行")
        print("2. 予測履歴を表示")
        print("3. 履歴をCSVファイルに保存")
        print("4. 画面をクリア")
        print("5. 終了")
        print("-" * 40)
    
    def run(self):
        """インタラクティブモードを実行"""
        self.clear_screen()
        self.print_header()
        
        if not self.load_model_interactive():
            print("モデルの読み込みに失敗しました。終了します。")
            return
        
        while True:
            self.show_menu()
            choice = input("選択してください (1-5): ").strip()
            
            if choice == '1':
                print("\n" + "=" * 50)
                print("           新しい予測")
                print("=" * 50)
                input_data = self.input_daily_data()
                self.predict_and_display(input_data)
                
            elif choice == '2':
                self.show_history()
                
            elif choice == '3':
                self.save_history()
                
            elif choice == '4':
                self.clear_screen()
                self.print_header()
                
            elif choice == '5':
                print("\n積雪深予測システムを終了します。")
                if self.prediction_history:
                    save_choice = input("終了前に履歴を保存しますか？ (y/n): ").strip().lower()
                    if save_choice == 'y':
                        self.save_history()
                break
                
            else:
                print("無効な選択です。1-5の数字を入力してください。")

def run_interactive_mode():
    """インタラクティブモードを実行"""
    app = SnowDepthInteractive()
    app.run()

if __name__ == "__main__":
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ["--interactive", "-i"]):
        print("=== 積雪深予測システム - インタラクティブモード ===")
        print("インタラクティブモードを起動しています...")
        run_interactive_mode()
        sys.exit(0)
    
    if len(sys.argv) < 3:
        print("使用方法:")
        print("  インタラクティブモード: python predict_snowdepth.py [--interactive | -i]")
        print("  CSV モード: python predict_snowdepth.py <model_path> <data_path> [output_dir]")
        print("例:")
        print("  python predict_snowdepth.py --interactive")
        print("  python predict_snowdepth.py output/model.pth data_2016_2025.csv results/")
        sys.exit(1)
    
    model_path = sys.argv[1]
    data_path = sys.argv[2]
    output_dir = pathlib.Path(sys.argv[3]) if len(sys.argv) > 3 else pathlib.Path("evaluation_results")
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    print("=== 積雪深予測評価システム - CSV モード ===")
    print(f"モデル: {model_path}")
    print(f"データ: {data_path}")
    print(f"出力先: {output_dir}")
    print()
    
    print("学習済みモデルを読み込み中...")
    model, scaler_x, scaler_y, input_features = load_model(model_path)
    print(f"入力特徴量: {input_features}")
    print()
    
    # データの読み込み
    print("評価データを読み込み中...")
    data = pd.read_csv(data_path)
    print(f"データサイズ: {data.shape}")
    
    missing_features = [f for f in input_features if f not in data.columns]
    if missing_features:
        print(f"エラー: 以下の特徴量がデータに含まれていません: {missing_features}")
        sys.exit(1)
    
    if 'snow_depth' not in data.columns:
        print("警告: 'snow_depth'列が見つかりません。予測のみを実行します。")
        evaluate_mode = False
    else:
        evaluate_mode = True
    
    print("データを前処理中...")
    x_tensor = preprocess_data(data, input_features, scaler_x)
    
    # 予測の実行
    print("予測を実行中...")
    y_pred = make_predictions(model, x_tensor, scaler_y)
    print(f"予測完了: {len(y_pred)}件")
    print()
    
    if evaluate_mode:
        y_true = data['snow_depth'].values
        print("=== 評価結果 ===")
        mse, rmse, r2 = evaluate_predictions(y_true, y_pred)
        print()
        
        plot_path = output_dir / "prediction_evaluation.png"
        plot_predictions(y_true, y_pred, plot_path)
    else:
        print("実際値がないため、評価をスキップします。")
    
    csv_path = output_dir / "predictions.csv"
    save_predictions(data, y_pred, csv_path)
    
    print("\n=== 予測統計 ===")
    print(f"予測値の平均: {np.mean(y_pred):.3f}")
    print(f"予測値の標準偏差: {np.std(y_pred):.3f}")
    print(f"予測値の最小値: {np.min(y_pred):.3f}")
    print(f"予測値の最大値: {np.max(y_pred):.3f}")
    
    print(f"\n評価完了。結果は {output_dir} に保存されました。")
