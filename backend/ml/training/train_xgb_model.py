import os
import sys
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Thêm đường dẫn gốc của dự án vào sys.path
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
sys.path.append(project_root)

from backend.ml.data.data_preprocessing import WeatherDataPreprocessor

def train_xgb_model():
    """Huấn luyện mô hình XGBoost cho dự báo thời tiết"""
    print("Bắt đầu huấn luyện mô hình XGBoost...")
    
    # Khởi tạo preprocessor
    preprocessor = WeatherDataPreprocessor()
    
    # Đường dẫn đến dữ liệu
    data_path = os.path.join(project_root, 'backend', 'ml', 'data', 'weather_dataset_with_season_terrain.csv')
    
    # Tiền xử lý dữ liệu
    print("Đang tiền xử lý dữ liệu...")
    df = preprocessor.preprocess(data_path)
    
    # Xác định các cột đích cần dự báo
    targets = ['temperature', 'humidity', 'wind_speed', 'pressure', 
              'precipitation', 'cloud', 'uv_index', 'visibility', 
              'rain_probability', 'dewpoint', 'gust_speed', 'snow_probability',
              'condition_code', 'wind_direction']
    
    # Tách features và targets
    X = df.drop(targets, axis=1)
    y_dict = {target: df[target] for target in targets}
    
    # Danh sách các cột cần chuẩn hóa
    feature_list_for_scale = [col for col in X.columns if col not in targets]
    
    # Dictionary lưu mô hình và scaler
    models = {}
    scalers = {}
    metrics = {}
    
    # Huấn luyện mô hình cho từng mục tiêu
    for target_name, y in y_dict.items():
        print(f"Đang huấn luyện mô hình cho {target_name}...")
        
        # Chia tập dữ liệu
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Chuẩn hóa dữ liệu
        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[feature_list_for_scale] = scaler.fit_transform(X_train[feature_list_for_scale])
        X_test_scaled[feature_list_for_scale] = scaler.transform(X_test[feature_list_for_scale])
        
        # Lưu scaler
        scalers[target_name] = scaler
        
        # Khởi tạo và huấn luyện mô hình
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
        model.fit(X_train_scaled, y_train)
        
        # Dự báo
        y_pred = model.predict(X_test_scaled)
        
        # Tính toán các chỉ số đánh giá
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Lưu mô hình và metrics
        models[target_name] = model
        metrics[target_name] = {
            'rmse': rmse,
            'r2': r2,
            'mae': mae
        }
        
        # Vẽ biểu đồ dự báo
        plt.figure(figsize=(10, 6))
        plt.plot(y_test[:100], label='Thực tế', color='black', linestyle='--')
        plt.plot(y_pred[:100], label='Dự báo', color='green')
        plt.title(f'XGBoost: Dự báo {target_name}')
        plt.xlabel('Mẫu dữ liệu')
        plt.ylabel(target_name)
        plt.legend()
        
        # Tạo thư mục để lưu biểu đồ
        plots_dir = os.path.join(project_root, 'backend', 'ml', 'evaluation', 'plots', 'xgb')
        os.makedirs(plots_dir, exist_ok=True)
        
        plt.savefig(os.path.join(plots_dir, f'{target_name}_prediction.png'))
        plt.close()
        
        print(f"Hoàn thành huấn luyện cho {target_name}. RMSE = {rmse:.4f}, R2 = {r2:.4f}")
    
    # Đóng gói kết quả
    data_to_save = {
        'models': models,
        'scalers': scalers,
        'metrics': metrics,
        'feature_list_for_scale': feature_list_for_scale
    }
    
    # Lưu mô hình
    models_dir = os.path.join(project_root, 'backend', 'ml', 'models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(data_to_save, os.path.join(models_dir, 'xgb_weather_models.joblib'))
    
    print(f"Đã lưu mô hình XGBoost tại {os.path.join(models_dir, 'xgb_weather_models.joblib')}")
    
    return data_to_save

if __name__ == "__main__":
    train_xgb_model()
