import sys
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

# Thêm đường dẫn gốc của dự án vào sys.path
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
sys.path.append(project_root)

from backend.ml.data.data_preprocessing import WeatherDataPreprocessor
from backend.ml.config.matplotlib_config import *

class ModelEvaluator:
    def __init__(self):
        self.preprocessor = WeatherDataPreprocessor()
        self.models = {
            'RandomForest': {},
            'XGBoost': {},
            'LightGBM': {},
        }
        self.metrics = {
            'RandomForest': {},
            'XGBoost': {},
            'LightGBM': {},
        }
        self.scalers = {
            'RandomForest': {},
            'XGBoost': {},
            'LightGBM': {},
        }
    
    def prepare_data(self, data_path):
        """Chuẩn bị dữ liệu để huấn luyện và đánh giá mô hình"""
        df = self.preprocessor.preprocess(data_path)
        
        # Update target list: bỏ 'condition_code', đổi 'cloud' thành 'cloud_cover' và thay 'wind_direction'
        targets = ['temperature', 'humidity', 'wind_speed', 'pressure', 
                   'precipitation', 'cloud_cover', 'uv_index', 'visibility', 
                   'rain_probability', 'dewpoint', 'gust_speed', 'snow_probability',
                   'wind_direction_sin', 'wind_direction_cos']
        
        X = df.drop(targets, axis=1)
        y_dict = {target: df[target] for target in targets}
        
        feature_list_for_scale = [col for col in X.columns if col not in targets]
        
        return X, y_dict, feature_list_for_scale
    
    def train_models(self, X, y_dict, feature_list_for_scale):
        """Huấn luyện cả ba mô hình cho mỗi mục tiêu"""
        for target_name, y in y_dict.items():
            print(f"Huấn luyện các mô hình cho {target_name}...")
            
            # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Chuẩn hóa dữ liệu
            scaler = StandardScaler()
            X_train_scaled = X_train.copy()
            X_test_scaled = X_test.copy()
            
            X_train_scaled[feature_list_for_scale] = scaler.fit_transform(X_train[feature_list_for_scale])
            X_test_scaled[feature_list_for_scale] = scaler.transform(X_test[feature_list_for_scale])
            
            # Lưu scaler cho từng model và target
            for model_name in self.models:
                self.scalers[model_name][target_name] = scaler
            
            # Huấn luyện RandomForest
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf_model.fit(X_train_scaled, y_train)
            rf_pred = rf_model.predict(X_test_scaled)
            
            # Huấn luyện XGBoost
            xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
            xgb_model.fit(X_train_scaled, y_train)
            xgb_pred = xgb_model.predict(X_test_scaled)
            
            # Huấn luyện LightGBM
            lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
            lgb_model.fit(X_train_scaled, y_train)
            lgb_pred = lgb_model.predict(X_test_scaled)
            
            # Lưu các mô hình
            self.models['RandomForest'][target_name] = rf_model
            self.models['XGBoost'][target_name] = xgb_model
            self.models['LightGBM'][target_name] = lgb_model
            
            # Tính toán metrics cho RandomForest
            self.metrics['RandomForest'][target_name] = self._calculate_metrics(y_test, rf_pred)
            
            # Tính toán metrics cho XGBoost
            self.metrics['XGBoost'][target_name] = self._calculate_metrics(y_test, xgb_pred)
            
            # Tính toán metrics cho LightGBM
            self.metrics['LightGBM'][target_name] = self._calculate_metrics(y_test, lgb_pred)
            
            # Vẽ biểu đồ so sánh
            self._plot_comparison(y_test, rf_pred, xgb_pred, lgb_pred, target_name)
    
    def _calculate_metrics(self, y_true, y_pred):
        """Tính toán các chỉ số đánh giá"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        # Tính MAPE với xử lý đặc biệt khi có giá trị bằng 0
        non_zero_mask = y_true != 0
        if np.any(non_zero_mask):
            mape = mean_absolute_percentage_error(
                y_true[non_zero_mask], y_pred[non_zero_mask]
            )
        else:
            mape = float('nan')
            
        return {
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'mape': mape
        }
    
    def _plot_comparison(self, y_true, rf_pred, xgb_pred, lgb_pred, target_name):
        """Vẽ biểu đồ so sánh kết quả của các mô hình"""
        # Vẽ biểu đồ thông thường
        plt.figure(figsize=(12, 6))
        plt.plot(y_true[:100], label='Thực tế', color='black', linestyle='--')
        plt.plot(rf_pred[:100], label='RandomForest', color='blue')
        plt.plot(xgb_pred[:100], label='XGBoost', color='green')
        plt.plot(lgb_pred[:100], label='LightGBM', color='red')
        plt.title(f'So sánh dự báo {target_name}')
        plt.xlabel('Mẫu dữ liệu')
        plt.ylabel(target_name)
        plt.legend()
        
        # Tạo thư mục nếu chưa tồn tại
        plots_dir = os.path.join(project_root, 'backend', 'ml', 'evaluation', 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Lưu biểu đồ
        plt.savefig(os.path.join(plots_dir, f'{target_name}_comparison.png'))
        plt.close()
        
        # Thêm biểu đồ scatter plot để hiển thị R²
        plt.figure(figsize=(18, 6))
        
        # RandomForest
        plt.subplot(1, 3, 1)
        plt.scatter(y_true, rf_pred, alpha=0.5, s=10)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
        rf_r2 = r2_score(y_true, rf_pred)
        plt.title(f'RandomForest: R² = {rf_r2:.4f}')
        plt.xlabel('Thực tế')
        plt.ylabel('Dự báo')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # XGBoost
        plt.subplot(1, 3, 2)
        plt.scatter(y_true, xgb_pred, alpha=0.5, s=10)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
        xgb_r2 = r2_score(y_true, xgb_pred)
        plt.title(f'XGBoost: R² = {xgb_r2:.4f}')
        plt.xlabel('Thực tế')
        plt.ylabel('Dự báo')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # LightGBM
        plt.subplot(1, 3, 3)
        plt.scatter(y_true, lgb_pred, alpha=0.5, s=10)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
        lgb_r2 = r2_score(y_true, lgb_pred)
        plt.title(f'LightGBM: R² = {lgb_r2:.4f}')
        plt.xlabel('Thực tế')
        plt.ylabel('Dự báo')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{target_name}_scatter_comparison.png'))
        plt.close()
    
    def compare_models(self):
        """So sánh các mô hình và trả về kết quả"""
        results = pd.DataFrame()
        
        for target_name in self.metrics['RandomForest'].keys():
            target_results = []
            
            for model_name in self.models.keys():
                metrics = self.metrics[model_name][target_name]
                target_results.append({
                    'Target': target_name,
                    'Model': model_name,
                    'RMSE': metrics['rmse'],
                    'R2': metrics['r2'],
                    'MAE': metrics['mae'],
                    'MAPE': metrics['mape']
                })
            
            target_df = pd.DataFrame(target_results)
            results = pd.concat([results, target_df], ignore_index=True)
        
        # Lưu kết quả so sánh
        results_dir = os.path.join(project_root, 'backend', 'ml', 'evaluation', 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Lưu dưới dạng CSV và Excel
        results.to_csv(os.path.join(results_dir, 'model_comparison.csv'), index=False)
        results.to_excel(os.path.join(results_dir, 'model_comparison.xlsx'), index=False)
        
        return results
    
    def get_best_models(self):
        """Xác định mô hình tốt nhất cho từng target dựa trên RMSE"""
        best_models = {}
        
        for target_name in self.metrics['RandomForest'].keys():
            best_model = None
            best_rmse = float('inf')
            
            for model_name in self.models.keys():
                rmse = self.metrics[model_name][target_name]['rmse']
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model_name
            
            best_models[target_name] = {
                'model_name': best_model,
                'rmse': best_rmse
            }
        
        # Lưu kết quả mô hình tốt nhất
        results_dir = os.path.join(project_root, 'backend', 'ml', 'evaluation', 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        with open(os.path.join(results_dir, 'best_models.txt'), 'w', encoding='utf-8') as f:
            f.write('Mô hình tốt nhất cho từng target:\n\n')
            for target, info in best_models.items():
                f.write(f"{target}: {info['model_name']} (RMSE = {info['rmse']:.4f})\n")
        
        return best_models
    
    def save_best_model(self):
        """Lưu mô hình tốt nhất cho từng target"""
        best_models = self.get_best_models()
        
        result_models = {}
        result_scalers = {}
        
        for target, info in best_models.items():
            model_name = info['model_name']
            result_models[target] = self.models[model_name][target]
            result_scalers[target] = self.scalers[model_name][target]
        
        # Tạo cấu trúc dữ liệu để lưu
        data_to_save = {
            'models': result_models,
            'scalers': result_scalers,
            'feature_list_for_scale': None,  # Sẽ được cập nhật khi sử dụng
            'best_model_info': best_models
        }
        
        # Lưu mô hình tốt nhất
        models_dir = os.path.join(project_root, 'backend', 'ml', 'models')
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(data_to_save, os.path.join(models_dir, 'best_weather_models.joblib'))
        
        print(f"Đã lưu các mô hình tốt nhất vào {os.path.join(models_dir, 'best_weather_models.joblib')}")
        
        return data_to_save
    
    def evaluate_on_real_data(self, real_data_path, historical_data_path, prediction_time):
        """Đánh giá mô hình trên dữ liệu thực tế"""
        # Đọc dữ liệu thực tế
        real_data = pd.read_csv(real_data_path)
        real_data['timestamp'] = pd.to_datetime(real_data['timestamp'])
        
        # Chọn dữ liệu tại thời điểm dự báo
        prediction_dt = pd.to_datetime(prediction_time)
        real_values = real_data[real_data['timestamp'] == prediction_dt]
        
        if len(real_values) == 0:
            print(f"Không tìm thấy dữ liệu thực tế tại thời điểm {prediction_time}")
            return None
        
        # Đọc dữ liệu lịch sử
        historical_data = pd.read_csv(historical_data_path)
        historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])
        
        # Lọc dữ liệu lịch sử trước thời điểm dự báo
        historical_filtered = historical_data[historical_filtered['timestamp'] < prediction_dt].sort_values('timestamp')
        
        # Lấy 12 giờ dữ liệu gần nhất
        hours_needed = 12
        if len(historical_filtered) > hours_needed:
            historical_filtered = historical_filtered.iloc[-hours_needed:]
        
        # Tiền xử lý dữ liệu lịch sử
        preprocessed_data = self.preprocessor.preprocess_real_time_data(historical_filtered)
        
        # Tạo input cho dự báo
        input_data = preprocessed_data.iloc[-1:].copy()
        
        # Thực hiện dự báo với mỗi mô hình
        predictions = {}
        # Update danh sách target cho đồng bộ
        targets = ['temperature', 'humidity', 'wind_speed', 'pressure', 
                   'precipitation', 'cloud_cover', 'uv_index', 'visibility', 
                   'rain_probability', 'dewpoint', 'gust_speed', 'snow_probability',
                   'wind_direction_sin', 'wind_direction_cos']
        
        for model_name in self.models:
            predictions[model_name] = {}
            
            for target in targets:
                if target in self.models[model_name] and target in real_values:
                    model = self.models[model_name][target]
                    scaler = self.scalers[model_name][target]
                    
                    # Áp dụng scaler cho input
                    feature_list = [col for col in input_data.columns if col not in targets]
                    input_scaled = input_data.copy()
                    input_scaled[feature_list] = scaler.transform(input_data[feature_list])
                    
                    # Dự báo
                    pred_value = model.predict(input_scaled)[0]
                    real_value = real_values[target].values[0]
                    
                    predictions[model_name][target] = {
                        'predicted': pred_value,
                        'real': real_value,
                        'error': abs(pred_value - real_value),
                        'error_percent': abs((pred_value - real_value) / real_value * 100) if real_value != 0 else float('inf')
                    }
        
        # Tạo báo cáo kết quả
        report = []
        
        for target in targets:
            target_report = {'Target': target}
            
            for model_name in predictions:
                if target in predictions[model_name]:
                    target_report[f'{model_name}_Predicted'] = predictions[model_name][target]['predicted']
                    target_report[f'{model_name}_Error'] = predictions[model_name][target]['error']
                    target_report[f'{model_name}_Error_Percent'] = predictions[model_name][target]['error_percent']
            
            if 'Real' not in target_report and target in real_values:
                target_report['Real'] = real_values[target].values[0]
                
            report.append(target_report)
        
        # Chuyển đổi báo cáo thành DataFrame
        report_df = pd.DataFrame(report)
        
        # Lưu báo cáo
        results_dir = os.path.join(project_root, 'backend', 'ml', 'evaluation', 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp_str = prediction_dt.strftime('%Y%m%d_%H%M')
        report_df.to_csv(os.path.join(results_dir, f'real_data_evaluation_{timestamp_str}.csv'), index=False)
        report_df.to_excel(os.path.join(results_dir, f'real_data_evaluation_{timestamp_str}.xlsx'), index=False)
        
        # Tính toán độ chính xác tổng thể và ghi nhận mô hình tốt nhất
        accuracy_report = self._calculate_overall_accuracy(predictions, targets)
        
        # Lưu báo cáo độ chính xác
        with open(os.path.join(results_dir, f'accuracy_report_{timestamp_str}.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Báo cáo đánh giá độ chính xác tại thời điểm {prediction_time}:\n\n")
            
            for model_name, accuracy in accuracy_report['model_accuracy'].items():
                f.write(f"Mô hình {model_name}: Độ chính xác trung bình {accuracy:.2f}%\n")
            
            f.write(f"\nMô hình tốt nhất: {accuracy_report['best_model']} với độ chính xác {accuracy_report['best_accuracy']:.2f}%\n\n")
            
            f.write("Độ chính xác theo từng target:\n")
            for target, accuracies in accuracy_report['target_accuracy'].items():
                f.write(f"\n{target}:\n")
                for model_name, accuracy in accuracies.items():
                    f.write(f"  - {model_name}: {accuracy:.2f}%\n")
        
        return {
            'report': report_df,
            'accuracy': accuracy_report
        }
    
    def _calculate_overall_accuracy(self, predictions, targets):
        """Tính toán độ chính xác tổng thể của các mô hình"""
        model_accuracy = {}
        target_accuracy = {}
        
        for model_name in predictions:
            total_accuracy = 0
            valid_targets = 0
            
            for target in targets:
                if target in predictions[model_name]:
                    error_percent = predictions[model_name][target]['error_percent']
                    
                    if not np.isinf(error_percent):
                        accuracy = max(0, 100 - error_percent)  # Đảm bảo độ chính xác không âm
                        
                        # Lưu độ chính xác theo target
                        if target not in target_accuracy:
                            target_accuracy[target] = {}
                        
                        target_accuracy[target][model_name] = accuracy
                        
                        total_accuracy += accuracy
                        valid_targets += 1
            
            if valid_targets > 0:
                model_accuracy[model_name] = total_accuracy / valid_targets
            else:
                model_accuracy[model_name] = 0
        
        # Tìm mô hình tốt nhất
        best_model = max(model_accuracy.items(), key=lambda x: x[1])
        
        return {
            'model_accuracy': model_accuracy,
            'target_accuracy': target_accuracy,
            'best_model': best_model[0],
            'best_accuracy': best_model[1]
        }

def main():
    """Hàm chính để thực thi việc đánh giá mô hình"""
    print("Bắt đầu đánh giá các mô hình dự báo thời tiết...")
    
    # Khởi tạo evaluator
    evaluator = ModelEvaluator()
    
    # Đường dẫn đến dữ liệu huấn luyện
    data_path = os.path.join(project_root, 'backend', 'ml', 'data', 'weather_dataset_with_season_terrain.csv')
    
    # Chuẩn bị dữ liệu
    print("Đang chuẩn bị dữ liệu...")
    X, y_dict, feature_list_for_scale = evaluator.prepare_data(data_path)
    
    # Huấn luyện các mô hình
    print("Đang huấn luyện các mô hình...")
    evaluator.train_models(X, y_dict, feature_list_for_scale)
    
    # So sánh các mô hình
    print("Đang so sánh các mô hình...")
    comparison_results = evaluator.compare_models()
    print("Kết quả so sánh các mô hình:")
    print(comparison_results)
    
    # Xác định và lưu mô hình tốt nhất
    print("Xác định mô hình tốt nhất cho từng target...")
    best_models = evaluator.get_best_models()
    for target, info in best_models.items():
        print(f"{target}: {info['model_name']} (RMSE = {info['rmse']:.4f})")
    
    # Lưu mô hình tốt nhất
    print("Đang lưu các mô hình tốt nhất...")
    evaluator.save_best_model()
    
    print("Đánh giá hoàn tất!")

if __name__ == "__main__":
    main()
