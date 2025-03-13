import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import os
import sys
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(project_root)
from backend.ml.data.data_preprocessing import WeatherDataPreprocessor

dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'weather_dataset_with_season_terrain.csv')
models_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'weather_models.joblib')

class WeatherPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        self.preprocessor = WeatherDataPreprocessor()
        self.feature_list_for_scale = None

    def prepare_data(self, data_path):
        """Prepare data for training."""
        df = self.preprocessor.preprocess(data_path)
        
        targets = [
            'temperature', 'humidity', 'wind_speed', 'pressure', 
            'precipitation', 'cloud', 'uv_index', 'visibility', 
            'rain_probability', 'dewpoint', 'gust_speed', 'snow_probability',
            'wind_direction'
        ]

        X = df.drop(targets, axis=1)
        y_dict = {target: df[target] for target in targets}

        # for target in targets:
        #     print(f"Distribution of {target}:")
        #     print(df[target].describe())
            
        # Tạo feature_list_for_scale sau khi tách target
        feature_list_for_scale = [col for col in X.columns if col not in targets]

        return X, y_dict, feature_list_for_scale
    
    def train(self, X, y_dict, feature_list_for_scale):
        """Train models for each weather parameter."""
        self.models = {}
        self.metrics = {}
        self.scalers = {}


        print("Training models...")
        for target_name, y in tqdm(y_dict.items()):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Scale data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train[feature_list_for_scale])
            X_test_scaled = scaler.transform(X_test[feature_list_for_scale])
            
            self.scalers[target_name] = scaler

            # Train model
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)
            
            # Dự đoán
            y_pred = model.predict(X_test_scaled)
            
            # Tính toán các metrics 
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            self.models[target_name] = model
            self.metrics[target_name] = {
                'rmse': np.sqrt(mse),
                'r2': r2,
                'mae': mean_absolute_error(y_test, y_pred),
                'mape': mean_absolute_percentage_error(y_test, y_pred) if np.all(y_test != 0) else float('nan')
            }
            
            # Vẽ biểu đồ
            self.plot_predictions(y_test, y_pred, target_name)
        
        print("Training complete.")
        
        # self.print_evaluation_metrics()
        
        return self.metrics
    
    def plot_predictions(self, y_true, y_pred, target_name):
        """Plot actual vs predicted values."""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5, c='blue', s=10, label='Predicted vs Actual')
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='Ideal Fit')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Actual vs Predicted {target_name}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'backend/ml/models/rating_chart1/{target_name}.png')
        plt.close()
    
    def save_models(self, path=models_path):
        """Save trained models and ensure 'feature_list_for_scale' is updated."""
        if self.feature_list_for_scale is None:
            # Cập nhật danh sách đặc trưng nếu chưa được gán
            self.feature_list_for_scale = [
                'hour', 'day', 'month', 'day_of_week', 'day_of_year',
                'is_spring', 'is_summer', 'is_autumn', 'is_winter',
                'season_sin', 'season_cos',
                'elevation_m', 'terrain_lowland', 'terrain_hills', 'terrain_highland', 'terrain_mountain',
                'temperature_lag_1', 'temperature_lag_2', 'temperature_lag_3',
                'temperature_rolling_mean_3', 'temperature_rolling_mean_6', 'temperature_rolling_mean_12',
                'humidity_lag_1', 'humidity_lag_2', 'humidity_lag_3',
                'humidity_rolling_mean_3', 'humidity_rolling_mean_6', 'humidity_rolling_mean_12',
                'wind_speed_lag_1', 'wind_speed_lag_2', 'wind_speed_lag_3',
                'wind_speed_rolling_mean_3', 'wind_speed_rolling_mean_6', 'wind_speed_rolling_mean_12',
                'pressure_lag_1', 'pressure_lag_2', 'pressure_lag_3',
                'pressure_rolling_mean_3', 'pressure_rolling_mean_6', 'pressure_rolling_mean_12',
                'precipitation_lag_1', 'precipitation_lag_2', 'precipitation_lag_3',
                'precipitation_rolling_mean_3', 'precipitation_rolling_mean_6', 'precipitation_rolling_mean_12',
                'cloud_lag_1', 'cloud_lag_2', 'cloud_lag_3',
                'cloud_rolling_mean_3', 'cloud_rolling_mean_6', 'cloud_rolling_mean_12',
                'uv_index_lag_1', 'uv_index_lag_2', 'uv_index_lag_3',
                'uv_index_rolling_mean_3', 'uv_index_rolling_mean_6', 'uv_index_rolling_mean_12',
                'visibility_lag_1', 'visibility_lag_2', 'visibility_lag_3',
                'visibility_rolling_mean_3', 'visibility_rolling_mean_6', 'visibility_rolling_mean_12',
                'rain_probability_lag_1', 'rain_probability_lag_2', 'rain_probability_lag_3',
                'rain_probability_rolling_mean_3', 'rain_probability_rolling_mean_6', 'rain_probability_rolling_mean_12',
                'dewpoint_lag_1', 'dewpoint_lag_2', 'dewpoint_lag_3',
                'dewpoint_rolling_mean_3', 'dewpoint_rolling_mean_6', 'dewpoint_rolling_mean_12',
                'airport_code_encoded'
            ]
            print("Updated 'feature_list_for_scale' before saving the model.")
        
        # Lưu mô hình và thông tin liên quan
        joblib.dump({'models': self.models, 'scalers': self.scalers, 'feature_list_for_scale': self.feature_list_for_scale}, path, compress=4)
        print(f"Models and feature list saved successfully to {path}.")
    
    def load_models(self, path=models_path):
        """Load trained models."""
        data = joblib.load(path)
        self.models = data['models']
        self.scalers = data['scalers']
        self.feature_list_for_scale = data['feature_list_for_scale']
    
    # def print_evaluation_metrics(self):
    #     """Print comprehensive evaluation metrics for all models."""
    #     print("\n" + "="*80)
    #     print(f"{'MODEL EVALUATION METRICS':^80}")
    #     print("="*80)
    #     print(f"{'Target':<15} | {'RMSE':^12} | {'MAE':^12} | {'R²':^12} | {'MAPE (%)':^12}")
    #     print("-"*80)
        
    #     for target, metrics in self.metrics.items():
    #         rmse = f"{metrics['rmse']:.4f}" if isinstance(metrics['rmse'], (int, float)) else metrics['rmse']
    #         mae = f"{metrics['mae']:.4f}" if isinstance(metrics['mae'], (int, float)) else metrics['mae']
    #         r2 = f"{metrics['r2']:.4f}" if isinstance(metrics['r2'], (int, float)) else metrics['r2']
    #         mape = f"{metrics['mape']:.4f}" if isinstance(metrics['mape'], (int, float)) else metrics['mape']
            
    #         print(f"{target:<15} | {rmse:^12} | {mae:^12} | {r2:^12} | {mape:^12}")
        
    #     print("="*80)
        
    #     # Also print top 5 most important features for each model
    #     print("\n" + "="*80)
    #     print(f"{'TOP 5 MOST IMPORTANT FEATURES':^80}")
    #     print("="*80)
        
    #     for target, importances in self.feature_importances.items():
    #         print(f"\nModel: {target}")
    #         sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    #         for i, (feature, importance) in enumerate(sorted_features, 1):
    #             print(f"{i}. {feature:<20}: {importance:.4f}")
    
    def generate_summary_report(self, output_path=None):
        """Generate a comprehensive summary report of model performance."""
        if output_path is None:
            output_path = os.path.join(project_root, 'backend', 'ml', 'models', 'model_evaluation_report.txt')
        
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"{'WEATHER PREDICTION MODEL EVALUATION REPORT':^80}\n")
            f.write(f"{'Generated on: ' + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'):^80}\n")
            f.write("="*80 + "\n\n")
            
            # Write model metrics
            f.write("MODEL PERFORMANCE METRICS\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Target':<15} | {'RMSE':^12} | {'MAE':^12} | {'R²':^12} | {'MAPE (%)':^12}\n")
            f.write("-"*80 + "\n")
            
            for target, metrics in self.metrics.items():
                rmse = f"{metrics['rmse']:.4f}" if isinstance(metrics['rmse'], (int, float)) else metrics['rmse']
                mae = f"{metrics['mae']:.4f}" if isinstance(metrics['mae'], (int, float)) else metrics['mae']
                r2 = f"{metrics['r2']:.4f}" if isinstance(metrics['r2'], (int, float)) else metrics['r2']
                mape = f"{metrics['mape']:.4f}" if isinstance(metrics['mape'], (int, float)) else metrics['mape']
                
                f.write(f"{target:<15} | {rmse:^12} | {mae:^12} | {r2:^12} | {mape:^12}\n")
            
            f.write("\n\n")
            
            # Write feature importances
            f.write("FEATURE IMPORTANCE ANALYSIS\n")
            f.write("-"*80 + "\n")
            
            for target, importances in self.feature_importances.items():
                f.write(f"\nModel: {target}\n")
                sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
                for i, (feature, importance) in enumerate(sorted_features, 1):
                    f.write(f"{i}. {feature:<20}: {importance:.4f}\n")
                f.write("\n")
        
        print(f"Summary report generated at: {output_path}")
        
def main():
    # Initialize predictor
    predictor = WeatherPredictor()
    
    # Prepare data
    X, y_dict, feature_list_for_scale = predictor.prepare_data(dataset_path)

    # Train models
    metrics = predictor.train(X, y_dict, feature_list_for_scale)
    
    # Print metrics
    for target, metric in metrics.items():
        print(f"{target} - RMSE: {metric['rmse']}, R²: {metric['r2']}")
    
    # Save models
    predictor.save_models()

    predictor.generate_summary_report()

if __name__ == "__main__":
    main()