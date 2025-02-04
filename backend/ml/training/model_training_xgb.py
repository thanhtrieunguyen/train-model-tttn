import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
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

dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'weather_dataset.csv')
models_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'weather_models.h5')

def is_gpu_available():
    """Kiểm tra xem có GPU không bằng cách xem có CUDA hỗ trợ không."""
    try:
        return 'gpu' in xgb.get_config().get('tree_method', '')
    except:
        return False
    
class WeatherPredictor:
    def __init__(self):
        gpu_available = is_gpu_available()
        
        self.model = XGBRegressor(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1,
            tree_method='gpu_hist' if gpu_available else 'hist',  # Nếu có GPU thì sử dụng 'gpu_hist', nếu không thì dùng 'hist'
        )
        self.preprocessor = WeatherDataPreprocessor()
        self.feature_list_for_scale = None

    def prepare_data(self, data_path):
        """Prepare data for training."""
        df = self.preprocessor.preprocess(data_path)
        
        targets = ['temperature', 'humidity', 'wind_speed', 'pressure', 
                  'precipitation', 'cloud', 'uv_index', 'visibility', 
                  'rain_probability', 'dewpoint', 'gust_speed', 'snow_probability',
                  'condition_code', 'wind_direction']

        X = df.drop(targets, axis=1)
        y_dict = {target: df[target] for target in targets}

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
            gpu_available = is_gpu_available()
            model = XGBRegressor(
                n_estimators=200,
                max_depth=20,
                random_state=42,
                n_jobs=-1,
                tree_method='gpu_hist' if gpu_available else 'hist', 
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
                'r2': r2
            }
            
            # Vẽ biểu đồ
            self.plot_predictions(y_test, y_pred, target_name)
        
        print("Training complete.")
        
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
        """Save trained models."""
        import h5py
        with h5py.File(path, 'w') as f:
            for target_name, model in self.models.items():
                model.save_model(f'{target_name}.json')
            f.create_dataset('feature_list_for_scale', data=np.array(self.feature_list_for_scale, dtype='S'))
    
    def load_models(self, path=models_path):
        """Load trained models."""
        import h5py
        self.models = {}
        self.scalers = {}
        with h5py.File(path, 'r') as f:
            for target_name in f.keys():
                if target_name != 'feature_list_for_scale':
                    model = XGBRegressor()
                    model.load_model(f[target_name].value)
                    self.models[target_name] = model
            self.feature_list_for_scale = [col.decode('utf-8') for col in f['feature_list_for_scale']]

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

if __name__ == "__main__":
    main()