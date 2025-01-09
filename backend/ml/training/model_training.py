import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
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

dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'historical_weather_data.csv')
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
        
    def prepare_data(self, data_path):
        """Prepare data for training."""
        df = self.preprocessor.preprocess(data_path)
        
        # Chuẩn bị dữ liệu cho model training (các dữ liệu này sẽ đưa vào model để train - ở đây không cần đến các dữ liệu như 'timestamp', 'airport_code', 'airport_name', 'condition', 'wind_direction_symbol' nên sẽ bỏ đi)
        targets = ['temperature', 'humidity', 'wind_speed', 'pressure', 
                  'precipitation', 'cloud', 'uv_index', 'visibility', 'rain_probability', 'dewpoint']
        
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

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, 30]
        }
            
        print("Training models...")
        for target_name, y in tqdm(y_dict.items()):
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale data
            X_train, X_test = self.preprocessor.scale_features(X_train, X_test, feature_list_for_scale)
            
            # Train model
            initial_model = RandomForestRegressor(random_state=42, n_jobs=-1)
            grid_search = GridSearchCV(initial_model, param_grid, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            
            # Tính toán các metrics 
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            self.models[target_name] = best_model
            self.metrics[target_name] = {
                'mse': mse,
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
        plt.savefig(f'backend/ml/models/plots/{target_name}.png')
        plt.close()
    
    def save_models(self, path=models_path):
        """Save trained models."""
        joblib.dump(self.models, path)
    
    def load_models(self, path=models_path):
        """Load trained models."""
        self.models = joblib.load(path)

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