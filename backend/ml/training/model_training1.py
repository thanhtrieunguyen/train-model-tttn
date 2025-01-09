import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import os
import sys
import warnings
from joblib import parallel_backend
warnings.filterwarnings("ignore", category=UserWarning, module="joblib.externals.loky.backend.resource_tracker")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(project_root)
from backend.ml.data.data_preprocessing1 import WeatherDataPreprocessor

dataset_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'historical_weather_data.csv')
models_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'enhanced_weather_models.joblib')

class EnhancedWeatherPredictor:
    def __init__(self):
        self.models = {}
        self.metrics = {}
        self.preprocessor = WeatherDataPreprocessor()

    def train_model(self, X, y, target_name):
        """Train a single model with hyperparameter tuning."""
        param_grid = {
            'n_estimators': [100, 200, 500],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10],
        }

        tscv = TimeSeriesSplit(n_splits=5)
        
        for train_index, test_index in tscv.split(X):
            print(f"Train size: {len(train_index)}, Test size: {len(test_index)}")
            
        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_grid,
            scoring='neg_mean_squared_error',
            cv=tscv,
            verbose=2,
            n_jobs=-1
        )
        
        with parallel_backend('loky', inner_max_num_threads=1):
            grid_search.fit(X, y)

        best_model = grid_search.best_estimator_
        return best_model

    def evaluate_model(self, model, X_test, y_test, target_name):
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        self.metrics[target_name] = {'mse': mse, 'rmse': np.sqrt(mse), 'r2': r2}

        # Plot predictions
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5, label='Predicted vs Actual')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal Fit')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Actual vs Predicted for {target_name}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'backend/ml/models/rating_chart/{target_name}.png')
        plt.close()

    def feature_importance(self, model, X, target_name):
        """Display and save feature importance."""
        importance = model.feature_importances_
        feature_names = X.columns
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        # Plot feature importance
        plt.figure(figsize=(12, 8))
        plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
        plt.xlabel('Importance')
        plt.title(f'Feature Importance for {target_name}')
        plt.gca().invert_yaxis()
        plt.grid(True)
        plt.savefig(f'backend/ml/models/rating_chart1/importance_{target_name}.png')
        plt.close()

    def train_all_targets(self, data_path):
        df = self.preprocessor.preprocess_data(data_path)

        df = df.sort_values(by=['airport_code', 'day_of_year', 'hour'])

        targets = ['temperature', 'humidity', 'wind_speed', 'pressure', 
                   'precipitation', 'cloud', 'visibility', 'uv_index', 
                   'dewpoint', 'rain_probability']
        
        # Tránh trường hợp không có target trong dataset
        X = df.drop(columns=[t for t in targets if t in df.columns], errors='ignore')
        y = df[targets].copy() if all(t in df.columns for t in targets) else None

        for target in targets:
            print(f"Training model for {target}...")
            y = df[target]

            train_size = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
            y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
            
            best_model = self.train_model(X_train, y_train, target)
            self.models[target] = best_model
            
            self.evaluate_model(best_model, X_test, y_test, target)
            self.feature_importance(best_model, X, target)

        # Save all models
        joblib.dump(self.models, models_path)
        print("All models saved.")

    def print_metrics(self):
        for target, metrics in self.metrics.items():
            print(f"Metrics for {target}:")
            print(f"MSE: {metrics['mse']}")
            print(f"RMSE: {metrics['rmse']}")
            print(f"R2: {metrics['r2']}")
            print() 

if __name__ == "__main__":
    predictor = EnhancedWeatherPredictor()
    predictor.train_all_targets(dataset_path)
    predictor.print_metrics()
