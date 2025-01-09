import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
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
from backend.ml.data.data_test import WeatherDataPreprocessor

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
        X, self.df = self.preprocessor.preprocess(data_path)
        
        # Prepare target variables
        targets = ['temperature', 'humidity', 'wind_speed', 'pressure', 
                  'precipitation', 'cloud']
        
        return X, {target: self.df[target] for target in targets}
    
    def train(self, X, y_dict):
        """Train models for each weather parameter."""
        self.models = {}
        self.metrics = {}
        
        print("Training models...")

        for target_name, y in tqdm(y_dict.items()):
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            self.models[target_name] = model
            self.metrics[target_name] = {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'r2': r2
            }
            
            # Plot actual vs predicted values
            self.plot_predictions(y_test, y_pred, target_name)
        
        return self.metrics
    
    def plot_predictions(self, y_true, y_pred, target_name):
        """Plot actual vs predicted values."""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Actual vs Predicted {target_name}')
        plt.savefig(f'predictions_{target_name}.png')
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
    X, y_dict = predictor.prepare_data(dataset_path)
    
    # print type data columns X
    print(X.dtypes)
    
    # Train models
    metrics = predictor.train(X, y_dict)
    
    # Print metrics
    for target, metric in metrics.items():
        print(f"\nMetrics for {target}:")
        print(f"RMSE: {metric['rmse']:.2f}")
        print(f"R2 Score: {metric['r2']:.2f}")
    
    # Save models
    predictor.save_models()

if __name__ == "__main__":
    main()