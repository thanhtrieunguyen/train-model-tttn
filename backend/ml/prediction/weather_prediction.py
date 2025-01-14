import joblib
import pandas as pd
import numpy as np
import os
import sys
import requests
from datetime import datetime, timedelta

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(project_root)
from backend.ml.data.data_preprocessing import WeatherDataPreprocessor

models_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'weather_models.joblib')

class WeatherPredictionService:
    def __init__(self, model_path=models_path):
        try:
            self.load_models(model_path)
        except Exception as e:
            print(f"Error loading models: {e}")
        self.preprocessor = WeatherDataPreprocessor()
    
    def load_models(self, path=models_path):
        """Load trained models."""
        data = joblib.load(path)
        self.models = data['models']
        self.scalers = data['scalers']
        self.feature_list_for_scale = data['feature_list_for_scale']  # Add this line

    def get_historical_weather(self, api_key, location, hours=3):
        """Get historical weather data for the last n hours."""
        historical_data = []
        current_time = datetime.now()
        
        # Get current weather
        current_weather = get_current_weather(api_key, location)
        historical_data.append(current_weather)
        
        # Get historical weather for each hour
        for i in range(1, hours + 1):
            time = current_time - timedelta(hours=i)
            print("Historical time:", time)

            historical_url = f"http://api.weatherapi.com/v1/history.json?key={api_key}&q={location}&dt={time.strftime('%Y-%m-%d')}"
            print("Historical URL:", historical_url)
            
            response = requests.get(historical_url)
            data = response.json()
            
            # Find the closest hour data
            target_hour = time.hour
            hour_data = None
            
            for hour in data['forecast']['forecastday'][0]['hour']:
                hour_time = datetime.strptime(hour['time'], '%Y-%m-%d %H:%M').hour
                if hour_time <= target_hour:
                    hour_data = hour
                else:
                    break
                    
            if hour_data:
                historical_entry = {
                    'timestamp': time.isoformat(),
                    'airport_code': location,
                    'temperature': hour_data['temp_c'],
                    'humidity': hour_data['humidity'],
                    'wind_speed': hour_data['wind_kph'],
                    'wind_direction_symbol': hour_data['wind_dir'],
                    'pressure': hour_data['pressure_mb'],
                    'precipitation': hour_data['precip_mm'],
                    'cloud': hour_data['cloud'],
                    'gust_speed': hour_data['gust_kph'],
                    'condition': hour_data['condition']['text'],
                    'rain_probability': hour_data.get('chance_of_rain', 0),
                    'snow_probability': hour_data.get('chance_of_snow', 0),
                    'visibility': hour_data['vis_km'],
                    'uv_index': hour_data['uv'],
                    'condition_code': hour_data['condition']['code'],
                    'dewpoint': hour_data.get('dewpoint_c', None)
                }
                historical_data.append(historical_entry)
        
        return historical_data

    def prepare_input(self, historical_data, future_times):
        """Prepare input data for prediction using historical data."""
        # Convert historical data to DataFrame
        df_historical = pd.DataFrame(historical_data)
        
        # Extract time features and encode categorical variables
        df_historical = self.preprocessor.extract_time_features(df_historical)
        df_historical = self.preprocessor.encode_categorical(df_historical)
        
        # Create future prediction times
        future_data = []
        for future_time in future_times:
            future_data.append({'timestamp': future_time})
        
        df_future = pd.DataFrame(future_data)
        df_future = self.preprocessor.encode_categorical(df_future)
        
        # Combine historical and future data
        df_combined = pd.concat([df_historical, df_future], ignore_index=True)
        
        # Create lag and rolling features
        columns = ['temperature', 'humidity', 'wind_speed', 'pressure', 'precipitation', 
                  'cloud', 'uv_index', 'visibility', 'rain_probability', 'dewpoint']
        df_combined = self.preprocessor.create_lag_and_rolling_features(df_combined, columns)
        
        # Handle missing values
        df_combined = self.preprocessor.handle_missing_values(df_combined)
        
        # Select features for prediction
        df_combined = self.preprocessor.select_features(df_combined)
        
        # Get only the future rows for prediction
        df_predict = df_combined.tail(len(future_times))
        
        # Ensure the feature list matches the training feature list
        df_predict = df_predict[self.feature_list_for_scale]  # Add this line
        
        return df_predict
    
    def predict(self, api_key, location, prediction_hours=3):
        """Predict weather for the next n hours using historical data."""
        # Get historical data
        historical_data = self.get_historical_weather(api_key, location)
        
        # Generate future times for prediction
        current_time = datetime.now()
        future_times = [current_time + timedelta(hours=i) for i in range(1, prediction_hours + 1)]
        
        # Prepare input data
        input_data = self.prepare_input(historical_data, future_times)
        
        # Predict weather parameters
        predictions = {}
        for target, model in self.models.items():
            scaler = self.scalers[target]
            input_data_scaled = scaler.transform(input_data)
            predictions[target] = model.predict(input_data_scaled)
        
        return predictions

def get_current_weather(api_key, location):
    """Get current weather data."""
    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={location}"
    response = requests.get(url)
    data = response.json()
    
    current_weather = {
        'timestamp': data['location']['localtime'],
        'airport_code': location,
        'temperature': data['current']['temp_c'],
        'humidity': data['current']['humidity'],
        'wind_speed': data['current']['wind_kph'],
        'wind_direction_symbol': data['current']['wind_dir'],
        'pressure': data['current']['pressure_mb'],
        'precipitation': data['current']['precip_mm'],
        'cloud': data['current']['cloud'],
        'gust_speed': data['current']['gust_kph'],
        'condition': data['current']['condition']['text'],
        'rain_probability': 0,
        'snow_probability': 0,
        'visibility': data['current']['vis_km'],
        'uv_index': data['current']['uv'],
        'condition_code': data['current']['condition']['code'],
        'dewpoint': data['current']['dewpoint_c'] if 'dewpoint_c' in data['current'] else None
    }
    
    return current_weather

# Example usage
if __name__ == "__main__":
    model_path = models_path
    prediction_service = WeatherPredictionService(model_path)
    
    api_key = 'a5b32b6e884e4b5aa5b95910241712'
    location = '12.668299675,108.120002747'
    
    # Get historical weather data for the last 3 hours
    historical_data = prediction_service.get_historical_weather(api_key, location, hours=3)
    print("\nHistorical weather data:")
    for data in historical_data:
        print(data)

    # # Get current weather
    current_weather = get_current_weather(api_key, location)
    print("\nCurrent weather:")
    print(current_weather)
    

    # # Get predictions for next 3 hours
    predictions = prediction_service.predict(api_key, location, prediction_hours=3)
    
    # # Print predictions
    for pred in predictions:
        print(f"\nPredictions for {pred['timestamp']}:")
        for target, value in pred['predictions'].items():
            print(f"  {target}: {value:.2f}")