import joblib
import pandas as pd
import os
import sys
import requests
from datetime import datetime, timedelta

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(project_root)
from backend.ml.data.data_preprocessing import WeatherDataPreprocessor

models_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'weather_models_xgb.joblib')

class WeatherPredictionService:
    def __init__(self, model_path=models_path):
        self.models = joblib.load(model_path)
        self.preprocessor = WeatherDataPreprocessor()
        
    def prepare_input(self, current_weather, current_time, time_offsets):
        """Prepare input data for prediction."""
        input_data = []
        
        for offset in time_offsets:
            future_time = current_time + timedelta(hours=offset)
            features = {
                'hour': future_time.hour,
                'day': future_time.day,
                'month': future_time.month,
                'day_of_week': future_time.weekday(),
                'day_of_year': future_time.timetuple().tm_yday,
                **current_weather
            }
            input_data.append(features)
        
        df = pd.DataFrame(input_data)
        df = self.preprocessor.encode_categorical(df)
        df = self.preprocessor.select_features(df)
        return df
    
    def predict(self, current_weather, current_time, time_offsets):
        """Predict weather for the specified time offsets."""
        input_data = self.prepare_input(current_weather, current_time, time_offsets)
        
        # Ensure the input data has the same features as the training data
        required_features = list(self.models['temperature'].feature_names_in_)
        input_data = input_data[required_features]
        
        predictions = {}
        
        for target, model in self.models.items():
            predictions[target] = model.predict(input_data)
        
        return predictions

def get_current_weather(api_key, location):
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
        'condition': data['current']['condition']['text'],
        'rain_probability': 0,  # WeatherAPI does not provide this directly
        'snow_probability': 0,  # WeatherAPI does not provide this directly
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
    
    api_key = 'a5b32b6e884e4b5aa5b95910241712'  # Replace with your WeatherAPI key
    location = '12.668299675,108.120002747'  # Replace with desired location
    current_weather = get_current_weather(api_key, location)
    
    current_time = datetime.now()
    
    # User input for time offsets
    time_offsets_input = input("Enter time offsets in hours (comma-separated, e.g., 0.5,1,3,6,12): ")
    time_offsets = [float(offset.strip()) for offset in time_offsets_input.split(',')]
    
    predictions = prediction_service.predict(current_weather, current_time, time_offsets)
    
    for offset in time_offsets:
        future_time = current_time + timedelta(hours=offset)
        print(f"Predictions for {future_time}:")
        for target in predictions.keys():
            prediction = predictions[target][time_offsets.index(offset)]
            print(f"  {target}: {prediction}")
        print('\n')