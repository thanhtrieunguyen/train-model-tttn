import requests
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

import os

models_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'enhanced_weather_models.joblib')


class WeatherForecastTester:
    def __init__(self, api_key, models_path, scaler_path):
        self.api_key = api_key
        self.models = joblib.load(models_path)
        self.scaler = MinMaxScaler()

    def fetch_weather_data(self, location):
        """Fetch current weather data from WeatherAPI."""
        url = f"http://api.weatherapi.com/v1/current.json?key={self.api_key}&q={location}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            current = data['current']
            return {
                'temperature': current['temp_c'],
                'humidity': current['humidity'],
                'wind_speed': current['wind_kph'] / 3.6,  # Convert kph to m/s
                'pressure': current['pressure_mb'],
                'precipitation': current['precip_mm'],
                'cloud': current['cloud'],
                'visibility': current['vis_km'],
                'rain_probability': 0,  # Placeholder as this info isn't in the API response
                'wind_direction_symbol': current['wind_dir'],
                'hour': datetime.now().hour,
                'day': datetime.now().day,
                'month': datetime.now().month,
                'day_of_week': datetime.now().weekday(),
                'day_of_year': datetime.now().timetuple().tm_yday,
            }
        else:
            raise Exception(f"Failed to fetch data from WeatherAPI: {response.status_code}, {response.text}")

    def preprocess_input(self, data):
        """Preprocess the input data for prediction."""
        df = pd.DataFrame([data])

        # Encode categorical variables
        df['wind_direction_symbol'] = pd.factorize(df['wind_direction_symbol'])[0]

        # Create interaction features
        df['wind_speed_rain'] = df['wind_speed'] * df['rain_probability']
        df['cloud_visibility'] = df['cloud'] / (df['visibility'] + 1e-5)

        # Scale features
        df = pd.DataFrame(self.scaler.fit_transform(df), columns=df.columns)

        return df

    def predict_future_weather(self, data, intervals):
        """Predict weather for specified future intervals."""
        predictions = {}
        for interval in intervals:
            future_data = data.copy()
            future_time = datetime.now() + timedelta(hours=interval)
            future_data['hour'] = future_time.hour
            future_data['day'] = future_time.day
            future_data['day_of_week'] = future_time.weekday()
            future_data['day_of_year'] = future_time.timetuple().tm_yday

            processed_data = self.preprocess_input(future_data)

            predictions[interval] = {
                target: model.predict(processed_data)[0] for target, model in self.models.items()
            }

        return predictions

if __name__ == "__main__":
    # Initialize the tester
    API_KEY = "a5b32b6e884e4b5aa5b95910241712"
    MODELS_PATH = models_path
    SCALER_PATH = "scaler.joblib"  # Update if scaler was saved separately

    tester = WeatherForecastTester(API_KEY, MODELS_PATH, SCALER_PATH)
    # Fetch current weather data
    location = "12.668299675,108.120002747"
    current_weather = tester.fetch_weather_data(location)
    print("Current weather data:", current_weather)

    # Predict future weather
    intervals = [1, 3, 6]  # After 1 hour, 3 hours, 6 hours
    predictions = tester.predict_future_weather(current_weather, intervals)

    # Display predictions
    for interval, pred in predictions.items():
        print(f"Predicted weather after {interval} hours:")
        for key, value in pred.items():
            print(f"  {key}: {value:.2f}")
