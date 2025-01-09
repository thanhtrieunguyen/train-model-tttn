import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from typing import Dict, List
from sklearn.preprocessing import MinMaxScaler

class WeatherPredictionTest:
    def __init__(self, api_key: str, model_path: str):
        """
        Initialize predictor with API key and model path
        api_key: WeatherAPI key
        model_path: Path to saved trained models
        """
        self.api_key = api_key
        self.models = joblib.load(model_path)
        self.scaler = MinMaxScaler()
        
    def get_current_weather(self, location: str) -> Dict:
        """Get current weather data from WeatherAPI"""
        try:
            url = f"http://api.weatherapi.com/v1/current.json?key={self.api_key}&q={location}&aqi=no"
            response = requests.get(url)
            response.raise_for_status()
            
            data = response.json()
            return {
                'timestamp': datetime.now(),
                'temperature': data['current']['temp_c'],
                'humidity': data['current']['humidity'],
                'wind_speed': data['current']['wind_kph'],
                'pressure': data['current']['pressure_mb'],
                'precipitation': data['current']['precip_mm'],
                'cloud': data['current']['cloud'],
                'visibility': data['current']['vis_km'],
                'uv_index': data['current']['uv'],
                'dewpoint': data['current']['dewpoint_c'],
                'wind_direction': data['current']['wind_degree'],
                'condition': data['current']['condition']['text'],
                'wind_direction_symbol': data['current']['wind_dir'],
                'rain_probability': 0 if data['current']['precip_mm'] == 0 else 100
            }
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch weather data: {str(e)}")
    
    def prepare_prediction_data(self, current_data: Dict, target_hour: datetime) -> pd.DataFrame:
        """Prepare data for prediction"""
        df = pd.DataFrame([current_data])
        
        # Add time features
        df['hour'] = target_hour.hour
        df['day'] = target_hour.day
        df['month'] = target_hour.month
        df['day_of_week'] = target_hour.weekday()
        df['day_of_year'] = target_hour.timetuple().tm_yday
        
        # Create interaction features
        df['wind_speed_rain'] = df['wind_speed'] * df['rain_probability']
        df['cloud_visibility'] = df['cloud'] / (df['visibility'] + 1e-5)
        
        # Encode categorical variables
        for col in ['condition', 'wind_direction_symbol']:
            if col in df.columns:
                df[col] = pd.factorize(df[col])[0]
        
        # Scale features
        df[df.columns] = self.scaler.transform(df)
        
        return df
    
    def predict_weather(self, location: str, hours_ahead: List[int] = [1, 3]) -> Dict:
        """
        Predict weather for specified hours ahead
        location: City name or coordinates
        hours_ahead: List of hours to predict ahead [default: [1, 3]]
        """
        try:
            current_data = self.get_current_weather(location)
            predictions = {}
            
            for hour in hours_ahead:
                target_time = datetime.now() + timedelta(hours=hour)
                
                if hour == 3:
                    # Predict for each hour up to 3 hours
                    hourly_predictions = {}
                    for h in range(1, 4):
                        curr_time = datetime.now() + timedelta(hours=h)
                        pred_data = self.prepare_prediction_data(current_data, curr_time)
                        
                        hour_pred = {}
                        for target in self.models.keys():
                            value = self.models[target].predict(pred_data)[0]
                            hour_pred[target] = round(value, 2)
                        
                        hourly_predictions[f"hour_{h}"] = {
                            "time": curr_time.strftime("%Y-%m-%d %H:%M"),
                            "predictions": hour_pred
                        }
                    predictions[f"{hour}_hours_ahead"] = hourly_predictions
                else:
                    # Single hour prediction
                    pred_data = self.prepare_prediction_data(current_data, target_time)
                    hour_pred = {}
                    for target in self.models.keys():
                        value = self.models[target].predict(pred_data)[0]
                        hour_pred[target] = round(value, 2)
                    
                    predictions[f"{hour}_hour_ahead"] = {
                        "time": target_time.strftime("%Y-%m-%d %H:%M"),
                        "predictions": hour_pred
                    }
            
            return predictions
            
        except Exception as e:
            raise Exception(f"Prediction failed: {str(e)}")

def main():
    # Configuration
    API_KEY = "a5b32b6e884e4b5aa5b95910241712"  # Replace with your WeatherAPI key
    MODEL_PATH = "backend/ml/models/weather_models.joblib"  # Replace with your model path
    LOCATION = "12.668299675,108.120002747"  # Replace with your location
    
    try:
        # Initialize predictor
        predictor = WeatherPredictionTest(API_KEY, MODEL_PATH)
        
        # Make predictions
        predictions = predictor.predict_weather(LOCATION)
        
        # Print predictions in a formatted way
        print("\nWeather Predictions:")
        print("\nCurrent Time:", datetime.now().strftime("%Y-%m-%d %H:%M"))
        print("\n1 Hour Ahead Prediction:")
        print(json.dumps(predictions['1_hour_ahead'], indent=2))
        
        print("\n3 Hours Ahead Predictions (Hourly):")
        for hour in predictions['3_hours_ahead']:
            print(f"\n{hour}:")
            print(json.dumps(predictions['3_hours_ahead'][hour], indent=2))
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    import json
    main()