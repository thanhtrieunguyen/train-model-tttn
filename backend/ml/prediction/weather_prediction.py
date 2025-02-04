import joblib
import pandas as pd
import numpy as np
import os
import sys
import requests
from datetime import datetime, timedelta
from math import ceil
import json

# Set up project paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(project_root)
from backend.ml.data.data_preprocessing import WeatherDataPreprocessor

# Define model path
conditions_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'conditions.json')

class WeatherPredictionService:
    def __init__(self, model_path=None):
        if model_path is None:
            raise ValueError("Cần cung cấp đường dẫn model khi khởi tạo WeatherPredictionService")
        try:
            self.load_models(model_path)
        except Exception as e:
            print(f"Error loading models: {e}")
        self.preprocessor = WeatherDataPreprocessor()
    
    def load_models(self, path):
        """Chỉ load metadata, không load tất cả models vào RAM"""
        self.path = path  # Lưu đường dẫn model
        data = joblib.load(path)  # Load metadata trước
        self.feature_list_for_scale = data['feature_list_for_scale']  # Lưu danh sách feature

    def get_historical_weather(self, api_key, location, hours=12):
        """Fetch historical weather data for the last n hours."""
        historical_data = []
        current_time = datetime.now()

        # Get current weather
        current_weather = get_current_weather(api_key, location)
        
        print(f"Current weather: {current_weather}")

        historical_data.append(current_weather)

        # Fetch historical data for each previous hour
        for i in range(1, hours + 1):
            time = current_time - timedelta(hours=i)
            historical_url = f"http://api.weatherapi.com/v1/history.json?key={api_key}&q={location}&dt={time.strftime('%Y-%m-%d')}"
            
            response = requests.get(historical_url)
            if response.status_code != 200:
                print(f"Error fetching data for {time}: {response.text}")
                continue
            
            data = response.json()
            target_hour = time.hour
            hour_data = None

            for hour in data['forecast']['forecastday'][0]['hour']:
                hour_time = datetime.strptime(hour['time'], '%Y-%m-%d %H:%M').hour
                if hour_time == target_hour:
                    hour_data = hour
                    break

            if hour_data:
                historical_data.append({
                    'timestamp': hour_data['time'],
                    'airport_code': location,
                    'temperature': hour_data['temp_c'],
                    'humidity': hour_data['humidity'],
                    'wind_speed': hour_data['wind_kph'],
                    'wind_direction': hour_data['wind_degree'],
                    'pressure': hour_data['pressure_mb'],
                    'precipitation': hour_data['precip_mm'],
                    'cloud': hour_data['cloud'],
                    'gust_speed': hour_data['gust_kph'],
                    'condition': hour_data['condition']['text'],
                    'rain_probability': hour_data['chance_of_rain'],
                    'snow_probability': hour_data['chance_of_snow'],
                    'visibility': hour_data['vis_km'],
                    'uv_index': hour_data['uv'],
                    'condition_code': hour_data['condition']['code'],
                    'dewpoint': hour_data.get('dewpoint_c')
                })

        return historical_data

    def prepare_input(self, historical_data, future_time):
        """Prepare input data for prediction using historical data."""
        df_historical = pd.DataFrame(historical_data)
        
        # Extract time features
        df_historical = self.preprocessor.extract_time_features(df_historical)
        
        # Create future prediction time
        future_data = [{'timestamp': future_time}]
        df_future = pd.DataFrame(future_data)
        
        # Combine historical and future data
        df_combined = pd.concat([df_historical, df_future], ignore_index=True)
        
        # Encode categorical before creating features
        df_combined = self.preprocessor.encode_categorical(df_combined)
        
        # Create lag and rolling features
        columns = ['temperature', 'humidity', 'wind_speed', 'pressure', 'precipitation', 
                'cloud', 'uv_index', 'visibility', 'rain_probability', 'dewpoint']
        df_combined = self.preprocessor.create_lag_and_rolling_features(df_combined, columns)
        
        # Handle missing values
        df_combined = self.preprocessor.handle_missing_values(df_combined)
        
        # Select features for prediction
        df_combined = self.preprocessor.select_features(df_combined)
        
        # Get only future row
        df_predict = df_combined.tail(1).copy()
        
        # Ensure columns match training data
        for col in self.feature_list_for_scale:
            if col not in df_predict.columns:
                df_predict[col] = 0
                
        return df_predict[self.feature_list_for_scale]

    # Hàm chuyển đổi hướng gió từ độ sang ký hiệu
    def convert_wind_direction_to_symbol(self, degree):
        """Convert wind direction from degrees to compass symbol."""
        directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                    'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
        idx = int((degree % 360) / 22.5)  # 360 độ chia thành 16 phần
        return directions[idx]
    
    # Hàm lấy thông tin về điều kiện thời tiết từ mã code
    def get_condition_info(self, code, is_day):
        with open(conditions_path, 'r', encoding='utf-8') as file:
            conditions = json.load(file)

        # Tìm code chính xác
        for condition in conditions:
            if condition["code"] == code:
                text = condition["day"] if is_day else condition["night"]
                icon = condition["icon"]
                closest_code = condition["code"]
                return {
                    "condition_code": closest_code,
                    "condition_text": text, 
                    "icon": icon}

        # Nếu không tìm thấy, tìm code gần nhất
        closest_condition = min(conditions, key=lambda x: abs(x["code"] - code))
        text = closest_condition["day"] if is_day else closest_condition["night"]
        icon = closest_condition["icon"]
        closest_code = closest_condition["code"]

        return {
            "condition_code": closest_code,  # Cập nhật code gần nhất
            "condition_text": text,
            "icon": icon
        }

    
    def predict(self, api_key, location, prediction_hours=24):
        if not hasattr(self, 'models'):  # Kiểm tra nếu model chưa được load
            data = joblib.load(self.path)  # Load model ngay khi cần
            self.models = data['models']
            self.scalers = data['scalers']
            
        """Predict weather conditions for the next n hours."""
        # Lấy dữ liệu thời tiết lịch sử
        historical_data = self.get_historical_weather(api_key, location, hours=12)

        location_info = historical_data[0]  # Giả sử thông tin vị trí từ bản ghi đầu tiên
        lat, lon = map(float, location.split(','))  # Tách lat và lon từ chuỗi location

        location_data = {
            "name": "Buon Me Thuot",  # Giả định tên
            "region": "",
            "country": "Vietnam",
            "lat": lat,
            "lon": lon,
            "tz_id": "Asia/Ho_Chi_Minh",
            "localtime": location_info.get("timestamp")  # Lấy thời gian hiện tại
        }
            
        # Tạo timestamp cho các thời điểm dự báo
        current_time = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        future_times = [current_time + timedelta(hours=i) for i in range(prediction_hours)]
        
        predictions = []

        for future_time in future_times:
            # Chuẩn bị dữ liệu đầu vào (dựa trên historical_data hiện tại)
            input_data = self.prepare_input(historical_data, future_time)

            # Dự báo cho từng mục tiêu (temperature, humidity, ...)
            prediction = {}
            for target, model in self.models.items():
                scaler = self.scalers[target]
                input_data_scaled = scaler.transform(input_data)
                prediction[target] = round(float(model.predict(input_data_scaled)[0]), 1)
                
                # Nếu dự đoán là wind_direction, chuyển sang ký tự
                if target == 'wind_direction':
                    prediction['wind_direction_symbol'] = self.convert_wind_direction_to_symbol(prediction[target])

            # Xử lý condition_code
            condition_code = int(prediction.get('condition_code', 0))
            is_day = 6 <= future_time.hour <= 18  # Giả định giờ ban ngày từ 6h sáng đến 18h
            condition_info = self.get_condition_info(condition_code, is_day)
            
            condition = {
                "code": condition_info["condition_code"],
                "text": condition_info["condition_text"],
                "icon": condition_info["icon"]
            }
                
            # Lưu kết quả dự báo vào historical_data
            historical_data.append({
                'timestamp': future_time.isoformat(),
                'airport_code': location,
                **prediction,  # Thêm các giá trị dự báo vào historical_data
                'wind_direction_symbol': prediction.get('wind_direction_symbol'),
            })

            # Cập nhật lại các đặc trưng lag và rolling
            df_historical = pd.DataFrame(historical_data)
            df_historical = self.preprocessor.extract_time_features(df_historical)
            columns = ['temperature', 'humidity', 'wind_speed', 'pressure', 
                    'precipitation', 'cloud', 'uv_index', 'visibility', 
                    'rain_probability', 'dewpoint']
            df_historical = self.preprocessor.create_lag_and_rolling_features(df_historical, columns)
            df_historical = self.preprocessor.handle_missing_values(df_historical)
            historical_data = df_historical.to_dict(orient='records')  # Cập nhật lại historical_data

            # Lưu kết quả dự báo
            predictions.append({
                "timestamp": future_time.isoformat(),
                "location": location,
                "temperature": prediction['temperature'],
                "humidity": prediction['humidity'],
                "wind_speed": prediction['wind_speed'],
                "wind_direction": prediction['wind_direction'],
                "wind_direction_symbol": prediction['wind_direction_symbol'],
                "pressure": prediction['pressure'],
                "precipitation": prediction['precipitation'],
                "cloud": prediction['cloud'],
                "gust_speed": prediction['gust_speed'],
                "condition": condition,
                "rain_probability": prediction['rain_probability'],
                "snow_probability": prediction['snow_probability'],
                "visibility": prediction['visibility'],
                "uv_index": prediction['uv_index'],
                "dewpoint": prediction['dewpoint']
            })

        return {"location": location_data, "predictions": predictions}

def get_current_weather(api_key, location):
    """Fetch current weather data."""
    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={location}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Error fetching current weather: {response.text}")
    
    data = response.json()
    return {
        'timestamp': data['location']['localtime'],
        'airport_code': location,
        'temperature': data['current']['temp_c'],
        'humidity': data['current']['humidity'],
        'wind_speed': data['current']['wind_kph'],
        'wind_direction': data['current']['wind_degree'],
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
        'dewpoint': data['current'].get('dewpoint_c')
    }

# Example usage
if __name__ == "__main__":
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models', 'weather_models.joblib'))
    
    # Khởi tạo dịch vụ dự báo
    prediction_service = WeatherPredictionService(model_path)

    # Dữ liệu test
    api_key = 'a5b32b6e884e4b5aa5b95910241712'
    location = '12.668299675,108.120002747'

    # Dự báo thời tiết 24 giờ
    output = prediction_service.predict(api_key, location, prediction_hours=24)
    print(json.dumps(output, indent=4, ensure_ascii=False))
