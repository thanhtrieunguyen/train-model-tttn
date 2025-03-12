import os
import json
import requests
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

from backend.ml.data.data_preprocessing import WeatherDataPreprocessor

# Đường dẫn đến các file dữ liệu
current_dir = os.path.dirname(os.path.abspath(__file__))
conditions_path = os.path.join(current_dir, '..', 'data', 'weather_conditions.json')
airports_path = os.path.join(current_dir, '..', 'data', 'airports.json')

def get_current_weather(api_key, location):
    """Lấy thông tin thời tiết hiện tại từ API."""
    try:
        response = requests.get(
            f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={location}&aqi=no"
        )
        data = response.json()
        
        current = data["current"]
        location_info = data["location"]
        
        result = {
            "timestamp": location_info["localtime"],
            "temperature": current["temp_c"],
            "humidity": current["humidity"],
            "wind_speed": current["wind_kph"],
            "wind_direction": current["wind_degree"],
            "wind_direction_symbol": current["wind_dir"],
            "pressure": current["pressure_mb"],
            "precipitation": current["precip_mm"],
            "cloud": current["cloud"],
            "visibility": current["vis_km"],
            "uv_index": current["uv"],
            "condition": {
                "code": current["condition"]["code"],
                "text": current["condition"]["text"],
                "icon": current["condition"]["icon"]
            },
            "gust_speed": current["gust_kph"],
            "dewpoint": current.get("dewpoint_c", 0),  # Có thể không có trong API
            "rain_probability": 0,  # Không có trong API current
            "snow_probability": 0,  # Không có trong API current
            "condition_code": current["condition"]["code"],
            "name": location_info["name"],
            "region": location_info["region"],
            "country": location_info["country"],
            "lat": location_info["lat"],
            "lon": location_info["lon"],
            "tz_id": location_info["tz_id"]
        }
        
        return result
    except Exception as e:
        print(f"Error fetching current weather: {e}")
        return None

class WeatherPredictionService:
    def __init__(self, model_path=None):
        """Khởi tạo dịch vụ dự báo thời tiết."""
        if model_path is None:
            # Mặc định sử dụng mô hình tốt nhất nếu có
            model_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            best_model_path = os.path.join(model_dir, 'models', 'best_weather_models.joblib')
            if os.path.exists(best_model_path):
                self.path = best_model_path
                print(f"Sử dụng mô hình tốt nhất từ: {best_model_path}")
            else:
                self.path = os.path.join(model_dir, 'models', 'rf_weather_models.joblib')
                print(f"Sử dụng mô hình RandomForest từ: {self.path}")
        else:
            self.path = model_path
        
        self.preprocessor = WeatherDataPreprocessor()
        # Model sẽ được load khi cần thiết để tiết kiệm bộ nhớ
        self.models = None
        self.scalers = None
        self.feature_list_for_scale = None
        self.best_model_info = None
        
        # Load model metadata để có thông tin về feature_list_for_scale
        self._load_model_metadata()

    def _load_model_metadata(self):
        """Load chỉ metadata của model mà không load toàn bộ model vào memory"""
        try:
            data = joblib.load(self.path)
            self.feature_list_for_scale = data.get('feature_list_for_scale')
            self.best_model_info = data.get('best_model_info')
            
            if self.best_model_info:
                print("Đã tải thông tin mô hình tốt nhất:")
                for target, info in self.best_model_info.items():
                    print(f" - {target}: {info['model_name']} (RMSE = {info['rmse']:.4f})")
            return True
        except Exception as e:
            print(f"Không thể load model metadata từ {self.path}: {e}")
            return False

    def load_models(self):
        """Load toàn bộ models vào memory khi cần thiết"""
        try:
            data = joblib.load(self.path)
            self.models = data['models']
            self.scalers = data['scalers']
            self.feature_list_for_scale = data.get('feature_list_for_scale')
            self.best_model_info = data.get('best_model_info')
            
            return True
        except Exception as e:
            print(f"Không thể load model từ {self.path}: {e}")
            return False

    def get_historical_weather(self, api_key, location, hours=12):
        """Lấy dữ liệu thời tiết lịch sử trong n giờ qua."""
        try:
            # Lấy thời tiết hiện tại
            current_weather = get_current_weather(api_key, location)
            if not current_weather:
                return []
            
            # Lấy thời tiết lịch sử
            now = datetime.now()
            historical_data = [current_weather]  # Bắt đầu với thời tiết hiện tại
            
            # Lặp qua các giờ trước đó
            for i in range(1, hours):
                past_time = now - timedelta(hours=i)
                
                # Chuyển đổi múi giờ nếu cần
                date_str = past_time.strftime("%Y-%m-%d")
                
                # Sử dụng API history
                try:
                    history_url = f"http://api.weatherapi.com/v1/history.json?key={api_key}&q={location}&dt={date_str}"
                    response = requests.get(history_url)
                    data = response.json()
                    
                    if 'forecast' in data and 'forecastday' in data['forecast']:
                        forecastday = data['forecast']['forecastday'][0]
                        hour_data = forecastday['hour'][past_time.hour]
                        
                        historical_point = {
                            "timestamp": hour_data['time'],
                            "airport_code": location,
                            "temperature": hour_data['temp_c'],
                            "humidity": hour_data['humidity'],
                            "wind_speed": hour_data['wind_kph'],
                            "wind_direction": hour_data['wind_degree'],
                            "wind_direction_symbol": hour_data['wind_dir'],
                            "pressure": hour_data['pressure_mb'],
                            "precipitation": hour_data['precip_mm'],
                            "cloud": hour_data['cloud'],
                            "visibility": hour_data['vis_km'],
                            "uv_index": hour_data.get('uv', 0),
                            "dewpoint": hour_data.get('dewpoint_c', 0),
                            "gust_speed": hour_data['gust_kph'],
                            "rain_probability": hour_data.get('chance_of_rain', 0),
                            "snow_probability": hour_data.get('chance_of_snow', 0),
                            "condition_code": hour_data['condition']['code'],
                        }
                        
                        historical_data.append(historical_point)
                except Exception as e:
                    print(f"Error fetching historical data for {date_str}: {e}")
                    continue
            
            # Sắp xếp dữ liệu theo thời gian
            historical_data.sort(key=lambda x: x['timestamp'])
            
            return historical_data
        except Exception as e:
            print(f"Error in get_historical_weather: {e}")
            return []

    def prepare_input(self, historical_data, future_time):
        """Chuẩn bị dữ liệu đầu vào cho dự báo từ dữ liệu lịch sử."""
        # Tạo DataFrame từ historical_data
        df = pd.DataFrame(historical_data)
        
        # Thêm các đặc trưng thời gian
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = self.preprocessor.extract_time_features(df)
        
        # Thêm đặc trưng mùa
        df = self.preprocessor.extract_season_features(df)
        
        # Xử lý các đặc trưng địa hình nếu cần
        df = self.preprocessor.add_terrain_features(df)
        
        # Mã hóa các biến categorical
        df = self.preprocessor.encode_categorical(df)
        
        # Tạo các đặc trưng lag và rolling
        columns = ['temperature', 'humidity', 'wind_speed', 'pressure', 'precipitation', 
                   'cloud', 'uv_index', 'visibility', 'rain_probability', 'dewpoint']
        df = self.preprocessor.create_lag_and_rolling_features(df, columns)
        
        # Xử lý giá trị thiếu
        df = self.preprocessor.handle_missing_values(df)
        
        # Chọn hàng dữ liệu cuối cùng (mới nhất) để dự báo
        last_row = df.iloc[-1:].copy()
        
        # Cập nhật timestamp thành thời gian tương lai cần dự báo
        last_row['timestamp'] = future_time
        
        # Cập nhật các đặc trưng thời gian
        last_row = self.preprocessor.extract_time_features(last_row)
        last_row = self.preprocessor.extract_season_features(last_row)
        
        # Chọn các cột cần thiết
        input_features = [col for col in last_row.columns if col in self.feature_list_for_scale]
        
        return last_row[input_features]

    def convert_wind_direction_to_symbol(self, degree):
        """Chuyển đổi hướng gió từ độ sang ký hiệu la bàn."""
        directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", 
                      "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW", "N"]
        index = round(degree / 22.5)
        return directions[index % 16]

    def get_condition_info(self, code, is_day):
        """Lấy thông tin điều kiện thời tiết từ mã code."""
        try:
            with open(conditions_path, 'r', encoding='utf-8') as file:
                conditions = json.load(file)
        except Exception as e:
            print(f"Lỗi khi đọc file conditions.json: {e}")
            return {
                "condition_code": code,
                "condition_text": "Unknown",
                "icon": "//cdn.weatherapi.com/weather/64x64/day/116.png"
            }

        # Tìm code chính xác
        for condition in conditions:
            if condition["code"] == code:
                text = condition["day"] if is_day else condition["night"]
                icon = condition["icon"].replace("day", "day" if is_day else "night")
                closest_code = condition["code"]
                return {
                    "condition_code": closest_code,
                    "condition_text": text, 
                    "icon": icon
                }

        # Nếu không tìm thấy, tìm code gần nhất
        closest_condition = min(conditions, key=lambda x: abs(x["code"] - code))
        text = closest_condition["day"] if is_day else closest_condition["night"]
        icon = closest_condition["icon"].replace("day", "day" if is_day else "night")
        closest_code = closest_condition["code"]

        return {
            "condition_code": closest_code,
            "condition_text": text,
            "icon": icon
        }

    def predict(self, api_key, location, prediction_hours=24):
        """Dự báo thời tiết cho n giờ tiếp theo."""
        if not self.models:
            # Load model ngay khi cần
            self.load_models()
        
        # Extract lat, lon from the user input for precision
        lat, lon = map(float, location.split(','))
        
        # Lấy dữ liệu thời tiết lịch sử
        historical_data = self.get_historical_weather(api_key, location, hours=12)
        if not historical_data:
            return {"error": "Không lấy được dữ liệu lịch sử"}
        
        # Updated location_data using precise input coordinates instead of possibly rounded values from API
        location_info = historical_data[0]
        location_data = {
            "name": location_info.get("name", "Unknown"),
            "region": location_info.get("region", ""),
            "country": location_info.get("country", ""),
            "lat": lat,
            "lon": lon,
            "tz_id": location_info.get("tz_id", ""),
            "localtime": location_info.get("timestamp", ""),
        }

        # Thêm thông tin airport nếu có
        try:
            with open(airports_path, 'r', encoding='utf-8') as file:
                airports = json.load(file)
            
            for airport in airports:
                airport_lat = airport.get('latitude')
                airport_lon = airport.get('longitude')
                if airport_lat and airport_lon and abs(lat - float(airport_lat)) < 0.01 and abs(lon - float(airport_lon)) < 0.01:
                    location_data["airport"] = {
                        "name": airport.get("name", ""),
                        "icao": airport.get("icao", ""),
                        "iata": airport.get("iata", ""),
                        "elevation": airport.get("elevation", 0),
                        "terrain": airport.get("terrain", "")
                    }
                    break
        except Exception as e:
            print(f"Lỗi khi tìm thông tin sân bay: {e}")
        
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

            # Đảm bảo giá trị không âm
            for key in ['temperature', 'humidity', 'wind_speed', 'pressure', 'visibility', 'uv_index']:
                if key in prediction and prediction[key] < 0:
                    prediction[key] = 0
                    
            # Giá trị humidity không vượt quá 100%
            if 'humidity' in prediction and prediction['humidity'] > 100:
                prediction['humidity'] = 100

            # Lưu kết quả dự báo
            predictions.append({
                "timestamp": future_time.isoformat(),
                "temperature": prediction["temperature"],
                "humidity": prediction["humidity"],
                "wind_speed": prediction["wind_speed"],
                "wind_direction": prediction["wind_direction"],
                "wind_direction_symbol": prediction.get("wind_direction_symbol", ""),
                "pressure": prediction["pressure"],
                "precipitation": prediction["precipitation"],
                "cloud": prediction["cloud"],
                "uv_index": prediction["uv_index"],
                "visibility": prediction["visibility"],
                "condition": condition,
                "dewpoint": prediction.get("dewpoint", 0),
                "gust_speed": prediction.get("gust_speed", 0),
                "rain_probability": prediction.get("rain_probability", 0),
                "snow_probability": prediction.get("snow_probability", 0),
                # Thêm thông tin về mô hình được sử dụng
                "model_info": self.best_model_info[target] if self.best_model_info and target in self.best_model_info else {}
            })
        
        # Trả về kết quả dự báo kèm thông tin địa điểm
        return {
            "location": location_data,
            "current_weather": historical_data[0] if len(historical_data) > 0 else {},
            "forecasts": predictions
        }

# Example usage
if __name__ == "__main__":
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'weather_models.joblib'))
    prediction_service = WeatherPredictionService(model_path)
    api_key = 'a5b32b6e884e4b5aa5b95910241712'
    location = '12.668299675,108.120002747'
    output = prediction_service.predict(api_key, location, prediction_hours=24)
    print(json.dumps(output, indent=4, ensure_ascii=False))
    # pass
