import requests
import pandas as pd
from datetime import datetime, timedelta
import joblib

# Load mô hình và các đối tượng cần thiết
def load_model_and_objects():
    models_dir = f"../models"
    model = joblib.load(f"{models_dir}/weather_rf_model.joblib")
    scaler = joblib.load(f"{models_dir}/weather_scaler.joblib")
    label_encoder = joblib.load(f"{models_dir}/weather_label_encoder.joblib")
    with open("../models/features.txt", 'r') as f:
        features = f.read().splitlines()
    return model, scaler, label_encoder, features

# Lấy dữ liệu thời tiết hiện tại từ WeatherAPI
def get_current_weather(api_key, lat, lon):
    url = "http://api.weatherapi.com/v1/current.json"
    params = {'key': api_key, 'q': f"{lat},{lon}"}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print("Failed to fetch current weather data.")
        return None

# Chuẩn bị dữ liệu đầu vào từ dữ liệu thời tiết hiện tại
def prepare_input(current_weather, features):
    data = {
        'timestamp': datetime.now(),
        'temperature': current_weather['current']['temp_c'],
        'feels_like': current_weather['current']['feelslike_c'],
        'humidity': current_weather['current']['humidity'],
        'wind_speed': current_weather['current']['wind_kph'],
        'wind_direction': current_weather['current']['wind_degree'],
        'gust_speed': current_weather['current']['gust_kph'],
        'pressure': current_weather['current']['pressure_mb'],
        'precipitation': current_weather['current']['precip_mm'],
        'uv_index': current_weather['current']['uv'],
        'visibility': current_weather['current']['vis_km'],
        'cloud': current_weather['current']['cloud']
    }
    df = pd.DataFrame([data])
    df = df[features]  # Sắp xếp theo đúng thứ tự các đặc trưng
    return df

# Dự đoán thời tiết cho các mốc thời gian
def predict_weather(model, scaler, input_data, hours):
    predictions = {}
    for hour in hours:
        future_data = input_data.copy()
        future_data['timestamp'] = future_data['timestamp'] + timedelta(hours=hour)
        future_scaled = scaler.transform(future_data)
        prediction = model.predict(future_scaled)
        predictions[f"After {hour} hours"] = prediction[0]
    return predictions

# Main function
if __name__ == "__main__":
    # Cấu hình
    WEATHER_API_KEY = "a5b32b6e884e4b5aa5b95910241712"  # Thay bằng API key của bạn
    LAT = 21.0285  # Vĩ độ (ví dụ Hà Nội)
    LON = 105.8542  # Kinh độ (ví dụ Hà Nội)

    # Load mô hình và các đối tượng
    print("Loading model and objects...")
    model, scaler, label_encoder, features = load_model_and_objects()

    # Lấy dữ liệu thời tiết hiện tại
    print("Fetching current weather...")
    current_weather = get_current_weather(WEATHER_API_KEY, LAT, LON)
    if not current_weather:
        exit()

    # Chuẩn bị dữ liệu đầu vào
    print("Preparing input data...")
    input_data = prepare_input(current_weather, features)

    # Dự đoán
    print("Predicting weather for the next hours...")
    hours = [3, 6, 12]
    predictions = predict_weather(model, scaler, input_data, hours)

    # Hiển thị kết quả
    print("\nWeather Predictions:")
    for key, value in predictions.items():
        print(f"{key}: {value}")
