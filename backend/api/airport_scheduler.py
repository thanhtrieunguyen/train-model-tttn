import os
import time
import json
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import sys

# Cấu hình các đường dẫn dựa vào vị trí hiện tại
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
airports_path = os.path.join(project_root, 'backend', 'ml', 'data', 'airports.json')
model_path = os.path.join(project_root, 'backend', 'ml', 'models', 'weather_models.joblib')
output_csv = os.path.join(project_root, 'backend', 'data', 'airport_forecast.csv')

# Thêm project vào sys.path để import module dự báo
sys.path.append(project_root)
from backend.ml.prediction.weather_prediction import WeatherPredictionService, get_current_weather

load_dotenv()
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
if not WEATHER_API_KEY:
    raise Exception("Không tìm thấy WEATHER_API_KEY trong môi trường.")

# Khởi tạo dịch vụ dự báo
prediction_service = WeatherPredictionService(model_path)

def load_airports():
    with open(airports_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Sửa hàm fetch_prediction_for_airport để trả về record dưới dạng dict.
def fetch_prediction_for_airport(airport):
    # location theo định dạng "lat,lon"
    location = f"{airport['latitude']},{airport['longitude']}"
    try:
        result = prediction_service.predict(WEATHER_API_KEY, location, prediction_hours=12)
        current = get_current_weather(WEATHER_API_KEY, location)
        
        # Xây dựng location_data từ thông tin sân bay và thời gian từ current weather.
        location_data = {
            "location": location,
            "city": airport.get("city"),
            "state": airport.get("region"),
            "country": airport.get("country"),
            "country_code": airport.get("country"),
            "timezone": airport.get("timezone"),
            "airport": {
                "icao": airport.get("icao"),
                "iata": airport.get("iata"),
                "name": airport.get("name"),
                "lat": airport.get("latitude"),
                "lon": airport.get("longitude"),
            },
            "timestamp": current.get("timestamp")
        }
        
        # Loại bỏ các trường không cần trong current và dự báo nếu cần (ví dụ "airport" hay "forecast_target")
        # ...existing code...
        
        return {
            "location_data": location_data,
            "current_weather": current,
            "prediction": result["predictions"]
        }
    except Exception as e:
        print(f"[{datetime.now().isoformat()}] Lỗi dự báo cho sân bay {airport.get('iata')}: {e}")
        return None

# def main_loop():
#     airports = load_airports()
#     print("Bắt đầu scheduler cho dự báo sân bay...")
#     while True:
#         now = datetime.now()
#         # Xác định mốc 30 phút tiếp theo (có thể là :00 hoặc :30)
#         if now.minute < 30:
#             target = now.replace(minute=30, second=0, microsecond=0)
#         else:
#             target = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
#         prefetch_start = target - timedelta(minutes=2)
#         sleep_seconds = (prefetch_start - now).total_seconds()
#         if sleep_seconds > 0:
#             print(f"Chờ {int(sleep_seconds)} giây đến prefetch_start ({prefetch_start.time()})")
#             time.sleep(sleep_seconds)
        
#         print(f"Bắt đầu prefetch dự báo cho mốc {target.time()}")
#         all_forecasts = []
#         failed_airports = []
#         # Thử lần đầu cho tất cả
#         for airport in airports:
#             forecasts = fetch_prediction_for_airport(airport)
#             if forecasts:
#                 all_forecasts.extend(forecasts)
#             else:
#                 failed_airports.append(airport)

#         # Trong khoảng thời gian còn lại, retry cho các sân bay lỗi
#         while datetime.now() < target and failed_airports:
#             print(f"Retry cho {len(failed_airports)} sân bay bị lỗi...")
#             for airport in failed_airports[:]:
#                 forecasts = fetch_prediction_for_airport(airport)
#                 if forecasts:
#                     all_forecasts.extend(forecasts)
#                     failed_airports.remove(airport)
#             time.sleep(10)
        
#         # Ghi CSV toàn bộ dữ liệu: current weather và hàng giờ trong 12h sau
#         if all_forecasts:
#             df = pd.DataFrame(all_forecasts)
#             df["update_time"] = datetime.now().isoformat()
#             df.to_csv(output_csv, index=False)
#             print(f"Đã lưu dữ liệu dự báo vào {output_csv}")
#         else:
#             print("Không có dự báo nào thành công trong phiên này.")

#         remaining = (target - datetime.now()).total_seconds()
#         if remaining > 0:
#             time.sleep(remaining)
#         print(f"Hoàn thành phiên dự báo cho mốc {target.time()}. Bắt đầu vòng lặp mới.\n")


# ...existing code...
# Comment lại phiên bản production main_loop
# def main_loop():
#     from backend.db.mongo_connector import store_weather_data
#     MONGO_DB_PASSWORD = os.getenv("MONGO_DB_PASSWORD")
#     airports = load_airports()
#     print("Bắt đầu scheduler cho dự báo sân bay theo chu kỳ 30 phút...")
#     
#     while True:
#         now = datetime.now()
#         if now.minute < 30:
#             target = now.replace(minute=30, second=0, microsecond=0)
#         else:
#             target = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
#         
#         prefetch_start = target - timedelta(minutes=5)
#         sleep_seconds = (prefetch_start - datetime.now()).total_seconds()
#         if sleep_seconds > 0:
#             print(f"Chờ {int(sleep_seconds)} giây đến thời điểm prefetch_start ({prefetch_start.time()})")
#             time.sleep(sleep_seconds)
#         
#         records = {}
#         failed_airports = []
#         for airport in airports:
#             record = fetch_prediction_for_airport(airport)
#             if record:
#                 records[airport.get("iata")] = record
#             else:
#                 failed_airports.append(airport)
#         
#         while datetime.now() < target and failed_airports:
#             print(f"Retry cho {len(failed_airports)} sân bay bị lỗi...")
#             time.sleep(10)
#             for airport in failed_airports[:]:
#                 record = fetch_prediction_for_airport(airport)
#                 if record:
#                     records[airport.get("iata")] = record
#                     failed_airports.remove(airport)
#         
#         final_records = list(records.values())
#         
#         if final_records:
#             df = pd.json_normalize(final_records)
#             df["update_time"] = datetime.now().isoformat()
#             df.to_csv(output_csv, index=False)
#             print(f"Đã lưu dữ liệu dự báo vào {output_csv}")
#         else:
#             print("Không có dự báo nào thành công trong phiên này.")
#         
#         from backend.db.mongo_connector import store_weather_data
#         if os.getenv("MONGO_DB_PASSWORD"):
#             for rec in final_records:
#                 store_weather_data(os.getenv("MONGO_DB_PASSWORD"), rec)
#             print("Đã lưu dữ liệu thời tiết vào MongoDB")
#         else:
#             print("MONGO_DB_PASSWORD chưa được thiết lập.")
#         
#         remaining = (target - datetime.now()).total_seconds()
#         if remaining > 0:
#             time.sleep(remaining)
#         print(f"Hoàn thành phiên dự báo cho mốc {target.time()}. Bắt đầu vòng lặp mới.\n")


# Hàm main_loop test với chu kỳ ngắn (mặc định 2 phút, bạn có thể đổi số phút nếu cần)
def main_loop(test_minutes=2):
    from backend.db.mongo_connector import store_weather_data
    MONGO_DB_PASSWORD = os.getenv("MONGO_DB_PASSWORD")
    airports = load_airports()
    print(f"Bắt đầu test scheduler cho dự báo sân bay với chu kỳ {test_minutes} phút...")
    
    while True:
        now = datetime.now()
        target = now + timedelta(minutes=test_minutes)
        # Prefetch bắt đầu 30 giây trước target
        prefetch_start = target - timedelta(seconds=30)
        sleep_seconds = (prefetch_start - datetime.now()).total_seconds()
        if sleep_seconds > 0:
            print(f"Chờ {int(sleep_seconds)} giây đến prefetch_start ({prefetch_start.time()})")
            time.sleep(sleep_seconds)
        
        records = {}
        failed_airports = []
        for airport in airports:
            record = fetch_prediction_for_airport(airport)
            if record:
                records[airport.get("iata")] = record
            else:
                failed_airports.append(airport)
        
        # Retry cho các sân bay lỗi trong khoảng 30 giây trước target
        while datetime.now() < target and failed_airports:
            print(f"Retry cho {len(failed_airports)} sân bay bị lỗi...")
            time.sleep(10)
            for airport in failed_airports[:]:
                record = fetch_prediction_for_airport(airport)
                if record:
                    records[airport.get("iata")] = record
                    failed_airports.remove(airport)
        
        final_records = list(records.values())
        
        if final_records:
            df = pd.json_normalize(final_records)
            df["update_time"] = datetime.now().isoformat()
            df.to_csv(output_csv, index=False)
            print(f"Đã lưu dữ liệu dự báo vào {output_csv}")
        else:
            print("Không có dự báo nào thành công trong phiên này.")
        
        if MONGO_DB_PASSWORD:
            for rec in final_records:
                store_weather_data(MONGO_DB_PASSWORD, rec)
            print("Đã lưu dữ liệu thời tiết vào MongoDB")
        else:
            print("MONGO_DB_PASSWORD chưa được thiết lập.")
        
        remaining = (target - datetime.now()).total_seconds()
        if remaining > 0:
            time.sleep(remaining)
        print(f"Hoàn thành phiên dự báo cho mốc {target.time()}. Bắt đầu vòng lặp mới.\n")
        
# ...existing code...

if __name__ == "__main__":
    # Chọn chạy chế độ test với chu kỳ 2 phút (có thể thay đổi test_minutes)
    main_loop(test_minutes=2)
