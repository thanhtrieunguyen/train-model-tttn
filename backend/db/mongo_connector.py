from pymongo import MongoClient
from datetime import datetime
# :/
MONGO_URI_TEMPLATE = "mongodb+srv://admin1:{db_password}@cluster0.nyw26.mongodb.net/"

def get_mongo_client(db_password):
    uri = MONGO_URI_TEMPLATE.format(db_password=db_password)
    client = MongoClient(uri)
    return client

def remove_nested_location(data):
    if isinstance(data, dict):
        if "location" in data:
            data.pop("location", None)
        for key in list(data.keys()):
            remove_nested_location(data[key])
    elif isinstance(data, list):
        for item in data:
            remove_nested_location(item)

def clean_record(record):
    """
    Loại bỏ các thuộc tính trùng lặp:
      - Giữ timestamp ở location_data, bỏ timestamp trong current_weather nếu trùng nhau.
      - Bỏ thuộc tính location trong location_data nếu nó đã trùng với (lat, lon) của airport.
    """
    loc_data = record.get("location_data", {})
    curr_weather = record.get("current_weather", {})

    # Nếu timestamp trong current_weather trùng với location_data, loại bỏ
    if curr_weather.get("timestamp") == loc_data.get("timestamp"):
        curr_weather.pop("timestamp", None)

    if "airport_code" in curr_weather:
        curr_weather.pop("airport_code", None)

    if "name"in curr_weather:
        curr_weather.pop("name", None)

    if "region" in curr_weather:
        curr_weather.pop("region", None)

    if "country" in curr_weather:
        curr_weather.pop("country", None)

    if "lat" in curr_weather:
        curr_weather.pop("lat", None)

    if "lon" in curr_weather:
        curr_weather.pop("lon", None)

    if "tz_id" in curr_weather:
        curr_weather.pop("tz_id", None) 

    # Nếu location_data có trường "location" (ví dụ chứa lat, lon trùng với airport), loại bỏ
    if "location" in loc_data:
        loc_data.pop("location", None)

    if "prediction" in record:
        remove_nested_location(record["prediction"])

    # Cập nhật lại record
    record["location_data"] = loc_data
    record["current_weather"] = curr_weather

    return record

def store_weather_data(db_password, record):
    """
    Lưu dữ liệu thời tiết của 1 sân bay: bao gồm location_data, current_weather và prediction.
    Trước khi lưu, lọc bỏ các thuộc tính trùng lặp không cần thiết.
    """
    # Xử lý record trước khi lưu
    record = clean_record(record)

    client = get_mongo_client(db_password)
    db = client['Airport_Weather']
    collection = db['data_weathers']
    
    # Thêm thời gian tạo record.
    record["created_at"] = datetime.now()
    collection.insert_one(record)
    client.close()