from pymongo import MongoClient
from datetime import datetime

MONGO_URI_TEMPLATE = "mongodb+srv://nthtrieu204:{db_password}@dbforlearn.hhc4g.mongodb.net/"

def get_mongo_client(db_password):
    uri = MONGO_URI_TEMPLATE.format(db_password=db_password)
    client = MongoClient(uri)
    return client

# Sửa lại hàm store_weather_data để lưu record theo dạng mới.
def store_weather_data(db_password, record):
    """
    Lưu dữ liệu thời tiết của 1 sân bay: bao gồm location_data, current_weather và prediction.
    """
    client = get_mongo_client(db_password)
    db = client['Airport_Weather']
    collection = db['airport_weather']
    
    # Thêm thời gian tạo record.
    record["created_at"] = datetime.now()
    collection.insert_one(record)
    client.close()
