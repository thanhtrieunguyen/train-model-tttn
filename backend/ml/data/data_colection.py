import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import json
from dotenv import load_dotenv
import os

load_dotenv()


ninja_api_key = os.getenv('NINJA_API_KEY')
weather_api_key = os.getenv('WEATHER_API_KEY')

class HistoricalWeatherCollector:
    def __init__(self, ninja_api_key, weather_api_key):
        self.ninja_api_key = ninja_api_key
        self.weather_api_key = weather_api_key
        self.airports_data = []
        self.filename = 'weather_dataset.csv'
        
    def get_vietnam_airports(self):
        url = "https://api.api-ninjas.com/v1/airports"
        headers = {'X-Api-Key': self.ninja_api_key}
        params = {'country': 'VN'}
        
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            self.airports_data = response.json()
            return True
        return False

    def get_historical_weather(self, lat, lon, date):
        url = "http://api.weatherapi.com/v1/history.json"
        params = {
            'key': self.weather_api_key,
            'q': f"{lat},{lon}",
            'dt': date.strftime('%Y-%m-%d')
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error fetching data for {date}: {str(e)}")
        return None

    def save_data(self, data_points):
        df = pd.DataFrame(data_points)
        if not df.empty:
            mode = 'a' if pd.io.common.file_exists(self.filename) else 'w'
            header = not pd.io.common.file_exists(self.filename)
            df.to_csv(self.filename, mode=mode, header=header, index=False)

    def collect_historical_data(self):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)

        for airport in self.airports_data:
            current_date = start_date
            while current_date <= end_date:
                daily_data = []

                weather = self.get_historical_weather(
                    airport['latitude'],
                    airport['longitude'],
                    current_date
                )
                    
                if weather and 'forecast' in weather:
                    for hour in weather['forecast']['forecastday'][0]['hour']:
                        data_point = {
                            'timestamp': hour['time'], # Thời gian dự báo
                            'airport_code': airport['iata'], # Mã sân bay
                            'airport_name': airport['name'], # Tên sân bay
                            'latitude': airport['latitude'], # Vĩ độ
                            'longitude': airport['longitude'], # Kinh độ
                            'temperature': hour['temp_c'], # Nhiệt độ (độ C)
                            'feels_like': hour['feelslike_c'], # Nhiệt độ cảm nhận (độ C)
                            'humidity': hour['humidity'], # Độ ẩm (%)
                            'wind_speed': hour['wind_mph'], # Tốc độ gió (mph)
                            'wind_speed': hour['wind_kph'], # Tốc độ gió (km/h)
                            'wind_direction': hour['wind_degree'], # Hướng gió (độ)
                            'wind_direction_symbol': hour['wind_dir'], # Ký hiệu hướng gió
                            'gust_speed': hour['gust_mph'], # Tốc độ gió giật (mph)
                            'pressure': hour['pressure_mb'], # Áp suất (mb)
                            'precipitation': hour['precip_mm'], # Lượng mưa (mm)
                            'rain_probability': hour['chance_of_rain'], # Xác suất mưa (%)
                            'snow_probability': hour['chance_of_snow'], # Xác suất tuyết (%)
                            'uv_index': hour['uv'], # Chỉ số UV
                            'dewpoint': hour['dewpoint_c'], # Điểm sương (độ C)
                            'visibility': hour['vis_km'], # Tầm nhìn (km)
                            'cloud': hour['cloud'],  # Mây (%)
                            'condition': hour['condition']['text'], # Tình trạng thời tiết
                            'condition_code': hour['condition']['code'], # Mã tình trạng thời tiết
                        }
                        daily_data.append(data_point)
    
                time.sleep(1)  # Rate limit prevention
                
                # Save data for current day
                self.save_data(daily_data)
                print(f"Saved data for {current_date.strftime('%Y-%m-%d')}")
                
                current_date += timedelta(days=1)

def main():
    collector = HistoricalWeatherCollector(
        ninja_api_key=os.getenv('NINJA_API_KEY'),
        weather_api_key=os.getenv('WEATHER_API_KEY')
    )
    
    if not collector.get_vietnam_airports():
        print("Could not fetch airports list!")
        return
    
    collector.collect_historical_data()

if __name__ == "__main__":
    main()