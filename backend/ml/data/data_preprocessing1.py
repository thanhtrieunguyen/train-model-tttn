import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class WeatherDataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def preprocess_data(self, data_path):
        df = pd.read_csv(data_path)

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by=['airport_code', 'timestamp'])

        numeric_columns = {
            'temperature': 'float64',
            'feels_like': 'float64',
            'humidity': 'int64',
            'wind_speed': 'float64',
            'wind_direction': 'float64',
            'gust_speed': 'float64',
            'pressure': 'float64',
            'precipitation': 'float64',
            'rain_probability': 'int64',
            'snow_probability': 'int64',
            'uv_index': 'float64',
            'dewpoint': 'float64',
            'visibility': 'float64',
            'cloud': 'int64',
            'condition_code': 'int64'
        }

        for col, dtype in numeric_columns.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(dtype)

        # Extract time features
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        
        # Drop unnecessary columns
        df = df.drop(['timestamp', 'airport_name', 'latitude', 'longitude', 'snow_probability'], axis=1, errors='ignore')

        # Encode categorical variables
        categorical_columns = ['airport_code', 'condition', 'wind_direction_symbol']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = pd.factorize(df[col])[0]

        # Create interaction features
        df['wind_speed_rain'] = df['wind_speed'] * df['rain_probability']
        df['cloud_visibility'] = df['cloud'] / (df['visibility'] + 1e-5)
        df['feels_like_diff'] = df['feels_like'] - df['temperature']

        # Remove outliers (using IQR)
        numeric_cols = ['temperature', 'humidity', 'wind_speed', 'pressure', 
                       'precipitation', 'cloud', 'visibility', 'rain_probability', 
                       'dewpoint', 'uv_index', 'wind_direction']
        
        for col in numeric_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

        # Handle missing values
        df = df.dropna()

        # Scale features
        df[df.columns] = self.scaler.fit_transform(df)
        
        return df

    