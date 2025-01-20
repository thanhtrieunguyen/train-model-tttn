import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime
import os

class WeatherDataPreprocessor:
    def __init__(self):
        self.airport_encoder = LabelEncoder()
        self.scaler = StandardScaler()  # Initialize the scaler
    
    def extract_time_features(self, df):
        """Extract time-based features from timestamp."""
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp']).copy()
        df.loc[:, 'hour'] = df['timestamp'].dt.hour
        df.loc[:, 'day'] = df['timestamp'].dt.day
        df.loc[:, 'month'] = df['timestamp'].dt.month
        df.loc[:, 'day_of_week'] = df['timestamp'].dt.dayofweek
        df.loc[:, 'day_of_year'] = df['timestamp'].dt.dayofyear
        return df
    
    def create_lag_and_rolling_features(self, df, columns):
        """Create lag and rolling mean features for specified columns."""
        for col in columns:
            for lag in range(1, 4):
                df[f'{col}_lag_{lag}'] = df[col].shift(lag) 
            for window in [3, 6, 12]:
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()

        lag_cols = [f'{col}_lag_{lag}' for col in columns for lag in range(1, 4)]
        rolling_cols = [f'{col}_rolling_mean_{window}' for col in columns for window in [3, 6, 12]]
        df[lag_cols + rolling_cols] = df[lag_cols + rolling_cols].round(1)

        return df
    
    def encode_categorical(self, df):
        """Encode categorical variables."""
        if 'airport_code' in df.columns:
            df['airport_code_encoded'] = self.airport_encoder.fit_transform(df['airport_code'])
            df = df.drop('airport_code', axis=1)
        
        if 'wind_direction_symbol' in df.columns:
            df = df.drop('wind_direction_symbol', axis=1)
        
        if 'condition' in df.columns:
            df = df.drop('condition', axis=1)
        
        if 'airport_name' in df.columns:
            df = df.drop(['airport_name'], axis=1)

        return df
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset."""
        # Chỉ chọn các cột kiểu số
        numeric_columns = df.select_dtypes(include=[np.number]).columns

        # Điền giá trị thiếu bằng trung bình của các cột số
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

        # Làm tròn tất cả các cột số đến 1 chữ số thập phân
        df[numeric_columns] = df[numeric_columns].round(1)
        
        return df
    
    # Chọn các feature cần thiết cho model để train và test
    def select_features(self, df):
        """Select relevant features for model training."""
        features = [
            'hour', 'day', 'month', 'day_of_week', 'day_of_year',
            'temperature', 'humidity', 'wind_speed', 'gust_speed',
            'pressure', 'precipitation', 'rain_probability', 'snow_probability',
            'visibility', 'uv_index', 'dewpoint', 'cloud', 'wind_direction', 'condition_code',
            'temperature_lag_1', 'temperature_lag_2', 'temperature_lag_3',
            'temperature_rolling_mean_3', 'temperature_rolling_mean_6', 'temperature_rolling_mean_12',
            'humidity_lag_1', 'humidity_lag_2', 'humidity_lag_3',
            'humidity_rolling_mean_3', 'humidity_rolling_mean_6', 'humidity_rolling_mean_12',
            'wind_speed_lag_1', 'wind_speed_lag_2', 'wind_speed_lag_3',
            'wind_speed_rolling_mean_3', 'wind_speed_rolling_mean_6', 'wind_speed_rolling_mean_12',
            'pressure_lag_1', 'pressure_lag_2', 'pressure_lag_3',
            'pressure_rolling_mean_3', 'pressure_rolling_mean_6', 'pressure_rolling_mean_12',
            'precipitation_lag_1', 'precipitation_lag_2', 'precipitation_lag_3',
            'precipitation_rolling_mean_3', 'precipitation_rolling_mean_6', 'precipitation_rolling_mean_12',
            'cloud_lag_1', 'cloud_lag_2', 'cloud_lag_3',
            'cloud_rolling_mean_3', 'cloud_rolling_mean_6', 'cloud_rolling_mean_12',
            'uv_index_lag_1', 'uv_index_lag_2', 'uv_index_lag_3',
            'uv_index_rolling_mean_3', 'uv_index_rolling_mean_6', 'uv_index_rolling_mean_12',
            'visibility_lag_1', 'visibility_lag_2', 'visibility_lag_3',
            'visibility_rolling_mean_3', 'visibility_rolling_mean_6', 'visibility_rolling_mean_12',
            'rain_probability_lag_1', 'rain_probability_lag_2', 'rain_probability_lag_3',
            'rain_probability_rolling_mean_3', 'rain_probability_rolling_mean_6', 'rain_probability_rolling_mean_12',
            'dewpoint_lag_1', 'dewpoint_lag_2', 'dewpoint_lag_3',
            'dewpoint_rolling_mean_3', 'dewpoint_rolling_mean_6', 'dewpoint_rolling_mean_12'
        ] + [col for col in df.columns if col.startswith('airport_')]
        
        return df[features]

    def preprocess(self, data_path):
        """Main preprocessing pipeline."""
        # Read data
        df = pd.read_csv(data_path)
        
        # Apply preprocessing steps
        df = self.extract_time_features(df)

        df = self.encode_categorical(df)

        # Create lag and rolling mean features
        columns = ['temperature', 'humidity', 'wind_speed', 'pressure', 'precipitation', 'cloud', 'uv_index', 'visibility', 'rain_probability', 'dewpoint', 'gust_speed', 'snow_probability']
        df = self.create_lag_and_rolling_features(df, columns)

        df = self.handle_missing_values(df)
        
        # Select features
        df = self.select_features(df)
        return df