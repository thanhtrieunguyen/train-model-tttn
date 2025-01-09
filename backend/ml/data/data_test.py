import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from datetime import datetime

class WeatherDataPreprocessor:
    def __init__(self):
        self.airport_encoder = LabelEncoder()
        self.condition_encoder = LabelEncoder()
        self.wind_dir_encoder = LabelEncoder()
        
    def extract_time_features(self, df):
        """Extract time-based features from timestamp."""
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        return df
    
    def encode_categorical(self, df):
        df = df.copy()
        """Encode categorical variables."""
        # Encode airport codes
        if 'airport_code' in df.columns:
            df['airport_code_encoded'] = self.airport_encoder.fit_transform(df['airport_code'])
            df = df.drop('airport_code', axis=1)

        # Encode wind direction and weather condition
        if 'wind_direction_symbol' in df.columns:
            df['wind_direction_encoded'] = self.wind_dir_encoder.fit_transform(df['wind_direction_symbol'])
            df = df.drop('wind_direction_symbol', axis=1)

        if 'condition' in df.columns:  
            df['condition_encoded'] = self.condition_encoder.fit_transform(df['condition'])
            df = df.drop('condition', axis=1)

        if 'airport_name' in df.columns:
            df = df.drop(['airport_name'], axis=1)

        return df
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset."""
        # Drop rows with missing values
        df = df.dropna()
        
        return df
    
    def select_features(self, df):
        """Select relevant features for model training."""
        features = [
            'hour', 'day', 'month', 'day_of_week', 'day_of_year',
            'temperature', 'humidity', 'wind_speed', 'wind_direction_encoded',
            'pressure', 'precipitation', 'rain_probability', 'snow_probability',
            'uv_index', 'cloud', 'condition_encoded'
        ] + [col for col in df.columns if col.startswith('airport_')]
        
        return df[features]
    
    def preprocess(self, data_path):
        """Main preprocessing pipeline."""
        # Read data
        df = pd.read_csv(data_path)
        
        # Apply preprocessing steps
        df = self.extract_time_features(df)
        df = self.encode_categorical(df)
        df = self.handle_missing_values(df)
        
        # Select features
        X = self.select_features(df)
        
        return X, df