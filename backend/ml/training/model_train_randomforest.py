import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, f1_score, accuracy_score
import xgboost as xgb
from sklearn.cluster import KMeans
import logging

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WeatherHybridModel:
    def __init__(self, data_path, model_dir='models', n_clusters=3, 
                 important_airports=None, target_cols=None):
        """
        Initialize the hybrid weather prediction model manager
        
        Args:
            data_path (str): Path to the weather dataset CSV
            model_dir (str): Directory to store trained models
            n_clusters (int): Number of airport clusters to create
            important_airports (list): List of airport codes that get dedicated models
            target_cols (list): List of target columns to predict
        """
        self.data_path = data_path
        self.model_dir = model_dir
        self.n_clusters = n_clusters
        self.important_airports = important_airports or ['VVNB', 'VVTS', 'VVDN']  # Default important airports
        self.target_cols = target_cols or ['temperature', 'precipitation', 'wind_speed', 'visibility']
        
        # Will store all models
        self.base_model = None
        self.cluster_models = {}
        self.airport_models = {}
        self.cluster_assignments = {}
        self.feature_preprocessor = None
        self.encoders = {}
        
        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)
        
    def _preprocess_data(self, df):
        """Preprocess data for model training"""
        # Clone dataframe to avoid modifying original
        data = df.copy()
        
        # Extract time features
        data['date'] = pd.to_datetime(data['timestamp'])
        data['hour'] = data['date'].dt.hour
        data['day'] = data['date'].dt.day
        data['month'] = data['date'].dt.month
        data['year'] = data['date'].dt.year
        data['dayofweek'] = data['date'].dt.dayofweek
        
        # Add seasonal features for Vietnam
        data['season'] = data['month'].apply(lambda m: 
            'winter' if m in [12, 1, 2] else 
            'spring' if m in [3, 4, 5] else 
            'summer' if m in [6, 7, 8] else 'fall')
        
        # Add terrain classification - extend this with your knowledge of airports
        # Airport in backend\ml\data\airports.json
        terrain_dict = {
            'HAN': 'lowland',   # Hanoi Noi Bai
            'SGN': 'lowland',   # Ho Chi Minh City
            'DAD': 'coastal',   # Da Nang
            'CXR': 'coastal',   # Cam Ranh
            'HPH': 'coastal',   # Hai Phong
            'DLI': 'highland',  # Da Lat
            'PQC': 'island',    # Phu Quoc
            'VCA': 'lowland',   # Can Tho
            'VCS': 'island',    # Con Dao
            'VCL': 'coastal',   # Chu Lai
            'BMV': 'coastal',   # Buon Ma Thuot
            'CAH': 'coastal',   # Ca Mau
            'DIN': 'coastal',   # Dien Bien
            'UIH': 'coastal',   # Phu Cat
            'VDH': 'coastal',   # Dong Hoi
            'VDO': 'coastal',   # Van Don
            'VKG': 'coastal',   # Rach Gia
            'VII': 'coastal',   # Vinh
            'VKS': 'coastal',   # Vung Tau
            'HIU': 'coastal',   # Phan Thiet
        }
        data['terrain'] = data['airport_code'].map(terrain_dict)
        data['terrain'].fillna('other', inplace=True)
        
        # Encode categorical features
        for col in ['airport_code', 'season', 'terrain', 'wind_direction_symbol']:
            if col in data.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    data[f'{col}_encoded'] = self.encoders[col].fit_transform(data[col].fillna('unknown'))
                else:
                    # Handle new categories during prediction
                    categories = set(data[col].unique())
                    new_categories = categories - set(self.encoders[col].classes_)
                    if new_categories:
                        logger.warning(f"Found new categories in {col}: {new_categories}")
                        data[col] = data[col].apply(lambda x: 'unknown' if x in new_categories else x)
                    data[f'{col}_encoded'] = self.encoders[col].transform(data[col].fillna('unknown'))
        
        # Calculate geographical clusters for airports
        if 'cluster' not in data.columns:
            self._cluster_airports(data)
            data = data.merge(
                pd.DataFrame({
                    'airport_code': list(self.cluster_assignments.keys()),
                    'cluster': list(self.cluster_assignments.values())
                }), 
                on='airport_code', how='left'
            )
        
        return data
    
    def _cluster_airports(self, data):
        """Group airports into geographical clusters"""
        # Get unique airports with their coordinates
        airports = data[['airport_code', 'latitude', 'longitude']].drop_duplicates()
        
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        airports['cluster'] = kmeans.fit_predict(airports[['latitude', 'longitude']])
        
        # Store cluster assignments
        self.cluster_assignments = dict(zip(airports['airport_code'], airports['cluster']))
        logger.info(f"Created {self.n_clusters} geographical clusters of airports")
        logger.info("Cluster assignments: " + str({k: v for i, (k, v) in enumerate(self.cluster_assignments.items()) if i < 5}) + "...")
        
        joblib.dump(self.cluster_assignments, os.path.join(self.model_dir, 'cluster_assignments.pkl'))
        return self.cluster_assignments
    
    def _get_features(self):
        """Define features to use for model training"""
        numeric_features = ['hour', 'day', 'month', 'dayofweek']
        categorical_features = ['airport_code_encoded', 'season_encoded', 'terrain_encoded']
        
        return numeric_features + categorical_features
    
    def _prepare_training_data(self, data, target_col):
        """Prepare data for a specific target column"""
        features = self._get_features()
        X = data[features].copy()
        y = data[target_col].copy()
        
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def _build_model(self, target_col, is_classification=False):
        """Build model pipeline for a specific target column"""
        if is_classification:
            model = RandomForestClassifier(random_state=42, n_estimators=100)
        else:
            # Use XGBoost for regression tasks
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                learning_rate=0.1,
                random_state=42
            )
        
        return model
    
    def train_base_model(self, data):
        """Train the base model using all data"""
        logger.info("Training base model using all data...")
        processed_data = self._preprocess_data(data)
        
        base_models = {}
        for target in self.target_cols:
            logger.info(f"Training base model for {target}")
            is_classification = target in ['condition_code', 'rain_probability']
            
            X_train, X_test, y_train, y_test = self._prepare_training_data(processed_data, target)
            model = self._build_model(target, is_classification)
            
            model.fit(X_train, y_train)
            
            # Evaluate
            if is_classification:
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                logger.info(f"Base model {target}: Accuracy={accuracy:.4f}, F1={f1:.4f}")
            else:
                y_pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                logger.info(f"Base model {target}: MAE={mae:.4f}")
            
            base_models[target] = model
        
        self.base_model = base_models
        # Save base model
        joblib.dump(base_models, os.path.join(self.model_dir, 'base_model.pkl'))
        logger.info("Base model saved")
        
    def train_cluster_models(self, data):
        """Train models specific to each geographical cluster"""
        logger.info("Training cluster-specific models...")
        processed_data = self._preprocess_data(data)
        
        for cluster in range(self.n_clusters):
            logger.info(f"Training models for cluster {cluster}")
            cluster_data = processed_data[processed_data['cluster'] == cluster]
            
            if len(cluster_data) < 1000:  # Skip if not enough data
                logger.warning(f"Not enough data for cluster {cluster} ({len(cluster_data)} samples). Skipping.")
                continue
                
            cluster_models = {}
            for target in self.target_cols:
                logger.info(f"Training cluster model for {target}")
                is_classification = target in ['condition_code', 'rain_probability']
                
                X_train, X_test, y_train, y_test = self._prepare_training_data(cluster_data, target)
                model = self._build_model(target, is_classification)
                
                model.fit(X_train, y_train)
                
                # Evaluate
                if is_classification:
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    logger.info(f"Cluster {cluster} model {target}: Accuracy={accuracy:.4f}, F1={f1:.4f}")
                else:
                    y_pred = model.predict(X_test)
                    mae = mean_absolute_error(y_test, y_pred)
                    logger.info(f"Cluster {cluster} model {target}: MAE={mae:.4f}")
                
                cluster_models[target] = model
            
            self.cluster_models[cluster] = cluster_models
            # Save cluster model
            joblib.dump(cluster_models, os.path.join(self.model_dir, f'cluster_{cluster}_model.pkl'))
        
        logger.info("Cluster models saved")
        
    def train_airport_models(self, data):
        """Train models specific to important airports"""
        logger.info(f"Training airport-specific models for {self.important_airports}...")
        processed_data = self._preprocess_data(data)
        
        for airport in self.important_airports:
            logger.info(f"Training models for airport {airport}")
            airport_data = processed_data[processed_data['airport_code'] == airport]
            
            if len(airport_data) < 500:  # Skip if not enough data
                logger.warning(f"Not enough data for airport {airport} ({len(airport_data)} samples). Skipping.")
                continue
                
            airport_models = {}
            for target in self.target_cols:
                logger.info(f"Training airport model for {target}")
                is_classification = target in ['condition_code', 'rain_probability'] 
                
                X_train, X_test, y_train, y_test = self._prepare_training_data(airport_data, target)
                model = self._build_model(target, is_classification)
                
                model.fit(X_train, y_train)
                
                # Evaluate
                if is_classification:
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    logger.info(f"Airport {airport} model {target}: Accuracy={accuracy:.4f}, F1={f1:.4f}")
                else:
                    y_pred = model.predict(X_test)
                    mae = mean_absolute_error(y_test, y_pred)
                    logger.info(f"Airport {airport} model {target}: MAE={mae:.4f}")
                
                airport_models[target] = model
            
            self.airport_models[airport] = airport_models
            # Save airport model
            joblib.dump(airport_models, os.path.join(self.model_dir, f'airport_{airport}_model.pkl'))
        
        logger.info("Airport models saved")
    
    def train_all_models(self):
        """Train base, cluster and airport-specific models"""
        logger.info("Loading and preprocessing data...")
        data = pd.read_csv(self.data_path)
        
        if data.empty:
            logger.error("No data found!")
            return False
            
        logger.info(f"Loaded {len(data)} records for training")
        
        self.train_base_model(data)
        self.train_cluster_models(data)
        self.train_airport_models(data)
        
        return True
    
    def load_models(self):
        """Load all saved models"""
        try:
            self.base_model = joblib.load(os.path.join(self.model_dir, 'base_model.pkl'))
            self.cluster_assignments = joblib.load(os.path.join(self.model_dir, 'cluster_assignments.pkl'))
            
            # Load cluster models
            for cluster in range(self.n_clusters):
                path = os.path.join(self.model_dir, f'cluster_{cluster}_model.pkl')
                if os.path.exists(path):
                    self.cluster_models[cluster] = joblib.load(path)
            
            # Load airport models
            for airport in self.important_airports:
                path = os.path.join(self.model_dir, f'airport_{airport}_model.pkl')
                if os.path.exists(path):
                    self.airport_models[airport] = joblib.load(path)
            
            logger.info("All models loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
    
    def predict(self, input_data):
        """
        Make a prediction using the appropriate model hierarchy:
        1. If airport has specific model, use it
        2. If not, use cluster model
        3. Fallback to base model
        """
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
            
        # Preprocess the input data
        processed_data = self._preprocess_data(input_data)
        features = self._get_features()
        
        results = {}
        for index, row in processed_data.iterrows():
            airport = row['airport_code']
            cluster = self.cluster_assignments.get(airport, 0)  # Default to cluster 0
            
            airport_predictions = {}
            for target in self.target_cols:
                X = row[features].values.reshape(1, -1)
                
                # Try airport-specific model first
                if airport in self.airport_models and target in self.airport_models[airport]:
                    prediction = self.airport_models[airport][target].predict(X)[0]
                    source = 'airport'
                # Try cluster model next
                elif cluster in self.cluster_models and target in self.cluster_models[cluster]:
                    prediction = self.cluster_models[cluster][target].predict(X)[0]
                    source = 'cluster'
                # Fall back to base model
                else:
                    prediction = self.base_model[target].predict(X)[0]
                    source = 'base'
                
                airport_predictions[target] = {
                    'value': float(prediction),
                    'model_source': source
                }
            
            results[airport] = airport_predictions
                
        return results

def train_models():
    """Entry point to train all models"""
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'weather_dataset1.csv')
    model_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
    print(data_path)
    print(model_dir)
    # logger.info(f"Training models using data from {data_path}")
    # model = WeatherHybridModel(
    #     data_path=data_path,
    #     model_dir=model_dir,
    #     n_clusters=4,  # Vietnam can be divided into North, Central, South, and Highland regions
    #     important_airports=['HAN', 'SGN', 'DAD', 'CXR', 'HPH'],  # Major airports
    #     target_cols=['temperature', 'precipitation', 'wind_speed', 'visibility', 'rain_probability']
    # )
    
    # success = model.train_all_models()
    # if success:
    #     logger.info("Model training completed successfully")
    # else:
    #     logger.error("Model training failed")

if __name__ == "__main__":
    train_models()