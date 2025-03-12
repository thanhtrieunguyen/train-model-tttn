import os
import pickle
import pandas as pd

class FeatureManager:
    @staticmethod
    def save_feature_names(feature_names, target, model_type):
        """Lưu tên các tính năng cho mô hình"""
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))
        output_path = os.path.join(base_dir, f'{target}_{model_type}_features.pkl')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(feature_names, f)
    
    @staticmethod
    def load_feature_names(target, model_type):
        """Tải tên các tính năng của mô hình"""
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))
        path = os.path.join(base_dir, f'{target}_{model_type}_features.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None
    
    @staticmethod
    def ensure_feature_order(df, target, model_type):
        """Đảm bảo thứ tự và tên tính năng đúng như lúc huấn luyện"""
        feature_names = FeatureManager.load_feature_names(target, model_type)
        if feature_names is None:
            raise ValueError(f"Không tìm thấy thông tin tính năng cho {target}_{model_type}")
        
        # Kiểm tra xem tất cả các tính năng cần thiết có tồn tại không
        missing_features = set(feature_names) - set(df.columns)
        if missing_features:
            raise ValueError(f"Thiếu các tính năng: {missing_features}")
        
        # Trả về dữ liệu với cùng thứ tự tính năng như khi huấn luyện
        return df[feature_names]
