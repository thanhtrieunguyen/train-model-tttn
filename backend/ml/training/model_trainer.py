# Thêm import
from backend.ml.config.feature_config import FeatureManager

# Thêm vào hàm train_model() hoặc tương tự sau khi chuẩn bị dữ liệu huấn luyện
# ...existing code...
def train_model(X_train, y_train, model_type, target, **params):
    # ...existing code...
    
    # Lưu tên các tính năng
    feature_names = list(X_train.columns)
    FeatureManager.save_feature_names(feature_names, target, model_type)
    
    # ...existing code...
