import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import argparse

# Thêm đường dẫn gốc của dự án vào sys.path
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
sys.path.append(project_root)

from backend.ml.evaluation.model_evaluator import ModelEvaluator

def parse_args():
    parser = argparse.ArgumentParser(description='Đánh giá mô hình dự báo thời tiết với dữ liệu thực tế')
    
    parser.add_argument(
        '--real-data', 
        type=str,
        default=os.path.join(project_root, 'backend', 'ml', 'data', 'weather_dataset_with_season_terrain.csv'),
        help='Đường dẫn đến file dữ liệu thực tế'
    )
    
    parser.add_argument(
        '--historical-data', 
        type=str,
        default=os.path.join(project_root, 'backend', 'ml', 'data', 'weather_dataset_with_season_terrain.csv'),
        help='Đường dẫn đến file dữ liệu lịch sử'
    )
    
    parser.add_argument(
        '--prediction-time',
        type=str,
        default=(datetime.now().replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S'),
        help='Thời điểm cần dự báo (định dạng YYYY-MM-DD HH:MM:SS)'
    )
    
    parser.add_argument(
        '--train',
        action='store_true',
        help='Huấn luyện lại các mô hình'
    )
    
    parser.add_argument(
        '--full-report',
        action='store_true',
        help='Xuất báo cáo chi tiết'
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Bắt đầu đánh giá mô hình dự báo thời tiết với dữ liệu thực tế tại thời điểm {args.prediction_time}...")
    
    # Khởi tạo evaluator
    evaluator = ModelEvaluator()
    
    if args.train:
        # Đường dẫn đến dữ liệu huấn luyện
        data_path = os.path.join(project_root, 'backend', 'ml', 'data', 'weather_dataset_with_season_terrain.csv')
        
        # Chuẩn bị dữ liệu
        print("Đang chuẩn bị dữ liệu...")
        X, y_dict, feature_list_for_scale = evaluator.prepare_data(data_path)
        
        # Huấn luyện các mô hình
        print("Đang huấn luyện các mô hình...")
        evaluator.train_models(X, y_dict, feature_list_for_scale)
        
        # So sánh các mô hình
        print("Đang so sánh các mô hình...")
        comparison_results = evaluator.compare_models()
        print("Kết quả so sánh các mô hình:")
        print(comparison_results)
        
        # Xác định và lưu mô hình tốt nhất
        print("Xác định mô hình tốt nhất cho từng target...")
        best_models = evaluator.get_best_models()
        for target, info in best_models.items():
            print(f"{target}: {info['model_name']} (RMSE = {info['rmse']:.4f})")
    
    # Đánh giá với dữ liệu thực tế
    print("Đang đánh giá với dữ liệu thực tế...")
    evaluation_result = evaluator.evaluate_on_real_data(
        args.real_data,
        args.historical_data,
        args.prediction_time
    )
    
    if evaluation_result is None:
        print("Không thể đánh giá do thiếu dữ liệu thực tế.")
        return
    
    accuracy_report = evaluation_result['accuracy']
    
    # In kết quả
    print("\n=== Kết quả đánh giá độ chính xác ===")
    print(f"Thời điểm dự báo: {args.prediction_time}")
    
    print("\nĐộ chính xác trung bình của các mô hình:")
    for model_name, accuracy in accuracy_report['model_accuracy'].items():
        print(f"- {model_name}: {accuracy:.2f}%")
    
    print(f"\nMô hình tốt nhất: {accuracy_report['best_model']} với độ chính xác {accuracy_report['best_accuracy']:.2f}%")
    
    if args.full_report:
        print("\nĐộ chính xác theo từng target:")
        for target, accuracies in accuracy_report['target_accuracy'].items():
            print(f"\n{target}:")
            for model_name, accuracy in accuracies.items():
                print(f"  - {model_name}: {accuracy:.2f}%")
    
    print("\nĐánh giá hoàn tất! Xem báo cáo chi tiết trong thư mục results.")

if __name__ == "__main__":
    main()
