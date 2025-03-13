
import os
import sys
import argparse
from datetime import datetime
import subprocess

current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(project_root)

def parse_args():
    parser = argparse.ArgumentParser(description='Chạy toàn bộ quy trình huấn luyện, đánh giá và so sánh các mô hình')
    
    parser.add_argument(
        '--api-key',
        type=str,
        required=True,
        help='API key cho dịch vụ thời tiết'
    )
    
    parser.add_argument(
        '--location',
        type=str,
        default='12.7,108.1',
        help='Vị trí dự báo (định dạng lat,lon)'
    )
    
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Bỏ qua bước huấn luyện các mô hình'
    )
    
    parser.add_argument(
        '--benchmark-iterations',
        type=int,
        default=5,
        help='Số lần lặp cho benchmark hiệu suất'
    )
    
    return parser.parse_args()

def run_command(command, description):
    """Chạy lệnh và hiển thị thông báo"""
    print(f"\n{'='*80}\n{description}\n{'='*80}\n")
    try:
        process = subprocess.run(command, check=True, shell=True)
        print(f"\n{description} đã hoàn thành thành công!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nLỗi: {description} không thành công.\n{e}\n")
        return False

def main():
    args = parse_args()
    run_command(
        f"python {os.path.join(project_root, 'backend', 'ml', 'evaluation', 'run_benchmark.py')} "
        f"--location {args.location} --api-key {args.api_key} --iterations {args.benchmark_iterations}",
        "5.2. So sánh hiệu suất các mô hình"
    )
    
if __name__ == '__main__':
    main()