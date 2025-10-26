# src/config.py

import torch
import os

# --- HẰNG SỐ CHUNG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Thiết bị sử dụng
INPUT_SIZE = 224    # Kích thước ảnh đầu vào tiêu chuẩn cho mô hình (224x224)
NUM_CLASSES = 102   # Số lượng lớp hoa cần phân loại
IMAGENET_MEAN = [0.485, 0.456, 0.406] # Giá trị Mean của ImageNet để chuẩn hóa
IMAGENET_STD = [0.229, 0.224, 0.225]  # Giá trị Standard Deviation của ImageNet để chuẩn hóa
PROJECT_PATH = '/content/drive/MyDrive/KPDL_BAINOP/' # Đường dẫn gốc của dự án
COLOR_MAP_PATH = os.path.join('data', 'flower_color_map_optimized.json') # Đường dẫn lưu file bản đồ màu
