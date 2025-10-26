# src/feedback_service.py (Giữ nguyên)

import os
import csv
import time
import pandas as pd
import sys

# Đường dẫn đến file feedback (Đặt trong thư mục data/)
FEEDBACK_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'feedback.csv')
HEADER = ['timestamp', 'model_choice', 'file_name', 'predicted_label', 'is_correct']

def initialize_feedback_file():
    """Khởi tạo file feedback.csv nếu nó chưa tồn tại."""
    if not os.path.exists(FEEDBACK_FILE):
        os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
        with open(FEEDBACK_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)
        print(f"✅ Khởi tạo file feedback.csv thành công tại: {FEEDBACK_FILE}")

def save_user_feedback(model_choice: str, file_name: str, predicted_label: str, is_correct: bool):
    """
    Ghi phản hồi của người dùng vào file CSV.
    Tham số: is_correct là True/False.
    """
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    
    if not os.path.exists(FEEDBACK_FILE):
        initialize_feedback_file()
        
    try:
        with open(FEEDBACK_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                model_choice,
                file_name,
                predicted_label,
                'True' if is_correct else 'False'
            ])
        return True
    except Exception as e:
        print(f"❌ Lỗi khi ghi feedback: {e}", file=sys.stderr)
        return False
        
def analyze_feedback_stats():
    """Đọc file feedback và trả về thống kê tổng hợp và chuỗi thời gian."""
    if not os.path.exists(FEEDBACK_FILE):
        return None
        
    try:
        df = pd.read_csv(FEEDBACK_FILE)
        if df.empty: return None

        df['is_correct'] = df['is_correct'].astype(str).str.lower() == 'true'
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 1. THỐNG KÊ TỔNG HỢP (cho bảng summary)
        stats = df.groupby('model_choice')['is_correct'].agg(['count', 'sum']).reset_index()
        stats['accuracy'] = (stats['sum'] / stats['count']) * 100
        stats = stats.rename(columns={'count': 'total_feedback'}).to_dict('records')
        
        # 2. DỮ LIỆU CHO BIỂU ĐỒ ĐƯỜNG (Nhóm theo ngày)
        df['date'] = df['timestamp'].dt.date
        df_time = df.groupby(['date', 'model_choice'])['is_correct'].agg(['count', 'sum']).reset_index()
        df_time['accuracy'] = (df_time['sum'] / df_time['count']) * 100
        
        df_time['date'] = df_time['date'].astype(str) 
        time_series_data = df_time.to_dict('records')
        
        return {'summary': stats, 'timeseries': time_series_data} # ⬅️ Trả về dictionary 2 cấp
        
    except Exception as e:
        print(f"❌ Lỗi khi phân tích feedback: {e}", file=sys.stderr)
        return None

initialize_feedback_file()