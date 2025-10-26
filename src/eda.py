# src/eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from PIL import Image
import os
import cv2
import numpy as np
import json
from sklearn.cluster import KMeans
from collections import Counter
from tqdm.auto import tqdm
# ĐÃ SỬA LỖI: Dùng Absolute Import (config)
from config import COLOR_MAP_PATH 

# --- EDA & PHÂN TÍCH MÀU ---
def visualize_all_102_flowers(dataframe):
    """Hiển thị một ảnh đại diện cho mỗi loài hoa trong 102 lớp."""
    print("--- Hiển thị 1 ảnh đại diện cho mỗi loài trong số 102 loài hoa ---")
    unique_flowers_df = dataframe.groupby('flower_name').sample(n=1, replace=True).sort_values('flower_name')
    num_images = len(unique_flowers_df); cols = 6; rows = math.ceil(num_images / cols)
    plt.figure(figsize=(cols * 3, rows * 3))
    for i, (idx, row) in enumerate(unique_flowers_df.iterrows()):
        plt.subplot(rows, cols, i + 1)
        try: img = Image.open(row['filepath']); plt.imshow(img)
        except FileNotFoundError: plt.text(0.5, 0.5, 'Image not found', ha='center', va='center')
        plt.axis('off'); title = row['flower_name']; title = title[:15] + '...' if len(title) > 15 else title
        plt.title(title, fontsize=10)
    plt.tight_layout(pad=1.0); plt.show()


def get_dominant_color_name(image_path, resize_to=80, n_clusters=3):
    """Sử dụng K-Means và lọc màu xanh lá để xác định màu chủ đạo của hoa."""
    try:
        img = cv2.imread(image_path)
        if img is None: return "Không xác định"
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        img_crop = img[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)]
        img_resized = cv2.resize(img_crop, (resize_to, resize_to), interpolation=cv2.INTER_AREA)
        img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)
        pixels_rgb = img_resized.reshape((-1, 3)); pixels_hsv = img_hsv.reshape((-1, 3))
        green_mask = np.logical_and(pixels_hsv[:, 0] > 30, pixels_hsv[:, 0] < 80)
        green_mask = np.logical_and(green_mask, pixels_hsv[:, 1] > 50)
        value_mask = np.logical_and(pixels_hsv[:, 2] > 50, pixels_hsv[:, 2] < 230)
        final_mask = np.logical_and(~green_mask, value_mask)
        filtered_pixels = pixels_rgb[final_mask]

        if len(filtered_pixels) < 10: filtered_pixels = pixels_rgb

        kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
        kmeans.fit(filtered_pixels)
        colors = kmeans.cluster_centers_.astype(int)
        counts = np.bincount(kmeans.labels_)
        dominant_color = colors[np.argmax(counts)]
        r, g, b = dominant_color

        if r > 180 and g < 100 and b < 100: return "Đỏ"
        elif r > 200 and g > 120 and b < 70: return "Cam"
        elif r > 190 and g > 170 and b < 100: return "Vàng"
        elif b > 150 and r < 100 and g < 100: return "Xanh dương"
        elif r > 180 and g < 120 and b > 180: return "Hồng"
        elif r > 150 and b > 150 and g < 120: return "Tím"
        elif g > r and g > b and r < 100 and b < 100: return "Xanh lá (Hoa hiếm)"
        else: return "Hồng nhạt/Trắng"

    except Exception: return "Không xác định"

def extract_color_map(df, save_path=COLOR_MAP_PATH, overwrite=True):
    """Khai phá và lưu bản đồ màu sắc chủ đạo cho mỗi loài hoa bằng cách lấy mẫu ảnh."""
    if os.path.exists(save_path) and not overwrite:
        print(f"✅ Bản đồ màu sắc đã tồn tại tại {save_path}")
        return {}

    print("--- 🔁 Đang (tái) khai phá màu sắc chủ đạo ---")
    flower_color_map = {}
    for flower_name in tqdm(df['flower_name'].unique(), desc="🔍 Đang xử lý"):
        sample_imgs = df[df['flower_name'] == flower_name]['filepath'].sample(n=min(5, len(df[df['flower_name'] == flower_name])), random_state=42)
        colors = [get_dominant_color_name(p) for p in sample_imgs]
        if colors: flower_color_map[flower_name] = Counter(colors).most_common(1)[0][0]

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(flower_color_map, f, indent=4, ensure_ascii=False)

    print(f"✅ Đã lưu (hoặc ghi đè) bản đồ màu vào {save_path}")
    return flower_color_map

def plot_color_distribution(df):
    """Thực thi việc khai phá màu sắc và trực quan hóa phân bố màu sắc chủ đạo của 102 loài hoa."""
    CUSTOM_COLOR_PALETTE = {
        "Hồng nhạt/Trắng": "#F08080", "Cam": "#FF8C00", "Đỏ": "#DC143C",
        "Xanh dương": "#4169E1", "Vàng": "#FFD700", "Hồng": "#FF69B4",
        "Tím": "#8A2BE2", "Xanh lá (Hoa hiếm)": "#3CB371", "Không xác định": "#A9A9A9"
    }
    flower_color_map = extract_color_map(df, overwrite=True)
    if flower_color_map:
        color_df = pd.DataFrame(flower_color_map.items(), columns=['Loài hoa', 'Màu chủ đạo'])
        order = color_df['Màu chủ đạo'].value_counts().index
        ordered_palette = [CUSTOM_COLOR_PALETTE[c] for c in order if c in CUSTOM_COLOR_PALETTE]
        plt.figure(figsize=(10, 6)); sns.countplot(y='Màu chủ đạo', data=color_df, order=order, palette=ordered_palette)
        plt.title('Phân bố Màu sắc Chủ đạo của 102 Loài Hoa (Đã tối ưu lọc nền)'); plt.show()

def analyze_image_size(df):
    """Phân tích và trực quan hóa phân bố Chiều Rộng và Chiều Cao của ảnh gốc."""
    print("\n--- 🔎 Phân tích Kích thước Ảnh Gốc (Height và Width) ---")

    def get_img_dims(filepath):
        try:
            with Image.open(filepath) as img:
                return pd.Series(img.size, index=['Width', 'Height'])
        except Exception:
            return pd.Series([None, None], index=['Width', 'Height'])

    df[['Width', 'Height']] = df['filepath'].apply(get_img_dims)
    df_dims = df.dropna(subset=['Width', 'Height'])
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); sns.histplot(df_dims['Width'], bins=30, kde=True, color='skyblue'); plt.title('Phân bố Chiều Rộng Ảnh Gốc')
    plt.subplot(1, 2, 2); sns.histplot(df_dims['Height'], bins=30, kde=True, color='lightcoral'); plt.title('Phân bố Chiều Cao Ảnh Gốc')
    plt.tight_layout(); plt.show()
    print(f"Tổng số ảnh được phân tích kích thước: {len(df_dims)}")
    print(f"Kích thước trung bình (W x H): {df_dims['Width'].mean():.0f} x {df_dims['Height'].mean():.0f}")
