# DỰ ÁN PHÂN LOẠI 102 LOÀI HOA (MLOPS & WEB APP)

Dự án triển khai một ứng dụng web Flask hiện đại để phân loại 102 loài hoa (Oxford 102 Category Flower Dataset). Hệ thống này được xây dựng trên kiến trúc đa mô hình và tích hợp **Vòng lặp Phản hồi (Feedback Loop)** để duy trì hiệu suất thực tế (MLOps).
---

## 🌟 Tính năng Chính

* **Đa Mô hình Lựa chọn:** Cho phép người dùng chọn và so sánh kết quả dự đoán giữa **ViT-B/16** và **EfficientNet-B1**.
* **Đầu vào Linh hoạt:** Hỗ trợ tải ảnh lên qua **File Upload** hoặc **Dán URL** ảnh trực tiếp.
* **Trực quan hóa Biểu đồ:** Hiển thị xác suất dự đoán Top-K bằng biểu đồ cột chuyên nghiệp.
* **Vòng lặp Phản hồi (MLOps):** Cơ chế thu thập phản hồi "Đúng/Sai" từ người dùng và lưu vào `data/feedback.csv`.
* **Trang Thống kê MLOps (`/stats`):** Phân tích dữ liệu phản hồi thực tế để tính toán **Độ chính xác Thực tế** và **Độ chính xác Theo Thời gian**.

---

## 1. 🛠️ Cấu trúc Dự án và Cài đặt
### 1.1. Cấu trúc Thư mục Dự án

```tree
flower_app_flask/
├── data/
│   ├── DOWNLOAD_LINKS.txt       # Chứa các URL để tải dữ liệu gốc
│   ├── feedback.csv             # Dữ liệu phản hồi thực tế (Feedback Loop)
│   ├── imagelabels.mat          # Metadata gốc (Nhãn ảnh)
│   ├── setid.mat                # Metadata gốc (Phân chia tập)
│   ├── cat_to_name.json         # Ánh xạ nhãn -> tên hoa
│   ├── flower_color_map_optimized.json # Kết quả EDA (Bản đồ màu)
│   └── jpg/                     # Thư mục chứa ảnh gốc đã giải nén (tùy chọn)
│
├── src/
│   ├── __init__.py              # (Cần thiết cho Python module)
│   ├── app.py                   # Ứng dụng Flask (Routing, Upload, Session)
│   ├── model_service.py         # Dịch vụ AI (Tải đa mô hình, Dự đoán tốc độ cao)
│   ├── feedback_service.py      # Logic ghi và phân tích Feedback Loop
│   ├── _config.py               # Cấu hình hằng số (nếu có)
│   ├── preprocessing.py         # Xử lý dữ liệu và tạo DataLoader
│   ├── model_training.py        # Logic Huấn luyện mô hình
│   └── evaluation.py            # Logic Đánh giá và phân tích (t-SNE, etc.)
│
├── web/
│   ├── models/                  # Mô hình đã huấn luyện
│   │   ├── exp_vit.pth          # ViT-B/16 Weights
│   │   ├── exp4_b1_continued.pth# EfficientNet-B1 Weights
│   │   └── wiki_cache.json      # Cache mô tả Wikipedia
│   │
│   ├── templates/               # Các file HTML
│   │   ├── index.html           # Trang phân loại chính
│   │   └── stats.html           # Trang thống kê MLOps
│   │
│   └── static/                  # Tài nguyên tĩnh
│       ├── uploads/             # Ảnh upload tạm thời
│       └── style.css            # Style giao diện
│
├── venv/                        # Môi trường ảo (KHÔNG commit)
├── README.md                    
├── report.pdf                   
└── requirements.txt
### 1.2. Hướng dẫn Cài đặt Môi trường

1.  **Cài đặt Thư viện Bắt buộc:**
    ```bash
    # Khởi tạo và kích hoạt môi trường ảo (Khuyến nghị)
    python -m venv venv
    source venv/bin/activate  
    # Cài đặt tất cả các thư viện
    pip install -r requirements.txt
    ```

### 1.3. Chuẩn bị Dữ liệu và Mô hình

1.  **Tải Dữ liệu và Metadata:**
    * Vui lòng tải các file dữ liệu cần thiết theo hướng dẫn trong **`data/DOWNLOAD_LINKS.txt`**.
    * Giải nén file ảnh gốc (`102flowers.tgz`) thành thư mục **`jpg/`** và đảm bảo nó có thể truy cập được bởi `src/`.
2.  **Tải Mô hình (`web/models/`):** Tải các file mô hình đã huấn luyện vào thư mục **`web/models/`**.

| Mô hình | Tên File (Bắt buộc) |
| :--- | :--- |
| **ViT (Vision Transformer)** | `exp_vit.pth` |
| **EfficientNet-B1** | `exp4_b1_continued.pth` |

---

## 2. 🚀 Khởi chạy và Sử dụng Web App
**Tải Mô hình (BẮT BUỘC):** Tải các file mô hình đã huấn luyện vào thư mục **`web/models/`**.

| Mô hình | Tên File (Bắt buộc) | Link Tải |
| :--- | :--- | :--- |
| **Vision Transformer (ViT-B/16)** | `exp_vit.pth` | [Link Tải ViT](https://drive.google.com/file/d/1ACAMxI0iTu3Y8NRFKWo64lczgRPbvxif/view?usp=sharing) |
| **EfficientNet-B1** | `exp4_b1_continued.pth` | [Link Tải B1](https://drive.google.com/file/d/1h2GYUs9qvywItWBk1lb22kBlKHS5_82P/view?usp=sharing) |
### 2.1. Khởi chạy Server

Đảm bảo bạn đang ở thư mục gốc của dự án và chạy:

```bash
python src/app.py
Truy cập: http://127.0.0.1:5000/

2.2. Hướng dẫn Sử dụng Đa Mô hình
Tại giao diện chính, chọn mô hình mong muốn (ViT hoặc B1) trong hộp chọn. (Các tùy chọn bị lỗi tải sẽ bị vô hiệu hóa).

Tải ảnh lên bằng File Upload hoặc Dán URL ảnh trực tiếp.

Nhấn Phân loại & Giải thích (XAI).

Sử dụng Vòng lặp Phản hồi: Sau khi xem kết quả, vui lòng nhấp vào nút "✅ Dự đoán Đúng" hoặc "❌ Dự đoán Sai" để đóng vòng lặp phản hồi.

2.3. Trang Thống kê MLOps
Để xem kết quả phân tích dữ liệu phản hồi đã thu thập, truy cập:

[http://127.0.0.1:5000/stats](http://127.0.0.1:5000/stats)
Trang này sẽ hiển thị Độ chính xác Thực tế Theo Thời gian và so sánh hiệu suất giữa hai mô hình.