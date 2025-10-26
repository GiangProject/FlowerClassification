# src/app.py

from flask import Flask, request, render_template, url_for, flash, redirect, session
from werkzeug.utils import secure_filename
import os
import uuid
import sys
import requests 
import io       
from PIL import Image 
import time

# Thêm thư mục gốc dự án vào Python path để import module (nếu cần)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
WEB_DIR = os.path.join(BASE_DIR, 'web')
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Import hàm predict, analyze_feedback_stats và MODEL_HEALTH_AND_ACC
from src.model_service import predict, MODEL_HEALTH_AND_ACC 
from src.feedback_service import save_user_feedback, analyze_feedback_stats

# --- CẤU HÌNH THƯ MỤC WEB/STATIC/TEMPLATES ---
UPLOAD_FOLDER = os.path.join(WEB_DIR, 'static', 'uploads')
MAX_IMAGE_DIM = 1024 # Kích thước tối đa cho cạnh ảnh đầu vào (cho UX tốt hơn)

app = Flask(__name__, 
            template_folder=os.path.join(WEB_DIR, 'templates'), 
            static_folder=os.path.join(WEB_DIR, 'static'))

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'your_secret_key_here' 
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Hàm tiện ích
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_uploads(max_age_seconds=3600):
    """Xóa các file trong uploads cũ hơn max_age_seconds (1 giờ)."""
    now = time.time()
    count = 0
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.isfile(filepath):
            try:
                if now - os.stat(filepath).st_ctime > max_age_seconds:
                    os.remove(filepath)
                    count += 1
            except Exception as e:
                print(f"Lỗi dọn dẹp file {filename}: {e}", file=sys.stderr)
    if count > 0:
        print(f"🧹 Đã dọn dẹp {count} file ảnh cũ.")


@app.route('/', methods=['GET', 'POST'])
def index():
    chosen_model = request.args.get('chosen_model', 'ViT') 

    if request.method == 'POST':
        
        # 1. XÓA ẢNH TỪ PHIÊN TRƯỚC (QUẢN LÝ SESSION)
        if 'current_file' in session:
            old_filepath = os.path.join(app.config['UPLOAD_FOLDER'], session['current_file'])
            if os.path.exists(old_filepath):
                os.remove(old_filepath)
                print(f"🗑️ Đã xóa file cũ: {session['current_file']}")
            session.pop('current_file', None)

        # 2. Lấy dữ liệu form
        model_choice = request.form.get('model_choice', 'ViT') 
        file = request.files.get('file') 
        image_url = request.form.get('image_url') 
        
        filepath = None
        filename = None
        img = None

        # --- XỬ LÝ TẢI ẢNH (FILE HOẶC URL) ---
        if image_url:
            try:
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                response = requests.get(image_url, timeout=10, headers=headers)
                
                if response.status_code != 200: raise ValueError(f'Không thể truy cập URL (Mã: {response.status_code}).')

                image_data = io.BytesIO(response.content)
                img = Image.open(image_data)
                file_format = img.format.lower()
                
                if file_format not in ['jpeg', 'png']: raise ValueError('Định dạng ảnh không được hỗ trợ (chỉ chấp nhận JPEG/PNG).')

                file_ext = '.jpg' if file_format == 'jpeg' else '.png'
                filename = str(uuid.uuid4()) + file_ext
                
            except Exception as e:
                 flash(f"Lỗi tải/xử lý ảnh từ URL: {e}", 'error')
                 return redirect(url_for('index', chosen_model=model_choice))

        elif file and file.filename != '':
            if allowed_file(file.filename):
                filename = secure_filename(file.filename)
                img = Image.open(file.stream) 
                file_ext = os.path.splitext(filename)[1].lower()
            else:
                flash('Định dạng file upload không được hỗ trợ.', 'error')
                return redirect(url_for('index'))
        
        else:
            flash('Vui lòng chọn file hoặc nhập URL ảnh hợp lệ.', 'error')
            return redirect(url_for('index', chosen_model=model_choice))

        # 3. XỬ LÝ RESIZE VÀ LƯU ẢNH CUỐI CÙNG
        if img:
            # Resize ảnh nếu quá lớn
            if img.width > MAX_IMAGE_DIM or img.height > MAX_IMAGE_DIM:
                img.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM))
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img.save(filepath)

            # 4. CHẠY DỰ ĐOÁN
            try:
                predictions = predict(filepath, model_choice=model_choice) 

                # LƯU TÊN FILE HIỆN TẠI VÀO SESSION để xóa trong lần request tiếp theo
                session['current_file'] = filename

                return render_template('index.html', 
                                       filename=filename, 
                                       predictions=predictions,
                                       chosen_model=model_choice,
                                       model_health=MODEL_HEALTH_AND_ACC) # Truyền trạng thái sức khỏe
            
            except Exception as e:
                # Đảm bảo file hiện tại bị xóa nếu dự đoán lỗi
                if os.path.exists(filepath):
                    os.remove(filepath)
                    
                flash(f"Lỗi dự đoán: {e}", 'error')
                return redirect(url_for('index', chosen_model=model_choice))

    # Phương thức GET
    return render_template('index.html', 
                           chosen_model=chosen_model,
                           model_health=MODEL_HEALTH_AND_ACC) # Truyền trạng thái sức khỏe


@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    data = request.form
    model_choice = data.get('model_choice')
    filename = data.get('filename')
    predicted_label = data.get('predicted_label')
    correctness = data.get('correctness') 

    is_correct = correctness.lower() == 'true'
    
    success = save_user_feedback(model_choice, filename, predicted_label, is_correct)
    
    if success:
        flash('✅ Cảm ơn bạn đã phản hồi! Dữ liệu đã được lưu.', 'success')
    else:
        flash('❌ Lỗi hệ thống khi lưu phản hồi.', 'error')
        
    return redirect(url_for('index'))


@app.route('/stats')
def stats_page():
    stats_data = analyze_feedback_stats()
    
    # stats_data là dictionary chứa 'summary' và 'timeseries'
    return render_template('stats.html', stats=stats_data)

if __name__ == '__main__':
    app.secret_key = app.config['SECRET_KEY'] 
    cleanup_uploads()
    app.run(debug=True, port=5000)