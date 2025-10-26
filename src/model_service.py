# src/model_service.py

import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import json
import torch.nn.functional as F
import os
import sys
import wikipediaapi
from sumy.summarizers.lsa import LsaSummarizer 
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer 

# --- CẤU HÌNH ĐƯỜNG DẪN TƯƠNG ĐỐI ---
BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'web', 'models')
CLASS_MAPPING_PATH = os.path.join(BASE_PATH, 'cat_to_name.json')
COLOR_MAP_PATH = os.path.join(BASE_PATH, 'flower_color_map_optimized.json') 
WIKI_CACHE_PATH = os.path.join(BASE_PATH, 'wiki_cache.json')

# --- ĐƯỜNG DẪN CỦA CÁC MÔ HÌNH ---
MODEL_PATHS = {
    'ViT': os.path.join(BASE_PATH, 'exp_vit.pth'), 
    'B1': os.path.join(BASE_PATH, 'exp4_b1_continued.pth') 
}

# --- HẰNG SỐ ---
INPUT_SIZE = 224
NUM_CLASSES = 102
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Khởi tạo Wikipedia ---
WIKI_VI = wikipediaapi.Wikipedia('FlowerApp (example@example.com)', 'vi')
WIKI_EN = wikipediaapi.Wikipedia('FlowerApp (example@example.com)', 'en')

# Bổ sung biến toàn cục: Lưu trạng thái và điểm của mô hình
MODEL_HEALTH_AND_ACC = {} 
MODEL_CACHE = {}

# ... (Logic Wikipedia giữ nguyên) ...

def smart_summarize(text, sentence_count=2):
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("vietnamese")) 
        summarizer = LsaSummarizer()
        summary_sentences = summarizer(parser.document, sentence_count)
        return " ".join(str(s) for s in summary_sentences)
    except Exception:
        return text[:500] + "..."

if os.path.exists(WIKI_CACHE_PATH):
    try:
        with open(WIKI_CACHE_PATH, 'r', encoding='utf-8') as f: WIKI_CACHE = json.load(f)
    except Exception: WIKI_CACHE = {}
else: WIKI_CACHE = {}

def get_wiki_summary_cached(flower_name):
    if flower_name in WIKI_CACHE: return WIKI_CACHE[flower_name]
    try:
        page_vi = WIKI_VI.page(flower_name)
        if page_vi.exists(): summary = smart_summarize(page_vi.summary)
        else:
            page_en = WIKI_EN.page(flower_name)
            summary = smart_summarize(page_en.summary) if page_en.exists() else "Không có thông tin mô tả."
    except Exception as e:
        print(f"[⚠️] Lỗi Wikipedia: {e}", file=sys.stderr)
        summary = "Không thể tải thông tin Wikipedia."
    WIKI_CACHE[flower_name] = summary
    try:
        with open(WIKI_CACHE_PATH, 'w', encoding='utf-8') as f:
            json.dump(WIKI_CACHE, f, ensure_ascii=False, indent=2)
    except Exception as e:
         print(f"[⚠️] Không thể ghi cache Wikipedia: {e}", file=sys.stderr)
    return summary

# ... (Logic build_model_b1, build_model_vit giữ nguyên) ...

def build_model_b1(num_classes=NUM_CLASSES):
    model = models.efficientnet_b1(weights=None) 
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features=num_ftrs, out_features=num_classes)
    return model

def build_model_vit(num_classes=NUM_CLASSES):
    model = models.vit_b_16(weights=None) 
    num_ftrs = model.heads.head.in_features
    model.heads.head = torch.nn.Linear(in_features=num_ftrs, out_features=num_classes)
    return model

def load_model_and_classes():
    """Tải tất cả mô hình, tên lớp và hàm tiền xử lý vào bộ nhớ."""
    print("🔧 Đang tải các mô hình AI...")

    # 1. Tải ViT
    try:
        model_vit = build_model_vit(num_classes=NUM_CLASSES).to(DEVICE)
        model_vit.load_state_dict(torch.load(MODEL_PATHS['ViT'], map_location=DEVICE))
        model_vit.eval()
        MODEL_CACHE['ViT'] = model_vit
        MODEL_HEALTH_AND_ACC['ViT'] = {'status': 'OK', 'acc': 91.9} # Điểm đã biết
        print(f"✅ Đã tải ViT-B/16.")
    except Exception as e:
        MODEL_HEALTH_AND_ACC['ViT'] = {'status': 'ERROR', 'acc': 0.0}
        print(f"❌ Lỗi tải ViT: {e}")

    # 2. Tải B1
    try:
        model_b1 = build_model_b1(num_classes=NUM_CLASSES).to(DEVICE)
        model_b1.load_state_dict(torch.load(MODEL_PATHS['B1'], map_location=DEVICE))
        model_b1.eval()
        MODEL_CACHE['B1'] = model_b1
        MODEL_HEALTH_AND_ACC['B1'] = {'status': 'OK', 'acc': 89.5} # Điểm đã biết
        print(f"✅ Đã tải EfficientNet-B1.")
    except Exception as e:
        MODEL_HEALTH_AND_ACC['B1'] = {'status': 'ERROR', 'acc': 0.0}
        print(f"❌ Lỗi tải B1: {e}")

    # 3. Tải tên lớp và tiền xử lý (Giữ nguyên)
    with open(CLASS_MAPPING_PATH, 'r', encoding='utf-8') as f:
        cat_to_name = json.load(f)
    class_names = [cat_to_name[str(i)] for i in range(1, NUM_CLASSES + 1)] 

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    return class_names, preprocess

# Khối khởi động Global Caching
try:
    CLASS_NAMES, PREPROCESS = load_model_and_classes() 
    with open(COLOR_MAP_PATH, 'r', encoding='utf-8') as f:
        COLOR_MAP = json.load(f)
    print("✅ Mô hình và tài nguyên đã sẵn sàng.")
except Exception as e:
    print(f"❌ Lỗi khởi động ứng dụng: {e}", file=sys.stderr)
    CLASS_NAMES, PREPROCESS, COLOR_MAP = [], None, {}

# ... (Hàm dự đoán predict giữ nguyên) ...

def predict(image_path, model_choice='ViT', top_k=3):
    """Dự đoán tốc độ cao (chỉ chạy suy luận cơ bản)."""
    
    MODEL = MODEL_CACHE.get(model_choice)
    if MODEL is None:
        return [{ "label": f"Lỗi Model: {model_choice} không tồn tại", "probability": "0.00%", "color": "Đen", "summary": "Không tải được mô hình." }]

    try:
        # 1. Xử lý ảnh đầu vào
        pil_img = Image.open(image_path).convert('RGB')
        input_tensor = PREPROCESS(pil_img).unsqueeze(0).to(DEVICE)

        # 2. Chạy dự đoán (Suy luận 1 lần)
        with torch.no_grad():
            output = MODEL(input_tensor)
            probs = F.softmax(output, dim=1)

        # 3. Lấy Top-K
        top_probs, top_indices = torch.topk(probs, top_k)
        top_probs = top_probs.squeeze().cpu().numpy()
        top_indices = top_indices.squeeze().cpu().numpy()
        
        # 4. TẠO KẾT QUẢ
        results = []
        for i in range(len(top_indices)):
            label_name = CLASS_NAMES[top_indices[i]]
            probability = top_probs[i]
            
            color = COLOR_MAP.get(label_name, "Không rõ")
            summary = get_wiki_summary_cached(label_name)

            item = {
                "label": label_name,
                "probability": f"{probability * 100:.2f}%",
                "color": color,
                "summary": summary
            }
            
            results.append(item)

        return results

    except Exception as e:
        print(f"Lỗi dự đoán: {e}", file=sys.stderr)
        return [{ "label": "Lỗi xử lý ảnh", "probability": "0.00%", "color": "Đen", "summary": "Không thể xử lý ảnh đầu vào." }]