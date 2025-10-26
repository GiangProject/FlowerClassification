# src/app/main.py

# ===================================================================
# ⭐️ CÀI ĐẶT THƯ VIỆN & IMPORTS (ĐIỀU PHỐI)
# ===================================================================
print("--- Đang cài đặt/kiểm tra thư viện phụ thuộc ---")
# Cần chạy lệnh này nếu môi trường chưa cài đặt
# !pip install -q torchsummary opencv-python scikit-learn
print("--- Hoàn tất cài đặt thư viện ---")

import os
import sys
import json
import scipy.io
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
import time
from tqdm.auto import tqdm # Thêm tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from sklearn.metrics import classification_report, accuracy_score # Thêm accuracy_score
from sklearn.preprocessing import label_binarize

# --- IMPORTS TỪ SRC ---
# SỬA LỖI MODULE NOT FOUND: BỎ COMMENT DÒNG THÊM PATH TƯƠNG ĐỐI
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
from config import DEVICE, NUM_CLASSES, PROJECT_PATH
from preprocessing import get_dataloaders, base_train_transforms, strong_train_transforms, valid_test_transforms 
from eda import visualize_all_102_flowers, plot_color_distribution, analyze_image_size
from model_training import build_model, build_model_b1, build_model_vit, train_phase
from evaluation import (
    evaluate_all_metrics_combined, plot_roc_curve_ovr, plot_training_history,
    analyze_occlusion_sensitivity, analyze_tsne, plot_confusion_matrix_subsets,
    plot_per_class_accuracy
)

# ===================================================================
# ⭐️ SETUP, TẢI DỮ LIỆU & TẠO DATAFRAME GỐC (BLOCK 1: LOGIC KHỞI TẠO)
# ===================================================================
# --- GẮN DRIVE VÀ THIẾT LẬP THƯ MỤC ---
from google.colab import drive
try: drive.mount('/content/drive')
except: pass
os.makedirs(PROJECT_PATH, exist_ok=True); os.chdir(PROJECT_PATH)
os.makedirs('data', exist_ok=True); os.makedirs('models', exist_ok=True)

def ensure_data_integrity():
    """Tải và kiểm tra tính toàn vẹn của dữ liệu gốc (ảnh và file .mat)."""
    MAT_FILE_PATH = 'data/imagelabels.mat'

    if not os.path.exists('jpg') or not os.path.exists(MAT_FILE_PATH) or (os.path.exists('jpg') and len(os.listdir('jpg/')) < 8000):
        print("--- 📥 Đang tải và kiểm tra Dữ liệu Gốc ---")
        # ĐÃ SỬA LỖI CÚ PHÁP: Thay thế !wget bằng os.system('wget ...')
        os.system('wget -q https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz -P data/')
        os.system('wget -q https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat -P data/')
        os.system('wget -q https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat -P data/')
        os.system('wget -q https://raw.githubusercontent.com/udacity/pytorch_challenge/master/cat_to_name.json -P data/')
        time.sleep(3)

        if os.path.exists('jpg'): os.system('rm -rf jpg')

        if os.path.exists('data/102flowers.tgz'):
             os.system('tar -xzf data/102flowers.tgz')
             print("Giải nén ảnh hoàn tất.")

        time.sleep(2)

        if not os.path.exists(MAT_FILE_PATH) or len(os.listdir('jpg/')) < 8000:
            print("❌ LỖI TRUY CẬP DRIVE: Dữ liệu bị thiếu sau nhiều lần thử.")
            raise FileNotFoundError("Không thể đảm bảo tính toàn vẹn của dữ liệu.")

    print(f"✅ Dữ liệu đã sẵn sàng. Số lượng file ảnh: {len(os.listdir('jpg/'))}")

ensure_data_integrity()

# --- TẠO DATAFRAME VÀ CLASS NAMES ---
image_labels = scipy.io.loadmat('data/imagelabels.mat')['labels'][0]
set_ids = scipy.io.loadmat('data/setid.mat')
with open('data/cat_to_name.json', 'r') as f: cat_to_name = json.load(f)

image_files = sorted(os.listdir('jpg/'))
df = pd.DataFrame({'filepath': 'jpg/' + pd.Series(image_files), 'label': image_labels - 1})
df['split'] = ''; df.loc[set_ids['trnid'][0] - 1, 'split'] = 'train'; df.loc[set_ids['valid'][0] - 1, 'split'] = 'valid'; df.loc[set_ids['tstid'][0] - 1, 'split'] = 'test'
df['class_id'] = (df['label'] + 1).astype(str); df['flower_name'] = df['class_id'].map(cat_to_name); df = df.drop(columns=['class_id'])
class_names_df = df.drop_duplicates(subset=['label']).sort_values('label'); class_names = class_names_df['flower_name'].tolist()

print(f"\nThiết bị sử dụng: {DEVICE}"); print("--- DataFrame df và class_names đã sẵn sàng ---")
# df.info() # Giữ nguyên df.info() nếu cần hiển thị thông tin dataframe

# ===================================================================
# ⭐️ THỰC THI EDA (BLOCK 2: GỌI CÁC HÀM EDA)
# ===================================================================
visualize_all_102_flowers(df)
print("\n--- Biểu đồ phân bố 20 loài hoa có nhiều ảnh nhất ---")
plt.figure(figsize=(12, 8))
sns.countplot(y='flower_name', data=df, order=df['flower_name'].value_counts().iloc[:20].index, palette='viridis')
plt.title('Phân bố 20 loài hoa hàng đầu'); plt.xlabel('Số lượng ảnh'); plt.ylabel('Tên loài hoa'); plt.show()
plot_color_distribution(df)
analyze_image_size(df)

# ===================================================================
# ⭐️ THỰC THI HUẤN LUYỆN MÔ HÌNH (BLOCK 4: GỌI CÁC HÀM TRAIN_PHASE)
# ===================================================================
all_histories = {}
criterion = torch.nn.CrossEntropyLoss()

# --- THỬ NGHIỆM 1: BASELINE B0 ---
print("🚀 Bắt đầu Thử nghiệm 1: Baseline Augmentation")
EXP_NAME = "exp1_baseline"; MODEL_SAVE_PATH = f'models/{EXP_NAME}.pth'
train_loader, valid_loader, test_loader_b0_base, test_df_baseline = get_dataloaders(df, base_train_transforms, batch_size=32)
model_exp1 = build_model().to(DEVICE)
optimizer_phase1 = optim.Adam(model_exp1.classifier.parameters(), lr=1e-3); scheduler_phase1 = lr_scheduler.ReduceLROnPlateau(optimizer_phase1, mode='min', patience=3)
model_exp1, history_exp1_p1 = train_phase(model_exp1, criterion, optimizer_phase1, scheduler_phase1, {'train': train_loader, 'valid': valid_loader}, num_epochs=15, patience=3, model_save_path=MODEL_SAVE_PATH)
for param in model_exp1.parameters(): param.requires_grad = True
optimizer_ft = optim.Adam(model_exp1.parameters(), lr=1e-5); scheduler_ft = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', patience=2)
model_exp1, history_exp1_p2 = train_phase(model_exp1, criterion, optimizer_ft, scheduler_ft, {'train': train_loader, 'valid': valid_loader}, num_epochs=10, patience=3, model_save_path=MODEL_SAVE_PATH)
all_histories['Baseline (B0)'] = {key: history_exp1_p1[key] + history_exp1_p2[key] for key in history_exp1_p1}
print(f"\n✅ Hoàn tất Thử nghiệm {EXP_NAME} và đã lưu lịch sử!")

# --- THỬ NGHIỆM 2: STRONG AUG B0 ---
print("🚀 Bắt đầu Thử nghiệm 2: Strong Augmentation")
EXP_NAME = "exp2_strong_aug"; MODEL_SAVE_PATH = f'models/{EXP_NAME}.pth'
train_loader, valid_loader, test_loader_b0_strong, test_df_strong = get_dataloaders(df, strong_train_transforms, batch_size=32)
model_exp2 = build_model().to(DEVICE)
optimizer_phase1 = optim.Adam(model_exp2.classifier.parameters(), lr=1e-4); scheduler_phase1 = lr_scheduler.ReduceLROnPlateau(optimizer_phase1, mode='min', patience=5)
model_exp2, history_exp2_p1 = train_phase(model_exp2, criterion, optimizer_phase1, scheduler_phase1, {'train': train_loader, 'valid': valid_loader}, num_epochs=15, patience=5, model_save_path=MODEL_SAVE_PATH)
for param in model_exp2.parameters(): param.requires_grad = True
optimizer_ft = optim.Adam(model_exp2.parameters(), lr=1e-4); scheduler_ft = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', patience=2)
model_exp2, history_exp2_p2 = train_phase(model_exp2, criterion, optimizer_ft, scheduler_ft, {'train': train_loader, 'valid': valid_loader}, num_epochs=15, patience=3, model_save_path=MODEL_SAVE_PATH)
all_histories['Strong Aug (B0)'] = {key: history_exp2_p1[key] + history_exp2_p2[key] for key in history_exp2_p1}
print(f"\n✅ Hoàn tất Thử nghiệm {EXP_NAME} và đã lưu lịch sử!")

# --- THỬ NGHIỆM 3/4: B1 & B1 CONTINUED ---
print("🚀 Bắt đầu Thử nghiệm 3/4: EfficientNet-B1")
EXP_NAME = "exp4_b1_continued"; PREVIOUS_BEST_MODEL = 'models/exp3_efficientnet_b1.pth'; MODEL_SAVE_PATH = f'models/{EXP_NAME}.pth'
train_loader, valid_loader, test_loader_b1, test_df_b1 = get_dataloaders(df, strong_train_transforms, batch_size=32)
model_exp3 = build_model_b1().to(DEVICE)
optimizer_phase1 = optim.Adam(model_exp3.classifier.parameters(), lr=1e-4); scheduler_phase1 = lr_scheduler.ReduceLROnPlateau(optimizer_phase1, mode='min', patience=5)
model_exp3, history_exp3_p1 = train_phase(model_exp3, criterion, optimizer_phase1, scheduler_phase1, {'train': train_loader, 'valid': valid_loader}, num_epochs=15, patience=5, model_save_path=PREVIOUS_BEST_MODEL)
for param in model_exp3.parameters(): param.requires_grad = True
optimizer_ft = optim.Adam(model_exp3.parameters(), lr=1e-4); scheduler_ft = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', patience=2)
model_exp3, history_exp3_p2 = train_phase(model_exp3, criterion, optimizer_ft, scheduler_ft, {'train': train_loader, 'valid': valid_loader}, num_epochs=15, patience=3, model_save_path=PREVIOUS_BEST_MODEL)
model_to_continue = build_model_b1().to(DEVICE); model_to_continue.load_state_dict(torch.load(PREVIOUS_BEST_MODEL, map_location=DEVICE))
for param in model_to_continue.parameters(): param.requires_grad = True
optimizer_continue = optim.Adam(model_to_continue.parameters(), lr=1e-4); scheduler_continue = lr_scheduler.ReduceLROnPlateau(optimizer_continue, mode='min', patience=2)
final_model, history_exp4 = train_phase(model_to_continue, criterion, optimizer_continue, scheduler_continue, {'train': train_loader, 'valid': valid_loader}, num_epochs=15, patience=4, model_save_path=MODEL_SAVE_PATH)
all_histories['Final Model (B1)'] = {key: history_exp3_p1[key] + history_exp3_p2[key] + history_exp4[key] for key in history_exp3_p1}
print(f"\n✅ Hoàn tất Thử nghiệm {EXP_NAME} và đã lưu lịch sử!")

# --- THỬ NGHIỆM 5: ViT ---
print("🚀 Bắt đầu Thử nghiệm 5: ViT-B/16")
EXP_NAME = "exp_vit"; MODEL_SAVE_PATH = f'models/{EXP_NAME}.pth'
train_loader, valid_loader, test_loader_vit, test_df_vit = get_dataloaders(df, strong_train_transforms, batch_size=32)
model_exp_vit = build_model_vit().to(DEVICE)
optimizer_phase1 = optim.Adam(model_exp_vit.heads.head.parameters(), lr=1e-4); scheduler_phase1 = lr_scheduler.ReduceLROnPlateau(optimizer_phase1, mode='min', patience=3)
model_exp_vit, history_vit_p1 = train_phase(model_exp_vit, criterion, optimizer_phase1, scheduler_phase1, {'train': train_loader, 'valid': valid_loader}, num_epochs=15, patience=5, model_save_path=MODEL_SAVE_PATH)
for param in model_exp_vit.parameters(): param.requires_grad = True
optimizer_ft = optim.Adam(model_exp_vit.parameters(), lr=5e-5); scheduler_ft = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', patience=3)
model_exp_vit, history_vit_p2 = train_phase(model_exp_vit, criterion, optimizer_ft, scheduler_ft, {'train': train_loader, 'valid': valid_loader}, num_epochs=15, patience=4, model_save_path=MODEL_SAVE_PATH)
all_histories['ViT-B/16'] = {key: history_vit_p1[key] + history_vit_p2[key] for key in history_vit_p1}
print(f"\n✅ Hoàn tất Thử nghiệm {EXP_NAME} và đã lưu lịch sử!")

torch.cuda.empty_cache(); gc.collect()

# ===================================================================
# ⭐️ ĐÁNH GIÁ VÀ PHÂN TÍCH TỔNG KẾT (BLOCK 5, 6, 7: GỌI CÁC HÀM EVALUATION)
# ===================================================================
best_model_labels, best_model_preds, best_model_probs = None, None, None
best_acc_so_far = -1.0
best_model_name = ""
results_summary = {}

# --- CẤU HÌNH ĐÁNH GIÁ ---
MODEL_PATHS_TO_EVAL = {
    'Baseline (B0)': ('models/exp1_baseline.pth', 'EfficientNet-B0', base_train_transforms, 32),
    'Strong Aug (B0)': ('models/exp2_strong_aug.pth', 'EfficientNet-B0', strong_train_transforms, 32),
    'Final Model (B1)': ('models/exp4_b1_continued.pth', 'EfficientNet-B1', strong_train_transforms, 32),
    'ViT-B/16': ('models/exp_vit.pth', 'ViT-B/16', strong_train_transforms, 32),
}

# --- THỰC THI ĐÁNH GIÁ TỔNG KẾT VÀ TÌM MÔ HÌNH TỐT NHẤT ---
for exp_name, (path, model_type, train_aug_func, batch_size) in MODEL_PATHS_TO_EVAL.items():
    metrics, labels, preds, probs = evaluate_all_metrics_combined(df, class_names, path, model_type, train_aug_func, batch_size)

    if metrics:
        results_summary[exp_name] = metrics

        # LOGIC TỰ ĐỘNG CHỌN MÔ HÌNH TỐT NHẤT
        if metrics.get('test_accuracy', 0) > best_acc_so_far:
            best_acc_so_far = metrics['test_accuracy']
            best_model_name = exp_name
            best_model_labels, best_model_preds, best_model_probs = labels, preds, probs # Lưu kết quả mô hình tốt nhất

# --- 5b: HIỂN THỊ BẢNG KẾT QUẢ VÀ BIỂU ĐỒ SO SÁNH ---
if results_summary:
    print(f"\n✅ Mô hình tốt nhất được chọn để phân tích chi tiết là: {best_model_name} (Accuracy: {best_acc_so_far:.4f})")
    df_results = pd.DataFrame(results_summary).T; results_df = df_results.sort_values('test_accuracy', ascending=False)

    print("\n" + "="*80); print("                    📊 BẢNG SO SÁNH HIỆU SUẤT TỔNG THỂ CÁC MÔ HÌNH"); print("="*80)
    formatters = {'train_accuracy': '{:,.2%}'.format, 'test_accuracy': '{:,.2%}'.format, 'test_auc': '{:,.3f}'.format, 'test_f1-score': '{:,.3f}'.format, 'test_precision': '{:,.3f}'.format, 'test_recall': '{:,.3f}'.format}
    print(results_df.to_string(formatters=formatters)); print("="*80)

    # Trực quan hóa Train vs Test Accuracy
    df_acc = results_df[['train_accuracy', 'test_accuracy']].reset_index().rename(columns={'index': 'model'})
    df_acc_melted = pd.melt(df_acc, id_vars='model', value_vars=['train_accuracy', 'test_accuracy'], var_name='Metric', value_name='Accuracy')
    plt.figure(figsize=(14, 7)); sns.barplot(data=df_acc_melted, x='model', y='Accuracy', hue='Metric', palette='Set1')
    plt.title('So sánh Độ chính xác Train vs. Test (Phân tích Overfitting)', fontsize=16); plt.xticks(rotation=15); plt.grid(axis='y', linestyle='--', alpha=0.7); plt.show()

    # Trực quan hóa Test F1-Score vs Test AUC
    df_perf_f1_auc = results_df[['test_f1-score', 'test_auc']].reset_index().rename(columns={'index': 'model'})
    df_perf_f1_auc_melted = pd.melt(df_perf_f1_auc, id_vars='model', value_vars=['test_f1-score', 'test_auc'], var_name='Metric', value_name='Score')
    plt.figure(figsize=(14, 7)); sns.barplot(data=df_perf_f1_auc_melted, x='model', y='Score', hue='Metric', palette='Set2')
    plt.title('So sánh F1-Score và AUC trên Tập Test', fontsize=16); plt.xticks(rotation=15); plt.grid(axis='y', linestyle='--', alpha=0.7); plt.show()

    # TRỰC QUAN HÓA ROC CURVE
    if best_model_labels is not None and best_model_probs is not None:
        print("\n\n" + "="*80); print(f"          📈 TRỰC QUAN HÓA ĐƯỜNG CONG ROC ({best_model_name})"); print("="*80)
        y_test_binarized = label_binarize(best_model_labels, classes=range(NUM_CLASSES)); y_score = np.array(best_model_probs)
        plot_roc_curve_ovr(y_test_binarized, y_score, class_names, best_model_name)
    
    # TRỰC QUAN HÓA LỊCH SỬ HUẤN LUYỆN
    if all_histories:
        print("\n\n" + "="*80); print("                     📈 LỊCH SỬ HUẤN LUYỆN CÁC MÔ HÌNH"); print("="*80)
        for model_name, history in all_histories.items(): plot_training_history(history, model_name)


# --- PHÂN TÍCH CHI TIẾT MÔ HÌNH TỐT NHẤT (BLOCK 6 & 7) ---
if best_model_labels is not None:
    print("\n" + "="*80); print("          🔬 BẮT ĐẦU PHÂN TÍCH DIỄN GIẢI MÔ HÌNH TỐT NHẤT"); print("="*80)

    # 1. Tải lại mô hình tốt nhất
    if best_model_name == 'ViT-B/16':
        BEST_MODEL_PATH = 'models/exp_vit.pth'; model_to_visualize = build_model_vit().to(DEVICE)
    elif best_model_name == 'Final Model (B1)':
        BEST_MODEL_PATH = 'models/exp4_b1_continued.pth'; model_to_visualize = build_model_b1().to(DEVICE)
    elif best_model_name == 'Strong Aug (B0)':
        BEST_MODEL_PATH = 'models/exp2_strong_aug.pth'; model_to_visualize = build_model().to(DEVICE)
    else:
        BEST_MODEL_PATH = 'models/exp1_baseline.pth'; model_to_visualize = build_model().to(DEVICE)

    try:
        model_to_visualize.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
        model_to_visualize.eval()
        
        # 2. Báo cáo phân loại chi tiết
        print("\n📊 Báo cáo Phân loại Chi tiết cho Mô hình Tốt nhất:\n")
        print(classification_report(best_model_labels, best_model_preds, target_names=class_names))
        
        # 3. Trực quan hóa độ chính xác từng lớp
        plot_per_class_accuracy(best_model_labels, best_model_preds, class_names, best_model_name)
        
        # 4. Phân tích Occlusion Sensitivity
        analyze_occlusion_sensitivity(df, model_to_visualize, best_model_name, class_names)
        
        # 5. Phân tích t-SNE
        analyze_tsne(df, model_to_visualize, best_model_name, class_names)
        
        # 6. Confusion Matrix chia nhỏ
        plot_confusion_matrix_subsets(best_model_labels, best_model_preds, class_names, best_model_name)

    except Exception as e:
        print(f"❌ Lỗi tải hoặc phân tích chi tiết mô hình: {e}")
