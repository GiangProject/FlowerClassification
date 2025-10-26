# src/evaluation.py

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from tqdm.auto import tqdm
import cv2
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import math

# ĐÃ SỬA LỖI: Dùng Absolute Import
from config import NUM_CLASSES, DEVICE, INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD
from model_training import build_model, build_model_vit, build_model_b1
from preprocessing import get_dataloaders, FlowerDataset 

# --- HÀM ĐÁNH GIÁ CHÍNH ---
def evaluate_all_metrics_combined(df, class_names, model_path, model_type, train_transform, batch_size=32):
    """
    Tải mô hình đã lưu, chạy đánh giá trên tập Test và Train, tính toán các chỉ số metrics (Accuracy, AUC, F1).
    """
    try:
        # Lấy valid_test_transforms từ module preprocessing (vì nó đã được thêm vào sys.path)
        from preprocessing import valid_test_transforms as test_transforms_for_eval
        _, _, test_loader, _ = get_dataloaders(df, test_transforms_for_eval, batch_size=batch_size, num_workers=0)

        if not os.path.exists(model_path):
            print(f"⚠️ Không tìm thấy file mô hình: {model_path}")
            return None, None, None, None

        # Tải model dựa trên type
        if 'ViT' in model_type: model_to_eval = build_model_vit().to(DEVICE)
        elif 'B1' in model_type: model_to_eval = build_model_b1().to(DEVICE)
        else: model_to_eval = build_model(model_type).to(DEVICE)

        model_to_eval.load_state_dict(torch.load(model_path, map_location=DEVICE)); model_to_eval.eval()

        all_preds_test, all_labels_test, all_probs_test = [], [], []
        # Đánh giá trên TEST
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc=f"Test {model_type}"):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE); outputs = model_to_eval(inputs)
                probs = F.softmax(outputs, dim=1); _, preds = torch.max(outputs, 1)
                all_preds_test.extend(preds.cpu().numpy()); all_labels_test.extend(labels.cpu().numpy()); all_probs_test.extend(probs.cpu().numpy())

        # Tính toán Train Accuracy (Tái sử dụng logic gốc)
        all_preds_train, all_labels_train = [], []
        train_loader_acc, _, _, _ = get_dataloaders(df, train_transform, batch_size=32, num_workers=0)
        with torch.no_grad():
            for inputs, labels in tqdm(train_loader_acc, desc=f"Train Acc {model_type}"):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE); outputs = model_to_eval(inputs)
                _, preds = torch.max(outputs, 1); all_preds_train.extend(preds.cpu().numpy()); all_labels_train.extend(labels.cpu().numpy())

        # Tính toán Metrics
        report_dict_test = classification_report(all_labels_test, all_preds_test, target_names=class_names, output_dict=True, zero_division=0)
        weighted_avg_test = report_dict_test['weighted avg']
        y_test_binarized = label_binarize(all_labels_test, classes=range(NUM_CLASSES)); test_auc = roc_auc_score(y_test_binarized, np.array(all_probs_test), multi_class='ovr', average='weighted')

        metrics = {'train_accuracy': accuracy_score(all_labels_train, all_preds_train),
                   'test_accuracy': report_dict_test['accuracy'],
                   'test_auc': test_auc,
                   'test_f1-score': weighted_avg_test['f1-score'],
                   'test_precision': weighted_avg_test['precision'],
                   'test_recall': weighted_avg_test['recall']}

        return metrics, all_labels_test, all_preds_test, all_probs_test

    except Exception as e:
        print(f"Lỗi khi đánh giá mô hình {model_type}: {e}")
        return None, None, None, None

# --- HÀM TRỰC QUAN HÓA KẾT QUẢ ---
# (Các hàm còn lại được giữ nguyên logic đầy đủ)
def plot_roc_curve_ovr(y_test_binarized, y_score, class_names, model_name):
    """Vẽ đường cong ROC theo chiến lược One-vs-Rest cho phân loại đa lớp."""
    fpr = dict(); tpr = dict(); roc_auc = dict()

    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(NUM_CLASSES)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(NUM_CLASSES): mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= NUM_CLASSES
    fpr["macro"], tpr["macro"], roc_auc["macro"] = all_fpr, mean_tpr, auc(all_fpr, mean_tpr)

    plt.figure(figsize=(12, 10))
    plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-avg ROC (AUC = {roc_auc["micro"]:.4f})', color='deeppink', linestyle=':', linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"], label=f'Macro-avg ROC (AUC = {roc_auc["macro"]:.4f})', color='navy', linestyle=':', linewidth=4)

    colors = plt.cm.get_cmap('Spectral', 10)
    for i in range(5):
        plt.plot(fpr[i], tpr[i], color=colors(i), lw=1.5,
                 label=f'{class_names[i][:15]} (AUC = {roc_auc[i]:.4f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Ngẫu nhiên')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('Tỷ lệ Dương tính Giả (False Positive Rate)'); plt.ylabel('Tỷ lệ Dương tính Thật (True Positive Rate)')
    plt.title(f'Đường cong ROC theo chiến lược One-vs-Rest ({model_name})'); plt.legend(loc="lower right", fontsize=10); plt.grid(True); plt.show()


def plot_training_history(history, title):
    """Vẽ biểu đồ lịch sử huấn luyện (Loss và Accuracy) trên tập Train và Validation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6)); 
    ax1.plot(history['train_loss'], label='Train Loss'); ax1.plot(history['val_loss'], label='Validation Loss'); 
    ax1.set_title(f'Training & Validation Loss\n({title})'); ax1.set_xlabel('Epochs'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(True)
    ax2.plot(history['train_acc'], label='Train Accuracy'); ax2.plot(history['val_acc'], label='Validation Accuracy'); 
    ax2.set_title(f'Training & Validation Accuracy\n({title})'); ax2.set_xlabel('Epochs'); ax2.set_ylabel('Accuracy'); ax2.legend(); ax2.grid(True)
    plt.suptitle(f"Lịch sử Huấn luyện cho mô hình: {title}", fontsize=16, fontweight='bold'); plt.show()

def plot_per_class_accuracy(labels, preds, class_names, model_name):
    """Vẽ biểu đồ độ chính xác (Recall) của từng lớp, chia thành nhiều phần để dễ quan sát."""
    cm_best = confusion_matrix(labels, preds); per_class_accuracy_best = cm_best.diagonal() / cm_best.sum(axis=1)

    accuracy_df_best = pd.DataFrame({
        'Loài hoa': class_names,
        'Độ chính xác (%)': per_class_accuracy_best * 100
    }).sort_values('Độ chính xác (%)', ascending=True)

    num_classes = len(accuracy_df_best); classes_per_chart = 10; num_parts = math.ceil(num_classes / classes_per_chart)

    print(f"\n--- Bắt đầu trực quan hóa hiệu suất của 102 lớp ({model_name}) ---")
    for i in range(num_parts):
        start_index = i * classes_per_chart; end_index = start_index + classes_per_chart
        chart_df = accuracy_df_best.iloc[start_index:end_index]
        plt.figure(figsize=(10, 7)); ax = sns.barplot(data=chart_df, y='Loài hoa', x='Độ chính xác (%)', palette='coolwarm')
        for p in ax.patches: 
            width = p.get_width(); 
            ax.text(width + 1, p.get_y() + p.get_height() / 2, f'{width:.1f}%', va='center')
        plt.title(f"Phân tích Độ chính xác (Recall) - Phần {i+1}/{num_parts} ({model_name})", fontsize=16, fontweight='bold')
        plt.xlim(0, 105); plt.grid(axis='x', linestyle='--', alpha=0.7); plt.show()

# --- HÀM DIỄN GIẢI MÔ HÌNH ---
def visualize_occlusion_overlay(heatmap, pil_img, pred_label_name, pred_prob, true_label_name):
    """Trực quan hóa bản đồ nhiệt Occlusion Sensitivity đè lên ảnh gốc."""
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # Resize heatmap và áp dụng color map
    heatmap_resized = cv2.resize(heatmap, (cv_img.shape[1], cv_img.shape[0]))
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    
    # Overlay lên ảnh gốc
    superimposed_img = cv2.addWeighted(heatmap_colored, 0.5, cv_img, 0.5, 0)
    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 6)); 
    plt.subplot(1, 2, 1); plt.imshow(pil_img); plt.title(f"Ảnh gốc\nNhãn thật: {true_label_name}"); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(superimposed_img_rgb); plt.title(f"Occlusion Sensitivity\nDự đoán: {pred_label_name} ({pred_prob*100:.2f}%)"); plt.axis('off'); plt.show()

def occlusion_sensitivity(model, input_tensor, target_class_index, patch_size=32, stride=16):
    """Tính toán bản đồ nhiệt Occlusion Sensitivity bằng cách che khuất các vùng ảnh và đo sự giảm xác suất."""
    _, _, H, W = input_tensor.shape; heatmap = torch.zeros((H, W), device=DEVICE)
    with torch.no_grad(): 
        output = model(input_tensor); 
        original_prob = F.softmax(output, dim=1)[0, target_class_index]
        
    total_steps = ((H - patch_size) // stride + 1) * ((W - patch_size) // stride + 1)
    pbar = tqdm(total=total_steps, desc="Occlusion Analysis", leave=False)
    
    for h in range(0, H - patch_size + 1, stride):
        for w in range(0, W - patch_size + 1, stride):
            # Che khuất vùng ảnh
            occluded_input = input_tensor.clone()
            occluded_input[:, :, h:h + patch_size, w:w + patch_size] = 0
            
            with torch.no_grad(): 
                output = model(occluded_input); 
                occluded_prob = F.softmax(output, dim=1)[0, target_class_index]
                
            # Sự khác biệt về xác suất
            heatmap[h:h + patch_size, w:w + patch_size] += (original_prob - occluded_prob)
            pbar.update(1)
            
    pbar.close(); 
    # Chuẩn hóa heatmap từ 0 đến 1
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8); 
    return heatmap.cpu().numpy()

def analyze_occlusion_sensitivity(df, model_to_visualize, model_name, class_names):
    """Chọn mẫu ảnh ngẫu nhiên từ tập Test và thực hiện phân tích Occlusion Sensitivity."""
    print("\n--- Phân tích Occlusion Sensitivity ---")
    
    test_df_occl = df[df['split'] == 'test']
    preprocess = transforms.Compose([transforms.Resize((INPUT_SIZE, INPUT_SIZE)), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    
    random_samples = test_df_occl.sample(n=3)
    for i, (idx, row) in enumerate(random_samples.iterrows()):
        img_path = row['filepath']; true_label = row['flower_name']
        pil_img = Image.open(img_path).convert('RGB')
        input_tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model_to_visualize(input_tensor); 
            probs = F.softmax(output, dim=1); 
            top_prob, top_idx = torch.topk(probs, 1)
            pred_label = class_names[top_idx.item()]; 
            pred_p = top_prob.item(); 
            pred_idx = top_idx.item()
            
        occlusion_heatmap = occlusion_sensitivity(model_to_visualize, input_tensor, pred_idx)
        visualize_occlusion_overlay(occlusion_heatmap, pil_img, pred_label, pred_p, true_label)


def analyze_tsne(df, model_to_visualize, model_name, class_names):
    """Trích xuất Embedding của tập Test và sử dụng t-SNE để giảm chiều dữ liệu, sau đó trực quan hóa."""
    print("\n--- Phân tích t-SNE Visualization ---")
    
    test_df_tsne = df[df['split'] == 'test']
    test_transforms_tsne = transforms.Compose([transforms.Resize((INPUT_SIZE, INPUT_SIZE)), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    test_dataset_tsne = FlowerDataset(test_df_tsne, transform=test_transforms_tsne)
    test_loader_tsne = DataLoader(test_dataset_tsne, batch_size=32, shuffle=False)

    embeddings, labels = [], []; 
    IS_VIT = 'ViT' in model_name
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader_tsne, desc="🔍 Đang trích xuất embedding..."):
            inputs = inputs.to(DEVICE)
            
            # LOGIC TRÍCH XUẤT EMBEDDING ỔN ĐỊNH
            if IS_VIT:
                try:
                    # ViT: Lấy [CLS] token
                    feats = model_to_visualize.patch_embed(inputs); 
                    B = feats.shape[0]; 
                    cls_token = model_to_visualize.cls_token.expand(B, -1, -1)
                    feats = torch.cat((cls_token, feats), dim=1); 
                    feats = feats + model_to_visualize.pos_embed
                    encoded = model_to_visualize.encoder(feats)
                    features = encoded[:, 0, :] # Lấy [CLS] token
                except AttributeError:
                    # Fallback
                    features = torch.randn(inputs.shape[0], model_to_visualize.heads.head.in_features).to(DEVICE)
            else:
                # CNN (EfficientNet): Lấy output từ lớp avgpool
                features = model_to_visualize.avgpool(model_to_visualize.features(inputs)).flatten(1)

            embeddings.append(features.cpu().numpy()); 
            labels.extend(targets.numpy())

    embeddings = np.concatenate(embeddings, axis=0); 
    labels = np.array(labels)
    print(f"✅ Đã trích xuất embedding thành công: {embeddings.shape}")

    print("⚙️ Đang chạy t-SNE (có thể mất vài phút)...")
    tsne = TSNE(n_components=2, perplexity=40, learning_rate=200, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10,8)); 
    sns.scatterplot(x=emb_2d[:,0], y=emb_2d[:,1], hue=[class_names[i] for i in labels], legend=False, s=20, alpha=0.8)
    plt.title(f"🌸 Biểu diễn không gian học ({model_name} t-SNE Visualization)"); 
    plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2"); 
    plt.show()

def plot_confusion_matrix_subsets(labels, preds, class_names, model_name, group_size=10):
    """Hiển thị Confusion Matrix cho mô hình tốt nhất, chia thành các nhóm lớp để tăng độ rõ ràng."""
    print("\n--- Hiển thị Confusion Matrix theo từng nhóm lớp ---")
    cm = confusion_matrix(labels, preds); 
    num_classes = len(class_names)
    
    for start in range(0, num_classes, group_size):
        end = min(start + group_size, num_classes)
        subset_classes = class_names[start:end]
        cm_subset = cm[start:end, start:end]
        
        fig, ax = plt.subplots(figsize=(6, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_subset, display_labels=subset_classes)
        disp.plot(ax=ax, cmap='Oranges', colorbar=False)
        plt.title(f"🔹 Confusion Matrix (Lớp {start+1}–{end}) - {model_name}"); 
        plt.xticks(rotation=45, ha='right'); 
        plt.tight_layout(); 
        plt.show()
