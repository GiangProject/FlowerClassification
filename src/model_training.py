# src/model_training.py

import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary
import copy
import time
from tqdm.auto import tqdm
import torch.optim as optim
from torch.optim import lr_scheduler

# ĐÃ SỬA LỖI: Dùng Absolute Import (config)
from config import INPUT_SIZE, NUM_CLASSES, DEVICE 

# --- MODEL BUILDERS ---
def build_model(model_name='EfficientNet-B0', num_classes=NUM_CLASSES):
    """
    Hàm xây dựng mô hình chung (B0, B1, ViT) từ kiến trúc tiền huấn luyện ImageNet.
    Thực hiện đóng băng các tham số Feature Extractor và thay thế lớp Head (Classifier) cuối cùng.
    """
    model_name = model_name.lower(); model = None
    if 'b0' in model_name: model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    elif 'b1' in model_name: model = models.efficientnet_b1(weights='IMAGENET1K_V1')
    elif 'vit' in model_name: model = models.vit_b_16(weights='IMAGENET1K_V1')
    else: raise ValueError(f"Kiến trúc {model_name} không được hỗ trợ trong hàm build_model.")

    # Đóng băng tất cả tham số
    for param in model.parameters(): param.requires_grad = False

    # Thay thế lớp phân loại cuối cùng
    if 'vit' in model_name:
        num_ftrs = model.heads.head.in_features
        model.heads.head = torch.nn.Linear(num_ftrs, num_classes)
    else: # EfficientNet (B0/B1)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)

    print(f"\n--- Kiến trúc {model_name.upper()} ---")
    try:
        # Bỏ qua summary cho ViT do không tương thích với torchsummary
        if 'vit' not in model_name: summary(model.to(DEVICE), (3, INPUT_SIZE, INPUT_SIZE))
        else: print("Tóm tắt kiến trúc bị bỏ qua do không tương thích với torchsummary.")
    except Exception: print("Lỗi khi hiển thị summary.")

    return model.to(DEVICE)

def build_model_b1(num_classes=NUM_CLASSES): 
    """Hàm tạo mô hình EfficientNet-B1."""
    return build_model('EfficientNet-B1', num_classes)
    
def build_model_vit(num_classes=NUM_CLASSES): 
    """Hàm tạo mô hình ViT-B/16."""
    return build_model('ViT-B/16', num_classes)

# --- TRAINING PHASE ---
def train_phase(model, criterion, optimizer, scheduler, dataloaders, num_epochs=20, patience=5, model_save_path=None):
    """
    Thực hiện vòng lặp huấn luyện chính với các giai đoạn Train và Validation.
    Bao gồm logic Early Stopping và lưu lại trọng số mô hình tốt nhất.
    """
    device = next(model.parameters()).device
    scaler = torch.cuda.amp.GradScaler() # Sử dụng GradScaler cho Mixed Precision Training (tăng tốc GPU)
    best_model_wts = copy.deepcopy(model.state_dict()); best_acc = 0.0
    epochs_no_improve = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}\n{'-'*20}")
        for phase in ['train','valid']:
            model.train() if phase=='train' else model.eval()
            running_loss, running_corrects = 0.0, 0
            if phase=='train': torch.cuda.empty_cache()

            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase} phase"):
                inputs, labels = inputs.to(device), labels.to(device); optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase=='train'):
                    with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
                        outputs = model(inputs); _, preds = torch.max(outputs, 1); loss = criterion(outputs, labels)
                    if phase=='train': 
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0); running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double().cpu().item() / len(dataloaders[phase].dataset)

            if phase=='train': 
                history['train_loss'].append(epoch_loss); history['train_acc'].append(epoch_acc)
            else:
                history['val_loss'].append(epoch_loss); history['val_acc'].append(epoch_acc)
                # Logic Early Stopping và lưu mô hình tốt nhất
                if epoch_acc > best_acc:
                    best_acc = epoch_acc; epochs_no_improve = 0
                    best_model_wts = copy.deepcopy(model.state_dict())
                    if model_save_path: torch.save(model.state_dict(), model_save_path); print(f"✨ Best model saved! Val Acc: {best_acc:.4f}")
                else: epochs_no_improve += 1

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

        if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau): scheduler.step(epoch_loss)

        if epochs_no_improve >= patience: 
            print(f"🛑 Early stopping triggered after {patience} epochs"); break
        print()

    time_elapsed = time.time() - start_time
    print(f"Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s"); print(f"Best Val Acc: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model, history
