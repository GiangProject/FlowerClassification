# src/preprocessing.py

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
# ĐÃ SỬA LỖI: Dùng Absolute Import (config)
from config import INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD

# --- HÀM/CLASS PREPROCESSING ---

class FlowerDataset(Dataset):
    """
    Class Dataset tùy chỉnh để tải ảnh và nhãn.
    """
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
    def __len__(self): 
        """Trả về tổng số lượng mẫu trong dataset."""
        return len(self.df)
    def __getitem__(self, idx):
        """Tải và áp dụng transform cho ảnh tại chỉ mục idx."""
        img_path = self.df.loc[idx, 'filepath']
        label = int(self.df.loc[idx, 'label'])
        image = Image.open(img_path).convert("RGB")
        if self.transform: image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

# Transforms
base_train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(INPUT_SIZE), transforms.RandomHorizontalFlip(), transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])
strong_train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.5, 1.0)), transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=40, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])
valid_test_transforms = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(INPUT_SIZE), transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

def get_dataloaders(df, train_transforms, batch_size=32, num_workers=0):
    """
    Tạo các DataLoader cho tập Train, Validation và Test.
    Áp dụng train_transforms cho tập train, valid_test_transforms cho tập valid/test.
    """
    test_transforms = valid_test_transforms
    train_df = df[df['split'] == 'train']; valid_df = df[df['split'] == 'valid']; test_df = df[df['split'] == 'test']
    
    train_loader = DataLoader(FlowerDataset(train_df, transform=train_transforms), batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(FlowerDataset(valid_df, transform=test_transforms), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(FlowerDataset(test_df, transform=test_transforms), batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    print(f"✅ DataLoaders đã sẵn sàng (Batch Size: {batch_size}).")
    return train_loader, valid_loader, test_loader, test_df
