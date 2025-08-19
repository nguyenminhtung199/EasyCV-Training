import torch
from PIL import Image
from torch.utils.data import Dataset
import os

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, transform=None):
        self.img_labels = []
        self.img_paths = []
        self.transform = transform
        
        # Đọc file dữ liệu
        with open(annotations_file, 'r') as f:
            for idx, line in enumerate(f):
                if idx == 0:
                    continue 
                parts = line.strip().split()
                img_path = parts[0]
                if not os.path.exists(img_path):
                    continue
                # Các nhãn tiếp theo (float) ở cuối dòng
                labels = [float(val) for val in parts[1:]]
                # Check if any label is between 0.2 and 0.9
                if any(0.2 < label < 0.9 for label in labels):
                    continue
                
                # Lưu đường dẫn và nhãn
                self.img_paths.append(img_path)
                self.img_labels.append(labels)

    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        # Đọc đường dẫn ảnh và nhãn
        img_path = self.img_paths[idx]
        labels = self.img_labels[idx]
        
        # Mở ảnh bằng PIL
        image = Image.open(img_path).convert("RGB")
        
        # Áp dụng transform nếu có
        if self.transform:
            image = self.transform(image)
        
        # Convert labels to tensor
        labels = torch.tensor(labels, dtype=torch.float32)
        
        return image, labels