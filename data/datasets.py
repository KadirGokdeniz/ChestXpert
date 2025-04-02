import os
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset

class ChestXrayDataset(Dataset):
    def __init__(self, image_paths, labels=None, preprocessor=None, transform=None):
        """
        NIH Chest X-ray Dataset sınıfı
        
        Args:
            image_paths: Görüntü dosyalarının yolları
            labels: Görüntülere ait hastalık etiketleri (pandas DataFrame veya numpy array)
            preprocessor: Görüntü ön işleme nesnesi
            transform: Veri artırma dönüşüm nesnesi (Albumentations)
        """
        self.image_paths = image_paths
        self.labels = labels
        self.preprocessor = preprocessor
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        try:
            # Görüntüyü yükle
            img_path = self.image_paths[idx]
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                print(f"Uyarı: {img_path} yüklenemedi")
                image = np.zeros((1024, 1024), dtype=np.uint8)
            
            # Veri artırma uygula
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed["image"]
            
            # Preprocessor kullanarak işle
            if self.preprocessor:
                image = self.preprocessor(image)
            else:
                # Basit normalizasyon
                image = image.astype(np.float32) / 255.0
                image = torch.from_numpy(image).unsqueeze(0)
            
            # Gerçek etiketleri döndür (eğer varsa)
            if self.labels is not None:
                label = torch.tensor(self.labels[idx], dtype=torch.float)
                return image, label
            else:
                # Etiket yoksa dummy
                return image, torch.tensor([0.0])
                
        except Exception as e:
            print(f"Görüntü yükleme hatası ({img_path}): {e}")
            # Hata durumunda dummy
            dummy_image = torch.zeros((1, 1024, 1024))
            
            if self.labels is not None:
                dummy_label = torch.zeros_like(torch.tensor(self.labels[0], dtype=torch.float))
                return dummy_image, dummy_label
            else:
                return dummy_image, torch.tensor([0.0])