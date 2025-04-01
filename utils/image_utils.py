import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def load_image(image_path):
    """Görüntüyü yükle ve RGB formatına dönüştür"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def sample_images_from_dataset(data_dir, image_dir, num_samples=5):
    """Veri setinden rastgele görüntüler örnekle"""
    # Tek bir görüntü dizini için
    folder_path = os.path.join(data_dir, image_dir)
    print(f"Görüntü dizini: {folder_path}")
    
    all_image_paths = []
    if os.path.exists(folder_path):
        image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        all_image_paths.extend(image_files)
    else:
        print(f"HATA: {folder_path} dizini bulunamadı!")
    
    # Rastgele örnekler seç
    if len(all_image_paths) > num_samples:
        return random.sample(all_image_paths, num_samples)
    return all_image_paths

def analyze_image_dimensions(data_dir, image_folders, sample_size=100):
    """Görüntü boyut dağılımını analiz et"""
    widths = []
    heights = []
    
    image_paths = sample_images_from_dataset(data_dir, image_folders, sample_size)
    
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            h, w = img.shape[:2]
            heights.append(h)
            widths.append(w)
    
    return {
        'width_stats': {
            'min': min(widths),
            'max': max(widths),
            'mean': np.mean(widths),
            'std': np.std(widths)
        },
        'height_stats': {
            'min': min(heights),
            'max': max(heights),
            'mean': np.mean(heights),
            'std': np.std(heights)
        }
    }