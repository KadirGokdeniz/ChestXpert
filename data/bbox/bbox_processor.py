import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

def load_bbox_data(bbox_path):
    """Bounding box CSV dosyasını yükle"""
    return pd.read_csv(bbox_path)

def analyze_bbox_data(bbox_df):
    """Bounding box verilerini analiz et"""
    # (Önceki yanıttaki analyze_bbox_data fonksiyonu)
    ...

def visualize_bbox_on_image(image, bbox_str, color=(255, 0, 0), thickness=2):
    """Görüntü üzerine bounding box çiz"""
    try:
        coords = bbox_str.replace('[', '').replace(']', '').split(',')
        x, y, w, h = map(float, coords)
        cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), color, thickness)
    except:
        pass
    return image

def generate_bbox_stats(bbox_df, output_dir=None):
    """Bounding box istatistiklerini çıkar ve sonuçları kaydet"""
    stats = {}
    
    # Toplam bbox sayısı
    stats['total_bbox_records'] = len(bbox_df)
    stats['total_images_with_bbox'] = bbox_df['Image Index'].nunique()
    
    # Hastalık türüne göre bbox dağılımı
    bbox_disease_counts, bbox_size_stats = analyze_bbox_data(bbox_df)
    stats['bbox_disease_counts'] = bbox_disease_counts.to_dict()
    stats['bbox_size_stats'] = bbox_size_stats
    
    # Sonuçları JSON olarak kaydet
    if output_dir:
        import json
        with open(os.path.join(output_dir, 'bbox_stats.json'), 'w') as f:
            json.dump(stats, f, indent=4)
    
    return stats