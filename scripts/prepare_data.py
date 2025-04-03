# scripts/prepare_data.py

import argparse
import os
import yaml
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Proje kök dizinini Python yoluna ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing.preprocessing import ChestXpertDataManager
from data.preprocessing.augmentation import get_safe_augmentations, apply_offline_augmentations
def parse_args():
    parser = argparse.ArgumentParser(description='ChestXpert veri hazırlama')
    parser.add_argument('--config', type=str, default='configs/data_prep_config.yaml',
                      help='Konfigürasyon dosyası')
    parser.add_argument('--quality_filter', action='store_true',
                      help='Kalite filtreleme uygula')
    parser.add_argument('--augment', action='store_true',
                      help='Veri artırma uygula')
    parser.add_argument('--full_dataset', action='store_true',
                      help='Tüm veri setini kullan (sınırlı örnek modunu kapat)')
    parser.add_argument('--balanced_aug', action='store_true',
                      help='Dengeli veri artırma uygula')
    parser.add_argument('--loss_type', type=str, default='weighted_bce',
                      choices=['bce', 'weighted_bce', 'focal', 'asymmetric'],
                      help='Kullanılacak kayıp fonksiyonu tipi')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Konfigürasyon dosyasını yükle
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Veri yöneticisini oluştur
    data_manager = ChestXpertDataManager(config)
    
    # Metadata yükle
    print("Metadata yükleniyor...")
    data_manager.load_metadata()
    
    # Kalite filtreleme
    if args.quality_filter:
        print("Düşük kaliteli görüntüler filtreleniyor...")
        quality_csv = config.get('quality_csv', 'data/quality_metrics.csv')
        data_manager.filter_by_quality(quality_csv)
    
    # Veri bölme
    print("Veriler bölünüyor...")
    train_df, val_df, test_df = data_manager.split_dataset()
    
    # Tam veri seti veya sınırlı örnek modu
    limit_samples = not args.full_dataset
    
    # Dataloader'ları oluştur ve test et
    print("Dataloaderlar test ediliyor...")
    train_dataset, val_dataset, test_dataset, class_weights = data_manager.create_datasets(limit_samples=limit_samples)
    train_loader, val_loader, test_loader, _ = data_manager.create_dataloaders()
    
    # Dengeli artırma (isteğe bağlı)
    if args.balanced_aug:
        print("Dengeli veri artırma uygulanıyor...")
        from data.preprocessing.augmentation import apply_balanced_augmentation
        
        disease_cols = [col for col in data_manager.train_df.columns if col in ['Atelectasis', 'Effusion', 'Nodule', 'Hernia', 'Cardiomegaly', 'Mass', 'Pneumothorax', 'Pleural_Thickening', 'Emphysema', 'Edema', 'Consolidation', 'Fibrosis', 'No Finding', 'Pneumonia', 'Infiltration']]
        
        output_dir = os.path.join(config.get('output_dir', 'data/processed'), 'augmented')
        apply_balanced_augmentation(
            data_manager.train_df, 
            disease_cols, 
            config['data_dir'], 
            output_dir, 
            target_ratio=0.5
        )
        print(f"Dengeli veri artırma tamamlandı, çıktı dizini: {output_dir}")
    
    # Örnek mini-batch kontrolü
    for images, labels in train_loader:
        print(f"Mini-batch boyutu: {images.shape}")
        print(f"Etiket boyutu: {labels.shape}")
        print(f"Etiket dağılımı: {labels.sum(dim=0)}")
        break
    
    print("Veri hazırlama tamamlandı!")

if __name__ == "__main__":
    main()