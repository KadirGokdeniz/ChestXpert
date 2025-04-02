import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from .datasets import ChestXrayDataset
from .preprocessing.preprocessing import XRayPreprocessor, filter_by_quality
from .preprocessing.augmentation import get_safe_augmentations
class ChestXpertDataManager:
    def __init__(self, config):
        self.config = config
        self.data_dir = config['data_dir']
        self.metadata_path = config['metadata_path']
        self.output_dir = config.get('output_dir', 'processed_data')
        
        # Veri bölünme oranları
        self.train_ratio = config.get('train_ratio', 0.7)
        self.val_ratio = config.get('val_ratio', 0.15)
        
        # Önişleme ve augmentasyon parametreleri
        self.preprocessor = XRayPreprocessor(config)
        self.quality_threshold = config.get('quality_threshold', {
            'snr': 1.5,
            'contrast': 220
        })
        
        # Veriler
        self.metadata_df = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        
    def load_metadata(self):
        """Metadata ve etiketleri yükle"""
        self.metadata_df = pd.read_csv(self.metadata_path)
        return self.metadata_df
    
    def filter_by_quality(self, quality_csv):
        """Düşük kaliteli görüntüleri filtrele"""
        quality_df = pd.read_csv(quality_csv)
        
        self.metadata_df = filter_by_quality(
            self.metadata_df, 
            quality_df,
            snr_threshold=self.quality_threshold['snr'],
            contrast_threshold=self.quality_threshold['contrast']
        )
        return self.metadata_df
    
    def split_dataset(self, stratify_col='Finding Label'):
        """Verileri eğitim/doğrulama/test olarak böl"""
        if self.metadata_df is None:
            self.load_metadata()
        
        # Hasta bazlı bölünme
        self.train_df, temp_df = train_test_split(
            self.metadata_df,
            train_size=self.train_ratio,
            random_state=42,
            stratify=self.metadata_df[stratify_col] if stratify_col else None
        )
        
        # Temp'ten doğrulama ve test setleri
        val_ratio = self.val_ratio / (1 - self.train_ratio)
        self.val_df, self.test_df = train_test_split(
            temp_df,
            train_size=val_ratio,
            random_state=42,
            stratify=temp_df[stratify_col] if stratify_col else None
        )
        
        print(f"Eğitim seti: {len(self.train_df)} görüntü")
        print(f"Doğrulama seti: {len(self.val_df)} görüntü")
        print(f"Test seti: {len(self.test_df)} görüntü")
        
        return self.train_df, self.val_df, self.test_df
    
    def create_datasets(self, apply_augmentation=True):
        """PyTorch dataset'leri oluştur"""
        if self.train_df is None:
            self.split_dataset()
        
        # Augmentasyon seçimi
        train_transform = get_safe_augmentations('medium') if apply_augmentation else None
        
        # Sınıf ağırlıkları hesaplama
        disease_cols = [col for col in self.train_df.columns if col in ['No Finding', 'Nodule', 'Pneumonia', ...]]
        class_weights = compute_class_weights(self.train_df[disease_cols])
        
        # Dataset'leri oluştur
        train_dataset = ChestXrayDataset(
            image_paths=self.train_df['path'].values,
            labels=self.train_df[disease_cols].values,
            preprocessor=self.preprocessor,
            transform=train_transform
        )
        
        val_dataset = ChestXrayDataset(
            image_paths=self.val_df['path'].values,
            labels=self.val_df[disease_cols].values,
            preprocessor=self.preprocessor,
            transform=None  # Doğrulamada augmentasyon yok
        )
        
        test_dataset = ChestXrayDataset(
            image_paths=self.test_df['path'].values,
            labels=self.test_df[disease_cols].values,
            preprocessor=self.preprocessor,
            transform=None  # Testte augmentasyon yok
        )
        
        return train_dataset, val_dataset, test_dataset, class_weights
    
    def create_dataloaders(self, batch_size=32, use_sampler=True):
        """PyTorch dataloaders oluştur"""
        train_dataset, val_dataset, test_dataset, class_weights = self.create_datasets()
        
        # Eğitim için dengeli örnekleyici
        if use_sampler:
            disease_cols = [col for col in self.train_df.columns if col in ['No Finding', 'Nodule', 'Pneumonia', ...]]
            sampler = create_balanced_sampler(train_dataset, self.train_df[disease_cols].values)
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4
            )
        else:
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
            )
        
        # Doğrulama ve test için standart dataloaders
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )
        
        return train_loader, val_loader, test_loader, class_weights
def create_balanced_sampler(dataset, labels, oversample_ratio=0.8):
    """Dengesiz veri için ağırlıklı örnekleyici"""
    if isinstance(labels, pd.DataFrame):
        labels = labels.values
    
    # Her örnek için sınıf indeksleri
    # Çok etiketli veriler için en nadir sınıfı kullan
    label_counts = np.sum(labels, axis=0)
    min_count = np.min(label_counts)
    
    # Her örnek için ağırlık hesapla
    sample_weights = np.zeros(len(labels))
    
    for i in range(len(labels)):
        sample_cls = np.where(labels[i] == 1)[0]
        if len(sample_cls) > 0:
            # En nadir sınıfı seç
            rarest_class = sample_cls[np.argmin(label_counts[sample_cls])]
            sample_weights[i] = 1.0 / label_counts[rarest_class]
        else:
            # 'No Finding' sınıfı için
            sample_weights[i] = 1.0 / len(labels)
    
    # Örnekleyici oluştur
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler


def create_stratified_batch_sampler(labels, batch_size=32, shuffle=True):
    """Her mini-batch'te dengeli sınıf dağılımı sağlayan örnekleyici"""
    # Sınıflara göre indeksleri grupla
    class_indices = [np.where(labels[:, i] == 1)[0] for i in range(labels.shape[1])]
    
    n_batches = len(labels) // batch_size
    batch_indices = []
    
    # Her batch için
    for i in range(n_batches):
        batch_idx = []
        # Her sınıftan örnekler ekle
        for cls_idx in class_indices:
            # Sınıf boyutuna göre örnekleme sayısını belirle
            n_samples = max(1, min(len(cls_idx), batch_size // len(class_indices)))
            
            # Rasgele örnekle
            if shuffle:
                sampled_idx = np.random.choice(cls_idx, size=n_samples, replace=False)
            else:
                start_idx = (i * n_samples) % len(cls_idx)
                sampled_idx = cls_idx[start_idx:start_idx + n_samples]
            
            batch_idx.extend(sampled_idx)
        
        # Batch boyutunu tamamla
        if len(batch_idx) < batch_size:
            # Tüm veriden rastgele örneklerle tamamla
            remaining = batch_size - len(batch_idx)
            all_indices = np.arange(len(labels))
            remaining_idx = np.setdiff1d(all_indices, batch_idx)
            additional_idx = np.random.choice(remaining_idx, size=remaining, replace=False)
            batch_idx.extend(additional_idx)
        
        # Batch'i ekle
        batch_indices.append(batch_idx[:batch_size])
    
    return batch_indices

def create_stratified_batch_sampler(labels, batch_size=32, shuffle=True):
    """Her mini-batch'te dengeli sınıf dağılımı sağlayan örnekleyici"""
    # Sınıflara göre indeksleri grupla
    class_indices = [np.where(labels[:, i] == 1)[0] for i in range(labels.shape[1])]
    
    n_batches = len(labels) // batch_size
    batch_indices = []
    
    # Her batch için
    for i in range(n_batches):
        batch_idx = []
        # Her sınıftan örnekler ekle
        for cls_idx in class_indices:
            # Sınıf boyutuna göre örnekleme sayısını belirle
            n_samples = max(1, min(len(cls_idx), batch_size // len(class_indices)))
            
            # Rasgele örnekle
            if shuffle:
                sampled_idx = np.random.choice(cls_idx, size=n_samples, replace=False)
            else:
                start_idx = (i * n_samples) % len(cls_idx)
                sampled_idx = cls_idx[start_idx:start_idx + n_samples]
            
            batch_idx.extend(sampled_idx)
        
        # Batch boyutunu tamamla
        if len(batch_idx) < batch_size:
            # Tüm veriden rastgele örneklerle tamamla
            remaining = batch_size - len(batch_idx)
            all_indices = np.arange(len(labels))
            remaining_idx = np.setdiff1d(all_indices, batch_idx)
            additional_idx = np.random.choice(remaining_idx, size=remaining, replace=False)
            batch_idx.extend(additional_idx)
        
        # Batch'i ekle
        batch_indices.append(batch_idx[:batch_size])
    
    return batch_indices
def test_class_balancing_strategies(train_df, disease_cols, batch_size=32):
    """Farklı sınıf dengeleme stratejilerinin etkisini gözlemle"""
    # Veri seti oluştur
    preprocessor = XRayPreprocessor({})
    dataset = ChestXrayDataset(
        image_paths=train_df['path'].values,
        labels=train_df[disease_cols].values,
        preprocessor=preprocessor
    )
    
    # Farklı örnekleme stratejileri
    results = {}
    
    # 1. Normal rastgele örnekleme
    random_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    
    # 2. Ağırlıklı örnekleyici
    weighted_sampler = create_balanced_sampler(dataset, train_df[disease_cols].values)
    weighted_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=weighted_sampler, num_workers=4
    )
    
    # 3. Katmanlı batch örnekleyici
    stratified_indices = create_stratified_batch_sampler(train_df[disease_cols].values, batch_size)
    stratified_batches = [train_df.iloc[indices] for indices in stratified_indices[:5]]
    
    # Sonuçları toplama
    loaders = {
        'Random': random_loader,
        'Weighted': weighted_loader,
    }
    
    # Her yükleyici için sınıf dağılımını gözlemle
    for name, loader in loaders.items():
        batch_distributions = []
        
        for i, (_, labels) in enumerate(loader):
            if i >= 5:  # İlk 5 batch yeterli
                break
                
            batch_dist = labels.sum(dim=0).numpy()
            batch_distributions.append(batch_dist)
        
        results[name] = np.array(batch_distributions)
    
    # Stratified batch'ler için
    stratified_dists = []
    for batch_df in stratified_batches:
        dist = batch_df[disease_cols].sum().values
        stratified_dists.append(dist)
    
    results['Stratified'] = np.array(stratified_dists)
    
    # Görselleştirme
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 4 * len(results)))
    
    for i, (name, distributions) in enumerate(results.items()):
        ax = axes[i]
        for j, dist in enumerate(distributions):
            ax.bar(np.arange(len(disease_cols)) + j*0.15, dist, width=0.15, 
                   label=f'Batch {j+1}')
            
        ax.set_title(f'{name} Sampling - Class Distribution')
        ax.set_xticks(np.arange(len(disease_cols)))
        ax.set_xticklabels(disease_cols, rotation=45, ha='right')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('sampling_strategies.png')
    plt.show()
    
    return results