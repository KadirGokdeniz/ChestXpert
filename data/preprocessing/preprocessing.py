import os
import numpy as np
import pandas as pd
import cv2
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from data.preprocessing.augmentation import get_safe_augmentations

class XRayPreprocessor:
    def __init__(self, config):
        """
        X-ray görüntüleri için ön işleme sınıfı.
        
        Args:
            config: Konfigürasyon parametreleri içeren sözlük.
                   Olası anahtarlar: 'resize_dim', 'normalization', 'apply_clahe', 'window'
        """
        preprocessing_config = config.get('preprocessing', {})
        self.resize_dim = preprocessing_config.get('resize_dim', (1024, 1024))
        self.norm_method = preprocessing_config.get('normalization', 'minmax')
        self.apply_clahe = preprocessing_config.get('apply_clahe', False)
        self.window = preprocessing_config.get('window', None)
        
    def __call__(self, image):
        # Boyut kontrolü ve düzenleme
        if image.shape[:2] != self.resize_dim:
            image = cv2.resize(image, self.resize_dim)
        
        # Gri tonlama kontrolü
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # CLAHE uygula
        if self.apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            image = clahe.apply(image)
        
        # Normalizasyon
        image = self.normalize_xray(image, self.norm_method, self.window)
        
        # Tensor'a çevirme
        image_tensor = torch.from_numpy(image).float()
        if len(image_tensor.shape) == 2:
            image_tensor = image_tensor.unsqueeze(0)  # Kanal boyutu ekle
        
        return image_tensor
    
    def normalize_xray(self, image, method='minmax', window=None):
        """Farklı normalizasyon stratejileri uygula"""
        if method == 'minmax':
            return (image - image.min()) / (image.max() - image.min() + 1e-10)
        elif method == 'zscore':
            mean = image.mean()
            std = image.std()
            return (image - mean) / (std + 1e-10)  # Sıfıra bölünmeyi önle
        elif method == 'histogram_eq':
            return cv2.equalizeHist(image.astype(np.uint8))
        elif method == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            return clahe.apply(image.astype(np.uint8))
        elif method == 'window':
            # Pencere seviyesi normalizasyonu (özellikle DICOM için)
            window_center, window_width = window
            lower = window_center - window_width/2
            upper = window_center + window_width/2
            image = np.clip(image, lower, upper)
            return (image - lower) / (upper - lower + 1e-10)
        return image  # Varsayılan: normalizasyon yok
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
        """NIH Chest X-ray veri seti metadata'sını yükle ve doğru dosya yollarını oluştur"""
        print(f"Metadata dosyası yükleniyor: {self.metadata_path}")
        df = pd.read_csv(self.metadata_path)
        
        # NIH görüntülerinin alt klasörlerde bulunduğunu hesaba kat
        if 'Image Index' in df.columns:
            def find_image_path(img_name):
                # Alt klasör sayısını belirle (001-012 arası)
                num_subfolders = 12
                
                # Her alt klasörde dosyayı ara
                for i in range(1, num_subfolders + 1):
                    subfolder = f"images_{i:03d}"  # 001, 002, ... 012 formatında
                    img_path = os.path.join(self.data_dir, subfolder, "images", img_name)
                    
                    if os.path.exists(img_path):
                        return img_path
                
                # Bulunamazsa None döndür
                print(f"UYARI: {img_name} hiçbir alt klasörde bulunamadı")
                return None
            
            # Görüntü yollarını oluştur
            print("Görüntü yolları oluşturuluyor (alt klasörlerde arama yapılıyor)...")
            df['path'] = df['Image Index'].apply(find_image_path)
            
            # Bulunamayan görüntüleri filtrele
            missing_count = df['path'].isna().sum()
            if missing_count > 0:
                print(f"UYARI: {missing_count} görüntü ({missing_count/len(df)*100:.2f}%) bulunamadı")
                # İsteğe bağlı: Bulunamayan görüntüleri filtrele
                df = df[df['path'].notna()].reset_index(drop=True)
                print(f"Bulunamayan görüntüler filtrelendi. Kalan görüntü sayısı: {len(df)}")
        
        # NIH veri seti sütunlarını hastalık sınıflarına ayır
        if 'Finding Labels' in df.columns:
            disease_labels = df['Finding Labels'].str.split('|')
            all_diseases = set()
            for labels in disease_labels:
                all_diseases.update(labels)
            
            # Tüm hastalık sınıfları için sütun oluştur
            for disease in all_diseases:
                df[disease] = df['Finding Labels'].apply(lambda x: 1 if disease in x else 0)
        
        self.metadata_df = df
        print(f"Metadata yüklendi. Toplam {len(df)} kayıt.")
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
    
    def split_dataset(self, stratify_col=None):
        """Verileri eğitim/doğrulama/test olarak böl"""
        if self.metadata_df is None:
            self.load_metadata()
        
        # Stratifikasyon için güvenli bir yaklaşım
        stratify_data = None
        if 'Finding Labels' in self.metadata_df.columns:
            # Finding Labels'ı basit sınıflara ayır
            # No Finding vs Any Finding (daha güvenli bir stratifikasyon)
            has_finding = self.metadata_df['Finding Labels'].apply(lambda x: 0 if x == 'No Finding' else 1)
            # Sınıf sayısını kontrol et, en az her sınıfta 5 örnek olsun
            if has_finding.value_counts().min() >= 5:
                stratify_data = has_finding
                print(f"Stratifikasyon için 'Has Finding' sütunu kullanılıyor")
            else:
                print(f"Yeterli örnek yok, stratifikasyon devre dışı bırakıldı")
        
        # Hasta bazlı bölünme
        patient_id_col = 'Patient ID' if 'Patient ID' in self.metadata_df.columns else None
        
        try:
            if patient_id_col:
                print(f"Hasta bazlı bölünme yapılıyor, hasta sütunu: {patient_id_col}")
                unique_patients = self.metadata_df[patient_id_col].unique()
                
                # Stratifikasyon güvenlik kontrolü
                if stratify_data is not None:
                    patient_group = self.metadata_df.groupby(patient_id_col)
                    # Her hasta için bir etiket (en yaygın etiket)
                    patient_labels = patient_group[stratify_data.name].agg(lambda x: x.mode()[0] if len(x) > 0 else None)
                    # Sınıf sayısını kontrol et
                    if patient_labels.value_counts().min() >= 5:
                        stratify_for_split = patient_labels
                        print(f"Hasta etiketleri stratifikasyon için kullanılıyor")
                    else:
                        stratify_for_split = None
                        print(f"Hasta etiketlerinde yeterli sınıf yok, stratifikasyon devre dışı")
                else:
                    stratify_for_split = None
                
                # Stratifikasyon güvenlik kontrolü
                # Eğitim ve geçici set - stratifikasyon olmadan
                train_patients, temp_patients = train_test_split(
                    unique_patients, 
                    train_size=self.train_ratio,
                    random_state=42,
                    stratify=None  # Stratifikasyonu devre dışı bırak
                )
                
                # Geçici setten doğrulama ve test setleri
                val_ratio_adjusted = self.val_ratio / (1 - self.train_ratio)
                val_patients, test_patients = train_test_split(
                    temp_patients, 
                    train_size=val_ratio_adjusted,
                    random_state=42
                )
                
                # Hasta ID'lerine göre filtreleme
                self.train_df = self.metadata_df[self.metadata_df[patient_id_col].isin(train_patients)]
                self.val_df = self.metadata_df[self.metadata_df[patient_id_col].isin(val_patients)]
                self.test_df = self.metadata_df[self.metadata_df[patient_id_col].isin(test_patients)]
            else:
                # Görüntü bazlı bölünme - stratifikasyon olmadan
                print("Görüntü bazlı bölünme yapılıyor (stratifikasyon olmadan)")
                self.train_df, temp_df = train_test_split(
                    self.metadata_df,
                    train_size=self.train_ratio,
                    random_state=42,
                    stratify=None  # Stratifikasyonu devre dışı bırak
                )
                
                # Temp'ten doğrulama ve test setleri
                val_ratio_adjusted = self.val_ratio / (1 - self.train_ratio)
                self.val_df, self.test_df = train_test_split(
                    temp_df,
                    train_size=val_ratio_adjusted,
                    random_state=42
                )
        except Exception as e:
            print(f"Stratifikasyon hatası: {e}")
            print("Basit rastgele bölünme yapılıyor...")
            # Herhangi bir hata durumunda basit rastgele bölünme yap
            self.train_df, temp_df = train_test_split(
                self.metadata_df,
                train_size=self.train_ratio,
                random_state=42
            )
            
            val_ratio_adjusted = self.val_ratio / (1 - self.train_ratio)
            self.val_df, self.test_df = train_test_split(
                temp_df,
                train_size=val_ratio_adjusted,
                random_state=42
            )
        
        print(f"Eğitim seti: {len(self.train_df)} görüntü")
        print(f"Doğrulama seti: {len(self.val_df)} görüntü")
        print(f"Test seti: {len(self.test_df)} görüntü")
        
        return self.train_df, self.val_df, self.test_df
        
    def create_datasets(self, apply_augmentation=True, limit_samples=False):
        """PyTorch dataset'leri oluştur"""
        if self.train_df is None:
            self.split_dataset()
        
        # Augmentasyon
        from albumentations import Compose, Rotate, RandomBrightnessContrast, HorizontalFlip, ShiftScaleRotate, GaussNoise
        
        def simple_augmentations():
            return Compose([
                Rotate(limit=10, p=0.5),
                RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
                HorizontalFlip(p=0.5),
                ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.3),
                GaussNoise(p=0.2),
            ])
        
        train_transform = simple_augmentations() if apply_augmentation else None
        
        # DÜZELTME: Sadece gerçek hastalık sütunlarını kullan
        known_disease_cols = ['Atelectasis', 'Effusion', 'Nodule', 'Hernia', 'Cardiomegaly', 
                            'Mass', 'Pneumothorax', 'Pleural_Thickening', 'Emphysema', 
                            'Edema', 'Consolidation', 'Fibrosis', 'No Finding', 
                            'Pneumonia', 'Infiltration']
        
        # Veri çerçevesinde var olan hastalık sütunlarını filtrele
        disease_cols = [col for col in known_disease_cols if col in self.train_df.columns]
        
        if not disease_cols:
            print("UYARI: Hastalık sütunları bulunamadı!")
            if 'Finding Labels' in self.train_df.columns:
                print("'Finding Labels' sütunu bulundu. Hastalık etiketlerini ayrıştırılıyor...")
                # Finding Labels sütunundan hastalık etiketlerini çıkar
                all_findings = set()
                for findings in self.train_df['Finding Labels']:
                    if isinstance(findings, str):
                        all_findings.update(findings.split('|'))
                
                # Hastalık etiketleri için sütunlar oluştur
                for finding in all_findings:
                    self.train_df[finding] = self.train_df['Finding Labels'].apply(
                        lambda x: 1 if isinstance(x, str) and finding in x else 0)
                    self.val_df[finding] = self.val_df['Finding Labels'].apply(
                        lambda x: 1 if isinstance(x, str) and finding in x else 0)
                    self.test_df[finding] = self.test_df['Finding Labels'].apply(
                        lambda x: 1 if isinstance(x, str) and finding in x else 0)
                
                # Güncellenen hastalık sütunlarını al
                disease_cols = list(all_findings)
            else:
                # Eğer gerçek hastalık sütunları bulunamazsa dummy etiket oluştur
                disease_cols = ['dummy_label']
                self.train_df['dummy_label'] = 0
                self.val_df['dummy_label'] = 0
                self.test_df['dummy_label'] = 0
        
        print(f"Etiket olarak kullanılacak hastalık sütunları: {disease_cols}")
        
        # Görüntü yollarını oluştur
        train_paths = self.train_df['path'].values
        val_paths = self.val_df['path'].values
        test_paths = self.test_df['path'].values
        
        # Etiketleri numpy dizilerine dönüştür
        train_labels = self.train_df[disease_cols].values
        val_labels = self.val_df[disease_cols].values
        test_labels = self.test_df[disease_cols].values
        
        print(f"Eğitim görüntü sayısı: {len(train_paths)}")
        print(f"Doğrulama görüntü sayısı: {len(val_paths)}")
        print(f"Test görüntü sayısı: {len(test_paths)}")
        
        # Sınıf ağırlıklarını hesapla - sadece sayısal değerlere sahip sütunları kullan
        try:
            class_weights = compute_class_weights(self.train_df[disease_cols])
            print(f"Hesaplanan sınıf ağırlıkları: {class_weights}")
        except Exception as e:
            print(f"Sınıf ağırlıkları hesaplanırken hata: {e}")
            class_weights = {i: 1.0 for i in range(len(disease_cols))}
        
        # Görüntü sayısını sınırla veya tümünü kullan
        if limit_samples:
            sample_limit = {'train': 100, 'val': 20, 'test': 20}
            print(f"UYARI: Sınırlı örnek modu aktif! Eğitim: {sample_limit['train']}, Doğrulama: {sample_limit['val']}, Test: {sample_limit['test']} görüntü")
        else:
            sample_limit = {'train': len(train_paths), 'val': len(val_paths), 'test': len(test_paths)}
        
        from data.datasets import ChestXrayDataset
        
        # Dataset'leri oluştur
        train_dataset = ChestXrayDataset(
            image_paths=train_paths[:sample_limit['train']],
            labels=train_labels[:sample_limit['train']],
            preprocessor=self.preprocessor,
            transform=train_transform
        )
        
        val_dataset = ChestXrayDataset(
            image_paths=val_paths[:sample_limit['val']],
            labels=val_labels[:sample_limit['val']],
            preprocessor=self.preprocessor,
            transform=None
        )
        
        test_dataset = ChestXrayDataset(
            image_paths=test_paths[:sample_limit['test']],
            labels=test_labels[:sample_limit['test']],
            preprocessor=self.preprocessor,
            transform=None
        )
        
        print("Dataset'ler başarıyla oluşturuldu")
        
        return train_dataset, val_dataset, test_dataset, class_weights
    def create_dataloaders(self, batch_size=32, num_workers=0):  # num_workers=0 yapın
        """PyTorch dataloaders oluştur"""
        train_dataset, val_dataset, test_dataset, class_weights = self.create_datasets()
        
        # Dataloader'ları oluştur - num_workers=0 ile tek işlem
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        print(f"DataLoader'lar başarıyla oluşturuldu (batch size: {batch_size}, single process)")
        
        return train_loader, val_loader, test_loader, class_weights
def filter_by_quality(image_df, quality_df, snr_threshold=1.0, contrast_threshold=200):
    """Düşük kaliteli görüntüleri filtrele"""
    merged_df = image_df.merge(quality_df, on='image_id', how='left')
    filtered_df = merged_df[
        (merged_df['snr'] >= snr_threshold) & 
        (merged_df['contrast'] >= contrast_threshold)
    ]
    return filtered_df
def compute_class_weights(labels_df, method='inverse'):
    """Farklı yöntemlerle sınıf ağırlıkları hesaplama"""
    try:
        # Toplam örnek sayısını ve pozitif örnek sayılarını hesapla
        n_samples = len(labels_df)
        
        # Eğer etiketler bir DataFrame
        if isinstance(labels_df, pd.DataFrame):
            classes = labels_df.columns
            pos_counts = labels_df.sum(axis=0).values
        # Eğer etiketler bir numpy dizisi
        elif isinstance(labels_df, np.ndarray):
            classes = range(labels_df.shape[1])
            pos_counts = labels_df.sum(axis=0)
        else:
            print("Uyarı: Etiket formatı desteklenmiyor, dummy ağırlıklar döndürülüyor")
            return {0: 1.0}  # Dummy ağırlık
        
        n_classes = len(classes)
        
        # Sıfıra bölünmeyi önle
        pos_counts = np.maximum(pos_counts, 1)
        
        if method == 'inverse':
            # Sınıf frekansının tersi ile ağırlıklandırma
            weights = n_samples / (n_classes * pos_counts)
        elif method == 'balanced':
            # Dengeli ağırlıklandırma
            weights = n_samples / (2 * pos_counts)
        elif method == 'effective_samples':
            # Etkili örnek sayısı yaklaşımı
            beta = 0.99
            weights = (1 - beta) / (1 - beta ** pos_counts)
        else:
            weights = np.ones(n_classes)
        
        # Ağırlıkları 1.0 etrafında normalize et
        if weights.sum() > 0:
            weights = weights / weights.sum() * n_classes
        
        return {str(class_name): float(weight) for class_name, weight in zip(classes, weights)}
    except Exception as e:
        print(f"Sınıf ağırlığı hesaplama hatası: {e}")
        return {0: 1.0}  # Hata durumunda dummy ağırlık döndür