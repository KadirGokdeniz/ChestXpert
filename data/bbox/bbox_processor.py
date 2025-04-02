import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class BBoxProcessor:
    def __init__(self, bbox_path):
        """
        Bounding box işleme sınıfı
        
        Args:
            bbox_path: Bounding box CSV dosyası yolu
        """
        self.bbox_path = bbox_path
        self.bbox_df = None
        self.load_bbox_data()
        
    def load_bbox_data(self):
        """Bounding box verilerini yükler"""
        self.bbox_df = pd.read_csv(self.bbox_path)
        return self.bbox_df
    
    def analyze_bbox_data(self):
        """Bounding box verilerini analiz eder"""
        # Hastalık bazında bbox sayısını inceleyin
        bbox_disease_counts = self.bbox_df['Finding Label'].value_counts()
        # bbox_processor.py dosyasına ekleyin (analyze_bbox_data fonksiyonunun başına)
        
        # Bbox boyutlarını inceleyin
        # x,y,w,h bilgilerini ayıklayın
        # bbox_processor.py dosyasını güncelleyin
        bbox_coords = []
        try:
            # 4 ayrı sütundan koordinatları al
            for idx, row in self.bbox_df.iterrows():
                try:
                    # Sütun değerlerinin tipini kontrol et ve uygun şekilde işle
                    x_val = row['Bbox [x']
                    y_val = row['y']
                    w_val = row['w']
                    h_val = row['h]']
                    
                    # Eğer string ise temizle, değilse doğrudan dönüştür
                    if isinstance(x_val, str):
                        x = float(x_val.replace('[', '').strip())
                    else:
                        x = float(x_val)
                        
                    if isinstance(y_val, str):
                        y = float(y_val.strip())
                    else:
                        y = float(y_val)
                        
                    if isinstance(w_val, str):
                        w = float(w_val.strip())
                    else:
                        w = float(w_val)
                        
                    if isinstance(h_val, str):
                        h = float(h_val.replace(']', '').strip())
                    else:
                        h = float(h_val)
                    
                    bbox_coords.append([x, y, w, h])
                except Exception as e:
                    print(f"Satır {idx} için koordinat dönüştürme hatası: {e}")
                    continue
        except Exception as e:
            print(f"BBox veri işleme hatası: {e}")

        if not bbox_coords:
            # Boş liste durumunda hata vermesini önle
            return bbox_disease_counts, {
                'width_mean': 0, 'width_std': 0,
                'height_mean': 0, 'height_std': 0,
                'area_mean': 0, 'area_std': 0
            }
            
        bbox_coords = np.array(bbox_coords)
        
        # İstatistikler
        bbox_stats = {
            'width_mean': np.mean(bbox_coords[:, 2]),
            'width_std': np.std(bbox_coords[:, 2]),
            'height_mean': np.mean(bbox_coords[:, 3]),
            'height_std': np.std(bbox_coords[:, 3]),
            'area_mean': np.mean(bbox_coords[:, 2] * bbox_coords[:, 3]),
            'area_std': np.std(bbox_coords[:, 2] * bbox_coords[:, 3])
        }
        
        return bbox_disease_counts, bbox_stats
    
    def visualize_bbox_distribution(self, save_path=None):
        """Hastalık türüne göre bbox dağılımını görselleştirir"""
        bbox_disease_counts, _ = self.analyze_bbox_data()
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=bbox_disease_counts.index, y=bbox_disease_counts.values)
        plt.xticks(rotation=45, ha='right')
        plt.title('Bounding Box Distribution by Disease')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()