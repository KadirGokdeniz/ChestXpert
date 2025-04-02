import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import cv2
from PIL import Image
import pydicom  # Opsiyonel: DICOM formatını işlemek için

class ImageQualityAnalyzer:
    def __init__(self, metadata_path):
        """
        Görüntü kalitesi analiz sınıfı
        
        Args:
            metadata_path: Metadata CSV dosyası yolu
        """
        self.metadata_path = metadata_path
        self.metadata_df = None
        self.load_metadata()
        
    def load_metadata(self):
        """Metadata dosyasını yükler ve görüntü kalite ilgili alanları ekler"""
        self.metadata_df = pd.read_csv(self.metadata_path)
        
        # Gerekli sütunların varlığını kontrol et
        required_columns = ['Image Index', 'OriginalImage[Width,Height]', 'OriginalImagePixelSpacing[x,y]']
        for col in required_columns:
            if col not in self.metadata_df.columns:
                if col == 'OriginalImage[Width,Height]':
                    print(f"Uyarı: {col} sütunu bulunamadı. Boş bir sütun oluşturuluyor.")
                    self.metadata_df['OriginalImage[Width,Height]'] = None
                elif col == 'OriginalImagePixelSpacing[x,y]':
                    print(f"Uyarı: {col} sütunu bulunamadı. Boş bir sütun oluşturuluyor.")
                    self.metadata_df['OriginalImagePixelSpacing[x,y]'] = None
                else:
                    raise ValueError(f"Gerekli sütun bulunamadı: {col}")
        
        # Görüntü boyutları ve piksel aralığı bilgilerini ayrıştır
        self._parse_image_dimensions()
        self._parse_pixel_spacing()
        
        # Kalite metriklerini hesapla
        self._calculate_quality_metrics()
        
        return self.metadata_df
    
    def _parse_image_dimensions(self):
        """Görüntü boyutlarını ayrıştırır ve yeni sütunlar ekler"""
        # OriginalImage[Width,Height] sütununu işle
        widths = []
        heights = []
        
        for dims in self.metadata_df['OriginalImage[Width,Height]']:
            if pd.isna(dims) or dims is None or not isinstance(dims, str):
                widths.append(np.nan)
                heights.append(np.nan)
                continue
                
            try:
                # Parantez içini temizle ve virgülle ayrılmış değerleri al
                dims_clean = dims.replace('[', '').replace(']', '')
                width, height = map(int, dims_clean.split(','))
                widths.append(width)
                heights.append(height)
            except:
                widths.append(np.nan)
                heights.append(np.nan)
        
        self.metadata_df['image_width'] = widths
        self.metadata_df['image_height'] = heights
        self.metadata_df['image_resolution'] = self.metadata_df['image_width'] * self.metadata_df['image_height']
    
    def _parse_pixel_spacing(self):
        """Piksel aralığı bilgilerini ayrıştırır ve yeni sütunlar ekler"""
        # OriginalImagePixelSpacing[x,y] sütununu işle
        spacing_x = []
        spacing_y = []
        
        for spacing in self.metadata_df['OriginalImagePixelSpacing[x,y]']:
            if pd.isna(spacing) or spacing is None or not isinstance(spacing, str):
                spacing_x.append(np.nan)
                spacing_y.append(np.nan)
                continue
                
            try:
                # Parantez içini temizle ve virgülle ayrılmış değerleri al
                spacing_clean = spacing.replace('[', '').replace(']', '')
                x, y = map(float, spacing_clean.split(','))
                spacing_x.append(x)
                spacing_y.append(y)
            except:
                spacing_x.append(np.nan)
                spacing_y.append(np.nan)
        
        self.metadata_df['pixel_spacing_x'] = spacing_x
        self.metadata_df['pixel_spacing_y'] = spacing_y
        
        # Fiziksel boyutu mm cinsinden hesapla (uzunluk = piksel sayısı * piksel aralığı)
        self.metadata_df['physical_width_mm'] = self.metadata_df['image_width'] * self.metadata_df['pixel_spacing_x']
        self.metadata_df['physical_height_mm'] = self.metadata_df['image_height'] * self.metadata_df['pixel_spacing_y']
    
    def _calculate_quality_metrics(self):
        """Görüntü kalitesi metriklerini hesaplar"""
        # Görüntü çözünürlüğüne göre kalite sınıflandırması
        conditions = [
            (self.metadata_df['image_resolution'] >= 6000000),  # 6MP ve üzeri
            (self.metadata_df['image_resolution'] >= 3000000) & (self.metadata_df['image_resolution'] < 6000000),  # 3-6MP
            (self.metadata_df['image_resolution'] >= 1000000) & (self.metadata_df['image_resolution'] < 3000000),  # 1-3MP
            (self.metadata_df['image_resolution'] < 1000000)  # 1MP altı
        ]
        choices = ['Yüksek', 'İyi', 'Orta', 'Düşük']
        self.metadata_df['resolution_quality'] = np.select(conditions, choices, default='Bilinmiyor')
        
        # Piksel aralığına göre kalite sınıflandırması (0.1mm veya daha az = yüksek kalite)
        conditions = [
            (self.metadata_df['pixel_spacing_x'] <= 0.1) & (self.metadata_df['pixel_spacing_y'] <= 0.1),
            (self.metadata_df['pixel_spacing_x'] <= 0.2) & (self.metadata_df['pixel_spacing_y'] <= 0.2),
            (self.metadata_df['pixel_spacing_x'] > 0.2) | (self.metadata_df['pixel_spacing_y'] > 0.2)
        ]
        choices = ['Yüksek', 'İyi', 'Düşük']
        self.metadata_df['spacing_quality'] = np.select(conditions, choices, default='Bilinmiyor')
        
        # Genel kalite değerlendirmesi (özel bir algoritma oluşturulabilir)
        # Burada basitçe çözünürlük ve piksel aralığı kalitelerini kombine ediyoruz
        quality_map = {
            ('Yüksek', 'Yüksek'): 'Çok Yüksek',
            ('Yüksek', 'İyi'): 'Yüksek',
            ('İyi', 'Yüksek'): 'Yüksek',
            ('İyi', 'İyi'): 'İyi',
            ('Orta', 'İyi'): 'İyi',
            ('İyi', 'Düşük'): 'Orta',
            ('Orta', 'Düşük'): 'Düşük',
            ('Düşük', 'Düşük'): 'Çok Düşük'
        }
        
        self.metadata_df['overall_quality'] = 'Bilinmiyor'
        for (res_qual, spac_qual), overall_qual in quality_map.items():
            mask = (self.metadata_df['resolution_quality'] == res_qual) & (self.metadata_df['spacing_quality'] == spac_qual)
            self.metadata_df.loc[mask, 'overall_quality'] = overall_qual
    
    def analyze_image_dimensions(self):
        """Görüntü boyutları istatistiklerini analiz eder"""
        dim_stats = {
            'width_mean': self.metadata_df['image_width'].mean(),
            'width_std': self.metadata_df['image_width'].std(),
            'width_min': self.metadata_df['image_width'].min(),
            'width_max': self.metadata_df['image_width'].max(),
            'height_mean': self.metadata_df['image_height'].mean(),
            'height_std': self.metadata_df['image_height'].std(),
            'height_min': self.metadata_df['image_height'].min(),
            'height_max': self.metadata_df['image_height'].max(),
            'resolution_mean': self.metadata_df['image_resolution'].mean(),
            'resolution_std': self.metadata_df['image_resolution'].std(),
            'resolution_min': self.metadata_df['image_resolution'].min(),
            'resolution_max': self.metadata_df['image_resolution'].max(),
        }
        
        return dim_stats
    
    def analyze_pixel_spacing(self):
        """Piksel aralığı istatistiklerini analiz eder"""
        spacing_stats = {
            'spacing_x_mean': self.metadata_df['pixel_spacing_x'].mean(),
            'spacing_x_std': self.metadata_df['pixel_spacing_x'].std(),
            'spacing_x_min': self.metadata_df['pixel_spacing_x'].min(),
            'spacing_x_max': self.metadata_df['pixel_spacing_x'].max(),
            'spacing_y_mean': self.metadata_df['pixel_spacing_y'].mean(),
            'spacing_y_std': self.metadata_df['pixel_spacing_y'].std(),
            'spacing_y_min': self.metadata_df['pixel_spacing_y'].min(),
            'spacing_y_max': self.metadata_df['pixel_spacing_y'].max(),
            'physical_width_mm_mean': self.metadata_df['physical_width_mm'].mean(),
            'physical_height_mm_mean': self.metadata_df['physical_height_mm'].mean(),
        }
        
        return spacing_stats
    
    def analyze_quality_distribution(self):
        """Kalite dağılımını analiz eder"""
        resolution_quality_counts = self.metadata_df['resolution_quality'].value_counts()
        spacing_quality_counts = self.metadata_df['spacing_quality'].value_counts()
        overall_quality_counts = self.metadata_df['overall_quality'].value_counts()
        
        return {
            'resolution_quality': resolution_quality_counts.to_dict(),
            'spacing_quality': spacing_quality_counts.to_dict(),
            'overall_quality': overall_quality_counts.to_dict()
        }
    
    def analyze_quality_by_disease(self):
        """Hastalık türüne göre görüntü kalitesi analizi yapar"""
        # Finding Labels sütununu kullanarak görüntü kalitesini hastalık türüne göre analiz et
        if 'Finding Labels' not in self.metadata_df.columns:
            return {"error": "Finding Labels sütunu bulunamadı"}
            
        # Her görüntüye ait hastalıkları ayır (çoklu hastalık olabilir)
        quality_by_disease = {}
        
        for finding in self.metadata_df['Finding Labels'].unique():
            # Tek bir hastalık için görüntü kalitesi dağılımı
            quality_dist = self.metadata_df[self.metadata_df['Finding Labels'] == finding]['overall_quality'].value_counts()
            quality_by_disease[finding] = quality_dist.to_dict()
            
        return quality_by_disease
    
    def visualize_image_dimensions(self, save_path=None):
        """Görüntü boyutları dağılımını görselleştirir"""
        plt.figure(figsize=(15, 10))
        
        # Veri olup olmadığını kontrol et
        has_resolution_data = not self.metadata_df['image_resolution'].dropna().empty
        has_width_data = not self.metadata_df['image_width'].dropna().empty
        has_height_data = not self.metadata_df['image_height'].dropna().empty
        
        # 1. Çözünürlük histogramı
        plt.subplot(2, 2, 1)
        if has_resolution_data:
            sns.histplot(self.metadata_df['image_resolution'].dropna() / 1000000, bins=20)
            plt.title('Görüntü Çözünürlüğü Dağılımı')
            plt.xlabel('Çözünürlük (Megapiksel)')
            plt.ylabel('Görüntü Sayısı')
        else:
            plt.text(0.5, 0.5, 'Çözünürlük verisi bulunamadı', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Görüntü Çözünürlüğü Dağılımı - VERİ YOK')
        
        # 2. Görüntü genişliği histogramı
        plt.subplot(2, 2, 2)
        if has_width_data:
            sns.histplot(self.metadata_df['image_width'].dropna(), bins=20)
            plt.title('Görüntü Genişliği Dağılımı')
            plt.xlabel('Genişlik (piksel)')
            plt.ylabel('Görüntü Sayısı')
        else:
            plt.text(0.5, 0.5, 'Genişlik verisi bulunamadı', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Görüntü Genişliği Dağılımı - VERİ YOK')
        
        # 3. Görüntü yüksekliği histogramı
        plt.subplot(2, 2, 3)
        if has_height_data:
            sns.histplot(self.metadata_df['image_height'].dropna(), bins=20)
            plt.title('Görüntü Yüksekliği Dağılımı')
            plt.xlabel('Yükseklik (piksel)')
            plt.ylabel('Görüntü Sayısı')
        else:
            plt.text(0.5, 0.5, 'Yükseklik verisi bulunamadı', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Görüntü Yüksekliği Dağılımı - VERİ YOK')
        
        # 4. Görüntü boyutu dağılımı
        plt.subplot(2, 2, 4)
        if has_width_data and has_height_data:
            valid_dim_data = self.metadata_df.dropna(subset=['image_width', 'image_height'])
            if not valid_dim_data.empty:
                sample_data = valid_dim_data.sample(min(1000, len(valid_dim_data)))
                sns.scatterplot(x='image_width', y='image_height', data=sample_data)
                plt.title('Görüntü Boyutları')
                plt.xlabel('Genişlik (piksel)')
                plt.ylabel('Yükseklik (piksel)')
            else:
                plt.text(0.5, 0.5, 'Yeterli veri bulunamadı', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Görüntü Boyutları - VERİ YOK')
        else:
            plt.text(0.5, 0.5, 'Boyut verisi bulunamadı', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Görüntü Boyutları - VERİ YOK')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
    
    def visualize_pixel_spacing(self, save_path=None):
        """Piksel aralığı dağılımını görselleştirir"""
        plt.figure(figsize=(15, 10))
        
        # Veri olup olmadığını kontrol et
        has_spacing_x = not self.metadata_df['pixel_spacing_x'].dropna().empty
        has_spacing_y = not self.metadata_df['pixel_spacing_y'].dropna().empty
        has_physical_width = not self.metadata_df['physical_width_mm'].dropna().empty
        has_physical_height = not self.metadata_df['physical_height_mm'].dropna().empty
        
        # 1. Piksel aralığı X histogramı
        plt.subplot(2, 2, 1)
        if has_spacing_x:
            sns.histplot(self.metadata_df['pixel_spacing_x'].dropna(), bins=20)
            plt.title('Piksel Aralığı X Dağılımı')
            plt.xlabel('Piksel Aralığı X (mm)')
            plt.ylabel('Görüntü Sayısı')
        else:
            plt.text(0.5, 0.5, 'Piksel aralığı X verisi bulunamadı', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Piksel Aralığı X Dağılımı - VERİ YOK')
        
        # 2. Piksel aralığı Y histogramı
        plt.subplot(2, 2, 2)
        if has_spacing_y:
            sns.histplot(self.metadata_df['pixel_spacing_y'].dropna(), bins=20)
            plt.title('Piksel Aralığı Y Dağılımı')
            plt.xlabel('Piksel Aralığı Y (mm)')
            plt.ylabel('Görüntü Sayısı')
        else:
            plt.text(0.5, 0.5, 'Piksel aralığı Y verisi bulunamadı', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Piksel Aralığı Y Dağılımı - VERİ YOK')
        
        # 3. Fiziksel boyut dağılımı
        plt.subplot(2, 2, 3)
        if has_physical_width and has_physical_height:
            valid_physical_data = self.metadata_df.dropna(subset=['physical_width_mm', 'physical_height_mm'])
            if not valid_physical_data.empty:
                sample_data = valid_physical_data.sample(min(1000, len(valid_physical_data)))
                sns.scatterplot(x='physical_width_mm', y='physical_height_mm', data=sample_data)
                plt.title('Fiziksel Boyutlar (mm)')
                plt.xlabel('Fiziksel Genişlik (mm)')
                plt.ylabel('Fiziksel Yükseklik (mm)')
            else:
                plt.text(0.5, 0.5, 'Yeterli fiziksel boyut verisi bulunamadı', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Fiziksel Boyutlar (mm) - VERİ YOK')
        else:
            plt.text(0.5, 0.5, 'Fiziksel boyut verisi bulunamadı', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Fiziksel Boyutlar (mm) - VERİ YOK')
        
        # 4. Piksel aralığı X ve Y karşılaştırması
        plt.subplot(2, 2, 4)
        if has_spacing_x and has_spacing_y:
            valid_spacing_data = self.metadata_df.dropna(subset=['pixel_spacing_x', 'pixel_spacing_y'])
            if not valid_spacing_data.empty:
                sample_data = valid_spacing_data.sample(min(1000, len(valid_spacing_data)))
                sns.scatterplot(x='pixel_spacing_x', y='pixel_spacing_y', data=sample_data)
                plt.title('Piksel Aralığı X vs Y')
                plt.xlabel('Piksel Aralığı X (mm)')
                plt.ylabel('Piksel Aralığı Y (mm)')
            else:
                plt.text(0.5, 0.5, 'Yeterli piksel aralığı verisi bulunamadı', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Piksel Aralığı X vs Y - VERİ YOK')
        else:
            plt.text(0.5, 0.5, 'Piksel aralığı verisi bulunamadı', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Piksel Aralığı X vs Y - VERİ YOK')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
    
    def visualize_quality_distribution(self, save_path=None):
        """Kalite dağılımını görselleştirir"""
        plt.figure(figsize=(15, 10))
        
        # 1. Çözünürlük kalitesi dağılımı
        plt.subplot(2, 2, 1)
        res_quality = self.metadata_df['resolution_quality'].value_counts()
        if not res_quality.empty and not all(res_quality.index == 'Bilinmiyor'):
            res_quality = res_quality[res_quality.index != 'Bilinmiyor']  # Bilinmiyor hariç
            if not res_quality.empty:
                plt.pie(res_quality, labels=res_quality.index, autopct='%1.1f%%')
                plt.title('Çözünürlük Kalitesi Dağılımı')
            else:
                plt.text(0.5, 0.5, 'Çözünürlük kalitesi verisi bulunamadı', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Çözünürlük Kalitesi Dağılımı - VERİ YOK')
        else:
            plt.text(0.5, 0.5, 'Çözünürlük kalitesi verisi bulunamadı', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Çözünürlük Kalitesi Dağılımı - VERİ YOK')
        
        # 2. Piksel aralığı kalitesi dağılımı
        plt.subplot(2, 2, 2)
        spacing_quality = self.metadata_df['spacing_quality'].value_counts()
        if not spacing_quality.empty and not all(spacing_quality.index == 'Bilinmiyor'):
            spacing_quality = spacing_quality[spacing_quality.index != 'Bilinmiyor']  # Bilinmiyor hariç
            if not spacing_quality.empty:
                plt.pie(spacing_quality, labels=spacing_quality.index, autopct='%1.1f%%')
                plt.title('Piksel Aralığı Kalitesi Dağılımı')
            else:
                plt.text(0.5, 0.5, 'Piksel aralığı kalitesi verisi bulunamadı', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Piksel Aralığı Kalitesi Dağılımı - VERİ YOK')
        else:
            plt.text(0.5, 0.5, 'Piksel aralığı kalitesi verisi bulunamadı', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Piksel Aralığı Kalitesi Dağılımı - VERİ YOK')
        
        # 3. Genel kalite dağılımı
        plt.subplot(2, 2, 3)
        overall_quality = self.metadata_df['overall_quality'].value_counts()
        if not overall_quality.empty and not all(overall_quality.index == 'Bilinmiyor'):
            overall_quality = overall_quality[overall_quality.index != 'Bilinmiyor']  # Bilinmiyor hariç
            if not overall_quality.empty:
                try:
                    overall_quality = overall_quality.sort_index()
                    sns.barplot(x=overall_quality.index, y=overall_quality.values)
                    plt.title('Genel Kalite Dağılımı')
                    plt.xticks(rotation=45)
                except Exception as e:
                    plt.text(0.5, 0.5, f'Genel kalite görselleştirme hatası: {str(e)}', 
                            ha='center', va='center', transform=plt.gca().transAxes)
                    plt.title('Genel Kalite Dağılımı - HATA')
            else:
                plt.text(0.5, 0.5, 'Genel kalite verisi bulunamadı', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Genel Kalite Dağılımı - VERİ YOK')
        else:
            plt.text(0.5, 0.5, 'Genel kalite verisi bulunamadı', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Genel Kalite Dağılımı - VERİ YOK')
        
        # 4. Kalite ve hastalık ilişkisi
        plt.subplot(2, 2, 4)
        
        if 'Finding Labels' in self.metadata_df.columns:
            # 'Finding Labels' sütunu var
            
            # Önce kalite verisi olup olmadığını kontrol et
            has_quality_data = not self.metadata_df[self.metadata_df['overall_quality'] != 'Bilinmiyor'].empty
            
            if has_quality_data:
                try:
                    # En yaygın 5 hastalık için kalite dağılımı
                    top_diseases = self.metadata_df['Finding Labels'].value_counts().head(5).index
                    
                    quality_counts = {}
                    for disease in top_diseases:
                        disease_quality = self.metadata_df[self.metadata_df['Finding Labels'] == disease]['overall_quality'].value_counts()
                        # Bilinmiyor kategorisini çıkar
                        disease_quality = disease_quality[disease_quality.index != 'Bilinmiyor']
                        if not disease_quality.empty:
                            quality_counts[disease] = disease_quality
                    
                    if quality_counts:
                        # Veriyi çizim için hazırla
                        diseases = []
                        qualities = []
                        counts = []
                        
                        for disease, quality_dist in quality_counts.items():
                            for quality, count in quality_dist.items():
                                diseases.append(disease)
                                qualities.append(quality)
                                counts.append(count)
                        
                        # DataFrame oluştur
                        plot_df = pd.DataFrame({
                            'Disease': diseases,
                            'Quality': qualities,
                            'Count': counts
                        })
                        
                        if not plot_df.empty and 'Disease' in plot_df.columns and 'Quality' in plot_df.columns and 'Count' in plot_df.columns:
                            # Stacked bar plot
                            plot_pivot = plot_df.pivot(index='Disease', columns='Quality', values='Count').fillna(0)
                            if not plot_pivot.empty:
                                plot_pivot.plot(kind='bar', stacked=True, ax=plt.gca())
                                plt.title('Hastalıklara Göre Görüntü Kalitesi')
                                plt.xlabel('Hastalık')
                                plt.ylabel('Görüntü Sayısı')
                                plt.xticks(rotation=45)
                            else:
                                plt.text(0.5, 0.5, 'Pivot tablo boş', 
                                        ha='center', va='center', transform=plt.gca().transAxes)
                                plt.title('Hastalıklara Göre Görüntü Kalitesi - VERİ YOK')
                        else:
                            plt.text(0.5, 0.5, 'Hastalık/kalite verileri yetersiz', 
                                    ha='center', va='center', transform=plt.gca().transAxes)
                            plt.title('Hastalıklara Göre Görüntü Kalitesi - VERİ YOK')
                    else:
                        plt.text(0.5, 0.5, 'Hastalıklara göre kalite verisi bulunamadı', 
                                ha='center', va='center', transform=plt.gca().transAxes)
                        plt.title('Hastalıklara Göre Görüntü Kalitesi - VERİ YOK')
                except Exception as e:
                    plt.text(0.5, 0.5, f'Hastalık/kalite görselleştirme hatası: {str(e)}', 
                            ha='center', va='center', transform=plt.gca().transAxes)
                    plt.title('Hastalıklara Göre Görüntü Kalitesi - HATA')
            else:
                plt.text(0.5, 0.5, 'Kalite verisi bulunamadı', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Hastalıklara Göre Görüntü Kalitesi - VERİ YOK')
        else:
            # 'Finding Labels' sütunu yok
            plt.text(0.5, 0.5, 'Finding Labels sütunu bulunamadı', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Hastalıklara Göre Görüntü Kalitesi - VERİ YOK')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
    
    def detect_outliers(self, method='iqr', threshold=1.5):
        """Aykırı değerleri tespit eder
        
        Args:
            method: 'iqr' için Interquartile Range, 'zscore' için Z-score
            threshold: IQR için çarpan, Z-score için eşik değeri
            
        Returns:
            outliers_df: Aykırı değerleri içeren DataFrame
        """
        outliers = {}
        
        # Analiz edilecek sayısal sütunlar
        numeric_cols = ['image_width', 'image_height', 'image_resolution',
                        'pixel_spacing_x', 'pixel_spacing_y',
                        'physical_width_mm', 'physical_height_mm']
        
        for col in numeric_cols:
            if method == 'iqr':
                # IQR yöntemi
                Q1 = self.metadata_df[col].quantile(0.25)
                Q3 = self.metadata_df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                col_outliers = self.metadata_df[(self.metadata_df[col] < lower_bound) | 
                                               (self.metadata_df[col] > upper_bound)]
            
            elif method == 'zscore':
                # Z-score yöntemi
                mean = self.metadata_df[col].mean()
                std = self.metadata_df[col].std()
                
                if std == 0:  # Standart sapma 0 ise aykırı değer yok
                    col_outliers = self.metadata_df[0:0]  # Boş DataFrame
                else:
                    z_scores = (self.metadata_df[col] - mean) / std
                    col_outliers = self.metadata_df[abs(z_scores) > threshold]
            
            else:
                raise ValueError(f"Bilinmeyen aykırı değer tespit yöntemi: {method}")
            
            outliers[col] = col_outliers
        
        # En yüksek ve en düşük kalite görüntülerini bul
        lowest_quality = self.metadata_df[self.metadata_df['overall_quality'] == 'Çok Düşük']
        highest_quality = self.metadata_df[self.metadata_df['overall_quality'] == 'Çok Yüksek']
        
        return {
            'numeric_outliers': outliers,
            'lowest_quality': lowest_quality,
            'highest_quality': highest_quality
        }
    
    def analyze_image_data_directly(self, data_dir, image_indices=None, sample_size=10):
        """Görüntü verilerini doğrudan analiz eder
        
        Args:
            data_dir: Görüntü dosyalarının bulunduğu dizin
            image_indices: Analiz edilecek görüntü indeksleri (None ise rastgele örnekleme)
            sample_size: Örneklem boyutu (image_indices None ise kullanılır)
            
        Returns:
            image_metrics: Görüntü metriklerini içeren DataFrame
        """
        # Analiz edilecek görüntüleri seç
        if image_indices is None:
            # Rastgele örnek seç
            image_indices = self.metadata_df['Image Index'].sample(sample_size).tolist()
        
        # Sonuç DataFrame'i hazırla
        metrics_data = []
        
        for img_idx in image_indices:
            # Görüntü dosyasını bul
            img_path = None
            
            # Görüntü klasörlerini ara
            image_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f)) and f.startswith('images_')]
            
            for folder in image_folders:
                potential_path = os.path.join(data_dir, folder, img_idx)
                if os.path.exists(potential_path):
                    img_path = potential_path
                    break
            
            if img_path is None:
                print(f"Uyarı: {img_idx} görüntüsü bulunamadı.")
                continue
            
            try:
                # Görüntüyü yükle
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Uyarı: {img_idx} görüntüsü yüklenemedi.")
                    continue
                
                # Görüntü metriklerini hesapla
                height, width = img.shape
                mean_intensity = np.mean(img)
                std_intensity = np.std(img)
                min_intensity = np.min(img)
                max_intensity = np.max(img)
                
                # Kontrast ölçümü
                contrast = max_intensity - min_intensity
                
                # SNR (Signal-to-Noise Ratio) tahmini
                # Sinyal ortalama yoğunluk, gürültü standart sapma olarak kabul edilir
                snr = mean_intensity / std_intensity if std_intensity > 0 else 0
                
                # Görüntü histogramı
                hist = cv2.calcHist([img], [0], None, [256], [0, 256])
                hist = hist.flatten() / hist.sum()  # Normalize
                
                # Entropi (bilgi içeriği)
                entropy = -np.sum(hist * np.log2(hist + 1e-7))
                
                # Lezyon kontrastı (örnek bölge - şu an basit bir tahmin)
                # Gerçek uygulamada, lezyon bölgesini tespit etmek için 
                # segmentasyon algoritmaları kullanılabilir
                center_x, center_y = width // 2, height // 2
                center_region = img[center_y-50:center_y+50, center_x-50:center_x+50]
                if center_region.size > 0:
                    center_mean = np.mean(center_region)
                    background_mean = np.mean(img) - (np.sum(center_region) / img.size)
                    lesion_contrast = abs(center_mean - background_mean)
                else:
                    lesion_contrast = 0
                
                # Sonuçları ekle
                metrics_data.append({
                    'Image Index': img_idx,
                    'measured_width': width,
                    'measured_height': height,
                    'mean_intensity': mean_intensity,
                    'std_intensity': std_intensity,
                    'contrast': contrast,
                    'snr': snr,
                    'entropy': entropy,
                    'lesion_contrast': lesion_contrast
                })
                
            except Exception as e:
                print(f"Hata: {img_idx} görüntüsü işlenirken bir hata oluştu: {str(e)}")
        
        # DataFrame oluştur
        metrics_df = pd.DataFrame(metrics_data)
        
        # Metadata ile birleştir
        if not metrics_df.empty:
            metrics_df = pd.merge(
                metrics_df,
                self.metadata_df[['Image Index', 'image_width', 'image_height', 'overall_quality', 'Finding Labels']],
                on='Image Index',
                how='left'
            )
            
            # Beklenen ve ölçülen boyutlar arasındaki farkları hesapla
            metrics_df['width_diff'] = metrics_df['measured_width'] - metrics_df['image_width']
            metrics_df['height_diff'] = metrics_df['measured_height'] - metrics_df['image_height']
        
        return metrics_df
    
    def get_summary_statistics(self):
        """Özet istatistikler döndürür"""
        dim_stats = self.analyze_image_dimensions()
        spacing_stats = self.analyze_pixel_spacing()
        quality_dist = self.analyze_quality_distribution()
        
        return {
            'dimensions': dim_stats,
            'pixel_spacing': spacing_stats,
            'quality_distribution': quality_dist,
            'missing_dimension_info': self.metadata_df['image_width'].isna().sum(),
            'missing_spacing_info': self.metadata_df['pixel_spacing_x'].isna().sum(),
            'complete_quality_records': self.metadata_df[self.metadata_df['overall_quality'] != 'Bilinmiyor'].shape[0]
        }