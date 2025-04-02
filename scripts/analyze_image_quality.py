import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from glob import glob
from tqdm import tqdm
import seaborn as sns

def numpy_to_python(obj):
    """
    NumPy veri tiplerini standart Python veri tiplerine dönüştürür.
    Bu fonksiyon JSON serileştirme hatalarını önlemek için kullanılır.
    
    Args:
        obj: Dönüştürülecek nesne
        
    Returns:
        Standart Python veri tiplerine dönüştürülmüş nesne
    """
    if isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(numpy_to_python(item) for item in obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return numpy_to_python(obj.tolist())
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, (np.void)):
        return None
    else:
        return obj

def find_image_files(data_dir):
    """Veri dizinindeki tüm görüntü dosyalarını bulur"""
    # Görüntü dosyalarını bul
    image_patterns = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    all_images = []
    
    # Doğrudan kök dizindeki görüntüleri ara
    for pattern in image_patterns:
        all_images.extend(glob(os.path.join(data_dir, pattern)))
    
    # Alt dizinlerdeki görüntüleri ara
    for pattern in image_patterns:
        all_images.extend(glob(os.path.join(data_dir, '**', pattern), recursive=True))
    
    print(f"Toplam {len(all_images)} görüntü dosyası bulundu.")
    return all_images

def analyze_image_quality(image_path):
    """Bir görüntünün kalite metriklerini hesaplar"""
    try:
        # Görüntüyü yükle
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        # Temel metrikler
        height, width = img.shape
        file_size = os.path.getsize(image_path) / 1024  # KB cinsinden
        mean_intensity = np.mean(img)
        std_intensity = np.std(img)
        min_intensity = np.min(img)
        max_intensity = np.max(img)
        
        # Kontrast hesapla
        contrast = max_intensity - min_intensity
        
        # SNR (Signal-to-Noise Ratio) tahmini
        snr = mean_intensity / std_intensity if std_intensity > 0 else 0
        
        # Görüntü histogramı
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()  # Normalize
        
        # Entropi (bilgi içeriği)
        non_zero_hist = hist[hist > 0]
        entropy = -np.sum(non_zero_hist * np.log2(non_zero_hist))
        
        # Laplace varyansı (kenar detayı ölçüsü)
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        laplacian_var = laplacian.var()
        
        # Sharpness metrikleri (Sobel kenar tespiti)
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobelx**2 + sobely**2)
        sharpness = np.mean(sobel_mag)
        
        # İsim ve uzantıyı ayır
        basename = os.path.basename(image_path)
        filename, ext = os.path.splitext(basename)
        
        return {
            'filename': basename,
            'width': width,
            'height': height,
            'resolution': width * height,
            'aspect_ratio': width / height,
            'file_size_kb': file_size,
            'bits_per_pixel': (file_size * 8 * 1024) / (width * height),
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'min_intensity': min_intensity,
            'max_intensity': max_intensity,
            'contrast': contrast,
            'snr': snr,
            'entropy': entropy,
            'sharpness': sharpness,
            'laplacian_variance': laplacian_var,
            'file_extension': ext
        }
    except Exception as e:
        print(f"Hata: {image_path} dosyası analiz edilirken bir sorun oluştu: {str(e)}")
        return None

def batch_analyze_images(image_paths, sample_size=None):
    """Birden fazla görüntüyü analiz eder"""
    if sample_size and sample_size < len(image_paths):
        # Rastgele örnekleme yap
        image_paths = np.random.choice(image_paths, sample_size, replace=False)
    
    results = []
    for img_path in tqdm(image_paths, desc="Görüntüler analiz ediliyor"):
        metrics = analyze_image_quality(img_path)
        if metrics:
            results.append(metrics)
    
    # DataFrame oluştur
    if results:
        return pd.DataFrame(results)
    return pd.DataFrame()

def analyze_quality_distribution(df):
    """Kalite metriklerinin dağılımını analiz eder"""
    # Çözünürlüğe göre kalite sınıflandırması
    conditions = [
        (df['resolution'] >= 6000000),  # 6MP ve üzeri
        (df['resolution'] >= 3000000) & (df['resolution'] < 6000000),  # 3-6MP
        (df['resolution'] >= 1000000) & (df['resolution'] < 3000000),  # 1-3MP
        (df['resolution'] < 1000000)  # 1MP altı
    ]
    choices = ['Yüksek', 'İyi', 'Orta', 'Düşük']
    df['resolution_quality'] = np.select(conditions, choices, default='Bilinmiyor')
    
    # SNR'ye göre kalite sınıflandırması
    conditions = [
        (df['snr'] >= 5),  # Yüksek SNR
        (df['snr'] >= 3) & (df['snr'] < 5),  # Orta SNR
        (df['snr'] < 3)  # Düşük SNR
    ]
    choices = ['Yüksek', 'Orta', 'Düşük']
    df['snr_quality'] = np.select(conditions, choices, default='Bilinmiyor')
    
    # Kontrast'a göre kalite sınıflandırması
    conditions = [
        (df['contrast'] >= 200),  # Yüksek kontrast
        (df['contrast'] >= 100) & (df['contrast'] < 200),  # Orta kontrast
        (df['contrast'] < 100)  # Düşük kontrast
    ]
    choices = ['Yüksek', 'Orta', 'Düşük']
    df['contrast_quality'] = np.select(conditions, choices, default='Bilinmiyor')
    
    # Keskinliğe göre kalite sınıflandırması
    # Sharpness değerleri veri setine göre değişebilir, bu yüzden persentil kullanıyoruz
    sharp_high = df['sharpness'].quantile(0.75)
    sharp_low = df['sharpness'].quantile(0.25)
    
    conditions = [
        (df['sharpness'] >= sharp_high),  # En keskin %25
        (df['sharpness'] >= sharp_low) & (df['sharpness'] < sharp_high),  # Orta keskin %50
        (df['sharpness'] < sharp_low)  # En az keskin %25
    ]
    choices = ['Yüksek', 'Orta', 'Düşük']
    df['sharpness_quality'] = np.select(conditions, choices, default='Bilinmiyor')
    
    # Genel kalite puanı (basit bir yöntem)
    # Her bir kalite metriğini puanla (Yüksek=3, Orta=2, Düşük=1, Bilinmiyor=0)
    quality_map = {'Yüksek': 3, 'İyi': 2.5, 'Orta': 2, 'Düşük': 1, 'Bilinmiyor': 0}
    
    df['resolution_score'] = df['resolution_quality'].map(quality_map)
    df['snr_score'] = df['snr_quality'].map(quality_map)
    df['contrast_score'] = df['contrast_quality'].map(quality_map)
    df['sharpness_score'] = df['sharpness_quality'].map(quality_map)
    
    # Ağırlıklı toplam (çözünürlük daha önemli)
    df['quality_score'] = (df['resolution_score'] * 0.4 + 
                          df['snr_score'] * 0.2 + 
                          df['contrast_score'] * 0.2 + 
                          df['sharpness_score'] * 0.2)
    
    # Puana göre genel kalite sınıflandırması
    conditions = [
        (df['quality_score'] >= 2.5),  # Yüksek
        (df['quality_score'] >= 2.0) & (df['quality_score'] < 2.5),  # İyi
        (df['quality_score'] >= 1.5) & (df['quality_score'] < 2.0),  # Orta
        (df['quality_score'] >= 1.0) & (df['quality_score'] < 1.5),  # Düşük
        (df['quality_score'] < 1.0)  # Çok Düşük
    ]
    choices = ['Çok Yüksek', 'Yüksek', 'Orta', 'Düşük', 'Çok Düşük']
    df['overall_quality'] = np.select(conditions, choices, default='Bilinmiyor')
    
    return df

def find_outliers(df, columns, method='iqr', threshold=1.5):
    """Metrik değerlerine göre aykırı değerleri bulur"""
    outliers = {}
    
    for col in columns:
        if method == 'iqr':
            # IQR yöntemi
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        elif method == 'zscore':
            # Z-score yöntemi
            mean = df[col].mean()
            std = df[col].std()
            
            z_scores = (df[col] - mean) / std
            outliers[col] = df[abs(z_scores) > threshold]
    
    return outliers

def visualize_quality_metrics(df, output_dir):
    """Kalite metriklerini görselleştirir"""
    # Çıktı dizinini oluştur
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Çözünürlük dağılımı
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    sns.histplot(df['resolution'] / 1000000, bins=20)
    plt.title('Görüntü Çözünürlüğü Dağılımı')
    plt.xlabel('Çözünürlük (Megapiksel)')
    plt.ylabel('Görüntü Sayısı')
    
    plt.subplot(2, 2, 2)
    sns.histplot(df['width'], bins=20, color='blue', label='Genişlik')
    sns.histplot(df['height'], bins=20, color='red', label='Yükseklik')
    plt.title('Görüntü Boyutları Dağılımı')
    plt.xlabel('Piksel Sayısı')
    plt.ylabel('Görüntü Sayısı')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=df, x='width', y='height')
    plt.title('Genişlik vs Yükseklik')
    plt.xlabel('Genişlik (piksel)')
    plt.ylabel('Yükseklik (piksel)')
    
    plt.subplot(2, 2, 4)
    res_quality = df['resolution_quality'].value_counts()
    plt.pie(res_quality, labels=res_quality.index, autopct='%1.1f%%')
    plt.title('Çözünürlük Kalitesi Dağılımı')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'resolution_analysis.png'), dpi=300)
    plt.close()
    
    # 2. Kontrast ve SNR analizi
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    sns.histplot(df['contrast'], bins=20)
    plt.title('Kontrast Dağılımı')
    plt.xlabel('Kontrast Değeri')
    plt.ylabel('Görüntü Sayısı')
    
    plt.subplot(2, 2, 2)
    sns.histplot(df['snr'], bins=20)
    plt.title('SNR Dağılımı')
    plt.xlabel('SNR Değeri')
    plt.ylabel('Görüntü Sayısı')
    
    plt.subplot(2, 2, 3)
    contrast_quality = df['contrast_quality'].value_counts()
    plt.pie(contrast_quality, labels=contrast_quality.index, autopct='%1.1f%%')
    plt.title('Kontrast Kalitesi Dağılımı')
    
    plt.subplot(2, 2, 4)
    snr_quality = df['snr_quality'].value_counts()
    plt.pie(snr_quality, labels=snr_quality.index, autopct='%1.1f%%')
    plt.title('SNR Kalitesi Dağılımı')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'contrast_snr_analysis.png'), dpi=300)
    plt.close()
    
    # 3. Keskinlik ve Detay analizi
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    sns.histplot(df['sharpness'], bins=20)
    plt.title('Keskinlik Dağılımı')
    plt.xlabel('Keskinlik Değeri')
    plt.ylabel('Görüntü Sayısı')
    
    plt.subplot(2, 2, 2)
    sns.histplot(df['laplacian_variance'], bins=20)
    plt.title('Laplacian Varyansı Dağılımı')
    plt.xlabel('Laplacian Varyansı')
    plt.ylabel('Görüntü Sayısı')
    
    plt.subplot(2, 2, 3)
    sharpness_quality = df['sharpness_quality'].value_counts()
    plt.pie(sharpness_quality, labels=sharpness_quality.index, autopct='%1.1f%%')
    plt.title('Keskinlik Kalitesi Dağılımı')
    
    plt.subplot(2, 2, 4)
    sns.histplot(df['entropy'], bins=20)
    plt.title('Entropi Dağılımı')
    plt.xlabel('Entropi Değeri')
    plt.ylabel('Görüntü Sayısı')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sharpness_detail_analysis.png'), dpi=300)
    plt.close()
    
    # 4. Genel kalite analizi
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    sns.histplot(df['quality_score'], bins=20)
    plt.title('Kalite Puanı Dağılımı')
    plt.xlabel('Kalite Puanı')
    plt.ylabel('Görüntü Sayısı')
    
    plt.subplot(2, 2, 2)
    overall_quality = df['overall_quality'].value_counts()
    plt.pie(overall_quality, labels=overall_quality.index, autopct='%1.1f%%')
    plt.title('Genel Kalite Dağılımı')
    
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=df, x='quality_score', y='file_size_kb')
    plt.title('Kalite Puanı vs Dosya Boyutu')
    plt.xlabel('Kalite Puanı')
    plt.ylabel('Dosya Boyutu (KB)')
    
    plt.subplot(2, 2, 4)
    sns.countplot(x='file_extension', data=df)
    plt.title('Dosya Formatı Dağılımı')
    plt.xlabel('Dosya Formatı')
    plt.ylabel('Görüntü Sayısı')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_quality_analysis.png'), dpi=300)
    plt.close()
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Doğrudan görüntü kalitesi analizi')
    parser.add_argument('--data_dir', type=str, required=True, help='Veri seti dizini yolu')
    parser.add_argument('--output_dir', type=str, default='./quality_analysis', help='Analiz sonuçlarının kaydedileceği dizin')
    parser.add_argument('--sample_size', type=int, default=None, help='Analiz edilecek görüntü sayısı (None=tümü)')
    
    args = parser.parse_args()
    
    # Görüntü dosyalarını bul
    all_images = find_image_files(args.data_dir)
    
    if not all_images:
        print("Hiç görüntü dosyası bulunamadı!")
        return
    
    # Görüntülerin kalitesini analiz et
    print(f"Görüntüler analiz ediliyor (örneklem boyutu: {args.sample_size if args.sample_size else 'tümü'})...")
    results_df = batch_analyze_images(all_images, args.sample_size)
    
    if results_df.empty:
        print("Görüntü analizi sonuçları boş!")
        return
    
    # Kalite dağılımını analiz et
    print("Kalite dağılımı analiz ediliyor...")
    results_df = analyze_quality_distribution(results_df)
    
    # Aykırı değerleri bul
    print("Aykırı değerler tespit ediliyor...")
    metrics_to_check = ['resolution', 'contrast', 'snr', 'sharpness', 'entropy', 'laplacian_variance']
    outliers = find_outliers(results_df, metrics_to_check)
    
    # En düşük ve en yüksek kaliteli görüntüleri bul
    lowest_quality = results_df[results_df['overall_quality'] == 'Çok Düşük'].sort_values('quality_score')
    highest_quality = results_df[results_df['overall_quality'] == 'Çok Yüksek'].sort_values('quality_score', ascending=False)
    
    # Görselleştirmeleri oluştur
    print("Görselleştirmeler oluşturuluyor...")
    figures_dir = os.path.join(args.output_dir, "figures")
    figures_dir = r"C:\\Users\\Asus F15\\Desktop\\ChestXpert\\data\\processed\\quality_analysis\\figures"
    os.makedirs(figures_dir, exist_ok=True)
    visualize_quality_metrics(results_df, figures_dir)
    
    # Sonuçları kaydet
    print("Sonuçlar kaydediliyor...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Tüm kalite sonuçlarını kaydet
    results_df.to_csv(os.path.join(args.output_dir, "image_quality_results.csv"), index=False)
    
    # En düşük kaliteli görüntüleri kaydet
    if not lowest_quality.empty:
        lowest_quality.head(50).to_csv(os.path.join(args.output_dir, "lowest_quality_images.csv"), index=False)
    
    # En yüksek kaliteli görüntüleri kaydet
    if not highest_quality.empty:
        highest_quality.head(50).to_csv(os.path.join(args.output_dir, "highest_quality_images.csv"), index=False)
    
    # Özet istatistikler
    summary = {
        'analyzed_images': len(results_df),
        'dimensions': {
            'width_mean': results_df['width'].mean(),
            'width_std': results_df['width'].std(),
            'height_mean': results_df['height'].mean(),
            'height_std': results_df['height'].std(),
            'resolution_mean': results_df['resolution'].mean(),
            'resolution_min': results_df['resolution'].min(),
            'resolution_max': results_df['resolution'].max(),
            'aspect_ratio_mean': results_df['aspect_ratio'].mean()
        },
        'file_stats': {
            'file_size_mean_kb': results_df['file_size_kb'].mean(),
            'bits_per_pixel_mean': results_df['bits_per_pixel'].mean()
        },
        'quality_metrics': {
            'contrast_mean': results_df['contrast'].mean(),
            'snr_mean': results_df['snr'].mean(),
            'sharpness_mean': results_df['sharpness'].mean(),
            'entropy_mean': results_df['entropy'].mean(),
            'laplacian_var_mean': results_df['laplacian_variance'].mean()
        },
        'quality_distribution': {
            'resolution_quality': results_df['resolution_quality'].value_counts().to_dict(),
            'contrast_quality': results_df['contrast_quality'].value_counts().to_dict(),
            'snr_quality': results_df['snr_quality'].value_counts().to_dict(),
            'sharpness_quality': results_df['sharpness_quality'].value_counts().to_dict(),
            'overall_quality': results_df['overall_quality'].value_counts().to_dict()
        },
        'outliers': {metric: len(df) for metric, df in outliers.items()},
        'file_formats': results_df['file_extension'].value_counts().to_dict()
    }
    summary_dir = r"C:\Users\Asus F15\Desktop\ChestXpert\data\processed\quality_analysis"
    # Özeti JSON olarak kaydet
    with open(summary_dir, 'w') as f:
        json.dump(numpy_to_python(summary), f, indent=4)
    
    print("\n=== ÖZET RAPOR ===")
    print(f"Analiz edilen görüntü sayısı: {len(results_df)}")
    print(f"Ortalama görüntü boyutları: {summary['dimensions']['width_mean']:.1f} x {summary['dimensions']['height_mean']:.1f} piksel")
    print(f"Ortalama görüntü çözünürlüğü: {summary['dimensions']['resolution_mean']/1000000:.2f} megapiksel")
    print(f"Ortalama dosya boyutu: {summary['file_stats']['file_size_mean_kb']:.1f} KB")
    print(f"Ortalama SNR: {summary['quality_metrics']['snr_mean']:.2f}")
    
    try:
        good_quality_pct = (summary['quality_distribution']['overall_quality'].get('Çok Yüksek', 0) + 
                           summary['quality_distribution']['overall_quality'].get('Yüksek', 0)) / len(results_df) * 100
        print(f"Yüksek kaliteli görüntü oranı: {good_quality_pct:.1f}%")
    except:
        pass
    
    print(f"Analiz çıktıları kaydedildi: {args.output_dir}")

if __name__ == "__main__":
    main()