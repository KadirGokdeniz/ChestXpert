import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import sys
import json

# Proje kök dizinini Python yoluna ekleyin
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Gerekli modülleri içe aktarın
# Not: Eğer bu modülleri henüz oluşturmadıysanız, oluşturmanız gerekecektir
try:
    from utils.image_utils import analyze_image_dimensions, sample_images_from_dataset, load_image
except ImportError:
    print("Uyarı: utils.image_utils modülü bulunamadı")

def load_metadata(metadata_path):
    """Metadata dosyasını yükle"""
    try:
        return pd.read_csv(metadata_path)
    except Exception as e:
        print(f"Metadata yükleme hatası: {e}")
        return None

def load_bbox_data(bbox_path):
    """Bounding box verilerini yükle"""
    try:
        return pd.read_csv(bbox_path)
    except Exception as e:
        print(f"Bbox verisi yükleme hatası: {e}")
        return None

def analyze_disease_distribution(metadata_df):
    """Hastalık dağılımını analiz et"""
    if metadata_df is None:
        return {}
        
    all_findings = []
    for labels in metadata_df['Finding Labels']:
        findings = labels.split('|')
        all_findings.extend(findings)
    
    from collections import Counter
    finding_counts = Counter(all_findings)
    return finding_counts

def main():
    parser = argparse.ArgumentParser(description='ChestXpert veri seti analiz betiği')
    parser.add_argument('--data_dir', type=str, required=True, help='Veri seti ana dizini')
    parser.add_argument('--output_dir', type=str, default='data\\processed\\stats', help='Çıktı dizini')
    
    args = parser.parse_args()
    
    # Tam yolları oluştur
    data_dir = args.data_dir
    output_dir = os.path.join(project_root, args.output_dir)
    
    # Çıktı dizinini oluştur
    os.makedirs(output_dir, exist_ok=True)
    
    # Dosya yollarını tanımla
    metadata_path = os.path.join(data_dir,"data","data","archive (22)","Data_Entry_2017.csv")
    bbox_path = os.path.join(data_dir, "data","data","archive (22)","BBox_List_2017.csv")
    image_folders = [f"images_{str(i).zfill(3)}" for i in range(1, 13)]  # images_001 - images_012
    
    
    print("Veri seti analizi başlatılıyor...")
    
    # Metadata analizi
    print("\n1. Metadata analizi yapılıyor...")
    metadata_df = load_metadata(metadata_path)
    if metadata_df is not None:
        total_images = len(metadata_df)
        total_patients = metadata_df['Patient ID'].nunique() if 'Patient ID' in metadata_df.columns else 0
        disease_counts = analyze_disease_distribution(metadata_df)
        
        print(f"Toplam görüntü sayısı: {total_images}")
        print(f"Toplam hasta sayısı: {total_patients}")
        print("En yaygın 5 hastalık:")
        for disease, count in sorted(disease_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  - {disease}: {count} ({count/total_images*100:.2f}%)")
    
    # Bounding box analizi
    print("\n2. Bounding box analizi yapılıyor...")
    bbox_df = load_bbox_data(bbox_path)
    if bbox_df is not None:
        total_bbox_records = len(bbox_df)
        total_images_with_bbox = bbox_df['Image Index'].nunique() if 'Image Index' in bbox_df.columns else 0
        
        print(f"Toplam bbox kaydı: {total_bbox_records}")
        print(f"Bbox verisi olan görüntü sayısı: {total_images_with_bbox}")
        
        if metadata_df is not None:
            bbox_coverage = total_images_with_bbox / total_images * 100
            print(f"Bbox kapsama oranı: {bbox_coverage:.2f}%")
    
    print("\n3. Görüntü boyutları analizi yapılıyor...")
    try:
        for i in range (12):
            image_dir = os.path.join(data_dir,"data","data","archive (22)",image_folders[i], "images")
            image_stats = analyze_image_dimensions(data_dir, image_dir)
            
            print("Görüntü boyut istatistikleri:")
            print(f"  - Genişlik: {image_stats['width_stats']['mean']:.1f} ± {image_stats['width_stats']['std']:.1f} piksel")
            print(f"  - Yükseklik: {image_stats['height_stats']['mean']:.1f} ± {image_stats['height_stats']['std']:.1f} piksel")
    except Exception as e:
        print(f"Görüntü analizi hatası: {e}")
        # Hata durumunda varsayılan değerler tanımla
        image_stats = {
            "width_stats": {"min": 0, "max": 0, "mean": 0, "std": 0},
            "height_stats": {"min": 0, "max": 0, "mean": 0, "std": 0},
            "error": str(e)
        }

    # Bu satırları try-except bloğunun dışına taşıyın
    try:
        # Görüntü istatistiklerini kaydet
        with open(os.path.join(output_dir, 'image_stats.json'), 'w') as f:
            json.dump(image_stats, f, indent=4)
        print(f"İstatistikler kaydedildi: {os.path.join(output_dir, 'image_stats.json')}")
    except Exception as e:
        print(f"İstatistikleri kaydetme hatası: {e}")

if __name__ == "__main__":
    main()