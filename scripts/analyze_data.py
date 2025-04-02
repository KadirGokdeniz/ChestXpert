import os
import argparse
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt

# Proje kök dizinini Python yoluna ekle
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Modülleri içe aktar
from data.metadata.metadata_processor import MetadataProcessor
from data.bbox.bbox_processor import BBoxProcessor
from utils.visualization import visualize_sample_images, check_image_organization

def main():
    parser = argparse.ArgumentParser(description='ChestXpert veri seti analizi')
    parser.add_argument('--data_dir', type=str, required=True, help='Veri seti dizini yolu')
    parser.add_argument('--output_dir', type=str, default=os.path.join(project_root, 'data', 'processed', 'analysis'), 
                        help='Analiz sonuçlarının kaydedileceği dizin')
    parser.add_argument('--visualize', action='store_true', help='Örnek görüntüleri görselleştir')
    parser.add_argument('--num_samples', type=int, default=5, help='Görselleştirilecek örnek sayısı')
    
    args = parser.parse_args()
    
    # Çıktı dizinini oluştur
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Dosya yollarını tanımla
    data_dir = args.data_dir
    metadata_path = os.path.join(data_dir, "Data_Entry_2017.csv")
    bbox_path = os.path.join(data_dir, "BBox_List_2017.csv")
    test_list_path = os.path.join(data_dir, "test_list.txt")
    train_val_list_path = os.path.join(data_dir, "train_val_list.txt")
    
    # Görüntü klasörlerini tanımla
    image_folders = [f"images_{str(i).zfill(3)}" for i in range(1, 13)]
    
    print("1. Metadata analizi yapılıyor...")
    metadata_processor = MetadataProcessor(metadata_path)
    metadata_df = metadata_processor.metadata_df
    
    # Metadata istatistiklerini oluştur
    disease_counts = metadata_processor.analyze_disease_distribution()
    demographic_stats = metadata_processor.analyze_demographics()
    view_position_stats = metadata_processor.analyze_view_positions()
    
    # Çoklu etiket analizini yap
    disease_counts_per_image, common_combinations = metadata_processor.analyze_multilabel_distribution()
    
    # Test ve eğitim listelerini yükle
    try:
        with open(test_list_path, 'r') as f:
            test_list = [line.strip() for line in f.readlines()]
        
        with open(train_val_list_path, 'r') as f:
            train_val_list = [line.strip() for line in f.readlines()]
    except Exception as e:
        print(f"Test/eğitim listelerini yükleme hatası: {e}")
        test_list = []
        train_val_list = []
    
    print("2. Bounding box analizi yapılıyor...")
    bbox_processor = BBoxProcessor(bbox_path)
    bbox_df = bbox_processor.bbox_df
    
    # Bbox istatistiklerini oluştur
    bbox_disease_counts, bbox_stats = bbox_processor.analyze_bbox_data()
    
    print("3. Veri kalitesi değerlendiriliyor...")
    # Eksik değer analizi
    metadata_null = metadata_df.isnull().sum()
    bbox_null = bbox_df.isnull().sum()
    
    # Bbox kapsama analizi
    total_images = len(metadata_df)
    images_with_bbox = bbox_df['Image Index'].nunique()
    bbox_coverage = (images_with_bbox / total_images) * 100 if total_images > 0 else 0
    
    # Eksik demografik bilgi
    missing_demographics = metadata_df[(metadata_df['Patient Age'].isnull()) | 
                             (metadata_df['Patient Gender'].isnull())].shape[0]
    
    print("4. Görüntü organizasyonu kontrol ediliyor...")
    image_counts = check_image_organization(data_dir, image_folders)
    found_images = sum(image_counts.values())
    
    # Özet rapor oluştur
    summary = {
        "metadata": {
            "total_images": total_images,
            "total_patients": metadata_df['Patient ID'].nunique(),
            "total_disease_classes": len(disease_counts),
            "top_5_diseases": {k: v for k, v in sorted(disease_counts.items(), key=lambda x: x[1], reverse=True)[:5]},
            "age_range": [float(metadata_df['Patient Age'].min()), float(metadata_df['Patient Age'].max())],
            "age_mean": float(metadata_df['Patient Age'].mean()),
            "gender_distribution": demographic_stats['gender_stats'].to_dict(),
            "view_positions": view_position_stats.to_dict(),
            "disease_per_image_distribution": disease_counts_per_image.to_dict(),
            "missing_demographics": missing_demographics,
            "missing_demographics_percent": (missing_demographics / total_images * 100) if total_images > 0 else 0
        },
        "bbox": {
            "total_bbox_records": len(bbox_df),
            "images_with_bbox": images_with_bbox,
            "bbox_coverage_percent": bbox_coverage,
            "bbox_stats": bbox_stats
        },
        "data_organization": {
            "found_images": found_images,
            "image_folder_counts": image_counts,
            "train_val_images": len(train_val_list),
            "test_images": len(test_list),
            "train_test_split_percent": [
                (len(train_val_list) / total_images * 100) if total_images > 0 else 0,
                (len(test_list) / total_images * 100) if total_images > 0 else 0
            ]
        }
    }
    
    # Özet raporu kaydet
    summary_path = os.path.join(args.output_dir, "data_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"Veri özeti kaydedildi: {summary_path}")
    
    # Görselleştirmeler
    figures_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # Metadata görselleştirmeleri
    metadata_processor.visualize_disease_distribution(
        save_path=os.path.join(figures_dir, "disease_distribution.png"))
    
    # Bbox görselleştirmeleri
    bbox_processor.visualize_bbox_distribution(
        save_path=os.path.join(figures_dir, "bbox_distribution.png"))
    
    # Örnek görüntüler
    if args.visualize:
        print("5. Örnek görüntüler görselleştiriliyor...")
        visualize_sample_images(
            metadata_df, bbox_df, data_dir, image_folders, args.num_samples,
            save_path=os.path.join(figures_dir, "sample_images.png"))
    
    print("\n=== ÖZET RAPOR ===")
    print(f"Toplam görüntü sayısı: {total_images}")
    print(f"Toplam hasta sayısı: {summary['metadata']['total_patients']}")
    print(f"Bbox verisi olan görüntü sayısı: {images_with_bbox} ({bbox_coverage:.2f}%)")
    print(f"Eğitim/Doğrulama: {len(train_val_list)} görüntü, Test: {len(test_list)} görüntü")
    print(f"Analiz çıktıları kaydedildi: {args.output_dir}")

if __name__ == "__main__":
    main()