import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def load_metadata(metadata_path):
    """Metadata CSV dosyasını yükle"""
    return pd.read_csv(metadata_path)

def analyze_disease_distribution(metadata_df):
    """Hastalık dağılımını analiz et"""
    # Finding Labels sütunundaki değerleri ayırın (birden fazla hastalık olabilir)
    all_findings = []
    for labels in metadata_df['Finding Labels']:
        findings = labels.split('|')
        all_findings.extend(findings)
    
    # Hastalık sayımlarını yapın
    finding_counts = Counter(all_findings)
    
    # Sonuçları görselleştirin
    plt.figure(figsize=(14, 8))
    labels, values = zip(*finding_counts.most_common())
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha='right')
    plt.title('Disease Distribution')
    plt.tight_layout()
    plt.show()
    
    return finding_counts

def analyze_demographics(metadata_df):
    """Yaş ve cinsiyet dağılımını analiz et"""
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(metadata_df['Patient Age'], bins=20, kde=True)
    plt.title('Age Distribution')
    
    # Cinsiyet dağılımı
    plt.subplot(1, 2, 2)
    gender_counts = metadata_df['Patient Gender'].value_counts()
    plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%')
    plt.title('Gender Distribution')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'age_stats': metadata_df['Patient Age'].describe(),
        'gender_stats': gender_counts
    }

def analyze_view_positions(metadata_df):
    view_counts = metadata_df['View Position'].value_counts()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=view_counts.index, y=view_counts.values)
    plt.title('View Position Distribution')
    plt.tight_layout()
    plt.show()
    
    return view_counts

def analyze_multilabel_distribution(metadata_df):
    """Çoklu etiket dağılımını analiz et"""
    # Hastalık bazında bbox sayısını inceleyin
    bbox_disease_counts = metadata_df['Finding Label'].value_counts()
    
    # Bbox boyutlarını inceleyin
    # x,y,w,h bilgilerini ayıklayın
    bbox_coords = []
    for bbox_str in metadata_df['Bbox [x,y,w,h]']:
        try:
            # Sütun formatına göre ayarlayın
            coords = bbox_str.replace('[', '').replace(']', '').split(',')
            x, y, w, h = map(float, coords)
            bbox_coords.append([x, y, w, h])
        except:
            continue
    
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
    
    # Bbox hastalık dağılımını görselleştirin
    plt.figure(figsize=(12, 6))
    sns.barplot(x=bbox_disease_counts.index, y=bbox_disease_counts.values)
    plt.xticks(rotation=45, ha='right')
    plt.title('Bounding Box Distribution by Disease')
    plt.tight_layout()
    plt.show()
    
    return bbox_disease_counts, bbox_stats

def generate_metadata_stats(metadata_df, output_dir=None):
    """Metadata istatistiklerini çıkar ve sonuçları kaydet"""
    stats = {}
    
    # Temel istatistikler
    stats['total_images'] = len(metadata_df)
    stats['total_patients'] = metadata_df['Patient ID'].nunique()
    
    # Hastalık dağılımı
    disease_counts = analyze_disease_distribution(metadata_df)
    stats['disease_counts'] = {k: v for k, v in disease_counts.most_common()}
    
    # Demografik bilgiler
    demographic_stats = analyze_demographics(metadata_df)
    stats['age_stats'] = demographic_stats['age_stats'].to_dict()
    stats['gender_stats'] = demographic_stats['gender_stats'].to_dict()
    
    # Çoklu etiket analizi
    disease_counts_per_image, common_combinations = analyze_multilabel_distribution(metadata_df)
    stats['disease_per_image'] = disease_counts_per_image.to_dict()
    stats['common_combinations'] = common_combinations.to_dict()
    
    # Sonuçları JSON olarak kaydet
    if output_dir:
        import json
        with open(os.path.join(output_dir, 'metadata_stats.json'), 'w') as f:
            json.dump(stats, f, indent=4)
    
    return stats
def analyze_multilabel_distribution(metadata_df):
    # Her bir görüntüdeki hastalık sayısını hesaplayın
    metadata_df['num_findings'] = metadata_df['Finding Labels'].apply(lambda x: len(x.split('|')))
    
    # Hastalık sayısı dağılımını görselleştirin
    plt.figure(figsize=(10, 6))
    sns.countplot(x='num_findings', data=metadata_df)
    plt.title('Number of Diseases per Image')
    plt.xlabel('Number of Diseases')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
    
    # En yaygın hastalık kombinasyonlarını bulun
    disease_combinations = metadata_df['Finding Labels'].value_counts().head(15)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=disease_combinations.values, y=disease_combinations.index)
    plt.title('Most Common Disease Combinations')
    plt.tight_layout()
    plt.show()
    
    return metadata_df['num_findings'].value_counts(), disease_combinations