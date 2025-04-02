import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

class MetadataProcessor:
    def __init__(self, metadata_path):
        """
        Metadata işleme sınıfı
        
        Args:
            metadata_path: Metadata CSV dosyası yolu
        """
        self.metadata_path = metadata_path
        self.metadata_df = None
        self.load_metadata()
        
    def load_metadata(self):
        """Metadata dosyasını yükler"""
        self.metadata_df = pd.read_csv(self.metadata_path)
        return self.metadata_df
    
    def analyze_disease_distribution(self):
        """Hastalık dağılımını analiz eder"""
        # Finding Labels sütunundaki değerleri ayırın (birden fazla hastalık olabilir)
        all_findings = []
        for labels in self.metadata_df['Finding Labels']:
            findings = labels.split('|')
            all_findings.extend(findings)
        
        # Hastalık sayımlarını yapın
        finding_counts = Counter(all_findings)
        return finding_counts
    
    def analyze_demographics(self):
        """Yaş ve cinsiyet dağılımını analiz eder"""
        # Yaş istatistikleri
        age_stats = self.metadata_df['Patient Age'].describe()
        
        # Cinsiyet dağılımı
        gender_counts = self.metadata_df['Patient Gender'].value_counts()
        
        return {
            'age_stats': age_stats,
            'gender_stats': gender_counts
        }
    
    def analyze_view_positions(self):
        """View Position dağılımını analiz eder"""
        return self.metadata_df['View Position'].value_counts()
    
    def analyze_multilabel_distribution(self):
        """Çoklu etiket dağılımını analiz eder"""
        # Her bir görüntüdeki hastalık sayısını hesaplayın
        self.metadata_df['num_findings'] = self.metadata_df['Finding Labels'].apply(lambda x: len(x.split('|')))
        
        # Hastalık sayısı dağılımı
        disease_counts_per_image = self.metadata_df['num_findings'].value_counts()
        
        # En yaygın hastalık kombinasyonları
        disease_combinations = self.metadata_df['Finding Labels'].value_counts().head(15)
        
        return disease_counts_per_image, disease_combinations
    
    def get_summary_statistics(self):
        """Özet istatistikler döndürür"""
        disease_counts = self.analyze_disease_distribution()
        demographic_stats = self.analyze_demographics()
        
        return {
            'total_images': len(self.metadata_df),
            'total_patients': self.metadata_df['Patient ID'].nunique(),
            'disease_counts': disease_counts,
            'demographics': demographic_stats
        }

    def visualize_disease_distribution(self, save_path=None):
        """Hastalık dağılımını görselleştirir"""
        finding_counts = self.analyze_disease_distribution()
        
        plt.figure(figsize=(14, 8))
        labels, values = zip(*finding_counts.most_common())
        plt.bar(labels, values)
        plt.xticks(rotation=45, ha='right')
        plt.title('Disease Distribution')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()