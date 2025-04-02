import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.patches import Rectangle

def visualize_sample_images(metadata_df, bbox_df, data_dir, image_folders, num_samples=5, save_path=None):
    """
    Rastgele örnek görüntüleri görselleştirir
    
    Args:
        metadata_df: Metadata DataFrame
        bbox_df: Bounding box DataFrame
        data_dir: Veri dizini
        image_folders: Görüntü klasörleri listesi
        num_samples: Görselleştirilecek örnek sayısı
        save_path: Kaydedilecek dosya yolu (opsiyonel)
    """
    # Rastgele görüntü indekslerini seçin
    sample_indices = random.sample(range(len(metadata_df)), num_samples)
    data_dir=os.path.normpath(data_dir)
    plt.figure(figsize=(15, 4*num_samples))
    for i, idx in enumerate(sample_indices):
        # Görüntü yolunu alın
        img_index = metadata_df.iloc[idx]['Image Index']
        img_folder = None
        
        # Initialize img_folder before the loop
        img_folder = None

        # Doğru klasörü bulun (images_001 - images_012)
        for folder in image_folders:
            folder_path = os.path.join(data_dir, folder,"images")
            # For debugging
            print(f"Checking folder: {folder_path}")
            
            # Make sure img_index is just the filename, not a path
            if os.path.exists(os.path.join(folder_path, img_index)):
                img_folder = folder
                print(f"Found image in folder: {folder}")
                break

        if img_folder is None:
            print(f"Image {img_index} not found in any folder")
            continue

        img_path = os.path.normpath(os.path.join(data_dir, img_folder,"images", img_index))
        print(img_path)
        # Görüntüyü yükleyin
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Metadata bilgilerini alın
        finding = metadata_df.iloc[idx]['Finding Labels']
        patient_age = metadata_df.iloc[idx]['Patient Age']
        patient_gender = metadata_df.iloc[idx]['Patient Gender']
        view_position = metadata_df.iloc[idx]['View Position']
        
        # Varsa bbox bilgilerini alın
        bbox_rows = bbox_df[bbox_df['Image Index'] == img_index]
        
        # Görüntüyü ve bilgileri gösterin
        plt.subplot(1, num_samples, i+1)  # 1 satır, num_samples sütun
        plt.imshow(img)
        # Varsa bbox'ları çizin
        for _, bbox_row in bbox_rows.iterrows():
            try:
                print("Accepted")
                x = bbox_row['Bbox [x']
                y = bbox_row['y']  
                w = bbox_row['w']
                h = bbox_row['h]']
                
                rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2)
                plt.gca().add_patch(rect)
            except Exception as e:
                print(f"Error processing row: {e}")
                continue
        
        plt.title(f"Image: {img_index}, \nFinding: {finding}\nAge: {patient_age}, \nGender: {patient_gender}, \nView: {view_position}")
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()

def check_image_organization(data_dir, image_folders):
    """
    Görüntü dosyalarının organizasyonunu kontrol eder
    
    Args:
        data_dir: Veri dizini
        image_folders: Görüntü klasörleri listesi
        
    Returns:
        image_counts: Klasör başına görüntü sayısı sözlüğü
    """
    image_counts = {}
    
    for folder in image_folders:
        folder_path = os.path.join(data_dir, folder,"images")
        if os.path.exists(folder_path):
            image_files = [f for f in os.listdir(folder_path) 
                          if f.endswith(('.png', '.jpg', '.jpeg'))]
            image_counts[folder] = len(image_files)
    
    return image_counts