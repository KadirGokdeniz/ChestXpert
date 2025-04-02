import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as A
def get_safe_augmentations(severity='medium'):
    """Tıbbi görüntüler için güvenli augmentasyonlar"""
    if severity == 'light':
        return A.Compose([
            A.Rotate(limit=5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.95, 1.05), translate_percent=(-0.05, 0.05), rotate=0, p=0.3),
        ])
    
    elif severity == 'medium':
        return A.Compose([
            A.Rotate(limit=10),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.95, 1.05), translate_percent=(-0.05, 0.05), rotate=0, p=0.3),
            A.GaussNoise(var=(0, 5), p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        ])
    
    elif severity == 'strong':
        return A.Compose([
            A.Rotate(limit=15),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.95, 1.05), translate_percent=(-0.05, 0.05), rotate=0, p=0.3),
            A.GaussNoise(var=(0, 10), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.RandomGamma(gamma_limit=(80, 120), p=0.2),
            A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2),
        ])
def get_disease_specific_augmentations(disease_type):
    """Hastalık tipine göre özelleştirilmiş augmentasyonlar"""
    base_augs = get_safe_augmentations('medium')
    
    if disease_type == 'nodule':
        # Nodüller için daha fazla rotasyon ve ölçeklendirme
        return A.Compose([
            A.Rotate(limit=20),
            A.Affine(scale=(0.95, 1.05), translate_percent=(-0.05, 0.05), rotate=0, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.2, p=0.5),
            A.HorizontalFlip(p=0.5),
        ])
    
    elif disease_type == 'effusion':
        # Efüzyon için daha fazla kontrast değişimi
        return A.Compose([
            A.Rotate(limit=5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.3, p=0.7),
            A.HorizontalFlip(p=0.5),
        ])
    
    elif disease_type == 'pneumonia':
        # Pnömoni için daha fazla yoğunluk değişimi
        return A.Compose([
            A.Rotate(limit=10),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.4),
        ])
    
    return base_augs
def apply_offline_augmentations(image_dir, output_dir, class_counts, 
                              target_ratio=0.5, transform=None):
    """Nadir sınıflar için offline augmentasyon uygulama"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Sınıf başına hedef görüntü sayısını hesapla
    max_count = max(class_counts.values())
    augmentation_targets = {}
    
    for cls, count in class_counts.items():
        target = int(max_count * target_ratio)
        if count < target:
            augmentation_targets[cls] = target - count
    
    # Augmentasyon uygula
    for cls, target in augmentation_targets.items():
        class_images = [f for f in os.listdir(image_dir) 
                      if f.endswith('.png') and get_class(f) == cls]
        
        if not class_images:
            continue
            
        augmentations_per_image = max(1, target // len(class_images))
        
        for img_file in class_images:
            img_path = os.path.join(image_dir, img_file)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            for i in range(augmentations_per_image):
                transformed = transform(image=image)
                aug_image = transformed["image"]
                
                # Yeni isim oluştur
                base_name = os.path.splitext(img_file)[0]
                new_name = f"{base_name}_aug_{i}.png"
                
                # Kaydet
                cv2.imwrite(os.path.join(output_dir, new_name), aug_image)

def visualize_augmentations(image_path, transform, num_samples=5):
    """Bir görüntüye uygulanan augmentasyonları görselleştir"""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    fig, axes = plt.subplots(1, num_samples + 1, figsize=(15, 3))
    
    # Orijinal görüntü
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Orijinal')
    axes[0].axis('off')
    
    # Augmentasyon örnekleri
    for i in range(num_samples):
        transformed = transform(image=image)
        aug_image = transformed["image"]
        
        axes[i+1].imshow(aug_image, cmap='gray')
        axes[i+1].set_title(f'Aug #{i+1}')
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_samples.png')
    plt.show()