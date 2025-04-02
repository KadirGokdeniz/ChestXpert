import numpy as np
import cv2
def normalize_xray(image, method='minmax', window=None):
    """Farklı normalizasyon stratejileri uygula"""
    if method == 'minmax':
        return (image - image.min()) / (image.max() - image.min())
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
        return (image - lower) / (upper - lower)