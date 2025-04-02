import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
def compute_class_weights(labels_df, method='inverse'):
    """Farklı yöntemlerle sınıf ağırlıkları hesaplama"""
    # Toplam örnek sayısını ve pozitif örnek sayılarını hesapla
    n_samples = len(labels_df)
    classes = labels_df.columns
    n_classes = len(classes)
    
    # Her sınıf için pozitif örnek sayısı
    pos_counts = labels_df.sum(axis=0).values
    
    if method == 'inverse':
        # Sınıf frekansının tersi ile ağırlıklandırma
        weights = n_samples / (n_classes * pos_counts)
    
    elif method == 'balanced':
        # Dengeli ağırlıklandırma
        weights = n_samples / (2 * pos_counts)
    
    elif method == 'effective_samples':
        # Etkili örnek sayısı yaklaşımı
        beta = 0.99
        weights = (1 - beta) / (1 - beta ** pos_counts)
    
    # Ağırlıkları 1.0 etrafında normalize et
    weights = weights / weights.sum() * n_classes
    
    return {class_name: weight for class_name, weight in zip(classes, weights)}


def weighted_bce_loss(outputs, targets, pos_weights):
    """Sınıf dengesizliği için ağırlıklı BCE kaybı"""
    loss = 0
    for i, pw in enumerate(pos_weights):
        # Binary cross entropy, her sınıf için ayrı hesaplanıyor
        bce = F.binary_cross_entropy_with_logits(
            outputs[:, i], targets[:, i], reduction='none')
        
        # Pozitif ve negatif örnekleri ayır
        pos_mask = targets[:, i] == 1
        neg_mask = targets[:, i] == 0
        
        # Ağırlıklı kayıp
        pos_loss = bce[pos_mask].mean() * pw if pos_mask.any() else 0
        neg_loss = bce[neg_mask].mean() if neg_mask.any() else 0
        
        class_loss = pos_loss + neg_loss
        loss += class_loss
        
    return loss / len(pos_weights)
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        """
        Sınıf dengesizliği için Focal Loss
        
        Args:
            alpha: Sınıf ağırlıkları. None veya her sınıf için bir değer içeren list/tensor
            gamma: Kolay/zor örnekleri ayarlamak için fokus parametresi
            reduction: 'mean', 'sum' veya 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none')
        
        # Modelin tahmin olasılığı
        pt = torch.exp(-BCE_loss)
        
        # alpha faktörü
        if self.alpha is not None:
            alpha_factor = self.alpha.view(-1, self.alpha.size(-1)).expand_as(targets)
            at = torch.where(targets == 1, alpha_factor, 1 - alpha_factor)
            BCE_loss = BCE_loss * at
        
        # Focal ağırlık hesaplama
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * BCE_loss
        
        # Azaltma tipine göre kayıp değerini döndür
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss