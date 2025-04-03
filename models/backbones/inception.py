# models/backbones/inception.py

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import Inception_V3_Weights

class InceptionBackbone(nn.Module):
    """
    Inception-v3 tabanlı öznitelik çıkarıcı omurga.
    """
    
    def __init__(self, pretrained=True, freeze_layers=True, aux_logits=False):
        """
        Args:
            pretrained: ImageNet ile önceden eğitilmiş ağırlıkları kullan
            freeze_layers: İlk katmanları dondur (transfer öğrenme için)
            aux_logits: Yardımcı sınıflandırıcıyı etkinleştir
        """
        super(InceptionBackbone, self).__init__()
        
        # Inception-v3 modelini yükle
        if pretrained:
            self.model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=aux_logits)
        else:
            self.model = models.inception_v3(weights=None, aux_logits=aux_logits)
        
        # Yardımcı sınıflandırıcı kullanımını kaydet
        self.aux_logits = aux_logits
        
        # Özellik boyutunu kaydet
        self.feature_dim = self.model.fc.in_features  # 2048
        
        # Son sınıflandırıcıyı kaldır (öznitelik çıkarıcı olarak kullanmak için)
        self.model.fc = nn.Identity()
        
        if aux_logits:
            self.model.AuxLogits.fc = nn.Identity()
        
        # İlk katmanları dondur (isteğe bağlı)
        if freeze_layers:
            # Inception-v3'ün ilk katmanlarını dondur
            # İlk 7 blok (17 katman) dondurulabilir
            freeze_layers = [
                self.model.Conv2d_1a_3x3,
                self.model.Conv2d_2a_3x3,
                self.model.Conv2d_2b_3x3,
                self.model.maxpool1,
                self.model.Conv2d_3b_1x1,
                self.model.Conv2d_4a_3x3,
                self.model.maxpool2
            ]
            
            for layer in freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Girdi görüntü tensörü [batch_size, channels, height, width]
            Not: Inception-v3 299x299 girdi bekler, ama 224x224 girdi de kabul eder
            
        Returns:
            features: Çıktı öznitelik tensörü [batch_size, feature_dim]
            aux: Yardımcı sınıflandırıcı çıktısı (aux_logits=True ise)
        """
        # Eğitim modunda aux_logits etkinse
        if self.aux_logits and self.training:
            # Yardımcı sınıflandırıcı çıktısı dahil
            features, aux = self.model(x)
            return features
        else:
            # Sadece ana öznitelikler
            return self.model(x)
        
    def extract_features(self, x):
        """
        Farklı katmanlardan öznitelik haritalarını çıkarır
        
        Args:
            x: Girdi görüntü tensörü [batch_size, channels, height, width]
            
        Returns:
            feature_maps: Farklı katmanlardan öznitelik haritaları sözlüğü
        """
        feature_maps = {}
        
        # Öznitelik haritalarını aşama aşama çıkar
        x = self.model.Conv2d_1a_3x3(x)
        feature_maps['Conv2d_1a_3x3'] = x
        
        x = self.model.Conv2d_2a_3x3(x)
        feature_maps['Conv2d_2a_3x3'] = x
        
        x = self.model.Conv2d_2b_3x3(x)
        feature_maps['Conv2d_2b_3x3'] = x
        
        x = self.model.maxpool1(x)
        
        x = self.model.Conv2d_3b_1x1(x)
        feature_maps['Conv2d_3b_1x1'] = x
        
        x = self.model.Conv2d_4a_3x3(x)
        feature_maps['Conv2d_4a_3x3'] = x
        
        x = self.model.maxpool2(x)
        
        # İlk Mixed (Inception) bloğa geçmeden önceki öznitelikler
        feature_maps['pre_mixed'] = x
        
        # Mixed 5b bloğu (ilk Inception bloğu)
        x = self.model.Mixed_5b(x)
        feature_maps['Mixed_5b'] = x
        
        # Mixed 5c bloğu
        x = self.model.Mixed_5c(x)
        feature_maps['Mixed_5c'] = x
        
        # Mixed 5d bloğu
        x = self.model.Mixed_5d(x)
        feature_maps['Mixed_5d'] = x
        
        # Mixed 6a bloğu
        x = self.model.Mixed_6a(x)
        feature_maps['Mixed_6a'] = x
        
        # Mixed 6b bloğu
        x = self.model.Mixed_6b(x)
        feature_maps['Mixed_6b'] = x
        
        # Mixed 6c bloğu
        x = self.model.Mixed_6c(x)
        feature_maps['Mixed_6c'] = x
        
        # Mixed 6d bloğu
        x = self.model.Mixed_6d(x)
        feature_maps['Mixed_6d'] = x
        
        # Mixed 6e bloğu
        x = self.model.Mixed_6e(x)
        feature_maps['Mixed_6e'] = x
        
        # Yardımcı sınıflandırıcı aktifse ve eğitim modundaysa
        if self.aux_logits and self.training:
            feature_maps['aux'] = self.model.AuxLogits(x)
        
        # Mixed 7a bloğu
        x = self.model.Mixed_7a(x)
        feature_maps['Mixed_7a'] = x
        
        # Mixed 7b bloğu
        x = self.model.Mixed_7b(x)
        feature_maps['Mixed_7b'] = x
        
        # Mixed 7c bloğu (son Inception bloğu)
        x = self.model.Mixed_7c(x)
        feature_maps['Mixed_7c'] = x
        
        # Adaptif ortalamalı havuzlama
        x = self.model.avgpool(x)
        feature_maps['avgpool'] = x
        
        # Dropout
        x = self.model.dropout(x)
        
        # Düzleştir
        x = torch.flatten(x, 1)
        feature_maps['flatten'] = x
        
        return feature_maps

# Test için
if __name__ == "__main__":
    # Yapay girdi tensörü oluştur
    x = torch.randn(4, 3, 299, 299)  # Inception-v3 için doğru boyut 299x299
    
    # Modeli test et
    model = InceptionBackbone(pretrained=True, aux_logits=False)
    print(f"Model yapısı: {model}")
    
    # Forward pass ile çıktı boyutunu kontrol et
    out = model(x)
    print(f"Çıktı tensör boyutu: {out.shape}")
    print(f"Öznitelik boyutu: {model.feature_dim}")
    
    # Eğitilebilir parametre sayısını hesapla
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Toplam parametre sayısı: {total_params:,}")
    print(f"Eğitilebilir parametre sayısı: {trainable_params:,}")
    print(f"Donmuş parametre oranı: {(total_params - trainable_params) / total_params:.2%}")
    
    # Öznitelik haritalarını çıkar
    feature_maps = model.extract_features(x)
    for name, fm in feature_maps.items():
        print(f"{name}: {fm.shape}")