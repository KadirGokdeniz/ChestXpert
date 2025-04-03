# models/backbones/vgg.py

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights, VGG19_Weights

class VGGBackbone(nn.Module):
    """
    VGG tabanlı öznitelik çıkarıcı omurga.
    VGG16 veya VGG19 kullanımını destekler.
    """
    
    def __init__(self, variant='vgg16', pretrained=True, freeze_layers=True):
        """
        Args:
            variant: VGG varyantı ('vgg16', 'vgg19')
            pretrained: ImageNet ile önceden eğitilmiş ağırlıkları kullan
            freeze_layers: İlk katmanları dondur (transfer öğrenme için)
        """
        super(VGGBackbone, self).__init__()
        
        # Model varyantını seç
        if variant == 'vgg16':
            if pretrained:
                self.model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            else:
                self.model = models.vgg16(weights=None)
        elif variant == 'vgg19':
            if pretrained:
                self.model = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
            else:
                self.model = models.vgg19(weights=None)
        else:
            raise ValueError(f"Desteklenmeyen VGG varyantı: {variant}")
        
        # Öznitelik çıkarıcı ve sınıflandırıcı kısımlarını ayır
        self.features = self.model.features
        self.avgpool = self.model.avgpool
        
        # Sınıflandırıcının son katmanını kaldır (öznitelik çıkarıcı olarak kullanmak için)
        # VGG'nin sınıflandırıcısı genellikle şöyle yapılandırılır:
        # Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear
        # Son Linear katmanını kaldıralım
        self.classifier = nn.Sequential(*list(self.model.classifier.children())[:-1])
        
        # Özellik boyutunu kaydet
        self.feature_dim = 4096  # VGG'nin son katmandan önceki boyutu
        
        # İlk katmanları dondur (isteğe bağlı)
        if freeze_layers:
            # İlk konvolüsyon bloklarını dondur
            freeze_indices = 24 if variant == 'vgg16' else 28  # İlk blokların yaklaşık sayısı
            
            for i, param in enumerate(self.features.parameters()):
                if i < freeze_indices:
                    param.requires_grad = False
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Girdi görüntü tensörü [batch_size, channels, height, width]
            
        Returns:
            features: Çıktı öznitelik tensörü [batch_size, feature_dim]
        """
        # Öznitelik haritalarını hesapla
        x = self.features(x)
        
        # Global average pooling uygula
        x = self.avgpool(x)
        
        # Düzleştir
        x = torch.flatten(x, 1)
        
        # Son FC katmanlar
        x = self.classifier(x)
        
        return x

# Test için
if __name__ == "__main__":
    # Yapay girdi tensörü oluştur
    x = torch.randn(4, 3, 224, 224)
    
    # Modeli test et
    model = VGGBackbone(variant='vgg16', pretrained=True)
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